from __future__ import annotations

"""训练侧筛选外部预训练BIOT生理嵌入及其与驾驶风格的组合。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_base_biot", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT_DIR = RUN_DIR / "run_10_inner_biot_screen"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
EMBED_PATH = RUN_DIR / "cache" / "biot_pretrained_embeddings.npz"
RECENT_PATH = RUN_DIR / "tables" / "recent_style_features.csv"
ARMS = ("D", "B_biot", "BP_biot_prior_style", "BR_biot_recent_style", "BPR_all")


class FoldTransform:
    def __init__(self, use_biot, use_prior, use_recent):
        self.use_biot = use_biot
        self.use_prior = use_prior
        self.use_recent = use_recent
        self.imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=16, random_state=20260830) if use_biot else None

    def fit_transform(self, d, emb, prior, recent):
        parts = [d]
        if self.use_biot:
            e = np.nan_to_num(emb, nan=0.0)
            valid = np.isfinite(emb).all(axis=1, keepdims=True).astype(float)
            parts.extend([self.pca.fit_transform(e), valid])
        if self.use_prior:
            parts.append(prior)
        if self.use_recent:
            parts.append(recent)
        x = np.concatenate(parts, axis=1)
        return self.scaler.fit_transform(self.imp.fit_transform(x))

    def transform(self, d, emb, prior, recent):
        parts = [d]
        if self.use_biot:
            e = np.nan_to_num(emb, nan=0.0)
            valid = np.isfinite(emb).all(axis=1, keepdims=True).astype(float)
            parts.extend([self.pca.transform(e), valid])
        if self.use_prior:
            parts.append(prior)
        if self.use_recent:
            parts.append(recent)
        x = np.concatenate(parts, axis=1)
        return self.scaler.transform(self.imp.transform(x))


def arm_flags(arm):
    return (
        arm != "D",
        arm in {"BP_biot_prior_style", "BPR_all"},
        arm in {"BR_biot_recent_style", "BPR_all"},
    )


def main():
    print("[Run64 BIOT inner screen] 目标：检验外部预训练生理表示是否提供专家路由净信息。")
    pfull = pd.read_csv(base.PFULL_PATH).set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    recent = pd.read_csv(RECENT_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    cache = np.load(EMBED_PATH, allow_pickle=True)
    emb_ids = cache["event_uid"].astype(str)
    emb_map = {uid: i for i, uid in enumerate(emb_ids)}
    emb_all = cache["embedding"].astype(float)
    recent_cols = [c for c in recent.columns if c.startswith("recent_")]
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        idx = np.asarray([emb_map[x] for x in ids], dtype=int)
        emb = emb_all[idx]
        prior = features.reindex(pd.Index(ids))[base.STYLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        recent_x = recent.reindex(pd.Index(ids))[recent_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        curves = base.load_inner_curves(frame)
        truth = pfull.loc[ids, base.truth_columns()].to_numpy(float)
        target, losses = base.centered_regret(curves, truth)
        tau = max(0.5, float(np.median(np.std(losses, axis=1))))
        d = base.disagreement_features(curves)
        base_curve = curves.mean(axis=1)
        base_error = base.event_mae(base_curve, truth)
        for arm in ARMS:
            use_biot, use_prior, use_recent = arm_flags(arm)
            pred_curve = np.full_like(truth, np.nan)
            for inner_fold in sorted(frame["inner_fold"].unique()):
                val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                fit = ~val
                transform = FoldTransform(use_biot, use_prior, use_recent)
                xf = transform.fit_transform(d[fit], emb[fit], prior[fit], recent_x[fit])
                xv = transform.transform(d[val], emb[val], prior[val], recent_x[val])
                model = Ridge(alpha=100.0)
                model.fit(
                    xf,
                    target[fit],
                    sample_weight=base.subject_weights(frame.loc[fit, "subject"].astype(str).to_numpy()),
                )
                regret = model.predict(xv)
                regret -= regret.mean(axis=1, keepdims=True)
                regret = np.clip(regret, np.percentile(target[fit], 1, axis=0), np.percentile(target[fit], 99, axis=0))
                q = base.stable_softmax(-regret / tau)
                w = (1.0 - base.TRUST_UPDATE) / 3.0 + base.TRUST_UPDATE * q
                pred_curve[val] = base.curves_from_weights(curves[val], w)
            model_error = base.event_mae(pred_curve, truth)
            cert = base.certification(
                frame["subject"].astype(str).to_numpy(),
                base_error,
                model_error,
                base.SEED + 41000 + outer_fold * 100 + ARMS.index(arm),
            )
            cert.update(
                {
                    "outer_fold": outer_fold,
                    "arm": arm,
                    "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
                    "biot_valid_fraction": float(np.isfinite(emb).all(axis=1).mean()),
                }
            )
            rows.append(cert)
            print(
                f"outer={outer_fold} arm={arm} gain={cert['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={cert['bootstrap_ci_lower_deg']:+.4f}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_biot_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("arm", as_index=False)
        .agg(
            mean_gain=("subject_macro_mae_improvement_deg", "mean"),
            min_gain=("subject_macro_mae_improvement_deg", "min"),
            positive_outer_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top=("leave_top_subject_improvement_deg", "mean"),
        )
        .sort_values("mean_gain", ascending=False)
    )
    d_gain = float(summary.loc[summary["arm"] == "D", "mean_gain"].iloc[0])
    summary["gain_over_D"] = summary["mean_gain"] - d_gain
    summary.to_csv(OUT_DIR / "tables" / "inner_biot_summary.csv", index=False, encoding="utf-8-sig")
    best_biot = summary.loc[summary["arm"] != "D"].iloc[0]
    advance = bool(best_biot["gain_over_D"] >= 0.02 and best_biot["positive_outer_count"] >= 4)
    decision = {
        "status": "inner_only_biot_screen_complete",
        "outer_test_opened": False,
        "best_biot_arm": best_biot.to_dict(),
        "advance_biot": advance,
        "all_arms": summary.to_dict(orient="records"),
        "boundary": "External EEG checkpoint is a frozen representation prior, not evidence that physiology improves steering.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 外部预训练BIOT训练侧筛选\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n是否进入outer：**{advance}**\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

