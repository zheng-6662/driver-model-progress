from __future__ import annotations

"""训练侧筛选当前事件前65–5秒近期驾驶行为代理，而非prior-session静态风格。"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_base_recent_style", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT_DIR = RUN_DIR / "run_9_inner_recent_style"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)
RECENT_PATH = RUN_DIR / "tables" / "recent_style_features.csv"
MODELS = ("ridge", "extratrees")
ARMS = ("D", "R_recent_style", "P_physio", "PR_physio_recent")


def prepare_x(ids, curves, feature_index, recent_index, arm):
    parts = [base.disagreement_features(curves)]
    if arm in {"P_physio", "PR_physio_recent"}:
        parts.append(feature_index.reindex(pd.Index(ids))[base.PHYS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    if arm in {"R_recent_style", "PR_physio_recent"}:
        cols = [c for c in recent_index.columns if c.startswith("recent_")]
        parts.append(recent_index.reindex(pd.Index(ids))[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    return np.concatenate(parts, axis=1)


def fit_regret(model_name, x_fit, target, subjects, x_val, seed):
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xf = imp.fit_transform(x_fit)
    xv = imp.transform(x_val)
    weights = base.subject_weights(subjects)
    if model_name == "ridge":
        scaler = StandardScaler()
        xf = scaler.fit_transform(xf)
        xv = scaler.transform(xv)
        model = Ridge(alpha=100.0)
    else:
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=int(seed),
            n_jobs=4,
        )
    model.fit(xf, target, sample_weight=weights)
    pred = model.predict(xv)
    return np.clip(pred, np.percentile(target, 1, axis=0), np.percentile(target, 99, axis=0))


def main():
    print("[Run64 recent style screen] 目标：检验65–5秒近期行为是否提供专家路由净信息。")
    pfull = pd.read_csv(base.PFULL_PATH).set_index("event_uid", verify_integrity=True)
    feature_index = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    recent_index = pd.read_csv(RECENT_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        curves = base.load_inner_curves(frame)
        truth = pfull.loc[ids, base.truth_columns()].to_numpy(float)
        target, losses = base.centered_regret(curves, truth)
        tau = max(0.5, float(np.median(np.std(losses, axis=1))))
        base_curve = curves.mean(axis=1)
        base_error = base.event_mae(base_curve, truth)
        for arm in ARMS:
            x = prepare_x(ids, curves, feature_index, recent_index, arm)
            for model_name in MODELS:
                pred_curve = np.full_like(truth, np.nan)
                for inner_fold in sorted(frame["inner_fold"].unique()):
                    val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                    fit = ~val
                    regret = fit_regret(
                        model_name,
                        x[fit],
                        target[fit],
                        frame.loc[fit, "subject"].astype(str).to_numpy(),
                        x[val],
                        base.SEED + 35000 + outer_fold * 100 + inner_fold * 10 + MODELS.index(model_name),
                    )
                    regret -= regret.mean(axis=1, keepdims=True)
                    q = base.stable_softmax(-regret / tau)
                    w = (1.0 - base.TRUST_UPDATE) / 3.0 + base.TRUST_UPDATE * q
                    pred_curve[val] = base.curves_from_weights(curves[val], w)
                model_error = base.event_mae(pred_curve, truth)
                cert = base.certification(
                    frame["subject"].astype(str).to_numpy(),
                    base_error,
                    model_error,
                    base.SEED + 37000 + outer_fold * 100 + ARMS.index(arm) * 2 + MODELS.index(model_name),
                )
                cert.update(
                    {
                        "outer_fold": outer_fold,
                        "candidate": f"G_{arm}_{model_name}",
                        "arm": arm,
                        "model": model_name,
                        "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
                        "feature_dimension": int(x.shape[1]),
                    }
                )
                rows.append(cert)
                print(
                    f"outer={outer_fold} {cert['candidate']} gain={cert['subject_macro_mae_improvement_deg']:+.4f} "
                    f"ci_lo={cert['bootstrap_ci_lower_deg']:+.4f}"
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_recent_style_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("candidate", as_index=False)
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
    for model_name in MODELS:
        d_gain = float(summary.loc[summary["candidate"] == f"G_D_{model_name}", "mean_gain"].iloc[0])
        mask = summary["candidate"].str.endswith(f"_{model_name}")
        summary.loc[mask, "gain_over_D_same_model"] = summary.loc[mask, "mean_gain"] - d_gain
    summary.to_csv(OUT_DIR / "tables" / "inner_recent_style_summary.csv", index=False, encoding="utf-8-sig")
    recent_best = summary.loc[summary["candidate"].str.contains("R_recent_style")].iloc[0]
    advance = bool(recent_best["gain_over_D_same_model"] >= 0.02 and recent_best["positive_outer_count"] >= 4)
    decision = {
        "status": "inner_only_recent_style_screen_complete",
        "outer_test_opened": False,
        "best_recent_style": recent_best.to_dict(),
        "advance_recent_style": advance,
        "all_candidates": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 近期驾驶行为/风格训练侧筛选\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n是否进入outer：**{advance}**\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

