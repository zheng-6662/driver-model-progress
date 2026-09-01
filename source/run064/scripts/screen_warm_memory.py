from __future__ import annotations

"""只在inner-OOF上筛选严格暖启动的残差记忆。

允许的信息变化只有一项：同一驾驶员先前事件的真实20点曲线在 `anchor+1s` 后已经
揭晓。生理仅用于在这些历史事件中做状态相似加权；既往session风格仅用于从
meta-fit其他驾驶员中构造跨人残差先验。所有候选都固定25%修正，不打开outer test。
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_local("run64_base_warm", RUN_DIR / "experiment.py")
OUT_DIR = RUN_DIR / "run_4_inner_warm_memory"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)

STRENGTH = 0.25
STYLE_K = 50
PHYS_VALUE_COLS = base.PHYS_COLS[:12]
STYLE_VALUE_COLS = base.STYLE_COLS[:16]
CANDIDATES = ["H_mean", "H_ewma", "HP_state_memory", "S_style_prior", "HS_mean_prior", "HPS_state_prior"]


def absolute_anchor(meta: pd.DataFrame) -> np.ndarray:
    session = pd.to_datetime(meta["session_stamp"], format="%Y_%m_%d_%H_%M_%S", errors="coerce")
    return session.astype("int64").to_numpy(float) / 1e9 + meta["primary_release_s"].to_numpy(float)


def fit_transform_features(fit_values: np.ndarray, val_values: np.ndarray):
    imp = SimpleImputer(strategy="median", add_indicator=False, keep_empty_features=True)
    scaler = StandardScaler()
    f = scaler.fit_transform(imp.fit_transform(fit_values))
    v = scaler.transform(imp.transform(val_values))
    return f, v


def style_prior(
    style_fit: np.ndarray,
    residual_fit: np.ndarray,
    style_val: np.ndarray,
) -> np.ndarray:
    """按既往session风格在meta-fit其他被试事件中检索残差先验。"""
    out = np.zeros((len(style_val), residual_fit.shape[1]), dtype=float)
    k = min(STYLE_K, len(style_fit))
    for i, row in enumerate(style_val):
        dist = np.mean((style_fit - row) ** 2, axis=1)
        idx = np.argpartition(dist, k - 1)[:k]
        local = dist[idx]
        scale = max(float(np.median(local)), 1e-6)
        w = np.exp(-local / scale)
        w = w / max(float(w.sum()), 1e-12)
        out[i] = np.sum(residual_fit[idx] * w[:, None], axis=0)
    return out


def sequential_memory(
    val_meta: pd.DataFrame,
    val_residual: np.ndarray,
    phys_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回历史均值、EWMA和生理相似加权残差；每个输出都在当前事件预测前形成。"""
    mean_out = np.zeros_like(val_residual)
    ewma_out = np.zeros_like(val_residual)
    phys_out = np.zeros_like(val_residual)
    meta = val_meta.copy()
    meta["row_index"] = np.arange(len(meta))
    meta["anchor_abs_s"] = absolute_anchor(meta)
    for _, group in meta.groupby("subject", sort=False):
        order = group.sort_values(["anchor_abs_s", "primary_release_s", "row_index"])
        revealed_residuals: list[np.ndarray] = []
        revealed_phys: list[np.ndarray] = []
        pending: list[tuple[float, np.ndarray, np.ndarray]] = []
        ewma = np.zeros(val_residual.shape[1], dtype=float)
        for row in order.itertuples():
            now = float(row.anchor_abs_s)
            ready = [x for x in pending if x[0] <= now + 1e-9]
            pending = [x for x in pending if x[0] > now + 1e-9]
            for _, residual, phys in sorted(ready, key=lambda x: x[0]):
                ewma = residual.copy() if not revealed_residuals else 0.5 * residual + 0.5 * ewma
                revealed_residuals.append(residual)
                revealed_phys.append(phys)
            idx = int(row.row_index)
            if revealed_residuals:
                r = np.stack(revealed_residuals)
                mean_out[idx] = r.mean(axis=0)
                ewma_out[idx] = ewma
                p = np.stack(revealed_phys)
                dist = np.mean((p - phys_val[idx]) ** 2, axis=1)
                scale = max(float(np.median(dist)), 1e-6)
                w = np.exp(-dist / scale)
                w = w / max(float(w.sum()), 1e-12)
                phys_out[idx] = np.sum(r * w[:, None], axis=0)
            pending.append((now + 1.0, val_residual[idx], phys_val[idx]))
    return mean_out, ewma_out, phys_out


def evaluate_outer_context(outer_fold, frame, curves, truth, feature_index, pfull_index):
    rows = []
    base_curve = curves.mean(axis=1)
    residual = truth - base_curve
    base_error = base.event_mae(base_curve, truth)
    ids = frame["event_uid"].astype(str).to_numpy()
    meta_all = pfull_index.reindex(pd.Index(ids))[["subject", "session_stamp", "primary_release_s"]].copy()
    raw_phys = feature_index.reindex(pd.Index(ids))[PHYS_VALUE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    raw_style = feature_index.reindex(pd.Index(ids))[STYLE_VALUE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    pred_by_candidate = {c: np.full_like(truth, np.nan) for c in CANDIDATES}
    for inner_fold in sorted(frame["inner_fold"].unique()):
        val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
        fit = ~val
        phys_fit, phys_val = fit_transform_features(raw_phys[fit], raw_phys[val])
        style_fit, style_val = fit_transform_features(raw_style[fit], raw_style[val])
        prior = style_prior(style_fit, residual[fit], style_val)
        mean_mem, ewma_mem, phys_mem = sequential_memory(
            meta_all.iloc[np.where(val)[0]].reset_index(drop=True), residual[val], phys_val
        )
        corrections = {
            "H_mean": mean_mem,
            "H_ewma": ewma_mem,
            "HP_state_memory": phys_mem,
            "S_style_prior": prior,
            "HS_mean_prior": 0.75 * mean_mem + 0.25 * prior,
            "HPS_state_prior": 0.75 * phys_mem + 0.25 * prior,
        }
        for candidate, correction in corrections.items():
            pred_by_candidate[candidate][val] = base_curve[val] + STRENGTH * correction
    for candidate, pred in pred_by_candidate.items():
        model_error = base.event_mae(pred, truth)
        cert = base.certification(
            frame["subject"].astype(str).to_numpy(),
            base_error,
            model_error,
            base.SEED + 17000 + outer_fold * 100 + CANDIDATES.index(candidate),
        )
        cert.update(
            {
                "outer_fold": outer_fold,
                "candidate": candidate,
                "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
                "correction_strength": STRENGTH,
            }
        )
        rows.append(cert)
    return rows


def main():
    print("[Run64 warm inner screen] 目标：筛选已揭晓历史、生理状态记忆和既往session风格先验。")
    pfull = pd.read_csv(base.PFULL_PATH)
    pfull_index = pfull.set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        curves = base.load_inner_curves(frame)
        truth = pfull_index.loc[frame["event_uid"], base.truth_columns()].to_numpy(float)
        current = evaluate_outer_context(
            outer_fold, frame, curves, truth, features, pfull_index
        )
        rows.extend(current)
        for row in current:
            print(
                f"outer={outer_fold} candidate={row['candidate']} "
                f"gain={row['subject_macro_mae_improvement_deg']:+.4f} "
                f"ci_lo={row['bootstrap_ci_lower_deg']:+.4f}"
            )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_warm_candidate_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("candidate", as_index=False)
        .agg(
            mean_inner_gain_deg=("subject_macro_mae_improvement_deg", "mean"),
            min_inner_gain_deg=("subject_macro_mae_improvement_deg", "min"),
            outer_positive_count=("subject_macro_mae_improvement_deg", lambda x: int((x > 0).sum())),
            outer_base_gate_pass_count=("base_gate_pass", "sum"),
            mean_ci_lower_deg=("bootstrap_ci_lower_deg", "mean"),
            mean_leave_top_gain_deg=("leave_top_subject_improvement_deg", "mean"),
            mean_event_improved_fraction=("event_improved_fraction", "mean"),
        )
        .sort_values(["mean_inner_gain_deg", "outer_positive_count"], ascending=False)
    )
    summary.to_csv(OUT_DIR / "tables" / "inner_warm_candidate_summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "inner_only_warm_screen_complete",
        "outer_test_new_candidates_opened": False,
        "best_candidate": summary.iloc[0].to_dict(),
        "all_candidates": summary.to_dict(orient="records"),
        "selection_rule": "No outer-test performance was used. Warm candidates require at least 4/5 positive outer contexts before final evaluation.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 严格暖启动训练侧筛选\n\n"
        "本表只使用outer训练侧inner-OOF；当前事件预测后，真值到达才更新下一事件。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

