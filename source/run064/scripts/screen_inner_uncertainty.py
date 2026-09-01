from __future__ import annotations

"""训练侧筛选：生理/风格是否能提升冻结B_all3的事件级误差与置信度估计。

这条支线不修改均值曲线，只检验多模态是否能成为可靠的不确定性头。若相对专家
分歧控制没有增量，就不能用“选择性预测”替代失败的均值增量主张。
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_base_unc", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT_DIR = RUN_DIR / "run_6_inner_uncertainty"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)

ARMS = {
    "D": (),
    "P": ("phys",),
    "S": ("style",),
    "PS": ("phys", "style"),
}
HEADS = ("ridge", "hist")


def make_x(ids, curves, feature_index, blocks):
    rows = feature_index.reindex(pd.Index(ids.astype(str)))
    parts = [base.disagreement_features(curves)]
    for block in blocks:
        cols = base.PHYS_COLS if block == "phys" else base.STYLE_COLS
        parts.append(rows[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    return np.concatenate(parts, axis=1)


def fit_predict(head, x_fit, y_fit, subjects_fit, x_val):
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xf = imp.fit_transform(x_fit)
    xv = imp.transform(x_val)
    weights = base.subject_weights(subjects_fit)
    if head == "ridge":
        scaler = StandardScaler()
        xf = scaler.fit_transform(xf)
        xv = scaler.transform(xv)
        model = Ridge(alpha=100.0)
    else:
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=20260830,
        )
    model.fit(xf, y_fit, sample_weight=weights)
    return model.predict(xv)


def subject_macro_spearman(subjects, truth, pred):
    values = []
    frame = pd.DataFrame({"subject": subjects, "truth": truth, "pred": pred})
    for _, g in frame.groupby("subject"):
        if len(g) >= 5 and g["truth"].nunique() > 1 and g["pred"].nunique() > 1:
            rho = spearmanr(g["truth"], g["pred"]).statistic
            if np.isfinite(rho):
                values.append(float(rho))
    return float(np.mean(values)) if values else float("nan")


def selective_mae(subjects, actual_error, predicted_risk, coverage=0.8):
    values = []
    frame = pd.DataFrame({"subject": subjects, "error": actual_error, "risk": predicted_risk})
    for _, g in frame.groupby("subject"):
        keep = max(1, int(np.ceil(len(g) * coverage)))
        values.append(float(g.nsmallest(keep, "risk")["error"].mean()))
    return float(np.mean(values))


def main():
    print("[Run64 uncertainty inner screen] 目标：检验生理/风格是否改进冻结曲线的误差风险预测。")
    pfull = pd.read_csv(base.PFULL_PATH).set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        curves = base.load_inner_curves(frame)
        truth = pfull.loc[ids, base.truth_columns()].to_numpy(float)
        base_curve = curves.mean(axis=1)
        actual_error = base.event_mae(base_curve, truth)
        target = np.log1p(actual_error)
        for arm, blocks in ARMS.items():
            x = make_x(ids, curves, features, blocks)
            for head in HEADS:
                pred_log = np.full(len(frame), np.nan)
                high_label = np.zeros(len(frame), dtype=int)
                for inner_fold in sorted(frame["inner_fold"].unique()):
                    val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                    fit = ~val
                    pred_log[val] = fit_predict(
                        head,
                        x[fit],
                        target[fit],
                        frame.loc[fit, "subject"].astype(str).to_numpy(),
                        x[val],
                    )
                    threshold = float(np.quantile(actual_error[fit], 0.75))
                    high_label[val] = (actual_error[val] >= threshold).astype(int)
                pred_risk = np.expm1(pred_log)
                auc = float(roc_auc_score(high_label, pred_risk)) if len(np.unique(high_label)) == 2 else float("nan")
                row = {
                    "outer_fold": outer_fold,
                    "candidate": f"U_{arm}_{head}",
                    "arm": arm,
                    "head": head,
                    "risk_log_mae": float(np.mean(np.abs(pred_log - target))),
                    "pooled_spearman": float(spearmanr(actual_error, pred_risk).statistic),
                    "subject_macro_spearman": subject_macro_spearman(
                        frame["subject"].astype(str).to_numpy(), actual_error, pred_risk
                    ),
                    "high_error_auc": auc,
                    "selective_subject_macro_mae_at_80pct": selective_mae(
                        frame["subject"].astype(str).to_numpy(), actual_error, pred_risk, 0.8
                    ),
                    "full_subject_macro_mae": float(
                        pd.DataFrame({"subject": frame["subject"], "error": actual_error})
                        .groupby("subject")["error"]
                        .mean()
                        .mean()
                    ),
                    "feature_dimension": int(x.shape[1]),
                }
                rows.append(row)
                print(
                    f"outer={outer_fold} {row['candidate']} auc={auc:.4f} "
                    f"rho={row['subject_macro_spearman']:.4f} sel80={row['selective_subject_macro_mae_at_80pct']:.4f}"
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_uncertainty_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby("candidate", as_index=False)
        .agg(
            mean_high_error_auc=("high_error_auc", "mean"),
            min_high_error_auc=("high_error_auc", "min"),
            mean_subject_macro_spearman=("subject_macro_spearman", "mean"),
            mean_selective_mae_80=("selective_subject_macro_mae_at_80pct", "mean"),
            mean_full_mae=("full_subject_macro_mae", "mean"),
            mean_risk_log_mae=("risk_log_mae", "mean"),
            feature_dimension=("feature_dimension", "max"),
        )
        .sort_values(["mean_high_error_auc", "mean_subject_macro_spearman"], ascending=False)
    )
    # 同一head内相对D控制的净增量。
    for head in HEADS:
        control = float(summary.loc[summary["candidate"] == f"U_D_{head}", "mean_high_error_auc"].iloc[0])
        for arm in ("P", "S", "PS"):
            key = f"U_{arm}_{head}"
            summary.loc[summary["candidate"] == key, "auc_gain_over_D"] = (
                summary.loc[summary["candidate"] == key, "mean_high_error_auc"] - control
            )
    summary.to_csv(OUT_DIR / "tables" / "inner_uncertainty_summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "inner_only_uncertainty_screen_complete",
        "outer_test_opened": False,
        "best_candidate": summary.iloc[0].to_dict(),
        "all_candidates": summary.to_dict(orient="records"),
        "advance_rule": "Both P and S must each improve high-error AUC over the same-head D control by at least 0.02 before any outer evaluation.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 多模态误差风险头训练侧筛选\n\n"
        "不修改均值曲线，仅检验生理/风格对事件级误差风险的净信息。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

