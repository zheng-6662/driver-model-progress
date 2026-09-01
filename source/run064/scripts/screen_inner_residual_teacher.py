from __future__ import annotations

"""只在inner-OOF上检验另一种教师接口：多模态预测 B_all3 剩余曲线残差。

第一波专家后悔学生表明生理/风格没有超过分歧-only。这里不打开outer test，先问：
两类模态是否更适合作为小幅曲线校准，而不是专家路由。固定比较强收缩Ridge和
较保守ExtraTrees残差头；修正强度固定25%。
"""

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


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_local("run64_base_for_residual", RUN_DIR / "experiment.py")
screen = load_local("run64_screen_for_residual", RUN_DIR / "scripts" / "screen_inner_candidates.py")

OUT_DIR = RUN_DIR / "run_3_inner_residual_screen"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)

BLOCKS = {
    "D": (),
    "P": ("phys",),
    "S": ("style",),
    "PS": ("phys", "style"),
    "H": ("history",),
    "HP": ("history", "phys"),
    "HS": ("history", "style"),
    "HPS": ("history", "phys", "style"),
}
MODELS = ("ridge", "extratrees")
CORRECTION_STRENGTH = 0.25


def make_x(ids, curves, feature_index, history, blocks):
    rows = feature_index.reindex(pd.Index(ids.astype(str)))
    parts = [base.disagreement_features(curves)]
    for block in blocks:
        if block == "phys":
            parts.append(rows[base.PHYS_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        elif block == "style":
            parts.append(rows[base.STYLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float))
        elif block == "history":
            parts.append(history)
        else:
            raise ValueError(block)
    return np.concatenate(parts, axis=1)


def fit_predict(model_name, x_fit, y_fit, subjects_fit, x_val, seed):
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xf = imputer.fit_transform(x_fit)
    xv = imputer.transform(x_val)
    weights = base.subject_weights(subjects_fit)
    if model_name == "ridge":
        scaler = StandardScaler()
        xf = scaler.fit_transform(xf)
        xv = scaler.transform(xv)
        model = Ridge(alpha=100.0, fit_intercept=True)
        model.fit(xf, y_fit, sample_weight=weights)
    else:
        model = ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            max_features=0.7,
            random_state=seed,
            n_jobs=4,
        )
        model.fit(xf, y_fit, sample_weight=weights)
    pred = model.predict(xv)
    lo = np.nanpercentile(y_fit, 1, axis=0)
    hi = np.nanpercentile(y_fit, 99, axis=0)
    return np.clip(pred, lo, hi)


def evaluate(outer_fold, block_name, blocks, model_name, frame, curves, truth, feature_index, pfull_index):
    base_curve = curves.mean(axis=1)
    residual = truth - base_curve
    history = screen.history_features(
        frame["event_uid"].astype(str).to_numpy(), curves, truth, pfull_index
    )
    x = make_x(frame["event_uid"].astype(str).to_numpy(), curves, feature_index, history, blocks)
    pred = np.full_like(truth, np.nan)
    for inner_fold in sorted(frame["inner_fold"].unique()):
        val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
        fit = ~val
        correction = fit_predict(
            model_name,
            x[fit],
            residual[fit],
            frame.loc[fit, "subject"].astype(str).to_numpy(),
            x[val],
            base.SEED + outer_fold * 1000 + inner_fold * 10 + list(MODELS).index(model_name),
        )
        pred[val] = base_curve[val] + CORRECTION_STRENGTH * correction
    b_error = base.event_mae(base_curve, truth)
    m_error = base.event_mae(pred, truth)
    cert = base.certification(
        frame["subject"].astype(str).to_numpy(),
        b_error,
        m_error,
        base.SEED + 13000 + outer_fold * 100 + list(BLOCKS).index(block_name) * 2 + list(MODELS).index(model_name),
    )
    cert.update(
        {
            "outer_fold": outer_fold,
            "feature_block": block_name,
            "residual_model": model_name,
            "candidate": f"R_{block_name}_{model_name}",
            "feature_dimension_before_imputation": int(x.shape[1]),
            "event_improved_fraction": float(np.mean(m_error < b_error - 1e-12)),
            "correction_strength": CORRECTION_STRENGTH,
        }
    )
    return cert


def main():
    print("[Run64 residual inner screen] 目标：判断生理/风格是否更适合小幅曲线残差教师接口。")
    pfull = pd.read_csv(base.PFULL_PATH)
    pfull_index = pfull.set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        curves = base.load_inner_curves(frame)
        truth = pfull_index.loc[frame["event_uid"], base.truth_columns()].to_numpy(float)
        for block_name, blocks in BLOCKS.items():
            for model_name in MODELS:
                row = evaluate(
                    outer_fold,
                    block_name,
                    blocks,
                    model_name,
                    frame,
                    curves,
                    truth,
                    features,
                    pfull_index,
                )
                rows.append(row)
                print(
                    f"outer={outer_fold} candidate={row['candidate']} "
                    f"gain={row['subject_macro_mae_improvement_deg']:+.4f} "
                    f"ci_lo={row['bootstrap_ci_lower_deg']:+.4f}"
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_residual_candidate_screen.csv", index=False, encoding="utf-8-sig")
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
            feature_dimension=("feature_dimension_before_imputation", "max"),
        )
        .sort_values(["mean_inner_gain_deg", "outer_positive_count"], ascending=False)
    )
    summary.to_csv(OUT_DIR / "tables" / "inner_residual_candidate_summary.csv", index=False, encoding="utf-8-sig")
    decision = {
        "status": "inner_only_residual_screen_complete",
        "outer_test_new_candidates_opened": False,
        "best_candidate": summary.iloc[0].to_dict(),
        "all_candidates": summary.to_dict(orient="records"),
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 多模态曲线残差教师训练侧筛选\n\n"
        "本表只使用outer训练侧inner-OOF，不读取新候选outer-test结果。\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

