from __future__ import annotations

"""训练侧筛选：生理/风格是否分别解释幅值、峰时、头段或尾段残差。"""

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
SPEC = importlib.util.spec_from_file_location("run64_base_struct", RUN_DIR / "experiment.py")
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

OUT_DIR = RUN_DIR / "run_8_inner_structured_targets"
(OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "outputs").mkdir(parents=True, exist_ok=True)

ARMS = {"D": (), "P": ("phys",), "S": ("style",), "PS": ("phys", "style")}
HEADS = ("ridge", "extratrees")
TARGET_NAMES = ("peak_amplitude_residual_deg", "peak_time_residual_s", "head5_mean_residual_deg", "tail5_mean_residual_deg")


def make_x(ids, curves, feature_index, blocks):
    rows = feature_index.reindex(pd.Index(ids.astype(str)))
    parts = [base.disagreement_features(curves)]
    for block in blocks:
        cols = base.PHYS_COLS if block == "phys" else base.STYLE_COLS
        parts.append(rows[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    return np.concatenate(parts, axis=1)


def targets(base_curve, truth):
    true_peak = np.max(np.abs(truth), axis=1)
    base_peak = np.max(np.abs(base_curve), axis=1)
    true_time = np.argmax(np.abs(truth), axis=1) * 0.05 + 0.05
    base_time = np.argmax(np.abs(base_curve), axis=1) * 0.05 + 0.05
    return np.column_stack(
        [
            true_peak - base_peak,
            true_time - base_time,
            np.mean(truth[:, :5] - base_curve[:, :5], axis=1),
            np.mean(truth[:, -5:] - base_curve[:, -5:], axis=1),
        ]
    )


def fit_predict(head, x_fit, y_fit, subjects, x_val, seed):
    imp = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xf = imp.fit_transform(x_fit)
    xv = imp.transform(x_val)
    weights = base.subject_weights(subjects)
    if head == "ridge":
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
    model.fit(xf, y_fit, sample_weight=weights)
    pred = model.predict(xv)
    return np.clip(pred, np.percentile(y_fit, 1, axis=0), np.percentile(y_fit, 99, axis=0))


def subject_macro_mae(subjects, truth, pred):
    frame = pd.DataFrame({"subject": subjects})
    values = []
    for j in range(truth.shape[1]):
        frame["err"] = np.abs(truth[:, j] - pred[:, j])
        values.append(float(frame.groupby("subject")["err"].mean().mean()))
    return np.asarray(values)


def main():
    print("[Run64 structured target screen] 目标：检验模态对幅值、峰时、头段、尾段残差的净信息。")
    pfull = pd.read_csv(base.PFULL_PATH).set_index("event_uid", verify_integrity=True)
    features = pd.read_csv(base.FEATURE_PATH).set_index("event_uid", verify_integrity=True)
    inner_all = pd.read_csv(base.INNER_PATH)
    rows = []
    for outer_fold in range(1, 6):
        frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        ids = frame["event_uid"].astype(str).to_numpy()
        curves = base.load_inner_curves(frame)
        truth_curve = pfull.loc[ids, base.truth_columns()].to_numpy(float)
        base_curve = curves.mean(axis=1)
        y = targets(base_curve, truth_curve)
        zero = np.zeros_like(y)
        zero_mae = subject_macro_mae(frame["subject"].astype(str).to_numpy(), y, zero)
        for arm, blocks in ARMS.items():
            x = make_x(ids, curves, features, blocks)
            for head in HEADS:
                pred = np.full_like(y, np.nan)
                for inner_fold in sorted(frame["inner_fold"].unique()):
                    val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
                    fit = ~val
                    pred[val] = fit_predict(
                        head,
                        x[fit],
                        y[fit],
                        frame.loc[fit, "subject"].astype(str).to_numpy(),
                        x[val],
                        base.SEED + 31000 + outer_fold * 100 + inner_fold * 10 + list(HEADS).index(head),
                    )
                mae = subject_macro_mae(frame["subject"].astype(str).to_numpy(), y, pred)
                for j, target_name in enumerate(TARGET_NAMES):
                    rows.append(
                        {
                            "outer_fold": outer_fold,
                            "candidate": f"K_{arm}_{head}",
                            "arm": arm,
                            "head": head,
                            "target": target_name,
                            "zero_residual_subject_macro_mae": float(zero_mae[j]),
                            "model_subject_macro_mae": float(mae[j]),
                            "improvement_vs_zero": float(zero_mae[j] - mae[j]),
                            "feature_dimension": int(x.shape[1]),
                        }
                    )
                print(
                    f"outer={outer_fold} K_{arm}_{head} gains="
                    + ",".join(f"{x:+.4f}" for x in (zero_mae - mae))
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "tables" / "inner_structured_target_screen.csv", index=False, encoding="utf-8-sig")
    summary = (
        detail.groupby(["candidate", "target"], as_index=False)
        .agg(
            mean_gain=("improvement_vs_zero", "mean"),
            min_gain=("improvement_vs_zero", "min"),
            outer_positive_count=("improvement_vs_zero", lambda x: int((x > 0).sum())),
            mean_model_mae=("model_subject_macro_mae", "mean"),
            mean_zero_mae=("zero_residual_subject_macro_mae", "mean"),
        )
    )
    # 与同一head的D控制比较，而不是只与零修正比较。
    summary["gain_over_D_same_head"] = np.nan
    for head in HEADS:
        for target_name in TARGET_NAMES:
            control = float(
                summary.loc[
                    (summary["candidate"] == f"K_D_{head}") & (summary["target"] == target_name),
                    "mean_gain",
                ].iloc[0]
            )
            mask = summary["candidate"].str.endswith(f"_{head}") & (summary["target"] == target_name)
            summary.loc[mask, "gain_over_D_same_head"] = summary.loc[mask, "mean_gain"] - control
    summary = summary.sort_values(["gain_over_D_same_head", "mean_gain"], ascending=False)
    summary.to_csv(OUT_DIR / "tables" / "inner_structured_target_summary.csv", index=False, encoding="utf-8-sig")
    phys_best = summary.loc[summary["candidate"].str.startswith("K_P_")].iloc[0].to_dict()
    style_best = summary.loc[summary["candidate"].str.startswith("K_S_")].iloc[0].to_dict()
    advance = bool(
        phys_best["gain_over_D_same_head"] > 0
        and phys_best["outer_positive_count"] >= 4
        and style_best["gain_over_D_same_head"] > 0
        and style_best["outer_positive_count"] >= 4
    )
    decision = {
        "status": "inner_only_structured_target_screen_complete",
        "outer_test_opened": False,
        "best_physio_target": phys_best,
        "best_style_target": style_best,
        "advance_to_structured_curve_calibration": advance,
        "boundary": "No result from this screen is an outer-test curve improvement.",
    }
    (OUT_DIR / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "outputs" / "RESULT_CN.md").write_text(
        "# Run64 结构化残差目标训练侧筛选\n\n"
        + summary.to_markdown(index=False, floatfmt=".4f")
        + f"\n\n是否进入结构化曲线校准：**{advance}**\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

