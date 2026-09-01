from __future__ import annotations

"""Run64 第一波：双时间尺度解析教师 -> 低容量专家权重学生。

本轮不重训 M2/M3/M4。训练标签来自每个 outer 上 Run63 已保存的 inner-OOF
三专家预测与真实20点曲线：教师把三个专家的逐事件损失中心化，学生只学习
预测这些相对后悔。生理与驾驶风格均不直接生成未来曲线。

这是发展性 OOF。四个固定学生臂分别回答：专家分歧本身、生理、风格以及两者
联合是否能识别“该信哪个冻结车辆专家”。
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
RUN63 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run63_protected_residual_and_soft_gating_20260829" / "run_1"
RUN60 = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run60_gbm_vs_noise_floor_20260828"
PFULL_PATH = (
    ROOT
    / "05_rebuild_from_raw_20260511"
    / "03_baselines"
    / "run57_a_full_release_population_causal_baseline_20260827"
    / "run_1"
    / "tables"
    / "pfull_event_manifest.csv"
)
FEATURE_PATH = RUN_DIR / "tables" / "multimodal_features.csv"
INNER_PATH = RUN63 / "tables" / "inner_oof_base_predictions.csv"

SEED = 20260830
BOOTSTRAP_REPS = 2000
ALPHA = 100.0
TRUST_UPDATE = 0.25
MIN_INNER_TOTAL_GAIN_DEG = 0.02
MIN_MODAL_GAIN_OVER_DISAGREEMENT_DEG = 0.02

PHYS_COLS = [
    "phys_emg_log_rms_z_2s",
    "phys_emg_envelope_slope_2s",
    "phys_emg_burst_fraction_2s",
    "phys_eda_tonic_z_30s",
    "phys_eda_phasic_area_per_s_15s",
    "phys_eda_phasic_slope_30s",
    "phys_hr_z_30s",
    "phys_hr_slope_15s",
    "phys_log_rmssd_30s",
    "phys_resp_rate_30s",
    "phys_resp_amplitude_z_30s",
    "phys_resp_interval_irregularity_30s",
    "phys_emg_valid_fraction_2s",
    "phys_eda_valid_fraction_30s",
    "phys_ecg_valid_fraction_30s",
    "phys_resp_valid_fraction_30s",
    "physio_source_available",
]
STYLE_COLS = [
    "style_steer_abs_mean__median",
    "style_steer_abs_p90__median",
    "style_steer_rate_abs_mean__median",
    "style_steer_rate_abs_p90__median",
    "style_brake_usage_ratio__median",
    "style_hard_brake_ratio__median",
    "style_throttle_usage_ratio__median",
    "style_hard_accel_ratio__median",
    "style_speed_mean__median",
    "style_speed_std__median",
    "style_ax_abs_mean__median",
    "style_ay_abs_mean__median",
    "style_yaw_rate_abs_mean__median",
    "style_lane_offset_abs_mean__median",
    "style_lane_offset_std__median",
    "style_log1p_prior_session_count",
    "style_available",
]

ARM_BLOCKS = {
    "D_disagreement": (),
    "P_physio": ("phys",),
    "S_style": ("style",),
    "PS_physio_style": ("phys", "style"),
}


@dataclass
class Student:
    imputer: SimpleImputer
    scaler: StandardScaler
    ridge: Ridge
    lower: np.ndarray
    upper: np.ndarray
    tau: float


def truth_columns() -> list[str]:
    return [f"target_t{i:02d}_deg" for i in range(1, 21)]


def pred_columns(prefix: str) -> list[str]:
    return [f"{prefix}_pred_t{i:02d}_deg" for i in range(1, 21)]


def stable_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(np.clip(z, -60.0, 60.0))
    return e / np.sum(e, axis=axis, keepdims=True)


def subject_weights(subjects: np.ndarray) -> np.ndarray:
    s = pd.Series(subjects.astype(str))
    counts = s.value_counts()
    w = s.map(lambda x: 1.0 / counts[x]).to_numpy(float)
    return w / np.mean(w)


def disagreement_features(curves: np.ndarray) -> np.ndarray:
    """从三条冻结专家曲线构造不依赖真值的低维分歧。"""
    point_std = np.std(curves, axis=1)
    pair01 = np.mean(np.abs(curves[:, 0] - curves[:, 1]), axis=1)
    pair02 = np.mean(np.abs(curves[:, 0] - curves[:, 2]), axis=1)
    pair12 = np.mean(np.abs(curves[:, 1] - curves[:, 2]), axis=1)
    peaks = np.max(np.abs(curves), axis=2)
    peak_times = np.argmax(np.abs(curves), axis=2).astype(float) * 0.05 + 0.05
    endpoints = curves[:, :, -1]
    areas = np.trapz(curves, dx=0.05, axis=2)
    return np.column_stack(
        [
            np.mean(point_std, axis=1),
            np.median(point_std, axis=1),
            np.mean(point_std[:, :5], axis=1),
            np.mean(point_std[:, 5:15], axis=1),
            np.mean(point_std[:, -5:], axis=1),
            pair01,
            pair02,
            pair12,
            np.std(peaks, axis=1),
            np.std(peak_times, axis=1),
            np.std(endpoints, axis=1),
            np.std(areas, axis=1),
        ]
    )


def event_feature_matrix(
    event_ids: list[str] | np.ndarray,
    curves: np.ndarray,
    feature_index: pd.DataFrame,
    arm: str,
) -> np.ndarray:
    ids = pd.Index(np.asarray(event_ids, dtype=str))
    rows = feature_index.reindex(ids)
    if rows.index.has_duplicates or rows.shape[0] != len(ids):
        raise ValueError("feature event join failed")
    blocks = [disagreement_features(curves)]
    for block in ARM_BLOCKS[arm]:
        columns = PHYS_COLS if block == "phys" else STYLE_COLS
        blocks.append(rows[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float))
    return np.concatenate(blocks, axis=1)


def centered_regret(curves: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    losses = np.mean(np.abs(curves - truth[:, None, :]), axis=2)
    return losses - np.mean(losses, axis=1, keepdims=True), losses


def fit_student(x: np.ndarray, target: np.ndarray, losses: np.ndarray, subjects: np.ndarray) -> Student:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    xi = imputer.fit_transform(x)
    scaler = StandardScaler()
    xs = scaler.fit_transform(xi)
    weights = subject_weights(subjects)
    ridge = Ridge(alpha=ALPHA, fit_intercept=True)
    ridge.fit(xs, target, sample_weight=weights)
    lower = np.nanpercentile(target, 1, axis=0)
    upper = np.nanpercentile(target, 99, axis=0)
    tau = max(0.5, float(np.median(np.std(losses, axis=1))))
    return Student(imputer=imputer, scaler=scaler, ridge=ridge, lower=lower, upper=upper, tau=tau)


def predict_student(model: Student, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = model.scaler.transform(model.imputer.transform(x))
    regret = model.ridge.predict(xs)
    regret = regret - np.mean(regret, axis=1, keepdims=True)
    regret = np.clip(regret, model.lower, model.upper)
    q = stable_softmax(-regret / model.tau)
    weights = (1.0 - TRUST_UPDATE) / 3.0 + TRUST_UPDATE * q
    return regret, weights


def curves_from_weights(curves: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(curves * weights[:, :, None], axis=1)


def event_mae(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - truth), axis=1)


def subject_delta_frame(subjects: np.ndarray, base_error: np.ndarray, model_error: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"subject": subjects.astype(str), "base": base_error, "model": model_error})
        .groupby("subject", as_index=False)[["base", "model"]]
        .mean()
        .assign(improvement=lambda d: d["base"] - d["model"])
    )


def certification(
    subjects: np.ndarray,
    base_error: np.ndarray,
    model_error: np.ndarray,
    rng_seed: int,
) -> dict[str, float | str | bool]:
    d = subject_delta_frame(subjects, base_error, model_error)
    estimate = float(d["improvement"].mean())
    rng = np.random.default_rng(rng_seed)
    values = d["improvement"].to_numpy(float)
    boots = np.empty(BOOTSTRAP_REPS, dtype=float)
    for b in range(BOOTSTRAP_REPS):
        boots[b] = float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
    top_idx = int(np.argmax(values))
    leave_top = float(np.mean(np.delete(values, top_idx))) if len(values) > 1 else float("nan")
    return {
        "subject_macro_mae_improvement_deg": estimate,
        "bootstrap_ci_lower_deg": float(np.percentile(boots, 2.5)),
        "bootstrap_ci_upper_deg": float(np.percentile(boots, 97.5)),
        "top_contributing_subject": str(d.iloc[top_idx]["subject"]),
        "leave_top_subject_improvement_deg": leave_top,
        "subject_improved_count": int((values > 0).sum()),
        "subject_worsened_count": int((values < 0).sum()),
        "base_gate_pass": bool(
            estimate >= MIN_INNER_TOTAL_GAIN_DEG
            and np.percentile(boots, 2.5) > 0
            and leave_top > 0
        ),
    }


def load_outer_predictions(pfull: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    files = [
        RUN60 / "predictions" / "M2_vehicle_anchor_oof_predictions.csv",
        RUN60 / "predictions" / "M3_lgbm_oof_predictions.csv",
        RUN60 / "predictions" / "M4_hist_oof_predictions.csv",
    ]
    arrays = []
    meta = None
    columns = [f"pred_t{i:02d}_deg" for i in range(1, 21)]
    for path in files:
        frame = pd.read_csv(path)
        if meta is None:
            meta = frame[["event_uid", "subject", "outer_fold", "support_class", "amplitude_bin", "floor_denominator_gauss_optimal_mae_deg"]].copy()
        elif not frame["event_uid"].equals(meta["event_uid"]):
            raise ValueError("outer prediction event order mismatch")
        arrays.append(frame[columns].to_numpy(float))
    assert meta is not None
    p_lookup = pfull.set_index("event_uid")
    truth = p_lookup.loc[meta["event_uid"], truth_columns()].to_numpy(float)
    for i, name in enumerate(truth_columns()):
        meta[name] = truth[:, i]
    return np.stack(arrays, axis=1), meta


def load_inner_curves(inner: pd.DataFrame) -> np.ndarray:
    arrays = []
    for prefix in ("M2_vehicle_anchor", "M3_lgbm", "M4_hist"):
        arrays.append(inner[pred_columns(prefix)].to_numpy(float))
    return np.stack(arrays, axis=1)


def evaluate_inner_arm(
    outer_fold: int,
    arm: str,
    frame: pd.DataFrame,
    curves: np.ndarray,
    truth: np.ndarray,
    feature_index: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    predictions = np.full_like(truth, np.nan)
    weights_all = np.full((len(frame), 3), np.nan, dtype=float)
    for inner_fold in sorted(frame["inner_fold"].unique()):
        val = frame["inner_fold"].to_numpy(int) == int(inner_fold)
        fit = ~val
        x_fit = event_feature_matrix(frame.loc[fit, "event_uid"].to_numpy(), curves[fit], feature_index, arm)
        x_val = event_feature_matrix(frame.loc[val, "event_uid"].to_numpy(), curves[val], feature_index, arm)
        target, losses = centered_regret(curves[fit], truth[fit])
        model = fit_student(x_fit, target, losses, frame.loc[fit, "subject"].astype(str).to_numpy())
        _, weights = predict_student(model, x_val)
        weights_all[val] = weights
        predictions[val] = curves_from_weights(curves[val], weights)
    if not np.isfinite(predictions).all():
        raise ValueError(f"inner prediction coverage failed: outer={outer_fold} arm={arm}")
    base = np.mean(curves, axis=1)
    base_error = event_mae(base, truth)
    model_error = event_mae(predictions, truth)
    cert = certification(
        frame["subject"].astype(str).to_numpy(),
        base_error,
        model_error,
        SEED + outer_fold * 100 + list(ARM_BLOCKS).index(arm),
    )
    cert.update(
        {
            "outer_fold": int(outer_fold),
            "arm": arm,
            "inner_events": int(len(frame)),
            "inner_subjects": int(frame["subject"].nunique()),
            "event_improved_fraction": float(np.mean(model_error < base_error - 1e-12)),
        }
    )
    return predictions, weights_all, cert


def fit_outer_arm(
    arm: str,
    train_frame: pd.DataFrame,
    train_curves: np.ndarray,
    train_truth: np.ndarray,
    test_ids: np.ndarray,
    test_curves: np.ndarray,
    feature_index: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = event_feature_matrix(train_frame["event_uid"].to_numpy(), train_curves, feature_index, arm)
    x_test = event_feature_matrix(test_ids, test_curves, feature_index, arm)
    target, losses = centered_regret(train_curves, train_truth)
    model = fit_student(x_train, target, losses, train_frame["subject"].astype(str).to_numpy())
    _, weights = predict_student(model, x_test)
    return curves_from_weights(test_curves, weights), weights


def metric_summary(per_event: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in per_event.groupby("model", sort=False):
        subject_mae = group.groupby("subject")["model_mae_deg"].mean()
        rows.append(
            {
                "model": model,
                "events": int(len(group)),
                "subjects": int(group["subject"].nunique()),
                "pooled_mae_deg": float(group["model_mae_deg"].mean()),
                "subject_macro_mae_deg": float(subject_mae.mean()),
                "event_worsened_fraction_vs_B_all3": float(group["worsened_vs_base"].mean()),
                "subject_worsened_count_vs_B_all3": int(
                    (group.groupby("subject")["improvement_vs_base_deg"].mean() < 0).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_models(per_event: pd.DataFrame) -> pd.DataFrame:
    base = per_event.loc[per_event["model"] == "B_all3", ["event_uid", "subject", "model_mae_deg"]].rename(
        columns={"model_mae_deg": "base_mae"}
    )
    rows = []
    for model in [m for m in per_event["model"].unique() if m != "B_all3"]:
        g = per_event.loc[per_event["model"] == model, ["event_uid", "model_mae_deg"]].rename(
            columns={"model_mae_deg": "candidate_mae"}
        )
        joined = base.merge(g, on="event_uid", validate="one_to_one")
        d = (
            joined.assign(improvement=lambda x: x["base_mae"] - x["candidate_mae"])
            .groupby("subject", as_index=False)["improvement"]
            .mean()
        )
        values = d["improvement"].to_numpy(float)
        rng = np.random.default_rng(SEED + 700 + list(per_event["model"].unique()).index(model))
        boots = np.asarray(
            [np.mean(values[rng.integers(0, len(values), size=len(values))]) for _ in range(BOOTSTRAP_REPS)]
        )
        top = int(np.argmax(values))
        rows.append(
            {
                "model": model,
                "improvement_subject_macro_mae_deg": float(values.mean()),
                "ci_lower_deg": float(np.percentile(boots, 2.5)),
                "ci_upper_deg": float(np.percentile(boots, 97.5)),
                "top_contributing_subject": str(d.iloc[top]["subject"]),
                "leave_top_subject_improvement_deg": float(np.mean(np.delete(values, top))),
                "subject_improved_count": int((values > 0).sum()),
                "subject_worsened_count": int((values < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def amplitude_relative_table(per_event: pd.DataFrame, pfull: pd.DataFrame) -> pd.DataFrame:
    amplitude = pfull[["event_uid", "subject", "amplitude_bin", "causal_pulse_amplitude_deg_at_release"]].copy()
    amplitude["release_amp"] = amplitude["causal_pulse_amplitude_deg_at_release"].abs()
    rows = []
    for (model, amp_bin), group in per_event.merge(
        amplitude[["event_uid", "release_amp"]], on="event_uid", how="left", validate="many_to_one"
    ).groupby(["model", "amplitude_bin"], sort=False):
        per_subject = []
        for _, sg in group.groupby("subject"):
            denom = float(np.nanmedian(sg["release_amp"]))
            if np.isfinite(denom) and denom > 0:
                per_subject.append(float(sg["model_mae_deg"].mean() / denom))
        rows.append(
            {
                "model": model,
                "amplitude_bin": amp_bin,
                "subject_macro_relative_mae_over_median_release_amplitude": float(np.mean(per_subject)) if per_subject else float("nan"),
                "subject_count": int(len(per_subject)),
                "event_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def support_table(per_event: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, support), group in per_event.groupby(["model", "support_class"], sort=False):
        per_subject = group.groupby("subject")["floor_ratio"].mean()
        rows.append(
            {
                "model": model,
                "support_class": support,
                "subject_macro_floor_ratio": float(per_subject.mean()),
                "subject_count": int(len(per_subject)),
                "event_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def save_figures(metrics: pd.DataFrame, per_event: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = metrics.sort_values("subject_macro_mae_deg")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(order["model"], order["subject_macro_mae_deg"], color="#4C78A8")
    ax.set_ylabel("Subject-macro curve MAE (deg)")
    ax.set_title("Run64: frozen-expert regret distillation")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_1_model_comparison.png", dpi=180)
    plt.close(fig)

    models = [m for m in per_event["model"].unique() if m != "B_all3"]
    base = per_event.loc[per_event["model"] == "B_all3", ["event_uid", "subject", "model_mae_deg"]].rename(
        columns={"model_mae_deg": "base"}
    )
    subject_rows = []
    for model in models:
        cand = per_event.loc[per_event["model"] == model, ["event_uid", "model_mae_deg"]].rename(
            columns={"model_mae_deg": "candidate"}
        )
        d = base.merge(cand, on="event_uid").assign(delta=lambda x: x["base"] - x["candidate"])
        for subject, value in d.groupby("subject")["delta"].mean().items():
            subject_rows.append({"model": model, "subject": subject, "improvement": value})
    sf = pd.DataFrame(subject_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        vals = sf.loc[sf["model"] == model, "improvement"].to_numpy(float)
        ax.scatter(np.full_like(vals, i, dtype=float), vals, s=22, alpha=0.7)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(len(models)), models, rotation=35, ha="right")
    ax.set_ylabel("Per-subject MAE improvement vs B_all3 (deg)")
    ax.set_title("Run64: cross-subject improvement distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "Figure_2_subject_improvement.png", dpi=180)
    plt.close(fig)


def main(out_dir: Path) -> int:
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("outputs", "tables", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    print("[Run64] 目标：检验预测起点前生理状态与既往驾驶风格能否蒸馏三专家相对后悔。")

    pfull = pd.read_csv(PFULL_PATH)
    features = pd.read_csv(FEATURE_PATH)
    if len(pfull) != 2323 or len(features) != 2323:
        raise ValueError("P_full/feature coverage mismatch")
    feature_index = features.set_index("event_uid", verify_integrity=True)
    truth_lookup = pfull.set_index("event_uid")[truth_columns()]

    inner_all = pd.read_csv(INNER_PATH)
    outer_curves, outer_meta = load_outer_predictions(pfull)
    outer_event_to_idx = pd.Series(np.arange(len(outer_meta)), index=outer_meta["event_uid"]).to_dict()

    cert_rows: list[dict[str, object]] = []
    outer_predictions: dict[str, np.ndarray] = {"B_all3": np.mean(outer_curves, axis=1)}
    outer_weights: dict[str, np.ndarray] = {}
    for arm in ARM_BLOCKS:
        outer_predictions[f"{arm}_raw"] = np.full((len(pfull), 20), np.nan)
        outer_predictions[f"{arm}_protected"] = np.full((len(pfull), 20), np.nan)
        outer_weights[arm] = np.full((len(pfull), 3), np.nan)

    oracle_rows = []
    for outer_fold in range(1, 6):
        train_frame = inner_all.loc[inner_all["outer_context_fold"] == outer_fold].reset_index(drop=True)
        train_curves = load_inner_curves(train_frame)
        train_truth = truth_lookup.loc[train_frame["event_uid"], truth_columns()].to_numpy(float)
        base_inner = np.mean(train_curves, axis=1)
        base_inner_error = event_mae(base_inner, train_truth)

        # 解析教师上限：真实专家后悔只在训练侧用于判断候选池是否值得路由。
        regret, losses = centered_regret(train_curves, train_truth)
        tau_oracle = max(0.5, float(np.median(np.std(losses, axis=1))))
        q_oracle = stable_softmax(-regret / tau_oracle)
        w_oracle = (1.0 - TRUST_UPDATE) / 3.0 + TRUST_UPDATE * q_oracle
        oracle_error = event_mae(curves_from_weights(train_curves, w_oracle), train_truth)
        oracle_cert = certification(
            train_frame["subject"].astype(str).to_numpy(),
            base_inner_error,
            oracle_error,
            SEED + outer_fold * 1000,
        )
        oracle_rows.append({"outer_fold": outer_fold, "tau_oracle": tau_oracle, **oracle_cert})

        inner_arm_predictions: dict[str, np.ndarray] = {}
        inner_arm_errors: dict[str, np.ndarray] = {}
        inner_arm_certs: dict[str, dict[str, object]] = {}
        for arm in ARM_BLOCKS:
            pred, _, cert = evaluate_inner_arm(
                outer_fold, arm, train_frame, train_curves, train_truth, feature_index
            )
            inner_arm_predictions[arm] = pred
            inner_arm_errors[arm] = event_mae(pred, train_truth)
            inner_arm_certs[arm] = cert

        d_error = inner_arm_errors["D_disagreement"]
        best_single_error = np.minimum(inner_arm_errors["P_physio"], inner_arm_errors["S_style"])
        for arm, cert in inner_arm_certs.items():
            if arm == "D_disagreement":
                gain_over_d = 0.0
                modality_specific = False
                certified = bool(cert["base_gate_pass"])
            else:
                gain_over_d = float(
                    subject_delta_frame(
                        train_frame["subject"].astype(str).to_numpy(),
                        d_error,
                        inner_arm_errors[arm],
                    )["improvement"].mean()
                )
                modality_specific = gain_over_d >= MIN_MODAL_GAIN_OVER_DISAGREEMENT_DEG
                certified = bool(cert["base_gate_pass"] and modality_specific)
                if arm == "PS_physio_style":
                    synergy = float(
                        subject_delta_frame(
                            train_frame["subject"].astype(str).to_numpy(),
                            best_single_error,
                            inner_arm_errors[arm],
                        )["improvement"].mean()
                    )
                    cert["gain_over_best_single_deg"] = synergy
                    # 联合臂必须至少不比最佳单模态差；否则不能用联合结果代替单模态证据。
                    certified = bool(certified and synergy >= 0.0)
            cert["gain_over_disagreement_only_deg"] = gain_over_d
            cert["modality_specific_gate_pass"] = modality_specific
            cert["certified_for_outer_test"] = certified
            cert_rows.append(cert)

        test_idx = np.where(outer_meta["outer_fold"].to_numpy(int) == outer_fold)[0]
        test_ids = outer_meta.loc[test_idx, "event_uid"].astype(str).to_numpy()
        test_curves = outer_curves[test_idx]
        for arm in ARM_BLOCKS:
            pred, weights = fit_outer_arm(
                arm,
                train_frame,
                train_curves,
                train_truth,
                test_ids,
                test_curves,
                feature_index,
            )
            outer_predictions[f"{arm}_raw"][test_idx] = pred
            outer_weights[arm][test_idx] = weights
            if inner_arm_certs[arm].get("certified_for_outer_test", False):
                outer_predictions[f"{arm}_protected"][test_idx] = pred
            else:
                outer_predictions[f"{arm}_protected"][test_idx] = np.mean(test_curves, axis=1)

    if any(not np.isfinite(v).all() for v in outer_predictions.values()):
        raise ValueError("outer prediction coverage failed")

    cert_table = pd.DataFrame(cert_rows)
    oracle_table = pd.DataFrame(oracle_rows)
    cert_table.to_csv(out_dir / "tables" / "inner_certification.csv", index=False, encoding="utf-8-sig")
    oracle_table.to_csv(out_dir / "tables" / "oracle_teacher_upper_bound.csv", index=False, encoding="utf-8-sig")

    truth = outer_meta[truth_columns()].to_numpy(float)
    base_error = event_mae(outer_predictions["B_all3"], truth)
    per_event_rows = []
    for model, pred in outer_predictions.items():
        model_error = event_mae(pred, truth)
        for i in range(len(outer_meta)):
            row = {
                "event_uid": outer_meta.iloc[i]["event_uid"],
                "subject": outer_meta.iloc[i]["subject"],
                "outer_fold": int(outer_meta.iloc[i]["outer_fold"]),
                "support_class": outer_meta.iloc[i]["support_class"],
                "amplitude_bin": outer_meta.iloc[i]["amplitude_bin"],
                "model": model,
                "model_mae_deg": float(model_error[i]),
                "base_mae_deg": float(base_error[i]),
                "improvement_vs_base_deg": float(base_error[i] - model_error[i]),
                "worsened_vs_base": bool(model_error[i] > base_error[i] + 1e-12),
                "floor_denominator_gauss_optimal_mae_deg": float(
                    outer_meta.iloc[i]["floor_denominator_gauss_optimal_mae_deg"]
                ),
                "floor_ratio": float(
                    model_error[i] / outer_meta.iloc[i]["floor_denominator_gauss_optimal_mae_deg"]
                ),
            }
            for t in range(20):
                row[f"pred_t{t+1:02d}_deg"] = float(pred[i, t])
                row[f"true_t{t+1:02d}_deg"] = float(truth[i, t])
            per_event_rows.append(row)
    per_event = pd.DataFrame(per_event_rows)
    per_event.to_csv(out_dir / "tables" / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

    weight_rows = []
    for arm, weights in outer_weights.items():
        for i in range(len(outer_meta)):
            weight_rows.append(
                {
                    "event_uid": outer_meta.iloc[i]["event_uid"],
                    "subject": outer_meta.iloc[i]["subject"],
                    "outer_fold": int(outer_meta.iloc[i]["outer_fold"]),
                    "arm": arm,
                    "M2_weight": float(weights[i, 0]),
                    "M3_weight": float(weights[i, 1]),
                    "M4_weight": float(weights[i, 2]),
                }
            )
    pd.DataFrame(weight_rows).to_csv(out_dir / "tables" / "student_weights.csv", index=False, encoding="utf-8-sig")

    metrics = metric_summary(per_event)
    bootstrap = bootstrap_models(per_event)
    amp = amplitude_relative_table(per_event, pfull)
    support = support_table(per_event)
    metrics.to_csv(out_dir / "tables" / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(out_dir / "tables" / "paired_bootstrap_subject.csv", index=False, encoding="utf-8-sig")
    amp.to_csv(out_dir / "tables" / "relative_mae_by_amplitude_bin.csv", index=False, encoding="utf-8-sig")
    support.to_csv(out_dir / "tables" / "floor_ratio_by_support_class.csv", index=False, encoding="utf-8-sig")
    save_figures(metrics, per_event, out_dir / "figures")

    metric_index = metrics.set_index("model")
    boot_index = bootstrap.set_index("model")
    claims = {}
    for model in ["P_physio_raw", "S_style_raw", "PS_physio_style_raw"]:
        claims[model] = {
            "subject_macro_mae_deg": float(metric_index.loc[model, "subject_macro_mae_deg"]),
            "improvement_deg": float(boot_index.loc[model, "improvement_subject_macro_mae_deg"]),
            "ci_lower_deg": float(boot_index.loc[model, "ci_lower_deg"]),
            "leave_top_subject_improvement_deg": float(
                boot_index.loc[model, "leave_top_subject_improvement_deg"]
            ),
            "supported": bool(
                boot_index.loc[model, "improvement_subject_macro_mae_deg"] >= 0.05
                and boot_index.loc[model, "ci_lower_deg"] > 0
                and boot_index.loc[model, "leave_top_subject_improvement_deg"] > 0
            ),
        }
    both_independently_supported = bool(claims["P_physio_raw"]["supported"] and claims["S_style_raw"]["supported"])
    decision = {
        "status": "developmental_result",
        "population": {"events": 2323, "subjects": 18, "recordings": 85},
        "baseline": {
            "model": "B_all3",
            "subject_macro_mae_deg": float(metric_index.loc["B_all3", "subject_macro_mae_deg"]),
        },
        "oracle_teacher": {
            "mean_inner_subject_macro_improvement_deg": float(
                oracle_table["subject_macro_mae_improvement_deg"].mean()
            ),
            "all_outer_base_gate_pass": bool(oracle_table["base_gate_pass"].all()),
        },
        "claims": claims,
        "physiology_and_style_both_independently_supported": both_independently_supported,
        "scientific_boundary": (
            "Same-population developmental OOF. The analytic teacher uses future truth only to create training labels; "
            "deployment students use only prediction-anchor-or-earlier physiology, prior-session style, and frozen-expert disagreement."
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    (out_dir / "outputs" / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Run64：生理与驾驶风格的专家后悔蒸馏",
        "",
        "## 结论",
        "",
        f"- B_all3 subject-macro MAE：{decision['baseline']['subject_macro_mae_deg']:.4f}°。",
        f"- 25%信任域解析教师在inner训练侧平均上限改善：{decision['oracle_teacher']['mean_inner_subject_macro_improvement_deg']:.4f}°。",
        f"- 生理与风格是否分别获得稳定发展性支持：{both_independently_supported}。",
        "- 本结果是同一2323事件、18被试上的发展性OOF，不是独立确认。",
        "",
        "## 主结果",
        "",
        metrics.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 被试配对bootstrap",
        "",
        bootstrap.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 各outer训练侧认证",
        "",
        cert_table.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 证据边界",
        "",
        "解析教师只在训练侧用真实未来曲线产生专家相对后悔标签；测试时不存在教师或未来真值。生理来自预测起点前原始通道，风格只来自当前session之前已经完成的session。任何分类/压力识别文献均未被当作转向曲线增量证据。",
    ]
    (out_dir / "outputs" / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "final_info.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="run_1")
    args = parser.parse_args()
    target = Path(args.out_dir)
    if not target.is_absolute():
        target = RUN_DIR / target
    sys.exit(main(target))

