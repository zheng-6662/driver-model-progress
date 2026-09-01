"""Run37 的输入门禁与四类预冻结裁决。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _subject_sign_flip_p(values: np.ndarray, repetitions: int = 5000, seed: int = 20260812) -> float:
    observed = float(np.mean(values))
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(repetitions, len(values)), replace=True)
    null = np.mean(signs * values[None, :], axis=1)
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (repetitions + 1))


def assess_support_gate(
    support: pd.DataFrame,
    edges: pd.DataFrame,
    transforms: pd.DataFrame,
    thresholds: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    support_thresholds = config["support_thresholds"]
    validity = config["distance_validity_gates"]
    dense = support["support_class"].eq("dense_overlap")
    out = support["support_class"].eq("out_of_support")
    global_fallback = support["match_level"].eq("global")
    insufficient = ~support["has_5_legal_neighbors"]
    subject_support = (
        support.assign(
            is_dense=dense,
            is_out=out,
            d5_over_fold_p95=support["d5"] / support["d5_p95"],
        )
        .groupby("subject", as_index=False)
        .agg(
            event_count=("event_uid", "size"),
            dense_count=("is_dense", "sum"),
            dense_fraction=("is_dense", "mean"),
            out_of_support_count=("is_out", "sum"),
            out_of_support_fraction=("is_out", "mean"),
            mean_d1=("d1", "mean"),
            mean_d5=("d5", "mean"),
            mean_d5_over_fold_p95=("d5_over_fold_p95", "mean"),
            global_fallback_fraction=("match_level", lambda values: float(np.mean(values == "global"))),
        )
        .sort_values("subject", ignore_index=True)
    )
    significant_shift_subjects = int((subject_support["mean_d5_over_fold_p95"] > 1.0).sum())

    # 每个真实查询被试与其对应外折的训练内部伪留一参考均值比较。
    fold_reference = thresholds.set_index("outer_fold")["pseudo_d5_mean"].to_dict()
    deltas = []
    for subject, group in support.groupby("subject", sort=True):
        fold = int(group["outer_fold"].iloc[0])
        deltas.append(float(group["d5"].mean() - fold_reference[fold]))
    shift_p = _subject_sign_flip_p(np.asarray(deltas, dtype=float))

    outer_edges = edges[~edges["is_pseudo_reference"]]
    usage = (
        outer_edges.groupby(["neighbor_subject", "neighbor_recording_uid"], as_index=False)
        .agg(edge_count=("query_event_uid", "size"), query_subject_count=("query_subject", "nunique"))
        .sort_values("edge_count", ascending=False, ignore_index=True)
    )
    subject_usage = (
        outer_edges.groupby("neighbor_subject", as_index=False)
        .agg(edge_count=("query_event_uid", "size"), query_subject_count=("query_subject", "nunique"))
        .sort_values("edge_count", ascending=False, ignore_index=True)
    )
    subject_usage["edge_share"] = subject_usage["edge_count"] / max(len(outer_edges), 1)
    max_subject_share = float(subject_usage["edge_share"].max())

    measures = {
        "event_count": int(len(support)),
        "subject_count": int(support["subject"].nunique()),
        "dense_coverage": float(dense.mean()),
        "borderline_coverage": float(support["support_class"].eq("borderline").mean()),
        "out_of_support_fraction": float(out.mean()),
        "severe_out_of_support_fraction": float(support["severe_out_of_support"].mean()),
        "subjects_with_at_least_3_dense": int((subject_support["dense_count"] >= 3).sum()),
        "global_fallback_fraction": float(global_fallback.mean()),
        "insufficient_neighbor_fraction": float(insufficient.mean()),
        "median_d5_over_d1": float(support["d5_over_d1"].median()),
        "leave_one_channel_jaccard_median": float(
            support["leave_one_channel_jaccard_median"].median()
        ),
        "maximum_constant_feature_fraction": float(
            transforms["constant_feature_fraction"].max()
        ),
        "maximum_single_neighbor_subject_edge_share": max_subject_share,
        "test_minus_pseudo_d5_subject_macro_mean": float(np.mean(deltas)),
        "test_vs_pseudo_d5_subject_sign_flip_p": shift_p,
        "subjects_above_fold_p95_mean_d5": significant_shift_subjects,
    }
    checks = {
        "constant_feature_fraction_pass": measures["maximum_constant_feature_fraction"]
        <= validity["constant_feature_fraction_max"],
        "distance_resolution_pass": measures["median_d5_over_d1"]
        >= validity["median_d5_over_d1_min"],
        "leave_one_channel_stability_pass": measures[
            "leave_one_channel_jaccard_median"
        ]
        >= validity["leave_one_channel_top5_jaccard_median_min"],
        "neighbor_subject_dominance_pass": measures[
            "maximum_single_neighbor_subject_edge_share"
        ]
        <= validity["single_neighbor_subject_edge_share_max"],
        "legal_neighbor_coverage_pass": measures["insufficient_neighbor_fraction"]
        <= validity["queries_without_5_legal_neighbors_max_fraction"],
    }
    support_checks = {
        "dense_coverage_pass": measures["dense_coverage"]
        >= support_thresholds["dense_coverage_min"],
        "out_of_support_pass": measures["out_of_support_fraction"]
        <= support_thresholds["out_of_support_max"],
        "subject_dense_coverage_pass": measures["subjects_with_at_least_3_dense"]
        >= support_thresholds["subjects_with_at_least_3_dense_min"],
        "global_fallback_pass": measures["global_fallback_fraction"]
        <= support_thresholds["global_fallback_max"],
    }
    distance_valid = all(checks.values())
    support_adequate = distance_valid and all(support_checks.values())
    if not distance_valid:
        support_status = "distance_audit_invalid"
    elif not support_adequate:
        support_status = "local_sparsity_or_insufficient_support"
    else:
        support_status = "adequate_for_outcome_unlock"
    shift_input_evidence = (
        measures["test_minus_pseudo_d5_subject_macro_mean"] > 0
        and shift_p < config["inference"]["permutation_alpha"]
        and significant_shift_subjects >= config["decision"]["covariate_shift_subjects_min"]
    )
    payload = {
        "status": "pass" if distance_valid else "fail",
        "support_status": support_status,
        "outcome_labels_unlocked": bool(support_adequate),
        "input_shift_evidence": bool(shift_input_evidence),
        "measures": measures,
        "distance_validity_checks": checks,
        "support_adequacy_checks": support_checks,
        "failed_gates": sorted(
            [name for name, passed in {**checks, **support_checks}.items() if not passed]
        ),
    }
    usage = usage.merge(
        subject_usage[["neighbor_subject", "edge_share"]], on="neighbor_subject", how="left"
    )
    return payload, subject_support, usage


def make_final_decision(
    support_gate: dict[str, Any],
    per_query: pd.DataFrame | None,
    bootstrap: pd.DataFrame | None,
    permutation: pd.DataFrame | None,
    gradients: pd.DataFrame | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not support_gate["outcome_labels_unlocked"]:
        dominant = (
            "distance_audit_invalid"
            if support_gate["support_status"] == "distance_audit_invalid"
            else "local_sparsity"
        )
        return {
            "support_status": support_gate["support_status"],
            "covariate_shift_status": "input_only_evidence"
            if support_gate["input_shift_evidence"]
            else "not_established",
            "identifiability_status": "not_evaluated_support_gate_locked",
            "underfitting_evidence_status": "not_evaluated_support_gate_locked",
            "dominant_failure_mode": dominant,
            "secondary_failure_modes": ["covariate_shift_input_evidence"]
            if support_gate["input_shift_evidence"]
            else [],
            "claim_boundary": "Input-only support audit; future outcomes remained locked.",
            "failed_gates": support_gate["failed_gates"],
            "recommended_next_step": "Acquire independent subjects/recordings in sparse or shifted input regions; keep Run35 model fixed.",
        }

    assert per_query is not None
    assert bootstrap is not None
    assert permutation is not None
    assert gradients is not None
    decision_cfg = config["decision"]
    boot = bootstrap.set_index("metric")
    perm = permutation.set_index("metric")
    curve = boot.loc["curve_rmse_deg"]
    amplitude = boot.loc["peak_amplitude_abs_diff_deg"]
    tail = boot.loc["tail5_rmse_deg"]
    subject_curve = per_query.groupby("subject")["ratio_curve_rmse_deg"].mean()
    consistent_subjects = int((subject_curve < 1.0).sum())
    gradient = gradients[gradients["comparison"].isin(["rank_1_5", "farthest_5"])]
    gradient_macro = gradient.groupby("comparison")["curve_rmse_deg"].mean()
    far_over_top = float(
        gradient_macro.get("farthest_5", np.nan)
        / max(gradient_macro.get("rank_1_5", np.nan), 1e-12)
    )
    weak_gradient = far_over_top <= decision_cfg[
        "weak_distance_gradient_farthest_over_top5_max"
    ]
    identifiability = (
        float(curve["estimate"]) >= decision_cfg["identifiability_curve_ratio_median_min"]
        and float(curve["ci_lower"])
        >= decision_cfg["identifiability_curve_ratio_ci_lower_min"]
        and (
            float(amplitude["estimate"])
            >= decision_cfg["identifiability_secondary_ratio_min"]
            or float(tail["estimate"])
            >= decision_cfg["identifiability_secondary_ratio_min"]
        )
        and weak_gradient
    )

    dense = per_query[per_query["support_class"].eq("dense_overlap")].copy()
    low_threshold = dense["top5_curve_rmse_deg"].median()
    low_variance = dense[dense["top5_curve_rmse_deg"] <= low_threshold]
    overall_error_median = per_query["nested_blend_oof_mae_deg"].median()
    high_error_fraction = float(
        (low_variance["nested_blend_oof_mae_deg"] > overall_error_median).mean()
    )
    underfitting = (
        float(curve["ci_upper"])
        <= decision_cfg["underfitting_curve_ratio_ci_upper_max"]
        and float(perm.loc["curve_rmse_deg", "two_sided_p"])
        < config["inference"]["permutation_alpha"]
        and consistent_subjects >= decision_cfg["underfitting_consistent_subjects_min"]
        and (
            float(amplitude["estimate"])
            <= decision_cfg["underfitting_secondary_ratio_max"]
            or float(tail["estimate"])
            <= decision_cfg["underfitting_secondary_ratio_max"]
        )
        and high_error_fraction
        >= decision_cfg["underfitting_dense_low_variance_high_error_fraction_min"]
    )
    out_error = float(
        per_query.loc[
            per_query["support_class"].eq("out_of_support"), "nested_blend_oof_mae_deg"
        ].mean()
    )
    dense_error = float(dense["nested_blend_oof_mae_deg"].mean())
    covariate_shift = bool(
        support_gate["input_shift_evidence"]
        and support_gate["measures"]["out_of_support_fraction"] > 0.20
        and out_error > dense_error
    )
    pseudo_repeat_curve = per_query.groupby("subject")[
        "unrestricted_top5_curve_rmse_deg"
    ].mean().mean()
    unique_curve = per_query.groupby("subject")["top5_curve_rmse_deg"].mean().mean()
    pseudo_repeat_reduction = float(1.0 - pseudo_repeat_curve / max(unique_curve, 1e-12))

    if covariate_shift:
        dominant = "covariate_shift"
        recommendation = "增加独立被试与道路工况覆盖，并建立输入域外检测；暂不增加点预测器复杂度。"
    elif identifiability:
        dominant = "dense_region_empirical_identifiability_limit"
        recommendation = "转向条件分布、校准预测区间和谨慎拒识；任何新增机制输入都必须在独立事件上验证。"
    elif underfitting:
        dominant = "model_underfitting_evidence"
        recommendation = "冻结已识别的低条件方差失败维度，先在新增独立数据上确认，再开发低自由度针对性模型。"
    else:
        dominant = "inconclusive_small_n"
        recommendation = "新增独立被试和recording，并原样复验Run37；不要在这365个事件上调整距离或模型。"
    secondary: list[str] = []
    if support_gate["input_shift_evidence"] and dominant != "covariate_shift":
        secondary.append("input_shift_evidence")
    if pseudo_repeat_reduction > 0.20:
        secondary.append("repeat_structure_would_inflate_neighbor_consistency")
    identifiability_checks = {
        "curve_ratio_estimate_at_least_0_90": float(curve["estimate"])
        >= decision_cfg["identifiability_curve_ratio_median_min"],
        "curve_ratio_ci_lower_at_least_0_80": float(curve["ci_lower"])
        >= decision_cfg["identifiability_curve_ratio_ci_lower_min"],
        "amplitude_or_tail_ratio_at_least_0_90": (
            float(amplitude["estimate"])
            >= decision_cfg["identifiability_secondary_ratio_min"]
            or float(tail["estimate"])
            >= decision_cfg["identifiability_secondary_ratio_min"]
        ),
        "distance_divergence_gradient_weak": weak_gradient,
    }
    underfitting_checks = {
        "curve_ratio_ci_upper_at_most_0_80": float(curve["ci_upper"])
        <= decision_cfg["underfitting_curve_ratio_ci_upper_max"],
        "curve_subject_block_permutation_p_below_0_01": float(
            perm.loc["curve_rmse_deg", "two_sided_p"]
        )
        < config["inference"]["permutation_alpha"],
        "subjects_with_aggregation_at_least_10": consistent_subjects
        >= decision_cfg["underfitting_consistent_subjects_min"],
        "amplitude_or_tail_ratio_at_most_0_80": (
            float(amplitude["estimate"])
            <= decision_cfg["underfitting_secondary_ratio_max"]
            or float(tail["estimate"])
            <= decision_cfg["underfitting_secondary_ratio_max"]
        ),
        "dense_low_variance_high_oof_error_fraction_at_least_0_50": high_error_fraction
        >= decision_cfg["underfitting_dense_low_variance_high_error_fraction_min"],
    }
    classification_failed_gates = [
        f"identifiability:{name}"
        for name, passed in identifiability_checks.items()
        if not passed
    ] + [
        f"underfitting:{name}"
        for name, passed in underfitting_checks.items()
        if not passed
    ]
    return {
        "support_status": support_gate["support_status"],
        "covariate_shift_status": "present" if covariate_shift else "not_established",
        "identifiability_status": "limited" if identifiability else "not_established",
        "underfitting_evidence_status": "present" if underfitting else "not_established",
        "dominant_failure_mode": dominant,
        "secondary_failure_modes": secondary,
        "primary_curve_ratio": float(curve["estimate"]),
        "primary_curve_ratio_ci": [float(curve["ci_lower"]), float(curve["ci_upper"])],
        "primary_permutation_p": float(perm.loc["curve_rmse_deg", "two_sided_p"]),
        "peak_amplitude_ratio": float(amplitude["estimate"]),
        "tail5_ratio": float(tail["estimate"]),
        "subjects_with_neighbor_aggregation": consistent_subjects,
        "events_with_neighbor_aggregation": int(
            (per_query["ratio_curve_rmse_deg"] < 1.0).sum()
        ),
        "event_count": int(len(per_query)),
        "farthest_over_top5_curve_divergence": far_over_top,
        "dense_low_variance_high_oof_error_fraction": high_error_fraction,
        "dense_oof_mae_deg": dense_error,
        "out_of_support_oof_mae_deg": out_error,
        "pseudo_repeat_divergence_reduction_fraction": pseudo_repeat_reduction,
        "classification_checks": {
            "dense_region_empirical_identifiability_limit": identifiability_checks,
            "model_underfitting_evidence": underfitting_checks,
        },
        "claim_boundary": "仅判断当前release时输入、365事件覆盖和按被试外推合同下的经验可分辨性；不是理论不可预测性或因果效应证明。",
        "failed_gates": classification_failed_gates,
        "recommended_next_step": recommendation,
    }
