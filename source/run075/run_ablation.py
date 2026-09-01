from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TABLES = HERE / "tables"
OUTPUTS = HERE / "outputs"
PREDICTIONS = HERE / "predictions"

MANIFEST = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/pfull_event_manifest.csv"
CACHE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/causal_input_cache.npz"
RUN57_SELECTION = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/model_selection.csv"
RUN60_CONFIG = REPO / "05_rebuild_from_raw_20260511/03_baselines/run60_gbm_vs_noise_floor_20260828/config.json"
RUN60_SELECTION = REPO / "05_rebuild_from_raw_20260511/03_baselines/run60_gbm_vs_noise_floor_20260828/tables/model_selection.csv"
RUN60_PREDICTIONS = REPO / "05_rebuild_from_raw_20260511/03_baselines/run60_gbm_vs_noise_floor_20260828/tables/per_event_predictions.csv"
RUN62_PREDICTIONS = REPO / "05_rebuild_from_raw_20260511/03_baselines/run62_amplitude_shape_factorized_20260829/tables/per_event_predictions.csv"

POINTS = 20
BASE = "B_all3_full"
CANDIDATE = "B_all3_without_yaw_roll_rate"


def fill_from_train(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values[train], axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.where(np.isfinite(values), values, median).astype(np.float64)


def weights(metadata: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    part = metadata.iloc[indices]
    counts = part.groupby(["subject", "episode_id"])["event_uid"].transform("count").astype(float)
    episodes = part.groupby("subject")["episode_id"].transform("nunique").astype(float)
    result = 1.0 / counts.to_numpy(float) / episodes.to_numpy(float)
    return result / result.mean()


def fit_points(worker) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=4) as pool:
        output = list(pool.map(worker, range(POINTS)))
    return np.column_stack(output)


def event_metrics(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    error = np.abs(prediction - truth)
    return pd.DataFrame(
        {
            "curve_mae_deg": error.mean(axis=1),
            "head5_mae_deg": error[:, :5].mean(axis=1),
            "tail5_mae_deg": error[:, -5:].mean(axis=1),
            "endpoint_mae_deg": error[:, -1],
            "peak_time_mae_s": np.abs(
                (np.argmax(np.abs(prediction), axis=1) - np.argmax(np.abs(truth), axis=1)) * 0.05
            ),
        }
    )


def subject_macro(metadata: pd.DataFrame, values: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(metadata), dtype=bool)
    frame = pd.DataFrame({"subject": metadata.loc[mask, "subject"].to_numpy(), "value": values[mask]})
    return float(frame.groupby("subject")["value"].mean().mean())


TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
PREDICTIONS.mkdir(parents=True, exist_ok=True)

metadata = pd.read_csv(
    MANIFEST,
    usecols=[
        "event_uid",
        "subject",
        "recording_uid",
        "outer_fold",
        "episode_id",
        "amplitude_bin",
        "causal_pulse_amplitude_deg_at_release",
    ],
    low_memory=False,
)
with np.load(CACHE, allow_pickle=False) as archive:
    full_summary = np.asarray(archive["summary"], dtype=np.float64)
    feature_names = archive["feature_names"].astype(str)

remove_mask = np.array(
    [("__yaw_rate__" in name) or ("__roll_rate__" in name) for name in feature_names], dtype=bool
)
keep_mask = ~remove_mask
summary = full_summary[:, keep_mask]
removed_features = feature_names[remove_mask].tolist()
kept_features = feature_names[keep_mask].tolist()
assert len(metadata) == 2323
assert full_summary.shape == (2323, 172)
assert summary.shape == (2323, 134)
assert len(removed_features) == 38
assert metadata["subject"].nunique() == 18
assert metadata["outer_fold"].value_counts().sort_index().to_dict() == {1: 352, 2: 471, 3: 435, 4: 539, 5: 526}

pred_columns = [f"pred_t{i:02d}_deg" for i in range(1, 21)]
true_columns = [f"true_t{i:02d}_deg" for i in range(1, 21)]
run60 = pd.read_csv(RUN60_PREDICTIONS, low_memory=False)
full_predictions = {}
for model in ["M2_vehicle_anchor", "M3_lgbm", "M4_hist"]:
    part = run60.loc[run60["model"].eq(model), ["event_uid", *pred_columns]]
    aligned = metadata[["event_uid"]].merge(part, on="event_uid", how="left", validate="one_to_one")
    full_predictions[model] = aligned[pred_columns].to_numpy(float)

run62 = pd.read_csv(RUN62_PREDICTIONS, low_memory=False)
stored_base = run62.loc[run62["model"].eq("C_B_all3"), ["event_uid", *pred_columns, *true_columns]]
stored_base = metadata[["event_uid"]].merge(stored_base, on="event_uid", how="left", validate="one_to_one")
truth = stored_base[true_columns].to_numpy(float)
base_prediction = stored_base[pred_columns].to_numpy(float)
reconstructed_base = sum(full_predictions.values()) / 3.0
assert np.max(np.abs(base_prediction - reconstructed_base)) < 1e-10

run60_config = json.loads(RUN60_CONFIG.read_text(encoding="utf-8"))
run57_selection = pd.read_csv(RUN57_SELECTION)
run60_selection = pd.read_csv(RUN60_SELECTION)

m2_selected = {
    int(row.outer_fold): json.loads(row.selected_config)
    for row in run57_selection.loc[run57_selection["model"].eq("M0_extratrees")].itertuples(index=False)
}
m3_selected = {
    int(row.outer_fold): {
        "config": json.loads(row.selected_config_json),
        "iterations": json.loads(row.selected_best_iterations_json),
    }
    for row in run60_selection.loc[run60_selection["model"].eq("M3_lgbm")].itertuples(index=False)
}
m4_selected = {
    int(row.outer_fold): json.loads(row.selected_config_json)
    for row in run60_selection.loc[run60_selection["model"].eq("M4_hist")].itertuples(index=False)
}

no_rate_predictions = {
    "M2_without_yaw_roll_rate": np.full_like(truth, np.nan),
    "M3_without_yaw_roll_rate": np.full_like(truth, np.nan),
    "M4_without_yaw_roll_rate": np.full_like(truth, np.nan),
}
training_rows = []

for fold in range(1, 6):
    train = np.flatnonzero(~metadata["outer_fold"].eq(fold).to_numpy())
    test = np.flatnonzero(metadata["outer_fold"].eq(fold).to_numpy())
    matrix = fill_from_train(summary, train)
    sample_weight = weights(metadata, train)

    m2_cfg = m2_selected[fold]
    m2 = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=int(m2_cfg["min_samples_leaf"]),
        max_features=float(m2_cfg["max_features"]),
        random_state=20260827 + fold * 1000,
        n_jobs=8,
    )
    m2.fit(matrix[train], truth[train], sample_weight=sample_weight)
    no_rate_predictions["M2_without_yaw_roll_rate"][test] = m2.predict(matrix[test])
    print(f"fold {fold}: M2完成", flush=True)

    m3_cfg = m3_selected[fold]["config"]
    m3_iterations = m3_selected[fold]["iterations"]
    lgb_fixed = run60_config["models"]["M3_lgbm"]["fixed"]

    def fit_lgbm_point(point: int) -> np.ndarray:
        model = lgb.LGBMRegressor(
            objective=lgb_fixed["objective"],
            metric=lgb_fixed["metric"],
            n_estimators=max(1, int(m3_iterations[point])),
            learning_rate=float(m3_cfg["learning_rate"]),
            num_leaves=int(m3_cfg["num_leaves"]),
            min_child_samples=int(m3_cfg["min_child_samples"]),
            max_depth=int(lgb_fixed["max_depth"]),
            subsample=float(lgb_fixed["subsample"]),
            colsample_bytree=float(lgb_fixed["colsample_bytree"]),
            reg_alpha=float(lgb_fixed["reg_alpha"]),
            reg_lambda=float(lgb_fixed["reg_lambda"]),
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            verbosity=-1,
            random_state=20260828 + fold * 100000 + point,
        )
        model.fit(matrix[train], truth[train, point], sample_weight=sample_weight)
        return model.predict(matrix[test])

    no_rate_predictions["M3_without_yaw_roll_rate"][test] = fit_points(fit_lgbm_point)
    print(f"fold {fold}: M3完成", flush=True)

    m4_cfg = m4_selected[fold]
    hist_fixed = run60_config["models"]["M4_hist"]["fixed"]

    def fit_hist_point(point: int) -> np.ndarray:
        model = HistGradientBoostingRegressor(
            loss=hist_fixed["loss"],
            learning_rate=float(m4_cfg["learning_rate"]),
            max_iter=int(hist_fixed["max_iter"]),
            max_leaf_nodes=int(m4_cfg["max_leaf_nodes"]),
            min_samples_leaf=int(hist_fixed["min_samples_leaf"]),
            l2_regularization=float(hist_fixed["l2_regularization"]),
            max_bins=int(hist_fixed["max_bins"]),
            early_stopping=False,
            random_state=20260828 + fold * 100000 + point,
        )
        model.fit(matrix[train], truth[train, point], sample_weight=sample_weight)
        return model.predict(matrix[test])

    no_rate_predictions["M4_without_yaw_roll_rate"][test] = fit_points(fit_hist_point)
    print(f"fold {fold}: M4完成", flush=True)
    training_rows.append(
        {
            "outer_fold": fold,
            "train_events": len(train),
            "test_events": len(test),
            "input_features": summary.shape[1],
            "removed_features": len(removed_features),
            "M2_config": json.dumps(m2_cfg, sort_keys=True),
            "M3_config": json.dumps(m3_cfg, sort_keys=True),
            "M4_config": json.dumps(m4_cfg, sort_keys=True),
        }
    )

for prediction in no_rate_predictions.values():
    assert np.isfinite(prediction).all()

candidate_prediction = sum(no_rate_predictions.values()) / 3.0
models = {
    BASE: base_prediction,
    CANDIDATE: candidate_prediction,
    "M2_full": full_predictions["M2_vehicle_anchor"],
    "M2_without_yaw_roll_rate": no_rate_predictions["M2_without_yaw_roll_rate"],
    "M3_full": full_predictions["M3_lgbm"],
    "M3_without_yaw_roll_rate": no_rate_predictions["M3_without_yaw_roll_rate"],
    "M4_full": full_predictions["M4_hist"],
    "M4_without_yaw_roll_rate": no_rate_predictions["M4_without_yaw_roll_rate"],
}

aggregate_rows = []
subject_rows = []
event_tables = {}
for model, prediction in models.items():
    metrics = event_metrics(truth, prediction)
    event_tables[model] = metrics
    aggregate_rows.append(
        {
            "model": model,
            "subject_macro_curve_mae_deg": subject_macro(metadata, metrics["curve_mae_deg"].to_numpy()),
            "subject_macro_head5_mae_deg": subject_macro(metadata, metrics["head5_mae_deg"].to_numpy()),
            "subject_macro_tail5_mae_deg": subject_macro(metadata, metrics["tail5_mae_deg"].to_numpy()),
            "subject_macro_endpoint_mae_deg": subject_macro(metadata, metrics["endpoint_mae_deg"].to_numpy()),
            "subject_macro_peak_time_mae_s": subject_macro(metadata, metrics["peak_time_mae_s"].to_numpy()),
            "pooled_curve_mae_deg_reference_only": float(metrics["curve_mae_deg"].mean()),
        }
    )
    for subject, indices in metadata.groupby("subject").groups.items():
        idx = np.asarray(list(indices), dtype=int)
        subject_rows.append(
            {
                "model": model,
                "subject": subject,
                "outer_fold": int(metadata.iloc[idx[0]]["outer_fold"]),
                "event_count": len(idx),
                "curve_mae_deg": float(metrics.iloc[idx]["curve_mae_deg"].mean()),
            }
        )

aggregate = pd.DataFrame(aggregate_rows)
subjects = pd.DataFrame(subject_rows)
aggregate.to_csv(TABLES / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
subjects.to_csv(TABLES / "subject_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(training_rows).to_csv(TABLES / "training_folds.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"removed_feature": removed_features}).to_csv(TABLES / "removed_features.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({"kept_feature": kept_features}).to_csv(TABLES / "kept_features.csv", index=False, encoding="utf-8-sig")

amplitude_rows = []
amplitude = pd.to_numeric(metadata["causal_pulse_amplitude_deg_at_release"], errors="raise").to_numpy(float)
for model in [BASE, CANDIDATE]:
    mae = event_tables[model]["curve_mae_deg"].to_numpy(float)
    for label in ["20_30", "30_45", "45_70", "ge70"]:
        mask = metadata["amplitude_bin"].eq(label).to_numpy()
        per_subject = []
        for subject in sorted(metadata.loc[mask, "subject"].unique()):
            subject_mask = mask & metadata["subject"].eq(subject).to_numpy()
            per_subject.append(float(mae[subject_mask].mean() / np.median(amplitude[subject_mask])))
        amplitude_rows.append(
            {
                "model": model,
                "amplitude_bin": label,
                "event_count": int(mask.sum()),
                "subject_count": len(per_subject),
                "subject_macro_relative_mae": float(np.mean(per_subject)),
            }
        )
amplitude_table = pd.DataFrame(amplitude_rows)
amplitude_table.to_csv(TABLES / "relative_mae_by_amplitude_bin.csv", index=False, encoding="utf-8-sig")

base_subject = subjects.loc[subjects["model"].eq(BASE)].set_index("subject")["curve_mae_deg"]
candidate_subject = subjects.loc[subjects["model"].eq(CANDIDATE)].set_index("subject")["curve_mae_deg"]
improvement = base_subject - candidate_subject
rng = np.random.default_rng(20260831)
values = improvement.to_numpy(float)
draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
fold_improvement = subjects.loc[subjects["model"].isin([BASE, CANDIDATE])].pivot(
    index="subject", columns="model", values="curve_mae_deg"
)
fold_improvement["outer_fold"] = fold_improvement.index.map(
    metadata.drop_duplicates("subject").set_index("subject")["outer_fold"]
)
fold_values = fold_improvement.groupby("outer_fold").apply(
    lambda frame: float((frame[BASE] - frame[CANDIDATE]).mean()), include_groups=False
)
comparison = {
    "subject_macro_improvement_full_minus_without_rates_deg": float(values.mean()),
    "candidate_minus_full_mae_deg": float(-values.mean()),
    "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
    "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
    "positive_outer_fold_count": int((fold_values > 0).sum()),
    "outer_fold_improvement_deg": {str(int(fold)): float(value) for fold, value in fold_values.items()},
    "improved_subject_count": int((values > 0).sum()),
    "harmed_subject_count": int((values < 0).sum()),
    "event_worsened_fraction": float(
        np.mean(event_tables[CANDIDATE]["curve_mae_deg"].to_numpy() > event_tables[BASE]["curve_mae_deg"].to_numpy())
    ),
}

amplitude_index = amplitude_table.set_index(["model", "amplitude_bin"])["subject_macro_relative_mae"]
amplitude_changes = {}
for label in ["20_30", "30_45", "45_70", "ge70"]:
    full_value = float(amplitude_index.loc[(BASE, label)])
    candidate_value = float(amplitude_index.loc[(CANDIDATE, label)])
    amplitude_changes[label] = {
        "full": full_value,
        "without_rates": candidate_value,
        "change_without_minus_full": candidate_value - full_value,
    }

mae_impact_negligible = comparison["candidate_minus_full_mae_deg"] <= 0.10
amplitude_impact_negligible = all(item["change_without_minus_full"] <= 0.01 for item in amplitude_changes.values())
decision = {
    "status": "REMOVAL_IMPACT_NEGLIGIBLE" if mae_impact_negligible and amplitude_impact_negligible else "REMOVAL_HAS_MATERIAL_IMPACT",
    "events": 2323,
    "subjects": 18,
    "input_dimension_full": 172,
    "input_dimension_without_rates": 134,
    "removed_feature_count": 38,
    "removed_channels": ["yaw_rate", "roll_rate"],
    "comparison": comparison,
    "amplitude_changes": amplitude_changes,
    "gates": {
        "subject_macro_mae_regression_no_more_than_0_10_deg": mae_impact_negligible,
        "each_ge20_amplitude_relative_regression_no_more_than_0_01": amplitude_impact_negligible,
    },
}
(OUTPUTS / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

prediction_frame = metadata[["event_uid", "subject", "recording_uid", "outer_fold", "amplitude_bin"]].copy()
for point in range(POINTS):
    prediction_frame[f"true_t{point + 1:02d}_deg"] = truth[:, point]
    prediction_frame[f"full_pred_t{point + 1:02d}_deg"] = base_prediction[:, point]
    prediction_frame[f"without_rates_pred_t{point + 1:02d}_deg"] = candidate_prediction[:, point]
prediction_frame["full_curve_mae_deg"] = event_tables[BASE]["curve_mae_deg"].to_numpy()
prediction_frame["without_rates_curve_mae_deg"] = event_tables[CANDIDATE]["curve_mae_deg"].to_numpy()
prediction_frame.to_csv(PREDICTIONS / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

aggregate_index = aggregate.set_index("model")
lines = [
    "# 去除 yaw_rate 和 roll_rate 的影响",
    "",
    f"- 状态：`{decision['status']}`",
    "- P_full=2323，18被试，原5折不变。",
    "- 输入维度：172 → 134；删除38个摘要特征。",
    "- M2/M3/M4均按原每折参数重训，没有重新搜索参数。",
    "",
    "## 当前组合基线",
    "",
    "| 模型 | subject-macro MAE° | head5° | tail5° | endpoint° | peak-time s | pooled MAE°参考 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model in [BASE, CANDIDATE]:
    row = aggregate_index.loc[model]
    lines.append(
        f"| {model} | {row.subject_macro_curve_mae_deg:.4f} | {row.subject_macro_head5_mae_deg:.4f} | "
        f"{row.subject_macro_tail5_mae_deg:.4f} | {row.subject_macro_endpoint_mae_deg:.4f} | "
        f"{row.subject_macro_peak_time_mae_s:.4f} | {row.pooled_curve_mae_deg_reference_only:.4f} |"
    )
lines += [
    "",
    "## 配对影响",
    "",
    f"- 去除后MAE变化（without-full）：{comparison['candidate_minus_full_mae_deg']:+.4f}°",
    f"- full-without改善量95%CI：[{comparison['bootstrap_ci_lower_deg']:+.4f}, {comparison['bootstrap_ci_upper_deg']:+.4f}]°",
    f"- 无角速度版本改善被试：{comparison['improved_subject_count']}/18",
    f"- 无角速度版本退化被试：{comparison['harmed_subject_count']}/18",
    f"- 正向外折：{comparison['positive_outer_fold_count']}/5",
    f"- 事件恶化比例：{comparison['event_worsened_fraction']:.3f}",
    "",
    "## >=20°幅值档relative MAE变化",
    "",
]
for label, item in amplitude_changes.items():
    lines.append(
        f"- {label}: {item['full']:.4f} → {item['without_rates']:.4f}，变化 {item['change_without_minus_full']:+.4f}"
    )
lines += [
    "",
    "## 判定",
    "",
    f"- subject-macro MAE退化≤0.10°：`{mae_impact_negligible}`",
    f"- 四个>=20°幅值档relative MAE退化均≤0.01：`{amplitude_impact_negligible}`",
]
(OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
