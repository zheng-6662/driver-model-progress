from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POINTS = 20
BASE = "B_all3_without_rates_old_training"
AUGMENTED = "B_all3_without_rates_augmented_training"
FULL_REFERENCE = "B_all3_full_reference"

MANIFEST = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/pfull_event_manifest.csv"
CACHE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/causal_input_cache.npz"
RUN57_SELECTION = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/model_selection.csv"
RUN60_CONFIG = REPO / "05_rebuild_from_raw_20260511/03_baselines/run60_gbm_vs_noise_floor_20260828/config.json"
RUN60_SELECTION = REPO / "05_rebuild_from_raw_20260511/03_baselines/run60_gbm_vs_noise_floor_20260828/tables/model_selection.csv"
RUN75_PREDICTIONS = REPO / "05_rebuild_from_raw_20260511/03_baselines/run75_remove_yaw_roll_rate_ablation_20260831/predictions/per_event_predictions.csv"
CAUSAL_PATH = REPO / "05_rebuild_from_raw_20260511/03_baselines/run56_frozen848_input_fidelity_ladder_20260827/causal_preprocess.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fill_from_train(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.where(np.isfinite(train_x), train_x, median), np.where(np.isfinite(test_x), test_x, median)


def sample_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby(["subject", "episode_id"])["event_uid"].transform("count").astype(float)
    episodes = metadata.groupby("subject")["episode_id"].transform("nunique").astype(float)
    values = 1.0 / counts.to_numpy(float) / episodes.to_numpy(float)
    return values / values.mean()


def fit_points(worker) -> np.ndarray:
    with ThreadPoolExecutor(max_workers=4) as pool:
        output = list(pool.map(worker, range(POINTS)))
    return np.column_stack(output)


def metrics(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    error = np.abs(prediction - truth)
    return pd.DataFrame(
        {
            "curve_mae_deg": error.mean(1),
            "head5_mae_deg": error[:, :5].mean(1),
            "tail5_mae_deg": error[:, -5:].mean(1),
            "endpoint_mae_deg": error[:, -1],
            "peak_time_mae_s": np.abs((np.argmax(np.abs(prediction), axis=1) - np.argmax(np.abs(truth), axis=1)) * 0.05),
        }
    )


def subject_macro(metadata: pd.DataFrame, values: np.ndarray) -> float:
    return float(pd.DataFrame({"subject": metadata["subject"], "value": values}).groupby("subject")["value"].mean().mean())


parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
args = parser.parse_args()
RUN = HERE / args.out_dir
TABLES = RUN / "tables"
OUTPUTS = RUN / "outputs"
PREDICTIONS = RUN / "predictions"
FIGURES = RUN / "figures"
for directory in [TABLES, OUTPUTS, PREDICTIONS, FIGURES]:
    directory.mkdir(parents=True, exist_ok=True)

print("实验目的：把148条8月事件只加入训练，检查原18人固定测试折是否改善。", flush=True)

old = pd.read_csv(
    MANIFEST,
    usecols=["event_uid", "subject", "recording_uid", "outer_fold", "episode_id", "amplitude_bin", "causal_pulse_amplitude_deg_at_release"],
    low_memory=False,
)
with np.load(CACHE, allow_pickle=False) as archive:
    old_full_summary = np.asarray(archive["summary"], dtype=float)
    feature_names = archive["feature_names"].astype(str)
remove = np.array(["__yaw_rate__" in name or "__roll_rate__" in name for name in feature_names])
keep = ~remove
old_summary = old_full_summary[:, keep]
assert old_summary.shape == (2323, 134)

run75 = pd.read_csv(RUN75_PREDICTIONS, encoding="utf-8-sig", low_memory=False)
run75 = old[["event_uid"]].merge(run75, on="event_uid", how="left", validate="one_to_one")
true_columns = [f"true_t{i:02d}_deg" for i in range(1, 21)]
truth = run75[true_columns].to_numpy(float)
old_no_rate_prediction = run75[[f"without_rates_pred_t{i:02d}_deg" for i in range(1, 21)]].to_numpy(float)
full_reference_prediction = run75[[f"full_pred_t{i:02d}_deg" for i in range(1, 21)]].to_numpy(float)

new = pd.read_csv(TABLES / "screened_events.csv", encoding="utf-8-sig", low_memory=False)
new = new.loc[new["screen_eligible"]].reset_index(drop=True)
new_truth = new[true_columns].to_numpy(float)
new["episode_id"] = new["event_uid"]

causal = load_module("run76_causal", CAUSAL_PATH)
new_full_summary = np.full((len(new), 172), np.nan, dtype=float)
event_index = {uid: index for index, uid in enumerate(new["event_uid"].astype(str))}
for source_path, group in new.groupby("source_path", sort=True):
    raw = causal.load_raw_recording(Path(source_path))
    times = raw["time"]
    smooth, rate, support, _, _ = causal.causal_endpoint_savgol(raw["steer_raw"], times, 0.1)
    arrays = {
        "steer_smooth": smooth,
        "steer_rate": rate,
        "speed_kmh": raw["speed_kmh"],
        "ay": raw["ay"],
        "yaw_rate": raw["yaw_rate"],
        "roll_rate": raw["roll_rate"],
        "roll": raw["roll"],
        "curvature": raw["curvature"],
        "lateral_distance": raw["lateral_distance"],
    }
    for row in group.itertuples(index=False):
        release = float(row.prediction_anchor_s)
        direction = float(row.direction)
        grid = np.linspace(release - 2.0, release, 401)
        blocks = {}
        for channel in causal.CHANNELS:
            if channel in {"yaw_rate", "roll_rate"}:
                blocks[channel] = np.zeros_like(grid)
                continue
            if channel in {"curvature", "lateral_distance"} and not bool(row.road_available):
                blocks[channel] = np.zeros_like(grid)
                continue
            source_support = support if channel in {"steer_smooth", "steer_rate"} else None
            block, _ = causal.causal_hold(times, arrays[channel], grid, source_support)
            blocks[channel] = block
        road_valid = bool(row.road_available)
        summary171 = causal.summary_features_from_200hz_blocks(grid, blocks, release, direction, road_valid)
        new_full_summary[event_index[row.event_uid], :171] = summary171
        new_full_summary[event_index[row.event_uid], 171] = 0.0 if road_valid else 1.0
new_summary = new_full_summary[:, keep]
assert new_summary.shape == (148, 134)
assert np.isfinite(new_summary[:, : 5 * 19]).all()

feature_frame = new[["event_uid", "subject", "recording_uid", "session_stamp", "prediction_anchor_s", "road_available"]].copy()
for index, name in enumerate(feature_names[keep]):
    feature_frame[name] = new_summary[:, index]
feature_frame.to_csv(TABLES / "new_event_features.csv", index=False, encoding="utf-8-sig")

run57_selection = pd.read_csv(RUN57_SELECTION)
run60_selection = pd.read_csv(RUN60_SELECTION)
run60_config = json.loads(RUN60_CONFIG.read_text(encoding="utf-8"))
m2_selected = {int(row.outer_fold): json.loads(row.selected_config) for row in run57_selection.loc[run57_selection.model.eq("M0_extratrees")].itertuples(index=False)}
m3_selected = {int(row.outer_fold): {"config": json.loads(row.selected_config_json), "iterations": json.loads(row.selected_best_iterations_json)} for row in run60_selection.loc[run60_selection.model.eq("M3_lgbm")].itertuples(index=False)}
m4_selected = {int(row.outer_fold): json.loads(row.selected_config_json) for row in run60_selection.loc[run60_selection.model.eq("M4_hist")].itertuples(index=False)}

augmented = {
    "M2_augmented": np.full_like(truth, np.nan),
    "M3_augmented": np.full_like(truth, np.nan),
    "M4_augmented": np.full_like(truth, np.nan),
}
training_rows = []

for fold in range(1, 6):
    old_train = np.flatnonzero(~old["outer_fold"].eq(fold).to_numpy())
    old_test = np.flatnonzero(old["outer_fold"].eq(fold).to_numpy())
    test_subjects = set(old.loc[old_test, "subject"].astype(str))
    new_train = np.flatnonzero(~new["subject"].astype(str).isin(test_subjects).to_numpy())
    train_x = np.vstack([old_summary[old_train], new_summary[new_train]])
    train_y = np.vstack([truth[old_train], new_truth[new_train]])
    train_metadata = pd.concat(
        [old.iloc[old_train][["event_uid", "subject", "episode_id"]], new.iloc[new_train][["event_uid", "subject", "episode_id"]]],
        ignore_index=True,
    )
    train_x, test_x = fill_from_train(train_x, old_summary[old_test])
    train_weight = sample_weights(train_metadata)

    cfg2 = m2_selected[fold]
    model2 = ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=int(cfg2["min_samples_leaf"]),
        max_features=float(cfg2["max_features"]),
        random_state=20260827 + fold * 1000,
        n_jobs=8,
    )
    model2.fit(train_x, train_y, sample_weight=train_weight)
    augmented["M2_augmented"][old_test] = model2.predict(test_x)
    print(f"fold {fold}: M2完成", flush=True)

    cfg3 = m3_selected[fold]["config"]
    iterations = m3_selected[fold]["iterations"]
    fixed3 = run60_config["models"]["M3_lgbm"]["fixed"]

    def lgb_point(point: int) -> np.ndarray:
        model = lgb.LGBMRegressor(
            objective=fixed3["objective"], metric=fixed3["metric"], n_estimators=max(1, int(iterations[point])),
            learning_rate=float(cfg3["learning_rate"]), num_leaves=int(cfg3["num_leaves"]),
            min_child_samples=int(cfg3["min_child_samples"]), max_depth=int(fixed3["max_depth"]),
            subsample=1.0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=1.0,
            deterministic=True, force_col_wise=True, n_jobs=1, verbosity=-1,
            random_state=20260828 + fold * 100000 + point,
        )
        model.fit(train_x, train_y[:, point], sample_weight=train_weight)
        return model.predict(test_x)

    augmented["M3_augmented"][old_test] = fit_points(lgb_point)
    print(f"fold {fold}: M3完成", flush=True)

    cfg4 = m4_selected[fold]
    fixed4 = run60_config["models"]["M4_hist"]["fixed"]

    def hist_point(point: int) -> np.ndarray:
        model = HistGradientBoostingRegressor(
            loss=fixed4["loss"], learning_rate=float(cfg4["learning_rate"]), max_iter=int(fixed4["max_iter"]),
            max_leaf_nodes=int(cfg4["max_leaf_nodes"]), min_samples_leaf=int(fixed4["min_samples_leaf"]),
            l2_regularization=float(fixed4["l2_regularization"]), max_bins=int(fixed4["max_bins"]),
            early_stopping=False, random_state=20260828 + fold * 100000 + point,
        )
        model.fit(train_x, train_y[:, point], sample_weight=train_weight)
        return model.predict(test_x)

    augmented["M4_augmented"][old_test] = fit_points(hist_point)
    print(f"fold {fold}: M4完成", flush=True)
    training_rows.append(
        {
            "outer_fold": fold,
            "old_train_events": len(old_train),
            "august_train_events": len(new_train),
            "august_train_subjects": int(new.iloc[new_train]["subject"].nunique()),
            "excluded_august_test_subjects": ";".join(sorted(test_subjects & set(new["subject"]))),
            "test_events": len(old_test),
        }
    )

for values in augmented.values():
    assert np.isfinite(values).all()
augmented_prediction = sum(augmented.values()) / 3.0

model_predictions = {
    FULL_REFERENCE: full_reference_prediction,
    BASE: old_no_rate_prediction,
    AUGMENTED: augmented_prediction,
}
aggregate_rows = []
subject_rows = []
event_tables = {}
for model, prediction in model_predictions.items():
    table = metrics(truth, prediction)
    event_tables[model] = table
    aggregate_rows.append(
        {
            "model": model,
            "subject_macro_curve_mae_deg": subject_macro(old, table.curve_mae_deg.to_numpy()),
            "subject_macro_head5_mae_deg": subject_macro(old, table.head5_mae_deg.to_numpy()),
            "subject_macro_tail5_mae_deg": subject_macro(old, table.tail5_mae_deg.to_numpy()),
            "subject_macro_endpoint_mae_deg": subject_macro(old, table.endpoint_mae_deg.to_numpy()),
            "subject_macro_peak_time_mae_s": subject_macro(old, table.peak_time_mae_s.to_numpy()),
            "pooled_curve_mae_deg_reference_only": float(table.curve_mae_deg.mean()),
        }
    )
    for subject, indices in old.groupby("subject").groups.items():
        idx = np.asarray(list(indices), int)
        subject_rows.append({"model": model, "subject": subject, "outer_fold": int(old.iloc[idx[0]].outer_fold), "event_count": len(idx), "curve_mae_deg": float(table.iloc[idx].curve_mae_deg.mean())})

aggregate = pd.DataFrame(aggregate_rows)
subjects = pd.DataFrame(subject_rows)
aggregate.to_csv(TABLES / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
subjects.to_csv(TABLES / "subject_metrics.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(training_rows).to_csv(TABLES / "training_folds.csv", index=False, encoding="utf-8-sig")

amplitude = pd.to_numeric(old.causal_pulse_amplitude_deg_at_release, errors="raise").to_numpy(float)
amplitude_rows = []
for model in [BASE, AUGMENTED]:
    mae = event_tables[model].curve_mae_deg.to_numpy(float)
    for label in ["20_30", "30_45", "45_70", "ge70"]:
        mask = old.amplitude_bin.eq(label).to_numpy()
        values = []
        for subject in sorted(old.loc[mask, "subject"].unique()):
            sm = mask & old.subject.eq(subject).to_numpy()
            values.append(float(mae[sm].mean() / np.median(amplitude[sm])))
        amplitude_rows.append({"model": model, "amplitude_bin": label, "event_count": int(mask.sum()), "subject_macro_relative_mae": float(np.mean(values))})
amplitude_table = pd.DataFrame(amplitude_rows)
amplitude_table.to_csv(TABLES / "relative_mae_by_amplitude_bin.csv", index=False, encoding="utf-8-sig")

pivot = subjects[subjects.model.isin([BASE, AUGMENTED])].pivot(index="subject", columns="model", values="curve_mae_deg")
improvement = pivot[BASE] - pivot[AUGMENTED]
rng = np.random.default_rng(20260831)
values = improvement.to_numpy(float)
draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
subject_fold = old.drop_duplicates("subject").set_index("subject").outer_fold
fold_improvement = pd.DataFrame({"improvement": improvement, "fold": improvement.index.map(subject_fold)}).groupby("fold").improvement.mean()

amplitude_index = amplitude_table.set_index(["model", "amplitude_bin"]).subject_macro_relative_mae
amplitude_changes = {}
for label in ["20_30", "30_45", "45_70", "ge70"]:
    base_value = float(amplitude_index.loc[(BASE, label)])
    augmented_value = float(amplitude_index.loc[(AUGMENTED, label)])
    amplitude_changes[label] = {"old_training": base_value, "augmented_training": augmented_value, "change": augmented_value - base_value}

comparison = {
    "subject_macro_improvement_deg": float(values.mean()),
    "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
    "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
    "positive_outer_fold_count": int((fold_improvement > 0).sum()),
    "outer_fold_improvement_deg": {str(int(fold)): float(value) for fold, value in fold_improvement.items()},
    "improved_subject_count": int((values > 0).sum()),
    "harmed_subject_count": int((values < 0).sum()),
    "event_worsened_fraction": float(np.mean(event_tables[AUGMENTED].curve_mae_deg.to_numpy() > event_tables[BASE].curve_mae_deg.to_numpy())),
}
gates = {
    "subject_macro_improvement_at_least_0_05_deg": comparison["subject_macro_improvement_deg"] >= 0.05,
    "bootstrap_ci_lower_above_zero": comparison["bootstrap_ci_lower_deg"] > 0,
    "positive_outer_folds_at_least_4": comparison["positive_outer_fold_count"] >= 4,
    "no_ge20_amplitude_relative_regression_over_0_01": all(item["change"] <= 0.01 for item in amplitude_changes.values()),
}
decision = {
    "status": "AUGMENTED_TRAINING_EFFECTIVE" if all(gates.values()) else "AUGMENTED_TRAINING_NOT_EFFECTIVE",
    "old_events": 2323,
    "august_training_events": len(new),
    "august_subjects": int(new.subject.nunique()),
    "comparison": comparison,
    "amplitude_changes": amplitude_changes,
    "gates": gates,
}
(OUTPUTS / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

output = old[["event_uid", "subject", "recording_uid", "outer_fold", "amplitude_bin"]].copy()
for point in range(POINTS):
    output[f"true_t{point+1:02d}_deg"] = truth[:, point]
    output[f"old_training_pred_t{point+1:02d}_deg"] = old_no_rate_prediction[:, point]
    output[f"augmented_training_pred_t{point+1:02d}_deg"] = augmented_prediction[:, point]
output["old_training_curve_mae_deg"] = event_tables[BASE].curve_mae_deg.to_numpy()
output["augmented_training_curve_mae_deg"] = event_tables[AUGMENTED].curve_mae_deg.to_numpy()
output.to_csv(PREDICTIONS / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 5))
gain = improvement.sort_index()
ax.bar(gain.index, gain.values, color=np.where(gain.values >= 0, "#1976D2", "#D95F59"))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("MAE improvement from August training data (deg)")
ax.set_title("Subject-level impact on the original 18 subjects")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(FIGURES / "Figure_1_subject_gain.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["20_30", "30_45", "45_70", "ge70"]
ax.plot(labels, [amplitude_index.loc[(BASE, label)] for label in labels], marker="o", label="old training")
ax.plot(labels, [amplitude_index.loc[(AUGMENTED, label)] for label in labels], marker="o", label="+ August training")
ax.set_ylabel("Subject-macro relative MAE")
ax.set_title("Amplitude-stratified effect of August training data")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "Figure_2_amplitude_relative_mae.png", dpi=180)
plt.close(fig)

aggregate_index = aggregate.set_index("model")
lines = [
    "# 8月被试加入训练后的原18人测试结果",
    "",
    f"- 状态：`{decision['status']}`",
    f"- 加入训练的8月事件：{len(new)}，被试：{new.subject.nunique()}。",
    "- 评估仍只在原18人固定外折测试事件上进行。",
    "",
    "| 模型 | subject-macro MAE° | head5° | tail5° | endpoint° | peak-time s | pooled MAE°参考 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model in [FULL_REFERENCE, BASE, AUGMENTED]:
    row = aggregate_index.loc[model]
    lines.append(f"| {model} | {row.subject_macro_curve_mae_deg:.4f} | {row.subject_macro_head5_mae_deg:.4f} | {row.subject_macro_tail5_mae_deg:.4f} | {row.subject_macro_endpoint_mae_deg:.4f} | {row.subject_macro_peak_time_mae_s:.4f} | {row.pooled_curve_mae_deg_reference_only:.4f} |")
lines += [
    "",
    "## 配对比较（augmented vs old no-rate）",
    "",
    f"- subject-macro改善：{comparison['subject_macro_improvement_deg']:+.4f}°",
    f"- 95%CI：[{comparison['bootstrap_ci_lower_deg']:+.4f}, {comparison['bootstrap_ci_upper_deg']:+.4f}]°",
    f"- 正向外折：{comparison['positive_outer_fold_count']}/5",
    f"- 改善/退化被试：{comparison['improved_subject_count']}/18，{comparison['harmed_subject_count']}/18",
    f"- 事件恶化比例：{comparison['event_worsened_fraction']:.3f}",
    "",
    "## >=20°幅值档变化",
    "",
]
for label, item in amplitude_changes.items():
    lines.append(f"- {label}: {item['old_training']:.4f} → {item['augmented_training']:.4f}，变化 {item['change']:+.4f}")
lines += ["", "## 门", ""]
for name, value in gates.items():
    lines.append(f"- {name}: `{value}`")
(OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
