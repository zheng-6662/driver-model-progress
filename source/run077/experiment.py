from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
POINTS = 20
RUN = HERE / "run_1"
TABLES = RUN / "tables"
OUTPUTS = RUN / "outputs"
PREDICTIONS = RUN / "predictions"
FIGURES = RUN / "figures"

SCREEN_ROOT = REPO / "05_rebuild_from_raw_20260511/03_baselines/run76_august_subject_augmented_training_20260831/run_1"
PHYSIO_CODE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run64_physio_style_regret_distillation_20260829/scripts/build_multimodal_features.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fill(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.where(np.isfinite(train_x), train_x, median), np.where(np.isfinite(test_x), test_x, median)


def event_metrics(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
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
    return float(pd.DataFrame({"subject": metadata.subject, "value": values}).groupby("subject").value.mean().mean())


for directory in [TABLES, OUTPUTS, PREDICTIONS, FIGURES]:
    directory.mkdir(parents=True, exist_ok=True)

events = pd.read_csv(SCREEN_ROOT / "tables/screened_events.csv", encoding="utf-8-sig", low_memory=False)
events = events.loc[events.screen_eligible].reset_index(drop=True)
vehicle_features = pd.read_csv(SCREEN_ROOT / "tables/new_event_features.csv", encoding="utf-8-sig", low_memory=False)
files = pd.read_csv(SCREEN_ROOT / "tables/file_summary.csv", encoding="utf-8-sig", low_memory=False)
feature_columns = [column for column in vehicle_features.columns if column.startswith("v0__") or column == "road_reference_missing_indicator"]
vehicle_features = events[["event_uid"]].merge(vehicle_features[["event_uid", *feature_columns]], on="event_uid", how="left", validate="one_to_one")
vehicle_x = vehicle_features[feature_columns].to_numpy(float)
truth_columns = [f"true_t{i:02d}_deg" for i in range(1, 21)]
truth = events[truth_columns].to_numpy(float)
assert len(events) == 148 and events.subject.nunique() == 18 and vehicle_x.shape == (148, 134)

physio_builder = load_module("run77_physio", PHYSIO_CODE)
file_index = files.set_index(["subject", "session_stamp"])
physio_rows = []
bad_timestamp_rows = 0
for (subject, stamp), group in events.groupby(["subject", "session_stamp"], sort=True):
    info = file_index.loc[(subject, stamp)]
    physio_path = str(info.physio_path) if pd.notna(info.physio_path) else ""
    signals = None
    times = None
    signal_fs = None
    if physio_path:
        raw = pd.read_csv(
            physio_path,
            usecols=lambda column: column == "StorageTime" or "PhysioLAB Pro1" in column,
            encoding="utf-8-sig",
            low_memory=False,
        )
        channel_map = {
            "ECG": next(column for column in raw.columns if "|CH1-ECG" in column),
            "EMG": next(column for column in raw.columns if "|CH2-EMG" in column),
            "EDA": next(column for column in raw.columns if "|CH3-EDA" in column),
            "RESP": next(column for column in raw.columns if "|CH4-RESP" in column),
        }
        absolute_time = pd.to_datetime(raw.StorageTime, format="mixed", errors="coerce")
        vehicle_start = pd.to_datetime(group.prediction_anchor_time.iloc[0]) - pd.to_timedelta(group.prediction_anchor_s.iloc[0], unit="s")
        relative_time = (absolute_time - vehicle_start).dt.total_seconds().to_numpy(float)
        bad_timestamp_rows += int((~np.isfinite(relative_time)).sum())
        signals = {}
        times = {}
        signal_fs = {}
        for name, column in channel_map.items():
            values = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(relative_time)
            signals[name] = values[valid]
            times[name] = relative_time[valid]
            signal_fs[name] = float(1.0 / np.median(np.diff(times[name])))

    for row in group.itertuples(index=False):
        if signals is None:
            output = {name: np.nan for name in physio_builder.PHYSIO_FEATURES}
            for name in physio_builder.PHYSIO_FEATURES[-4:]:
                output[name] = 0.0
        else:
            input_row = pd.Series(
                {
                    "primary_release_s": float(row.prediction_anchor_s),
                    "physio_sync_offset_s": 0.0,
                    "physio_emg_recording_usable": True,
                    "physio_eda_recording_usable": True,
                    "physio_hr_recording_usable": True,
                    "physio_resp_recording_usable": True,
                }
            )
            output = physio_builder._feature_row(input_row, signals, times, signal_fs)
        output.update({"event_uid": row.event_uid, "subject": subject, "session_stamp": stamp})
        physio_rows.append(output)

physio = pd.DataFrame(physio_rows)
physio = events[["event_uid"]].merge(physio, on="event_uid", how="left", validate="one_to_one")
physio_columns = list(physio_builder.PHYSIO_FEATURES)
physio_x = physio[physio_columns].to_numpy(float)
physio.to_csv(TABLES / "physio_features.csv", index=False, encoding="utf-8-sig")

counts = events.groupby("subject").size().sort_values(ascending=False)
fold_event_counts = {fold: 0 for fold in range(1, 6)}
subject_fold = {}
for subject, count in counts.items():
    fold = min(fold_event_counts, key=lambda item: (fold_event_counts[item], item))
    subject_fold[subject] = fold
    fold_event_counts[fold] += int(count)
events["outer_fold"] = events.subject.map(subject_fold).astype(int)
events["episode_id"] = events.event_uid
pd.DataFrame({"subject": sorted(subject_fold), "outer_fold": [subject_fold[s] for s in sorted(subject_fold)], "event_count": [int(counts[s]) for s in sorted(subject_fold)]}).to_csv(TABLES / "subject_folds.csv", index=False, encoding="utf-8-sig")

predictions = {
    "V_vehicle": np.full_like(truth, np.nan),
    "VP_vehicle_physio": np.full_like(truth, np.nan),
}
training_rows = []
for fold in range(1, 6):
    train = np.flatnonzero(~events.outer_fold.eq(fold).to_numpy())
    test = np.flatnonzero(events.outer_fold.eq(fold).to_numpy())
    train_weights = 1.0 / events.iloc[train].groupby("subject").event_uid.transform("count").to_numpy(float)
    train_weights /= train_weights.mean()
    for model_name, matrix in {
        "V_vehicle": vehicle_x,
        "VP_vehicle_physio": np.column_stack([vehicle_x, physio_x]),
    }.items():
        train_x, test_x = fill(matrix[train], matrix[test])
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=20260831 + fold,
            n_jobs=8,
        )
        model.fit(train_x, truth[train], sample_weight=train_weights)
        predictions[model_name][test] = model.predict(test_x)
    training_rows.append({"outer_fold": fold, "train_subjects": events.iloc[train].subject.nunique(), "test_subjects": events.iloc[test].subject.nunique(), "train_events": len(train), "test_events": len(test)})
    print(f"fold {fold}/5 完成", flush=True)

for prediction in predictions.values():
    assert np.isfinite(prediction).all()
pd.DataFrame(training_rows).to_csv(TABLES / "training_folds.csv", index=False, encoding="utf-8-sig")

aggregate_rows = []
subject_rows = []
metric_tables = {}
for model_name, prediction in predictions.items():
    table = event_metrics(truth, prediction)
    metric_tables[model_name] = table
    aggregate_rows.append(
        {
            "model": model_name,
            "subject_macro_curve_mae_deg": subject_macro(events, table.curve_mae_deg.to_numpy()),
            "subject_macro_head5_mae_deg": subject_macro(events, table.head5_mae_deg.to_numpy()),
            "subject_macro_tail5_mae_deg": subject_macro(events, table.tail5_mae_deg.to_numpy()),
            "subject_macro_endpoint_mae_deg": subject_macro(events, table.endpoint_mae_deg.to_numpy()),
            "subject_macro_peak_time_mae_s": subject_macro(events, table.peak_time_mae_s.to_numpy()),
            "pooled_curve_mae_deg_reference_only": float(table.curve_mae_deg.mean()),
        }
    )
    for subject, indices in events.groupby("subject").groups.items():
        idx = np.asarray(list(indices), int)
        subject_rows.append({"model": model_name, "subject": subject, "outer_fold": int(events.iloc[idx[0]].outer_fold), "event_count": len(idx), "curve_mae_deg": float(table.iloc[idx].curve_mae_deg.mean())})

aggregate = pd.DataFrame(aggregate_rows)
subjects = pd.DataFrame(subject_rows)
aggregate.to_csv(TABLES / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
subjects.to_csv(TABLES / "subject_metrics.csv", index=False, encoding="utf-8-sig")

amplitude = pd.to_numeric(events.pulse_amplitude_deg_report_only, errors="raise").to_numpy(float)
bins = pd.cut(amplitude, [0, 20, 30, 45, 70, np.inf], labels=["lt20", "20_30", "30_45", "45_70", "ge70"], right=False).astype(str)
events["amplitude_bin"] = bins
amplitude_rows = []
for model_name in predictions:
    mae = metric_tables[model_name].curve_mae_deg.to_numpy(float)
    for label in ["20_30", "30_45", "45_70", "ge70"]:
        mask = events.amplitude_bin.eq(label).to_numpy()
        if not mask.any():
            continue
        values = []
        for subject in events.loc[mask, "subject"].unique():
            sm = mask & events.subject.eq(subject).to_numpy()
            values.append(float(mae[sm].mean() / np.median(amplitude[sm])))
        amplitude_rows.append({"model": model_name, "amplitude_bin": label, "event_count": int(mask.sum()), "subject_count": len(values), "subject_macro_relative_mae": float(np.mean(values))})
amplitude_table = pd.DataFrame(amplitude_rows)
amplitude_table.to_csv(TABLES / "relative_mae_by_amplitude_bin.csv", index=False, encoding="utf-8-sig")

pivot = subjects.pivot(index="subject", columns="model", values="curve_mae_deg")
improvement = pivot["V_vehicle"] - pivot["VP_vehicle_physio"]
values = improvement.to_numpy(float)
rng = np.random.default_rng(20260831)
draws = np.asarray([rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)])
fold_improvement = pd.DataFrame({"improvement": improvement, "fold": improvement.index.map(pd.Series(subject_fold))}).groupby("fold").improvement.mean()
amplitude_index = amplitude_table.set_index(["model", "amplitude_bin"]).subject_macro_relative_mae
amplitude_changes = {}
for label in sorted(set(amplitude_table.amplitude_bin)):
    base = float(amplitude_index.loc[("V_vehicle", label)])
    candidate = float(amplitude_index.loc[("VP_vehicle_physio", label)])
    amplitude_changes[label] = {"vehicle": base, "vehicle_physio": candidate, "change": candidate - base}

comparison = {
    "subject_macro_improvement_deg": float(values.mean()),
    "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
    "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
    "positive_outer_fold_count": int((fold_improvement > 0).sum()),
    "outer_fold_improvement_deg": {str(int(fold)): float(value) for fold, value in fold_improvement.items()},
    "improved_subject_count": int((values > 0).sum()),
    "harmed_subject_count": int((values < 0).sum()),
    "event_worsened_fraction": float(np.mean(metric_tables["VP_vehicle_physio"].curve_mae_deg.to_numpy() > metric_tables["V_vehicle"].curve_mae_deg.to_numpy())),
}
gates = {
    "subject_macro_improvement_at_least_0_05_deg": comparison["subject_macro_improvement_deg"] >= 0.05,
    "bootstrap_ci_lower_above_zero": comparison["bootstrap_ci_lower_deg"] > 0,
    "positive_outer_folds_at_least_4": comparison["positive_outer_fold_count"] >= 4,
    "no_amplitude_relative_regression_over_0_01": all(item["change"] <= 0.01 for item in amplitude_changes.values()),
}
decision = {
    "status": "PHYSIO_INCREMENT_EFFECTIVE" if all(gates.values()) else "PHYSIO_INCREMENT_NOT_EFFECTIVE",
    "events": len(events),
    "subjects": events.subject.nunique(),
    "physio_feature_count": len(physio_columns),
    "bad_physio_timestamp_rows": bad_timestamp_rows,
    "comparison": comparison,
    "amplitude_changes": amplitude_changes,
    "gates": gates,
}
(OUTPUTS / "decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

output = events[["event_uid", "subject", "recording_uid", "outer_fold", "amplitude_bin", "physio_2s_coverage", "physio_30s_coverage"]].copy()
for point in range(POINTS):
    output[f"true_t{point+1:02d}_deg"] = truth[:, point]
    output[f"vehicle_pred_t{point+1:02d}_deg"] = predictions["V_vehicle"][:, point]
    output[f"vehicle_physio_pred_t{point+1:02d}_deg"] = predictions["VP_vehicle_physio"][:, point]
output["vehicle_curve_mae_deg"] = metric_tables["V_vehicle"].curve_mae_deg.to_numpy()
output["vehicle_physio_curve_mae_deg"] = metric_tables["VP_vehicle_physio"].curve_mae_deg.to_numpy()
output.to_csv(PREDICTIONS / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 5))
gain = improvement.sort_index()
ax.bar(gain.index, gain.values, color=np.where(gain.values >= 0, "#1976D2", "#D95F59"))
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Vehicle - vehicle+physio MAE improvement (deg)")
ax.set_title("August subjects: physiology increment by subject")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(FIGURES / "Figure_1_subject_physio_gain.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
labels = sorted(set(amplitude_table.amplitude_bin), key=lambda x: ["20_30", "30_45", "45_70", "ge70"].index(x))
ax.plot(labels, [amplitude_index.loc[("V_vehicle", label)] for label in labels], marker="o", label="vehicle")
ax.plot(labels, [amplitude_index.loc[("VP_vehicle_physio", label)] for label in labels], marker="o", label="vehicle+physio")
ax.set_ylabel("Subject-macro relative MAE")
ax.set_title("August subjects: amplitude-stratified physiology comparison")
ax.legend(frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "Figure_2_amplitude_relative_mae.png", dpi=180)
plt.close(fig)

aggregate_index = aggregate.set_index("model")
lines = [
    "# 8月18名被试内部车辆+生理实验",
    "",
    f"- 状态：`{decision['status']}`",
    f"- 事件：{len(events)}，被试：{events.subject.nunique()}，被试不相交5折。",
    f"- 生理特征：{len(physio_columns)}维；所有事件保留。",
    "",
    "| 模型 | subject-macro MAE° | head5° | tail5° | endpoint° | peak-time s | pooled MAE°参考 |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for model_name in ["V_vehicle", "VP_vehicle_physio"]:
    row = aggregate_index.loc[model_name]
    lines.append(f"| {model_name} | {row.subject_macro_curve_mae_deg:.4f} | {row.subject_macro_head5_mae_deg:.4f} | {row.subject_macro_tail5_mae_deg:.4f} | {row.subject_macro_endpoint_mae_deg:.4f} | {row.subject_macro_peak_time_mae_s:.4f} | {row.pooled_curve_mae_deg_reference_only:.4f} |")
lines += [
    "",
    "## 配对比较",
    "",
    f"- 生理改善：{comparison['subject_macro_improvement_deg']:+.4f}°",
    f"- 95%CI：[{comparison['bootstrap_ci_lower_deg']:+.4f}, {comparison['bootstrap_ci_upper_deg']:+.4f}]°",
    f"- 正向外折：{comparison['positive_outer_fold_count']}/5",
    f"- 改善/退化被试：{comparison['improved_subject_count']}/18，{comparison['harmed_subject_count']}/18",
    f"- 事件恶化比例：{comparison['event_worsened_fraction']:.3f}",
    "",
    "## 幅值档变化",
    "",
]
for label, item in amplitude_changes.items():
    lines.append(f"- {label}: {item['vehicle']:.4f} → {item['vehicle_physio']:.4f}，变化 {item['change']:+.4f}")
lines += ["", "## 门", ""]
for name, value in gates.items():
    lines.append(f"- {name}: `{value}`")
(OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
