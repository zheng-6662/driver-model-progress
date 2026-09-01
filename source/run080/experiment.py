from __future__ import annotations

"""Run80：在同一八月事件集上比较旧生理特征与Run79清洗特征。"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RUN = HERE / "run_1"
TABLES = RUN / "tables"
OUTPUTS = RUN / "outputs"
PREDICTIONS = RUN / "predictions"
FIGURES = RUN / "figures"
LOGS = RUN / "logs"

EVENT_PATH = Path(CONFIG["event_table"])
CLEAN_PATH = Path(CONFIG["clean_physio_table"])
RUN79_INVENTORY = REPO / "05_rebuild_from_raw_20260511/03_baselines/run79_august_physio_preprocessing_20260831/run_1/tables/source_inventory.csv"
RUN57_CACHE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1/tables/causal_input_cache.npz"
CAUSAL_CODE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run56_frozen848_input_fidelity_ladder_20260827/causal_preprocess.py"
OLD_PHYSIO_CODE = REPO / "05_rebuild_from_raw_20260511/03_baselines/run64_physio_style_regret_distillation_20260829/scripts/build_multimodal_features.py"

POINTS = 20
MODEL_NAMES = ["V_vehicle", "VP_old16", "VP_clean16"]


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    if specification.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    specification.loader.exec_module(module)
    return module


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fill_from_training(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return (
        np.where(np.isfinite(train_x), train_x, median),
        np.where(np.isfinite(test_x), test_x, median),
    )


def prepare_model_input(
    vehicle_train: np.ndarray,
    vehicle_test: np.ndarray,
    physio_train: np.ndarray | None,
    physio_test: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    vehicle_train_filled, vehicle_test_filled = fill_from_training(vehicle_train, vehicle_test)
    if physio_train is None or physio_test is None:
        return vehicle_train_filled, vehicle_test_filled
    train_missing = ~np.isfinite(physio_train)
    test_missing = ~np.isfinite(physio_test)
    physio_train_filled, physio_test_filled = fill_from_training(physio_train, physio_test)
    return (
        np.column_stack([vehicle_train_filled, physio_train_filled, train_missing.astype(float)]),
        np.column_stack([vehicle_test_filled, physio_test_filled, test_missing.astype(float)]),
    )


def subject_weights(metadata: pd.DataFrame) -> np.ndarray:
    counts = metadata.groupby("subject")["event_uid"].transform("count").to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.mean()


def event_metrics(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    absolute_error = np.abs(prediction - truth)
    return pd.DataFrame(
        {
            "curve_mae_deg": absolute_error.mean(axis=1),
            "head5_mae_deg": absolute_error[:, :5].mean(axis=1),
            "tail5_mae_deg": absolute_error[:, -5:].mean(axis=1),
            "endpoint_mae_deg": absolute_error[:, -1],
            "peak_time_mae_s": np.abs(
                (np.argmax(np.abs(prediction), axis=1) - np.argmax(np.abs(truth), axis=1)) * 0.05
            ),
        }
    )


def subject_macro(metadata: pd.DataFrame, values: np.ndarray) -> float:
    table = pd.DataFrame({"subject": metadata["subject"], "value": values})
    return float(table.groupby("subject")["value"].mean().mean())


def build_vehicle_features(events: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    causal = load_module("run80_causal", CAUSAL_CODE)
    with np.load(RUN57_CACHE, allow_pickle=False) as archive:
        feature_names = archive["feature_names"].astype(str)
    remove = np.asarray(
        ["__yaw_rate__" in name or "__roll_rate__" in name for name in feature_names], dtype=bool
    )
    keep = ~remove
    full = np.full((len(events), 172), np.nan, dtype=float)
    event_index = {uid: index for index, uid in enumerate(events["event_uid"].astype(str))}

    for source_path, group in events.groupby("source_path", sort=True):
        raw = causal.load_raw_recording(Path(str(source_path)))
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
            blocks: dict[str, np.ndarray] = {}
            road_available = as_bool(row.road_available)
            for channel in causal.CHANNELS:
                if channel in {"yaw_rate", "roll_rate"}:
                    blocks[channel] = np.zeros_like(grid)
                    continue
                if channel in {"curvature", "lateral_distance"} and not road_available:
                    blocks[channel] = np.zeros_like(grid)
                    continue
                source_support = support if channel in {"steer_smooth", "steer_rate"} else None
                block, _ = causal.causal_hold(times, arrays[channel], grid, source_support)
                blocks[channel] = block
            summary171 = causal.summary_features_from_200hz_blocks(
                grid, blocks, release, direction, road_available
            )
            index = event_index[str(row.event_uid)]
            full[index, :171] = summary171
            full[index, 171] = 0.0 if road_available else 1.0
    output = full[:, keep]
    if output.shape != (len(events), 134):
        raise ValueError(f"车辆特征维度错误: {output.shape}")
    return output, feature_names[keep].tolist()


def build_old_physio_features(events: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    builder = load_module("run80_old_physio", OLD_PHYSIO_CODE)
    inventory = pd.read_csv(RUN79_INVENTORY, encoding="utf-8-sig")
    source_index = inventory.set_index(["subject", "session_stamp"])["source_path"].to_dict()
    rows: list[dict[str, object]] = []
    suffixes = {
        "ECG": "|CH1-ECG",
        "EMG": "|CH2-EMG",
        "EDA": "|CH3-EDA",
        "RESP": "|CH4-RESP",
    }
    for (subject, stamp), group in events.groupby(["subject", "session_stamp"], sort=True):
        source = source_index.get((str(subject), str(stamp)))
        signals = None
        times = None
        signal_fs = None
        if source is not None:
            path = Path(str(source))
            header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
            mapping = {
                name: next(column for column in header if column.lower().endswith(suffix.lower()))
                for name, suffix in suffixes.items()
            }
            raw = pd.read_csv(
                path,
                usecols=["StorageTime", *mapping.values()],
                encoding="utf-8-sig",
                low_memory=False,
            )
            absolute_time = pd.to_datetime(raw["StorageTime"], format="mixed", errors="coerce")
            first_event = group.iloc[0]
            vehicle_start = pd.Timestamp(first_event["prediction_anchor_time"]) - pd.to_timedelta(
                float(first_event["prediction_anchor_s"]), unit="s"
            )
            relative_time = (absolute_time - vehicle_start).dt.total_seconds().to_numpy(float)
            signals = {}
            times = {}
            signal_fs = {}
            for name, column in mapping.items():
                values = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
                valid = np.isfinite(values) & np.isfinite(relative_time)
                channel_time = relative_time[valid]
                channel_values = values[valid]
                order = np.argsort(channel_time, kind="stable")
                channel_time = channel_time[order]
                channel_values = channel_values[order]
                unique = np.r_[True, np.diff(channel_time) > 0]
                times[name] = channel_time[unique]
                signals[name] = channel_values[unique]
                signal_fs[name] = float(1.0 / np.median(np.diff(times[name])))

        for event in group.itertuples(index=False):
            if signals is None or times is None or signal_fs is None:
                output = {name: np.nan for name in builder.PHYSIO_FEATURES}
                for name in builder.PHYSIO_FEATURES[-4:]:
                    output[name] = 0.0
            else:
                input_row = pd.Series(
                    {
                        "primary_release_s": float(event.prediction_anchor_s),
                        "physio_sync_offset_s": 0.0,
                        "physio_emg_recording_usable": True,
                        "physio_eda_recording_usable": True,
                        "physio_hr_recording_usable": True,
                        "physio_resp_recording_usable": True,
                    }
                )
                output = builder._feature_row(input_row, signals, times, signal_fs)
            output.update(
                {
                    "event_uid": event.event_uid,
                    "subject": subject,
                    "session_stamp": stamp,
                    "old_physio_source_available": source is not None,
                }
            )
            rows.append(output)
    table = pd.DataFrame(rows)
    table = events[["event_uid"]].merge(table, on="event_uid", how="left", validate="one_to_one")
    return table, list(builder.PHYSIO_FEATURES)


def greedy_subject_folds(events: pd.DataFrame) -> dict[str, int]:
    counts = events.groupby("subject").size().sort_values(ascending=False)
    fold_counts = {fold: 0 for fold in range(1, int(CONFIG["outer_split"]["folds"]) + 1)}
    assignment: dict[str, int] = {}
    for subject, count in counts.items():
        fold = min(fold_counts, key=lambda candidate: (fold_counts[candidate], candidate))
        assignment[str(subject)] = int(fold)
        fold_counts[fold] += int(count)
    return assignment


def amplitude_metrics(
    events: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    amplitude: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, metrics in metric_tables.items():
        mae = metrics["curve_mae_deg"].to_numpy(float)
        for label in ["20_30", "30_45", "45_70", "ge70"]:
            mask = events["amplitude_bin"].eq(label).to_numpy()
            subject_values = []
            for subject in events.loc[mask, "subject"].unique():
                subject_mask = mask & events["subject"].eq(subject).to_numpy()
                subject_values.append(float(mae[subject_mask].mean() / np.median(amplitude[subject_mask])))
            rows.append(
                {
                    "model": model,
                    "amplitude_bin": label,
                    "event_count": int(mask.sum()),
                    "subject_count": len(subject_values),
                    "subject_macro_relative_mae": float(np.mean(subject_values)),
                }
            )
    return pd.DataFrame(rows)


def paired_comparison(
    comparator: str,
    candidate: str,
    subject_table: pd.DataFrame,
    events: pd.DataFrame,
    metric_tables: dict[str, pd.DataFrame],
    amplitude_table: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    pivot = subject_table.pivot(index="subject", columns="model", values="curve_mae_deg")
    improvement = pivot[comparator] - pivot[candidate]
    values = improvement.to_numpy(float)
    draws = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(int(CONFIG["bootstrap"]["draws"]))],
        dtype=float,
    )
    fold_series = improvement.index.map(events.drop_duplicates("subject").set_index("subject")["outer_fold"])
    fold_improvement = pd.DataFrame(
        {"fold": fold_series.to_numpy(int), "improvement": values}
    ).groupby("fold")["improvement"].mean()

    amplitude_index = amplitude_table.set_index(["model", "amplitude_bin"])["subject_macro_relative_mae"]
    amplitude_changes: dict[str, dict[str, float]] = {}
    for label in ["20_30", "30_45", "45_70", "ge70"]:
        base = float(amplitude_index.loc[(comparator, label)])
        current = float(amplitude_index.loc[(candidate, label)])
        amplitude_changes[label] = {
            "comparator": base,
            "candidate": current,
            "change_candidate_minus_comparator": current - base,
        }

    event_base = metric_tables[comparator]["curve_mae_deg"].to_numpy(float)
    event_candidate = metric_tables[candidate]["curve_mae_deg"].to_numpy(float)
    summary: dict[str, object] = {
        "comparison": f"{candidate}_vs_{comparator}",
        "comparator": comparator,
        "candidate": candidate,
        "subject_macro_improvement_deg": float(values.mean()),
        "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
        "positive_outer_fold_count": int((fold_improvement > 0).sum()),
        "outer_fold_improvement_deg": {
            str(int(fold)): float(value) for fold, value in fold_improvement.items()
        },
        "improved_subject_count": int((values > 0).sum()),
        "harmed_subject_count": int((values < 0).sum()),
        "event_worsened_fraction": float(np.mean(event_candidate > event_base)),
        "amplitude_changes": amplitude_changes,
    }
    gate_config = CONFIG["gates"]
    gates = {
        "subject_macro_improvement_at_least_0_05_deg": summary["subject_macro_improvement_deg"]
        >= float(gate_config["subject_macro_improvement_deg_min"]),
        "bootstrap_ci_lower_above_zero": summary["bootstrap_ci_lower_deg"] > 0.0,
        "positive_outer_folds_at_least_4": summary["positive_outer_fold_count"]
        >= int(gate_config["positive_outer_folds_min"]),
        "no_amplitude_relative_regression_over_0_01": all(
            item["change_candidate_minus_comparator"]
            <= float(gate_config["maximum_amplitude_relative_regression"])
            for item in amplitude_changes.values()
        ),
    }
    summary["gates"] = gates
    summary["all_gates_pass"] = bool(all(gates.values()))
    subject_detail = pd.DataFrame(
        {
            "comparison": summary["comparison"],
            "subject": improvement.index,
            "outer_fold": fold_series.to_numpy(int),
            "comparator_mae_deg": pivot.loc[improvement.index, comparator].to_numpy(float),
            "candidate_mae_deg": pivot.loc[improvement.index, candidate].to_numpy(float),
            "improvement_deg": values,
        }
    )
    return summary, subject_detail


def main() -> int:
    started = time.time()
    for directory in [TABLES, OUTPUTS, PREDICTIONS, FIGURES, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENT_PATH, encoding="utf-8-sig", low_memory=False)
    events = events.loc[events["screen_eligible"].map(as_bool)].copy().reset_index(drop=True)
    if len(events) != 275 or events["subject"].nunique() != 26 or events["event_uid"].nunique() != 275:
        raise ValueError("Run80事件集必须是275事件/26被试/275唯一UID")
    truth_columns = [f"true_t{point:02d}_deg" for point in range(1, POINTS + 1)]
    truth = events[truth_columns].to_numpy(float)
    if not np.isfinite(truth).all():
        raise ValueError("20点目标不完整")

    print("[1/4] 构造134维车辆特征", flush=True)
    vehicle_x, vehicle_names = build_vehicle_features(events)
    vehicle_table = pd.concat(
        [events[["event_uid", "subject", "recording_uid", "session_stamp"]], pd.DataFrame(vehicle_x, columns=vehicle_names)],
        axis=1,
    )
    vehicle_table.to_csv(TABLES / "vehicle_features_134.csv", index=False, encoding="utf-8-sig")

    print("[2/4] 重建Run77旧16维生理特征", flush=True)
    old_table, old_columns = build_old_physio_features(events)
    old_table.to_csv(TABLES / "old16_physio_features.csv", index=False, encoding="utf-8-sig")
    old_x = old_table[old_columns].to_numpy(float)

    clean_table = pd.read_csv(CLEAN_PATH, encoding="utf-8-sig", low_memory=False)
    clean_columns = [f"clean_{name}" for name in old_columns]
    missing_clean = [column for column in clean_columns if column not in clean_table.columns]
    if missing_clean:
        raise ValueError(f"Run79缺少对应特征: {missing_clean}")
    clean_table = events[["event_uid"]].merge(
        clean_table[["event_uid", *clean_columns]],
        on="event_uid",
        how="left",
        validate="one_to_one",
    )
    clean_x = clean_table[clean_columns].to_numpy(float)
    if old_x.shape != (275, 16) or clean_x.shape != (275, 16):
        raise ValueError("生理特征维度必须是275x16")

    subject_fold = greedy_subject_folds(events)
    events["outer_fold"] = events["subject"].map(subject_fold).astype(int)
    fold_table = pd.DataFrame(
        {
            "subject": sorted(subject_fold),
            "outer_fold": [subject_fold[subject] for subject in sorted(subject_fold)],
            "event_count": [int(events["subject"].eq(subject).sum()) for subject in sorted(subject_fold)],
        }
    )
    fold_table.to_csv(TABLES / "subject_folds.csv", index=False, encoding="utf-8-sig")

    regressor = CONFIG["regressor"]
    predictions = {name: np.full_like(truth, np.nan) for name in MODEL_NAMES}
    training_rows: list[dict[str, object]] = []
    print("[3/4] 三模型被试不相交5折OOF训练", flush=True)
    for fold in range(1, 6):
        train = np.flatnonzero(~events["outer_fold"].eq(fold).to_numpy())
        test = np.flatnonzero(events["outer_fold"].eq(fold).to_numpy())
        if set(events.iloc[train]["subject"]) & set(events.iloc[test]["subject"]):
            raise ValueError("外折被试发生交叉")
        weights = subject_weights(events.iloc[train])
        matrices = {
            "V_vehicle": (None, None),
            "VP_old16": (old_x[train], old_x[test]),
            "VP_clean16": (clean_x[train], clean_x[test]),
        }
        for model_name, (physio_train, physio_test) in matrices.items():
            train_x, test_x = prepare_model_input(
                vehicle_x[train], vehicle_x[test], physio_train, physio_test
            )
            model = ExtraTreesRegressor(
                n_estimators=int(regressor["n_estimators"]),
                min_samples_leaf=int(regressor["min_samples_leaf"]),
                max_features=float(regressor["max_features"]),
                n_jobs=int(regressor["n_jobs"]),
                random_state=int(regressor["random_seed_base"]) + fold,
            )
            model.fit(train_x, truth[train], sample_weight=weights)
            predictions[model_name][test] = model.predict(test_x)
        training_rows.append(
            {
                "outer_fold": fold,
                "train_subjects": int(events.iloc[train]["subject"].nunique()),
                "test_subjects": int(events.iloc[test]["subject"].nunique()),
                "train_events": len(train),
                "test_events": len(test),
            }
        )
        print(f"fold {fold}/5 完成", flush=True)
    for name, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise ValueError(f"{name} OOF预测不完整")
    pd.DataFrame(training_rows).to_csv(TABLES / "training_folds.csv", index=False, encoding="utf-8-sig")

    print("[4/4] 指标、分层和配对bootstrap", flush=True)
    metric_tables: dict[str, pd.DataFrame] = {}
    aggregate_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    for model_name, prediction in predictions.items():
        metrics = event_metrics(truth, prediction)
        metric_tables[model_name] = metrics
        aggregate_rows.append(
            {
                "model": model_name,
                "subject_macro_curve_mae_deg": subject_macro(events, metrics["curve_mae_deg"].to_numpy()),
                "subject_macro_head5_mae_deg": subject_macro(events, metrics["head5_mae_deg"].to_numpy()),
                "subject_macro_tail5_mae_deg": subject_macro(events, metrics["tail5_mae_deg"].to_numpy()),
                "subject_macro_endpoint_mae_deg": subject_macro(events, metrics["endpoint_mae_deg"].to_numpy()),
                "subject_macro_peak_time_mae_s": subject_macro(events, metrics["peak_time_mae_s"].to_numpy()),
                "pooled_curve_mae_deg_reference_only": float(metrics["curve_mae_deg"].mean()),
            }
        )
        for subject, indices in events.groupby("subject").groups.items():
            index = np.asarray(list(indices), dtype=int)
            subject_rows.append(
                {
                    "model": model_name,
                    "subject": subject,
                    "outer_fold": int(events.iloc[index[0]]["outer_fold"]),
                    "event_count": len(index),
                    "curve_mae_deg": float(metrics.iloc[index]["curve_mae_deg"].mean()),
                }
            )
    aggregate = pd.DataFrame(aggregate_rows)
    subjects = pd.DataFrame(subject_rows)
    aggregate.to_csv(TABLES / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")
    subjects.to_csv(TABLES / "subject_metrics.csv", index=False, encoding="utf-8-sig")

    amplitude = pd.to_numeric(events["pulse_amplitude_deg_report_only"], errors="raise").to_numpy(float)
    events["amplitude_bin"] = pd.cut(
        amplitude,
        [0, 20, 30, 45, 70, np.inf],
        labels=["lt20", "20_30", "30_45", "45_70", "ge70"],
        right=False,
    ).astype(str)
    amplitude_table = amplitude_metrics(events, metric_tables, amplitude)
    amplitude_table.to_csv(TABLES / "relative_mae_by_amplitude_bin.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    clean_vehicle, detail_vehicle = paired_comparison(
        "V_vehicle", "VP_clean16", subjects, events, metric_tables, amplitude_table, rng
    )
    clean_old, detail_old = paired_comparison(
        "VP_old16", "VP_clean16", subjects, events, metric_tables, amplitude_table, rng
    )
    old_vehicle, detail_old_vehicle = paired_comparison(
        "V_vehicle", "VP_old16", subjects, events, metric_tables, amplitude_table, rng
    )
    comparisons = [clean_vehicle, clean_old, old_vehicle]
    pd.concat([detail_vehicle, detail_old, detail_old_vehicle], ignore_index=True).to_csv(
        TABLES / "paired_subject_improvements.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {
                "comparison": item["comparison"],
                "subject_macro_improvement_deg": item["subject_macro_improvement_deg"],
                "bootstrap_ci_lower_deg": item["bootstrap_ci_lower_deg"],
                "bootstrap_ci_upper_deg": item["bootstrap_ci_upper_deg"],
                "positive_outer_fold_count": item["positive_outer_fold_count"],
                "improved_subject_count": item["improved_subject_count"],
                "harmed_subject_count": item["harmed_subject_count"],
                "event_worsened_fraction": item["event_worsened_fraction"],
                "all_gates_pass": item["all_gates_pass"],
            }
            for item in comparisons
        ]
    ).to_csv(TABLES / "paired_bootstrap_subject.csv", index=False, encoding="utf-8-sig")

    if clean_vehicle["all_gates_pass"]:
        status = "CLEAN_PHYSIO_INCREMENT_EFFECTIVE"
    elif clean_old["all_gates_pass"]:
        status = "PREPROCESSING_HELPED_BUT_PHYSIO_NOT_EFFECTIVE"
    else:
        status = "CLEAN_PHYSIO_INCREMENT_NOT_EFFECTIVE"
    decision = {
        "run_id": CONFIG["run_id"],
        "status": status,
        "events": len(events),
        "subjects": int(events["subject"].nunique()),
        "vehicle_feature_count": vehicle_x.shape[1],
        "physio_feature_count": old_x.shape[1],
        "physio_missing_indicator_count": old_x.shape[1],
        "comparisons": {item["comparison"]: item for item in comparisons},
        "evidence_boundary": CONFIG["evidence_boundary"],
        "elapsed_seconds": float(time.time() - started),
    }
    (OUTPUTS / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output = events[
        ["event_uid", "subject", "recording_uid", "session_stamp", "outer_fold", "amplitude_bin"]
    ].copy()
    for point in range(POINTS):
        output[f"true_t{point + 1:02d}_deg"] = truth[:, point]
        for model_name in MODEL_NAMES:
            output[f"{model_name}_pred_t{point + 1:02d}_deg"] = predictions[model_name][:, point]
    for model_name in MODEL_NAMES:
        output[f"{model_name}_curve_mae_deg"] = metric_tables[model_name]["curve_mae_deg"].to_numpy(float)
    output.to_csv(PREDICTIONS / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

    aggregate_index = aggregate.set_index("model")
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [aggregate_index.loc[name, "subject_macro_curve_mae_deg"] for name in MODEL_NAMES]
    ax.bar(MODEL_NAMES, values, color=["#6E6E6E", "#D95F02", "#1976D2"])
    ax.set_ylabel("subject-macro curve MAE (deg)")
    ax.set_title("Run80: old vs processed physiology")
    ax.grid(axis="y", alpha=0.2)
    for index, value in enumerate(values):
        ax.text(index, value + 0.08, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_1_model_comparison.png", dpi=180)
    plt.close(fig)

    pivot = subjects.pivot(index="subject", columns="model", values="curve_mae_deg")
    clean_gain = pivot["V_vehicle"] - pivot["VP_clean16"]
    old_gain = pivot["V_vehicle"] - pivot["VP_old16"]
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - 0.2, old_gain, width=0.4, label="old16 vs vehicle", color="#D95F02")
    ax.bar(x + 0.2, clean_gain, width=0.4, label="clean16 vs vehicle", color="#1976D2")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, pivot.index, rotation=45)
    ax.set_ylabel("vehicle - physiology MAE improvement (deg)")
    ax.set_title("Subject-level physiology increment")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_2_subject_improvement.png", dpi=180)
    plt.close(fig)

    amplitude_index = amplitude_table.set_index(["model", "amplitude_bin"])
    labels = ["20_30", "30_45", "45_70", "ge70"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in zip(MODEL_NAMES, ["#6E6E6E", "#D95F02", "#1976D2"]):
        ax.plot(
            labels,
            [amplitude_index.loc[(name, label), "subject_macro_relative_mae"] for label in labels],
            marker="o",
            label=name,
            color=color,
        )
    ax.set_ylabel("subject-macro relative MAE")
    ax.set_title("Amplitude-stratified comparison")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_3_amplitude_relative_mae.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    old_error = metric_tables["VP_old16"]["curve_mae_deg"]
    clean_error = metric_tables["VP_clean16"]["curve_mae_deg"]
    ax.scatter(old_error, clean_error, s=18, alpha=0.55, color="#1976D2")
    limit = float(max(old_error.max(), clean_error.max()))
    ax.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("old16 event MAE (deg)")
    ax.set_ylabel("clean16 event MAE (deg)")
    ax.set_title("Per-event error: old vs processed physiology")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_4_old_vs_clean_event_error.png", dpi=180)
    plt.close(fig)

    lines = [
        "# Run80：处理后生理特征A/B",
        "",
        f"- 状态：`{status}`",
        f"- 事件：{len(events)}；被试：{events['subject'].nunique()}；被试不相交5折。",
        "- 三模型使用相同134维车辆特征、ExtraTrees参数、样本权重和外折。",
        "",
        "## 主结果",
        "",
        "| 模型 | subject-macro MAE° | head5° | tail5° | endpoint° | peak-time s | pooled MAE°参考 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in MODEL_NAMES:
        row = aggregate_index.loc[model_name]
        lines.append(
            f"| {model_name} | {row.subject_macro_curve_mae_deg:.4f} | "
            f"{row.subject_macro_head5_mae_deg:.4f} | {row.subject_macro_tail5_mae_deg:.4f} | "
            f"{row.subject_macro_endpoint_mae_deg:.4f} | {row.subject_macro_peak_time_mae_s:.4f} | "
            f"{row.pooled_curve_mae_deg_reference_only:.4f} |"
        )
    lines += ["", "## 配对比较", ""]
    for comparison in comparisons:
        lines += [
            f"### {comparison['comparison']}",
            "",
            f"- subject-macro改善：{comparison['subject_macro_improvement_deg']:+.4f}°。",
            f"- 95%CI：[{comparison['bootstrap_ci_lower_deg']:+.4f}, {comparison['bootstrap_ci_upper_deg']:+.4f}]°。",
            f"- 正向外折：{comparison['positive_outer_fold_count']}/5。",
            f"- 改善/退化被试：{comparison['improved_subject_count']}/{comparison['harmed_subject_count']}。",
            f"- 事件退化比例：{comparison['event_worsened_fraction']:.3f}。",
            f"- 四门全过：`{comparison['all_gates_pass']}`。",
            "",
        ]
    lines += [
        "## clean16相对车辆的四门",
        "",
    ]
    for name, value in clean_vehicle["gates"].items():
        lines.append(f"- {name}: `{value}`")
    lines += [
        "",
        "## clean16相对车辆的幅值档变化",
        "",
    ]
    for label, item in clean_vehicle["amplitude_changes"].items():
        lines.append(
            f"- {label}: {item['comparator']:.4f} → {item['candidate']:.4f}，"
            f"变化 {item['change_candidate_minus_comparator']:+.4f}。"
        )
    lines += [
        "## 结论边界",
        "",
        "本轮只检验Run79预处理能否改善固定ExtraTrees中的生理增量；不证明因果生理机制，也不证明对原18人或外部场景泛化。",
        "若clean16只超过old16但未超过车辆，结论只能是预处理改善表征，不能宣称生理已经提升预测。",
    ]
    (OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
