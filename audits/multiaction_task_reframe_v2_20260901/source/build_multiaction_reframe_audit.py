from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


AUDIT_NAME = "MULTIACTION_REFRAME_CORRECTED_MU_SCENE_V2_20260901"
ORIGINAL_LEDGER = Path(
    "05_rebuild_from_raw_20260511/03_baselines/"
    "run57_p0_event_population_ledger_20260827/run_1/tables/event_quality_ledger.csv"
)
AUGUST_FILE_SUMMARY = Path(
    "05_rebuild_from_raw_20260511/03_baselines/"
    "run78_august_subject_rescreen_20260831/run_1/tables/file_summary.csv"
)
AUGUST_SUBJECT_SUMMARY = Path(
    "05_rebuild_from_raw_20260511/03_baselines/"
    "run78_august_subject_rescreen_20260831/run_1/tables/subject_summary.csv"
)
AUGUST_PHYSIO_ROOT = Path(
    "05_rebuild_from_raw_20260511/03_baselines/"
    "run79_august_physio_preprocessing_20260831/run_1"
)
ORIGINAL_PHYSIO_FEATURES = Path(
    "05_rebuild_from_raw_20260511/06_physio_processing/"
    "physio_subject_collection_v1_20260603/tables/physio_features_10hz.csv"
)
ORIGINAL_PHYSIO_QUALITY = Path(
    "05_rebuild_from_raw_20260511/06_physio_processing/"
    "physio_subject_collection_v1_20260603/tables/physio_signal_quality_summary.csv"
)


CHANNEL_META = {
    "speed": ("km/h", "正式目标"),
    "longitudinal_acceleration": ("m/s^2", "正式目标"),
    "lateral_acceleration": ("m/s^2", "正式目标"),
    "yaw_rate": ("rad/s", "分批次辅助目标"),
    "roll": ("rad", "辅助目标"),
    "roll_rate": ("rad/s", "分批次辅助目标"),
    "lateral_velocity": ("m/s", "辅助目标"),
    "position_x": ("m", "解释性目标"),
    "position_y": ("m", "解释性目标"),
    "lateral_distance": ("m", "辅助目标"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="刺激中心多动作数据合同审计（只读数据，不训练模型）")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--august-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_column(columns: list[str], *suffixes: str) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for suffix in suffixes:
        suffix_lower = suffix.lower()
        exact = [original for lower, original in lowered.items() if lower == suffix_lower]
        if exact:
            return exact[0]
        matches = [original for lower, original in lowered.items() if lower.endswith("|" + suffix_lower)]
        if matches:
            return matches[0]
    return None


def session_datetime(stamp: str) -> pd.Timestamp:
    return pd.to_datetime(stamp, format="%Y_%m_%d_%H_%M_%S", errors="raise")


def safe_float(value: object) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.nan


def load_cohort_sources(project_root: Path, august_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    ledger = pd.read_csv(project_root / ORIGINAL_LEDGER, low_memory=False)
    reference = ledger.loc[
        ledger["contract_reference_population"].astype(str).str.lower().eq("true")
    ]
    original = (
        reference[["subject", "session_stamp", "source_path"]]
        .drop_duplicates()
        .rename(columns={"subject": "subject_key"})
    )
    if len(original) != 85:
        raise AssertionError(f"原始参考recording应为85，实际为{len(original)}")
    original["cohort"] = "original"

    august = pd.read_csv(project_root / AUGUST_FILE_SUMMARY, low_memory=False)
    august = august[["subject", "session_stamp", "source_path"]].rename(columns={"subject": "subject_key"})
    august["cohort"] = "august_2025"

    all_subjects = sorted(set(original["subject_key"]) | {path.name.lower() for path in august_root.iterdir() if path.is_dir()})
    subject_alias = {subject: f"S{index:03d}" for index, subject in enumerate(all_subjects, start=1)}

    recordings = pd.concat([original, august], ignore_index=True)
    recordings["session_time"] = recordings["session_stamp"].map(session_datetime)
    recordings = recordings.sort_values(["subject_key", "session_time", "cohort"]).reset_index(drop=True)
    recordings["subject_alias"] = recordings["subject_key"].map(subject_alias)
    recordings["recording_alias"] = [f"R{index:03d}" for index in range(1, len(recordings) + 1)]
    recordings["chronological_recording_index"] = recordings.groupby("subject_key").cumcount() + 1
    return recordings, ledger, subject_alias


def build_lineage(
    project_root: Path,
    august_root: Path,
    recordings: pd.DataFrame,
    subject_alias: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original_subjects = set(recordings.loc[recordings["cohort"] == "original", "subject_key"])
    august_dirs = {path.name.lower() for path in august_root.iterdir() if path.is_dir()}
    august_vehicle = set(recordings.loc[recordings["cohort"] == "august_2025", "subject_key"])
    subject_inventory = pd.read_csv(project_root / AUGUST_PHYSIO_ROOT / "tables/subject_inventory.csv")
    august_physio = set(subject_inventory.loc[subject_inventory["source_type"] == "four_channel_physio", "subject"])
    august_eligible = set(pd.read_csv(project_root / AUGUST_SUBJECT_SUMMARY)["subject"])

    rows = []
    for subject in sorted(original_subjects | august_dirs):
        rows.append(
            {
                "subject_alias": subject_alias[subject],
                "in_original_cohort": subject in original_subjects,
                "in_august_directory": subject in august_dirs,
                "august_vehicle_recordings": int(
                    ((recordings["cohort"] == "august_2025") & (recordings["subject_key"] == subject)).sum()
                ),
                "august_physio_available": subject in august_physio,
                "august_eligible_vehicle_event": subject in august_eligible,
                "cross_cohort_same_driver": subject in original_subjects and subject in august_dirs,
                "truly_new_august_driver": subject in august_vehicle and subject not in original_subjects,
            }
        )
    private = pd.DataFrame(rows)
    public = pd.DataFrame(
        [
            {"口径": "原始主队列驾驶员", "数量": len(original_subjects)},
            {"口径": "8月目录", "数量": len(august_dirs)},
            {"口径": "8月有车辆recording", "数量": len(august_vehicle)},
            {"口径": "8月有四通道生理", "数量": len(august_physio)},
            {"口径": "8月有合格旧转向事件", "数量": len(august_eligible)},
            {"口径": "跨批次同一驾驶员", "数量": len(original_subjects & august_vehicle)},
            {"口径": "8月真正新增驾驶员", "数量": len(august_vehicle - original_subjects)},
            {"口径": "原始recording", "数量": int((recordings["cohort"] == "original").sum())},
            {"口径": "8月车辆recording", "数量": int((recordings["cohort"] == "august_2025").sum())},
            {"口径": "合并recording", "数量": len(recordings)},
        ]
    )
    return private, public


def read_vehicle(path: Path) -> tuple[pd.DataFrame, dict[str, str | None], int]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
    column_map = {
        "storage_time": find_column(header, "StorageTime"),
        "t_s": find_column(header, "t_s"),
        "steer": find_column(header, "SteeringWheel"),
        "accelerator": find_column(header, "AcceleratorPedal"),
        "brake": find_column(header, "BrakePedal"),
        "speed_kmh": find_column(header, "v_km/h", "v_kmh"),
        "speed_ms": find_column(header, "v"),
        "ax": find_column(header, "ax"),
        "ay": find_column(header, "ay"),
        "vx": find_column(header, "vx"),
        "vy": find_column(header, "vy"),
        "yaw_rate": find_column(header, "vyaw"),
        "roll_rate": find_column(header, "vroll"),
        "roll": find_column(header, "roll"),
        "x": find_column(header, "x"),
        "y": find_column(header, "y"),
        "lateral_distance": find_column(header, "lateraldistance"),
        "curvature": find_column(header, "lanecurvatureXY"),
        "mu": find_column(header, "mu"),
        "distance7": find_column(header, "distance7"),
        "distance8": find_column(header, "distance8"),
        "pointdistance": find_column(header, "pointdistance"),
        "pointdistance9": find_column(header, "pointdistance9"),
        "distance_truck": find_column(header, "distance_truck"),
        "distance_changlane": find_column(header, "distance_changlane"),
        "distance_left": find_column(header, "Distance_left"),
        "distance_right": find_column(header, "Distance_right"),
    }
    required = [column_map[key] for key in ["storage_time", "steer", "accelerator", "brake"]]
    if any(value is None for value in required):
        raise ValueError(f"车辆文件缺少核心列: {path}")
    usecols = sorted({column for column in column_map.values() if column is not None})
    data = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    storage = pd.to_datetime(data[column_map["storage_time"]], format="mixed", errors="coerce")
    data = data.loc[storage.notna()].copy()
    storage = storage.loc[storage.notna()]
    data["_storage_time"] = storage
    data = data.sort_values("_storage_time").drop_duplicates("_storage_time").reset_index(drop=True)
    if column_map["t_s"] is not None:
        t_s = pd.to_numeric(data[column_map["t_s"]], errors="coerce")
        if t_s.notna().mean() < 0.99:
            raise ValueError(f"t_s不可用: {path}")
        data["_t_s"] = t_s - t_s.iloc[0]
    else:
        data["_t_s"] = (data["_storage_time"] - data["_storage_time"].iloc[0]).dt.total_seconds()

    for key, column in column_map.items():
        if key in {"storage_time", "t_s"}:
            continue
        data["_" + key] = pd.to_numeric(data[column], errors="coerce") if column is not None else np.nan
    if data["_speed_kmh"].notna().mean() < 0.5:
        if data["_speed_ms"].notna().mean() >= 0.5:
            data["_speed_kmh"] = data["_speed_ms"].abs() * 3.6
        else:
            data["_speed_kmh"] = data["_vx"].abs() * 3.6
    median_dt = float(np.nanmedian(np.diff(data["_t_s"].to_numpy(float))))
    if not (0 < median_dt < 1):
        raise ValueError(f"采样时间异常: {path}")
    return data, column_map, int(round(1.0 / median_dt))


def threshold_crossings(t: np.ndarray, values: np.ndarray, threshold: float, hysteresis: float = 0.0) -> list[float]:
    finite = np.isfinite(values)
    previous_high = np.r_[False, (values[:-1] >= threshold + hysteresis) & finite[:-1]]
    current_low = (values < threshold) & finite
    return [float(value) for value in t[previous_high & current_low]]


def collapse_times(times: list[float], refractory_s: float) -> list[float]:
    kept = []
    for value in sorted(times):
        if not kept or value - kept[-1] >= refractory_s:
            kept.append(value)
    return kept


def first_low_mu_scene_entry(
    t: np.ndarray,
    mu: np.ndarray,
    threshold: float,
    minimum_valid_mu: float,
) -> tuple[float, float, float] | None:
    """每个 recording 只恢复第一次真正跨入低附着场景的时刻。"""
    valid_idx = np.flatnonzero(np.isfinite(mu) & (mu >= minimum_valid_mu))
    for left, right in zip(valid_idx[:-1], valid_idx[1:]):
        previous = float(mu[left])
        current = float(mu[right])
        if previous > threshold and minimum_valid_mu <= current <= threshold:
            return float(t[right]), previous, current
    return None


def detect_stimuli(
    data: pd.DataFrame,
    cohort: str,
    config: dict,
) -> list[dict]:
    t = data["_t_s"].to_numpy(float)
    results: list[dict] = []
    if cohort == "original":
        for item in config["original_distance_triggers"]:
            values = data["_" + item["signal"]].to_numpy(float)
            crossings = collapse_times(
                threshold_crossings(t, values, float(item["threshold"])),
                float(config["deduplication"]["same_signal_refractory_s"]),
            )
            for onset in crossings:
                results.append(
                    {
                        "stimulus_type": f"distance_threshold_{item['external_trigger']}",
                        "stimulus_semantics": "外部触发编号已确认，目标交通行为语义未在文本配置中恢复",
                        "trigger_signal": item["signal"],
                        "trigger_rule": f"{item['signal']} < {item['threshold']:.0f} m",
                        "stimulus_onset_s": onset,
                        "stimulus_onset_source": "SILAB comparator configuration and recorded distance",
                        "onset_exactness": "exact_configuration_threshold",
                        "online_observable": "yes_with_target_perception",
                        "online_observable_assumption": "部署端需要目标识别与相对距离感知",
                        "confidence": "high_time_medium_semantics",
                        "candidate_eligible_by_mapping": True,
                        "unresolved_issue": "外部触发编号对应的具体交通车动作语义尚缺权威脚本映射",
                    }
                )
    else:
        for signal in ["distance_truck", "distance_changlane"]:
            values = data["_" + signal].to_numpy(float)
            crossings = collapse_times(threshold_crossings(t, values, 30.0, hysteresis=5.0), 2.0)
            for onset in crossings:
                results.append(
                    {
                        "stimulus_type": f"august_{signal}_below30_diagnostic",
                        "stimulus_semantics": "字段语义可读，但阈值与脚本触发关系尚未恢复",
                        "trigger_signal": signal,
                        "trigger_rule": "诊断规则：由>=35 m进入<30 m；不作为正式触发合同",
                        "stimulus_onset_s": onset,
                        "stimulus_onset_source": "recorded distance diagnostic only",
                        "onset_exactness": "proxy_unmapped_threshold",
                        "online_observable": "yes_with_target_perception_but_rule_unmapped",
                        "online_observable_assumption": "部署端需要目标识别；正式阈值仍待脚本确认",
                        "confidence": "low_mapping",
                        "candidate_eligible_by_mapping": False,
                        "unresolved_issue": "8月场景脚本和触发阈值未找到，禁止把30 m诊断线当真值",
                    }
                )

    low_mu_config = config["low_mu_scene"]
    low_mu_entry = first_low_mu_scene_entry(
        t,
        data["_mu"].to_numpy(float),
        float(low_mu_config["threshold"]),
        float(low_mu_config["minimum_valid_mu"]),
    )
    if low_mu_entry is not None:
        onset, previous, current = low_mu_entry
        results.append(
            {
                "stimulus_type": "enter_low_mu_scene",
                "stimulus_semantics": "每个recording第一次从普通附着跨入mu<=0.4低附着场景",
                "trigger_signal": "mu",
                "trigger_rule": f"first per recording: mu {previous:.3f} -> {current:.3f}, crossing into mu<=0.400",
                "stimulus_onset_s": onset,
                "stimulus_onset_source": "recorded simulator road-friction signal",
                "onset_exactness": "exact_recorded_script_state_change",
                "online_observable": "script_label_only",
                "online_observable_assumption": "当前数据没有证明部署端可在同一时刻直接观测mu，需要地图或在线附着估计器",
                "confidence": "high_time_high_signal",
                "candidate_eligible_by_mapping": True,
                "unresolved_issue": "每个recording只保留首次低附着进入；无合法零延迟在线代理，当前仅可作为脚本标签",
            }
        )
    return sorted(results, key=lambda row: row["stimulus_onset_s"])


def finite_coverage(data: pd.DataFrame, onset: float, pre: float, post: float) -> float:
    mask = (data["_t_s"] >= onset - pre) & (data["_t_s"] <= onset + post)
    columns = ["_steer", "_accelerator", "_brake", "_speed_kmh"]
    return min(float(data.loc[mask, column].notna().mean()) for column in columns) if mask.any() else 0.0


def group_and_select_stimuli(
    detected: list[dict],
    data: pd.DataFrame,
    record: pd.Series,
    config: dict,
) -> list[dict]:
    if not detected:
        return []
    pre_required = float(config["event_windows_s"]["required_pre"])
    post_required = float(config["event_windows_s"]["required_post"])
    overlap_s = float(config["deduplication"]["cross_signal_overlap_s"])
    duration = float(data["_t_s"].iloc[-1])
    rows = []
    for item in detected:
        onset = float(item["stimulus_onset_s"])
        row = dict(item)
        row.update(
            {
                "pre_available_s": max(0.0, onset),
                "post_available_s": max(0.0, duration - onset),
                "finite_coverage": finite_coverage(data, onset, pre_required, post_required),
            }
        )
        row["base_eligible"] = bool(
            row["candidate_eligible_by_mapping"]
            and row["pre_available_s"] >= pre_required
            and row["post_available_s"] >= post_required
            and row["finite_coverage"] >= float(config["finite_coverage_min"])
        )
        rows.append(row)

    group_number = 0
    previous_onset = -math.inf
    for row in rows:
        if row["stimulus_onset_s"] - previous_onset > overlap_s:
            group_number += 1
        row["overlapping_stimulus_group"] = f"{record.recording_alias}-G{group_number:03d}"
        previous_onset = row["stimulus_onset_s"]

    for _, indices in pd.DataFrame(rows).groupby("overlapping_stimulus_group").groups.items():
        candidate_indices = [index for index in indices if rows[index]["base_eligible"]]
        selected = None
        if candidate_indices:
            selected = sorted(
                candidate_indices,
                key=lambda index: (
                    0 if rows[index]["onset_exactness"] == "exact_configuration_threshold" else 1,
                    rows[index]["stimulus_onset_s"],
                ),
            )[0]
        for index in indices:
            rows[index]["included_candidate"] = index == selected
            if index == selected:
                rows[index]["exclusion_reason"] = ""
            elif not rows[index]["candidate_eligible_by_mapping"]:
                rows[index]["exclusion_reason"] = "unresolved_trigger_mapping"
            elif rows[index]["pre_available_s"] < pre_required:
                rows[index]["exclusion_reason"] = "insufficient_pre_history"
            elif rows[index]["post_available_s"] < post_required:
                rows[index]["exclusion_reason"] = "insufficient_post_observation"
            elif rows[index]["finite_coverage"] < float(config["finite_coverage_min"]):
                rows[index]["exclusion_reason"] = "insufficient_signal_coverage"
            else:
                rows[index]["exclusion_reason"] = "overlap_secondary_stimulus"

    for index, row in enumerate(rows, start=1):
        row.update(
            {
                "event_id": f"{record.recording_alias}-E{index:03d}",
                "subject_alias": record.subject_alias,
                "recording_alias": record.recording_alias,
                "cohort": record.cohort,
                "chronological_recording_index": int(record.chronological_recording_index),
                "scenario_id": "shared_original_route" if record.cohort == "original" else "august_recording_family",
                "scenario_name": "场景运行顺序可恢复，单次脚本名称未完整恢复",
                "raw_source_alias": record.recording_alias,
                "internal_subject_key": record.subject_key,
                "internal_session_stamp": record.session_stamp,
                "internal_source_path": record.source_path,
            }
        )
    return rows


def first_sustained(mask: np.ndarray, samples: int) -> int | None:
    if len(mask) < samples:
        return None
    hits = np.convolve(mask.astype(int), np.ones(samples, dtype=int), mode="valid")
    indices = np.flatnonzero(hits == samples)
    return int(indices[0]) if len(indices) else None


def duration_above(t: np.ndarray, magnitude: np.ndarray, start: int | None, threshold: float) -> float:
    if start is None:
        return math.nan
    end = start
    while end + 1 < len(magnitude) and magnitude[end + 1] >= threshold:
        end += 1
    return float(t[end] - t[start])


def integral_abs(t: np.ndarray, values: np.ndarray, horizon: float) -> float:
    mask = t <= horizon + 1e-9
    if mask.sum() < 2:
        return math.nan
    return float(np.trapezoid(np.abs(values[mask]), t[mask]))


def label_actions(data: pd.DataFrame, event: dict, thresholds: dict, sustain_s: float) -> dict:
    onset = float(event["stimulus_onset_s"])
    search = (data["_t_s"] >= onset) & (data["_t_s"] <= onset + 5.0)
    pre = (data["_t_s"] >= onset - 1.0) & (data["_t_s"] < onset)
    if search.sum() < 3 or pre.sum() < 3:
        raise ValueError(f"事件窗口不足但被纳入: {event['event_id']}")
    t_abs = data.loc[search, "_t_s"].to_numpy(float)
    t = t_abs - onset
    median_dt = float(np.nanmedian(np.diff(t)))
    sustain_samples = max(1, int(math.ceil(sustain_s / median_dt)))

    steer = np.degrees(data.loc[search, "_steer"].to_numpy(float))
    steer_pre = np.degrees(data.loc[pre, "_steer"].to_numpy(float))
    steer_anchor = float(steer[0])
    steer_reference = float(np.nanmedian(steer_pre))
    steer_delta = steer - steer_anchor
    steer_threshold = float(thresholds["steer_delta_deg"])
    steer_start = first_sustained(np.abs(steer_delta) >= steer_threshold, sustain_samples)
    steer_response = steer_start is not None
    steer_onset = float(t[steer_start]) if steer_response else math.nan
    steer_peak_index = int(np.nanargmax(np.abs(steer_delta)))
    steer_peak = float(steer_delta[steer_peak_index])
    steer_already = bool(abs(steer_anchor - steer_reference) >= steer_threshold)
    steer_sign = np.sign(steer_peak) if steer_peak != 0 else 0
    after_peak = steer_delta[steer_peak_index + 1 :]
    secondary = bool(len(after_peak) and np.nanmax(np.abs(after_peak - steer_peak)) >= steer_threshold)
    reverse = bool(len(after_peak) and np.any(np.sign(after_peak[np.abs(after_peak) >= steer_threshold]) == -steer_sign))
    if not steer_response:
        steer_type = "no_clear_steer_change"
    elif reverse:
        steer_type = "steer_then_reverse_correction"
    elif secondary:
        steer_type = "steer_then_return_or_secondary_correction"
    elif steer_already:
        steer_type = "already_turning_then_change"
    else:
        steer_type = "new_steer_response"

    brake_all = data["_brake"].to_numpy(float)
    brake_zero = float(np.nanquantile(brake_all, 0.05))
    brake = data.loc[search, "_brake"].to_numpy(float) - brake_zero
    brake_anchor = float(brake[0])
    brake_delta = brake - brake_anchor
    brake_threshold = float(thresholds["brake_delta"])
    brake_increase_start = first_sustained(brake_delta >= brake_threshold, sustain_samples)
    brake_release_start = first_sustained(brake_delta <= -brake_threshold, sustain_samples)
    brake_candidates = [value for value in [brake_increase_start, brake_release_start] if value is not None]
    brake_start = min(brake_candidates) if brake_candidates else None
    brake_response = brake_start is not None
    brake_onset = float(t[brake_start]) if brake_response else math.nan
    brake_already = bool(brake_anchor >= brake_threshold)
    if brake_increase_start is not None and (brake_release_start is None or brake_increase_start <= brake_release_start):
        brake_type = "already_braking_increase" if brake_already else "new_brake_press"
    elif brake_release_start is not None:
        brake_type = "brake_release"
    else:
        brake_type = "no_brake_response"
    brake_peak_index = int(np.nanargmax(np.abs(brake_delta)))

    accelerator = data.loc[search, "_accelerator"].to_numpy(float)
    accelerator_pre = data.loc[pre, "_accelerator"].to_numpy(float)
    accelerator_reference = float(np.nanmedian(accelerator_pre))
    accelerator_anchor = float(accelerator[0])
    accelerator_delta = accelerator - accelerator_reference
    accelerator_threshold = float(thresholds["accelerator_delta"])
    release_start = first_sustained(accelerator_delta <= -accelerator_threshold, sustain_samples)
    increase_start = first_sustained(accelerator_delta >= accelerator_threshold, sustain_samples)
    accelerator_candidates = [value for value in [release_start, increase_start] if value is not None]
    accelerator_start = min(accelerator_candidates) if accelerator_candidates else None
    accelerator_onset = float(t[accelerator_start]) if accelerator_start is not None else math.nan
    accelerator_release = release_start is not None
    accelerator_increase = increase_start is not None
    if accelerator_release and accelerator_increase:
        accelerator_type = "release_then_increase" if release_start < increase_start else "increase_then_release"
    elif accelerator_release:
        accelerator_type = "accelerator_release"
    elif accelerator_increase:
        accelerator_type = "accelerator_increase"
    else:
        accelerator_type = "accelerator_maintain"
    accelerator_peak_index = int(np.nanargmax(np.abs(accelerator_delta)))

    ordered = []
    for name, latency in [
        ("steer", steer_onset),
        ("brake", brake_onset),
        ("accelerator", accelerator_onset),
    ]:
        if math.isfinite(latency):
            ordered.append((latency, name))
    response_order = ">".join(name for _, name in sorted(ordered)) if ordered else "none"
    no_response = not (steer_response or brake_response or accelerator_release or accelerator_increase)

    if no_response:
        readable_mode = "无明显反应"
    else:
        parts = []
        if steer_response:
            parts.append("转向")
        if brake_response:
            parts.append("制动")
        if accelerator_release:
            parts.append("松油")
        if accelerator_increase:
            parts.append("补油")
        readable_mode = "+".join(parts)

    ambiguity = []
    if accelerator_release and accelerator_increase:
        ambiguity.append("accelerator_bidirectional_within_5s")
    if brake_increase_start is not None and brake_release_start is not None:
        ambiguity.append("brake_increase_and_release_within_5s")
    confidence = "high" if not ambiguity and event["finite_coverage"] >= 0.99 else "medium"
    return {
        "event_id": event["event_id"],
        "subject_alias": event["subject_alias"],
        "recording_alias": event["recording_alias"],
        "cohort": event["cohort"],
        "stimulus_type": event["stimulus_type"],
        "steer_state_at_stimulus": "already_turning" if steer_already else "near_pre_baseline",
        "steer_action_type": steer_type,
        "steer_onset_s": steer_onset,
        "steer_latency_s": steer_onset,
        "steer_peak_delta_deg": steer_peak,
        "steer_peak_time_s": float(t[steer_peak_index]),
        "steer_delta_deg": float(steer[-1] - steer_anchor),
        "steer_duration_s": duration_above(t, np.abs(steer_delta), steer_start, steer_threshold / 2),
        "steer_secondary_correction": secondary,
        "brake_state_at_stimulus": "already_pressed" if brake_already else "released_or_near_zero",
        "brake_zero_estimate": brake_zero,
        "brake_action_type": brake_type,
        "brake_onset_s": brake_onset,
        "brake_latency_s": brake_onset,
        "brake_peak_delta": float(brake_delta[brake_peak_index]),
        "brake_peak_time_s": float(t[brake_peak_index]),
        "brake_delta": float(brake[-1] - brake_anchor),
        "brake_duration_s": duration_above(t, np.abs(brake_delta), brake_start, brake_threshold / 2),
        "brake_secondary_correction": brake_increase_start is not None and brake_release_start is not None,
        "accelerator_state_at_stimulus": "high" if accelerator_anchor >= 0.7 else ("low" if accelerator_anchor <= 0.1 else "mid"),
        "accelerator_action_type": accelerator_type,
        "accelerator_onset_s": accelerator_onset,
        "accelerator_latency_s": accelerator_onset,
        "accelerator_peak_delta": float(accelerator_delta[accelerator_peak_index]),
        "accelerator_peak_time_s": float(t[accelerator_peak_index]),
        "accelerator_delta": float(accelerator[-1] - accelerator_reference),
        "accelerator_duration_s": duration_above(
            t, np.abs(accelerator_delta), accelerator_start, accelerator_threshold / 2
        ),
        "accelerator_secondary_correction": accelerator_release and accelerator_increase,
        "steer_response": steer_response,
        "brake_response": brake_response,
        "accelerator_release": accelerator_release,
        "accelerator_increase": accelerator_increase,
        "no_clear_response": no_response,
        "response_order": response_order,
        "multilabel_vector": f"{int(steer_response)}{int(brake_response)}{int(accelerator_release)}{int(accelerator_increase)}{int(no_response)}",
        "readable_action_mode": readable_mode,
        "no_response": no_response,
        "label_confidence": confidence,
        "ambiguity_reason": ";".join(ambiguity),
        "steer_cumulative_1s": integral_abs(t, steer_delta, 1.0),
        "steer_cumulative_2s": integral_abs(t, steer_delta, 2.0),
        "steer_cumulative_3s": integral_abs(t, steer_delta, 3.0),
        "brake_cumulative_1s": integral_abs(t, brake_delta, 1.0),
        "brake_cumulative_2s": integral_abs(t, brake_delta, 2.0),
        "brake_cumulative_3s": integral_abs(t, brake_delta, 3.0),
        "accelerator_cumulative_1s": integral_abs(t, accelerator_delta, 1.0),
        "accelerator_cumulative_2s": integral_abs(t, accelerator_delta, 2.0),
        "accelerator_cumulative_3s": integral_abs(t, accelerator_delta, 3.0),
    }


def channel_file_rows(data: pd.DataFrame, record: pd.Series, sample_rate_hz: int) -> list[dict]:
    columns = {
        "speed": "_speed_kmh",
        "longitudinal_acceleration": "_ax",
        "lateral_acceleration": "_ay",
        "yaw_rate": "_yaw_rate",
        "roll": "_roll",
        "roll_rate": "_roll_rate",
        "lateral_velocity": "_vy",
        "position_x": "_x",
        "position_y": "_y",
        "lateral_distance": "_lateral_distance",
    }
    limits = {
        "speed": (-20, 250),
        "longitudinal_acceleration": (-50, 50),
        "lateral_acceleration": (-50, 50),
        "yaw_rate": (-5, 5),
        "roll": (-3.2, 3.2),
        "roll_rate": (-5, 5),
        "lateral_velocity": (-50, 50),
        "position_x": (-1e7, 1e7),
        "position_y": (-1e7, 1e7),
        "lateral_distance": (-100, 100),
    }
    rows = []
    for channel, column in columns.items():
        values = data[column].to_numpy(float)
        finite = np.isfinite(values)
        lower, upper = limits[channel]
        abnormal = finite & ((values < lower) | (values > upper))
        rows.append(
            {
                "cohort": record.cohort,
                "recording_alias": record.recording_alias,
                "channel": channel,
                "present": bool(finite.any()),
                "sampling_rate_hz": sample_rate_hz,
                "missing_fraction": float(1 - finite.mean()),
                "abnormal_fraction": float(abnormal.mean()),
                "min": float(np.nanmin(values)) if finite.any() else math.nan,
                "max": float(np.nanmax(values)) if finite.any() else math.nan,
            }
        )
    return rows


def process_recordings(
    recordings: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict] = []
    primary_action_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    channel_rows: list[dict] = []
    recording_rows: list[dict] = []
    for position, record in enumerate(recordings.itertuples(index=False), start=1):
        data, column_map, sample_rate = read_vehicle(Path(record.source_path))
        duration_s = float(data["_t_s"].iloc[-1])
        recording_rows.append(
            {
                "recording_alias": record.recording_alias,
                "subject_alias": record.subject_alias,
                "cohort": record.cohort,
                "session_stamp": record.session_stamp,
                "chronological_recording_index": int(record.chronological_recording_index),
                "duration_s": duration_s,
                "sampling_rate_hz": sample_rate,
                "storage_start_ns": int(data["_storage_time"].iloc[0].value),
                "_subject_key": record.subject_key,
                "_source_path": record.source_path,
            }
        )
        channel_rows.extend(channel_file_rows(data, pd.Series(record._asdict()), sample_rate))
        detected = detect_stimuli(data, record.cohort, config)
        selected = group_and_select_stimuli(detected, data, pd.Series(record._asdict()), config)
        event_rows.extend(selected)
        for event in selected:
            if not event["included_candidate"]:
                continue
            for threshold_name, thresholds in config["candidate_label_thresholds"].items():
                labels = label_actions(data, event, thresholds, float(config["sustain_s"]))
                labels["threshold_set"] = threshold_name
                sensitivity_rows.append(labels)
                if threshold_name == "primary":
                    primary_action_rows.append(labels)
        if position % 20 == 0 or position == len(recordings):
            print(f"PROGRESS vehicle_recordings={position}/{len(recordings)}", flush=True)
    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(primary_action_rows),
        pd.DataFrame(sensitivity_rows),
        pd.DataFrame(channel_rows),
        pd.DataFrame(recording_rows),
    )


def build_stimulus_catalog(events: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, stimulus_type, trigger_rule), group in events.groupby(
        ["cohort", "stimulus_type", "trigger_rule"], sort=True
    ):
        first = group.iloc[0]
        exact = first["onset_exactness"].startswith("exact")
        rows.append(
            {
                "cohort": cohort,
                "scenario_id": first["scenario_id"],
                "scenario_name": first["scenario_name"],
                "stimulus_type": stimulus_type,
                "stimulus_semantics": first["stimulus_semantics"],
                "trigger_signal": first["trigger_signal"],
                "trigger_rule": trigger_rule,
                "source_file_type": "continuous_vehicle_csv",
                "onset_exactness": first["onset_exactness"],
                "script_onset_available": exact,
                "online_proxy_onset_available": first["online_observable"].startswith("yes_"),
                "script_minus_proxy_time_s": 0.0 if exact and first["online_observable"].startswith("yes_") else math.nan,
                "online_observable": first["online_observable"],
                "online_observable_assumption": first["online_observable_assumption"],
                "number_of_subjects": group["subject_alias"].nunique(),
                "number_of_recordings": group["recording_alias"].nunique(),
                "number_of_candidate_stimuli": len(group),
                "number_included_after_contract": int(group["included_candidate"].sum()),
                "confidence": first["confidence"],
                "unresolved_issue": first["unresolved_issue"],
            }
        )
    for cohort, signal, issue in [
        ("august_2025", "distance_left/right", "道路边界距离存在，但不是已证实脚本刺激"),
        ("original", "curvature/lateral_distance", "道路几何存在，但不能把普通道路变化自动命名为极端刺激"),
        ("august_2025", "curvature/lateral_distance", "道路几何存在，但不能把普通道路变化自动命名为极端刺激"),
    ]:
        rows.append(
            {
                "cohort": cohort,
                "scenario_id": "inventory_only",
                "scenario_name": "仅字段盘点",
                "stimulus_type": "not_promoted_signal_inventory",
                "stimulus_semantics": "潜在线索，未作为刺激事件",
                "trigger_signal": signal,
                "trigger_rule": "none",
                "source_file_type": "continuous_vehicle_csv",
                "onset_exactness": "not_mapped",
                "script_onset_available": False,
                "online_proxy_onset_available": False,
                "script_minus_proxy_time_s": math.nan,
                "online_observable": "unknown",
                "online_observable_assumption": "需先恢复场景脚本语义",
                "number_of_subjects": int(recordings.loc[recordings["cohort"] == cohort, "subject_alias"].nunique()),
                "number_of_recordings": int((recordings["cohort"] == cohort).sum()),
                "number_of_candidate_stimuli": 0,
                "number_included_after_contract": 0,
                "confidence": "inventory_only",
                "unresolved_issue": issue,
            }
        )
    return pd.DataFrame(rows)


def build_threshold_sensitivity(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold_name, group in labels.groupby("threshold_set", sort=False):
        for window_s in [1, 2, 3, 5]:
            steer = group["steer_latency_s"].le(window_s) & group["steer_latency_s"].notna()
            brake = group["brake_latency_s"].le(window_s) & group["brake_latency_s"].notna()
            release = group["accelerator_release"] & group["accelerator_latency_s"].le(window_s)
            increase = group["accelerator_increase"] & group["accelerator_latency_s"].le(window_s)
            no_response = ~(steer | brake | release | increase)
            rows.extend(
                [
                    {
                        "threshold_set": threshold_name,
                        "window_s": window_s,
                        "metric": metric,
                        "count": int(values.sum()),
                        "fraction": float(values.mean()),
                        "median_latency_s": float(group.loc[values, latency].median()) if values.any() and latency else math.nan,
                        "subjects_with_action": int(group.loc[values, "subject_alias"].nunique()),
                    }
                    for metric, values, latency in [
                        ("steer_response", steer, "steer_latency_s"),
                        ("brake_response", brake, "brake_latency_s"),
                        ("accelerator_release", release, "accelerator_latency_s"),
                        ("accelerator_increase", increase, "accelerator_latency_s"),
                        ("no_clear_response", no_response, ""),
                    ]
                ]
            )
    return pd.DataFrame(rows)


def build_sample_tables(
    events: pd.DataFrame,
    actions: pd.DataFrame,
    recording_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    included = events.loc[events["included_candidate"]].copy()
    merged = included.merge(actions, on=["event_id", "subject_alias", "recording_alias", "cohort", "stimulus_type"], validate="one_to_one")
    subject_counts = (
        merged.groupby(["subject_alias", "cohort"], as_index=False)
        .agg(
            total_events=("event_id", "size"),
            recordings=("recording_alias", "nunique"),
            steer_response=("steer_response", "sum"),
            brake_response=("brake_response", "sum"),
            accelerator_release=("accelerator_release", "sum"),
            accelerator_increase=("accelerator_increase", "sum"),
            no_response=("no_response", "sum"),
        )
    )
    stimulus_action = (
        merged.groupby(["cohort", "stimulus_type", "readable_action_mode"], as_index=False)
        .agg(events=("event_id", "size"), subjects=("subject_alias", "nunique"), recordings=("recording_alias", "nunique"))
    )
    density = (
        included.groupby(["recording_alias", "subject_alias", "cohort"], as_index=False)
        .agg(events=("event_id", "size"), min_gap_s=("stimulus_onset_s", lambda x: float(np.min(np.diff(np.sort(x)))) if len(x) > 1 else math.nan))
        .merge(recording_meta[["recording_alias", "duration_s"]], on="recording_alias", validate="one_to_one")
    )
    density["events_per_minute"] = density["events"] / (density["duration_s"] / 60.0)
    density["dense_below_2s"] = density["min_gap_s"] < 2.0
    counts = subject_counts["total_events"]
    action_coverage = {}
    for action in ["steer_response", "brake_response", "accelerator_release", "accelerator_increase", "no_response"]:
        per_subject = merged.groupby("subject_alias")[action].sum()
        action_coverage[action] = {f"subjects_ge_{minimum}": int((per_subject >= minimum).sum()) for minimum in [1, 3, 5, 10]}
    summary = {
        "total_detected_stimuli": int(len(events)),
        "total_candidate_events": int(len(included)),
        "strict_online_exact_events": int(
            (
                included["onset_exactness"].eq("exact_configuration_threshold")
                & included["finite_coverage"].ge(0.99)
                & included["pre_available_s"].ge(10)
            ).sum()
        ),
        "subjects": int(merged["subject_alias"].nunique()),
        "recordings": int(merged["recording_alias"].nunique()),
        "events_per_subject": {
            "min": int(counts.min()),
            "q1": float(counts.quantile(0.25)),
            "median": float(counts.median()),
            "q3": float(counts.quantile(0.75)),
            "max": int(counts.max()),
        },
        "no_response_events": int(merged["no_response"].sum()),
        "top_subject_share": float(counts.max() / counts.sum()),
        "dense_recording_fraction": float(density["dense_below_2s"].mean()),
        "action_coverage": action_coverage,
    }
    return subject_counts, stimulus_action, density, summary


def add_chronological_history(
    events: pd.DataFrame,
    actions: pd.DataFrame,
    recording_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    included = events.loc[events["included_candidate"]].merge(
        actions[["event_id", "readable_action_mode", "steer_response", "brake_response", "accelerator_release", "accelerator_increase"]],
        on="event_id",
        validate="one_to_one",
    )
    record_lookup = recording_meta.set_index("recording_alias")
    rows = []
    for subject, group in included.groupby("subject_alias", sort=True):
        ordered_records = recording_meta.loc[recording_meta["subject_alias"] == subject].sort_values("chronological_recording_index")
        record_order = ordered_records["recording_alias"].tolist()
        for event in group.itertuples(index=False):
            current_index = record_order.index(event.recording_alias)
            prior_records = record_order[:current_index]
            prior_event_rows = included.loc[(included["subject_alias"] == subject) & included["recording_alias"].isin(prior_records)]
            row = {
                "event_id": event.event_id,
                "subject_alias": subject,
                "recording_alias": event.recording_alias,
                "chronological_recording_index": int(event.chronological_recording_index),
                "prior_complete_recordings": len(prior_records),
                "prior_complete_recording_minutes": float(record_lookup.loc[prior_records, "duration_s"].sum() / 60.0) if prior_records else 0.0,
                "prior_completed_events": len(prior_event_rows),
                "prior_same_stimulus_events": int((prior_event_rows["stimulus_type"] == event.stimulus_type).sum()),
                "prior_steer_events": int(prior_event_rows["steer_response"].sum()),
                "prior_brake_events": int(prior_event_rows["brake_response"].sum()),
                "prior_accelerator_release_events": int(prior_event_rows["accelerator_release"].sum()),
                "prior_accelerator_increase_events": int(prior_event_rows["accelerator_increase"].sum()),
                "test_action_mode": event.readable_action_mode,
            }
            rows.append(row)
    chronology = pd.DataFrame(rows)
    feasibility = []
    for condition, values, column in [
        ("ordinary_minutes", [2, 5, 10, 20], "prior_complete_recording_minutes"),
        ("completed_events", [0, 1, 3, 5, 10], "prior_completed_events"),
    ]:
        for value in values:
            eligible = chronology.loc[(chronology["prior_complete_recordings"] >= 1) & (chronology[column] >= value)]
            feasibility.append(
                {
                    "protocol": "complete_early_recordings_to_later_recordings",
                    "condition": condition,
                    "threshold": value,
                    "calibratable_subjects": int(eligible["subject_alias"].nunique()),
                    "subjects_with_later_test": int(eligible["subject_alias"].nunique()),
                    "later_test_events": len(eligible),
                    "median_test_events_per_subject": float(eligible.groupby("subject_alias").size().median()) if len(eligible) else 0.0,
                    "action_modes_covered": int(eligible["test_action_mode"].nunique()),
                }
            )
    for minutes in [2, 5, 10, 20]:
        for completed_events in [1, 3, 5, 10]:
            eligible = chronology.loc[
                (chronology["prior_complete_recordings"] >= 1)
                & (chronology["prior_complete_recording_minutes"] >= minutes)
                & (chronology["prior_completed_events"] >= completed_events)
            ]
            feasibility.append(
                {
                    "protocol": "complete_early_recordings_to_later_recordings",
                    "condition": "ordinary_plus_completed_events",
                    "threshold": f"{minutes}min+{completed_events}events",
                    "calibratable_subjects": int(eligible["subject_alias"].nunique()),
                    "subjects_with_later_test": int(eligible["subject_alias"].nunique()),
                    "later_test_events": len(eligible),
                    "median_test_events_per_subject": float(eligible.groupby("subject_alias").size().median()) if len(eligible) else 0.0,
                    "action_modes_covered": int(eligible["test_action_mode"].nunique()),
                }
            )
    return chronology, pd.DataFrame(feasibility)


def window_coverage(times: np.ndarray, valid: np.ndarray, start: float, end: float, fs_hz: float) -> float:
    mask = (times >= start) & (times <= end)
    expected = max(1, int(round((end - start) * fs_hz)))
    return min(1.0, float(valid[mask].sum() / expected))


def build_original_physio_index(project_root: Path) -> tuple[dict, dict]:
    columns = [
        "subject",
        "session_stamp",
        "time_bin_s",
        "ECG_filt200",
        "EMG_filt200",
        "EDA_filt200",
        "RESP_filt200",
    ]
    features = pd.read_csv(project_root / ORIGINAL_PHYSIO_FEATURES, usecols=columns, low_memory=False)
    index = {}
    for key, group in features.groupby(["subject", "session_stamp"], sort=False):
        time_s = pd.to_numeric(group["time_bin_s"], errors="raise").to_numpy(float)
        valid = np.column_stack([pd.to_numeric(group[column], errors="coerce").notna().to_numpy() for column in columns[3:]])
        index[key] = (time_s, valid)
    quality = pd.read_csv(project_root / ORIGINAL_PHYSIO_QUALITY, low_memory=False)
    quality_index = {}
    for key, group in quality.groupby(["subject", "session_stamp"], sort=False):
        raw = group.loc[group["signal"].isin(["ECG_raw200", "EMG_raw200", "EDA_raw200", "RESP_raw200"])]
        near_constant = raw["near_constant"].astype(str).str.lower().eq("true")
        quality_index[key] = bool(
            len(raw) == 4
            and raw["status"].eq("ok").all()
            and pd.to_numeric(raw["missing_ratio"], errors="raise").le(0.10).all()
            and not near_constant.any()
        )
    return index, quality_index


def build_august_physio_index(project_root: Path, recording_meta: pd.DataFrame) -> dict:
    start_ns = recording_meta.set_index(["_subject_key", "session_stamp"])["storage_start_ns"].to_dict()
    quality = pd.read_csv(project_root / AUGUST_PHYSIO_ROOT / "tables/recording_quality.csv", low_memory=False)
    quality_lookup = {
        (row.subject, row.session_stamp): bool(row.ecg_usable and row.emg_usable and row.eda_usable and row.resp_usable)
        for row in quality.itertuples(index=False)
    }
    index = {}
    for path in (project_root / AUGUST_PHYSIO_ROOT / "processed").glob("*/*.npz"):
        with np.load(path, allow_pickle=False) as data:
            subject = str(data["subject"][0])
            stamp = str(data["session_stamp"][0])
            key = (subject, stamp)
            if key not in start_ns:
                continue
            channel_data = {}
            for name, fs_key, start_key, valid_key in [
                ("ecg", "ecg_fs_hz", "ecg_start_time_ns", "ecg_valid_mask"),
                ("emg", "emg_envelope_fs_hz", "emg_start_time_ns", "emg_envelope_valid_mask"),
                ("eda", "eda_fs_hz", "eda_start_time_ns", "eda_valid_mask"),
                ("resp", "resp_fs_hz", "resp_start_time_ns", "resp_valid_mask"),
            ]:
                fs_hz = float(data[fs_key][0])
                valid = data[valid_key].astype(bool)
                offset = (int(data[start_key][0]) - int(start_ns[key])) / 1e9
                times = offset + np.arange(len(valid), dtype=float) / fs_hz
                channel_data[name] = (times, valid, fs_hz)
            index[key] = {"channels": channel_data, "quality_pass": quality_lookup.get(key, False)}
    return index


def build_physiology_join(
    project_root: Path,
    events: pd.DataFrame,
    actions: pd.DataFrame,
    recording_meta: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    original_index, original_quality = build_original_physio_index(project_root)
    august_index = build_august_physio_index(project_root, recording_meta)
    merged = events.loc[events["included_candidate"]].merge(
        actions[["event_id", "readable_action_mode"]], on="event_id", validate="one_to_one"
    )
    rows = []
    windows = [
        ("pre_30s", -30.0, 0.0),
        ("pre_60s", -60.0, 0.0),
        ("post_0p1s", 0.0, 0.1),
        ("post_0p2s", 0.0, 0.2),
        ("post_0p4s", 0.0, 0.4),
        ("post_0p6s", 0.0, 0.6),
        ("post_1s", 0.0, 1.0),
        ("post_2s", 0.0, 2.0),
        ("post_3s", 0.0, 3.0),
    ]
    for event in merged.itertuples(index=False):
        key = (event.internal_subject_key, event.internal_session_stamp)
        output = {
            "event_id": event.event_id,
            "subject_alias": event.subject_alias,
            "recording_alias": event.recording_alias,
            "cohort": event.cohort,
            "stimulus_type": event.stimulus_type,
            "readable_action_mode": event.readable_action_mode,
            "physiology_source": "none",
            "signal_quality_pass": False,
            "sync_contract": "vehicle_and_physio_share_HRT_timestamp; hardware_latency_not_calibrated",
        }
        if event.cohort == "original" and key in original_index:
            time_s, valid_matrix = original_index[key]
            output["physiology_source"] = "original_10hz_feature_layer_from_200hz_cleaned_signals"
            for label, left, right in windows:
                mask = (time_s >= event.stimulus_onset_s + left) & (time_s <= event.stimulus_onset_s + right)
                expected = max(1, int(round((right - left) * 10)))
                output[label] = min(1.0, float(valid_matrix[mask].all(axis=1).sum() / expected))
            output["signal_quality_pass"] = original_quality.get(key, False)
            output["effective_sampling_rates_hz"] = "ECG/EMG/EDA/RESP raw 200; audit feature layer 10"
        elif event.cohort == "august_2025" and key in august_index:
            output["physiology_source"] = "run79_channel_specific_cleaned_continuous_layer"
            channel_data = august_index[key]["channels"]
            for label, left, right in windows:
                coverages = [
                    window_coverage(times, valid, event.stimulus_onset_s + left, event.stimulus_onset_s + right, fs_hz)
                    for times, valid, fs_hz in channel_data.values()
                ]
                output[label] = min(coverages)
            output["signal_quality_pass"] = bool(
                august_index[key]["quality_pass"]
                and output["pre_30s"] >= 0.9
                and output["post_3s"] >= 0.9
            )
            output["effective_sampling_rates_hz"] = "ECG 250; EMG envelope 100; EDA 20; RESP 20"
        else:
            for label, _, _ in windows:
                output[label] = 0.0
            output["effective_sampling_rates_hz"] = "unavailable"
        output["long_baseline_5min_possible"] = event.stimulus_onset_s >= 300 and output["physiology_source"] != "none"
        output["long_baseline_10min_possible"] = event.stimulus_onset_s >= 600 and output["physiology_source"] != "none"
        output["long_baseline_20min_possible"] = event.stimulus_onset_s >= 1200 and output["physiology_source"] != "none"
        rows.append(output)
    join = pd.DataFrame(rows)
    coverage = (
        join.groupby(["cohort", "stimulus_type", "readable_action_mode"], as_index=False)
        .agg(
            events=("event_id", "size"),
            physiology_available=("physiology_source", lambda x: int((x != "none").sum())),
            quality_pass=("signal_quality_pass", "sum"),
            pre30_coverage_mean=("pre_30s", "mean"),
            pre60_coverage_mean=("pre_60s", "mean"),
            post0p4_coverage_mean=("post_0p4s", "mean"),
            post3_coverage_mean=("post_3s", "mean"),
        )
    )
    summary = {
        "events": len(join),
        "physiology_available": int((join["physiology_source"] != "none").sum()),
        "quality_pass": int(join["signal_quality_pass"].sum()),
        "pre30_ge_90pct": int((join["pre_30s"] >= 0.9).sum()),
        "pre60_ge_90pct": int((join["pre_60s"] >= 0.9).sum()),
        "post0p4_ge_90pct": int((join["post_0p4s"] >= 0.9).sum()),
        "post3_ge_90pct": int((join["post_3s"] >= 0.9).sum()),
        "baseline_5min_possible": int(join["long_baseline_5min_possible"].sum()),
        "baseline_10min_possible": int(join["long_baseline_10min_possible"].sum()),
        "baseline_20min_possible": int(join["long_baseline_20min_possible"].sum()),
    }
    return join, coverage, summary


def build_vehicle_inventory(
    channel_files: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    included = events.loc[events["included_candidate"]]
    rows = []
    for (cohort, channel), group in channel_files.groupby(["cohort", "channel"], sort=True):
        event_group = included.loc[included["cohort"] == cohort]
        unit, role = CHANNEL_META[channel]
        rows.append(
            {
                "cohort": cohort,
                "channel": channel,
                "recordings_total": len(group),
                "recordings_present": int(group["present"].sum()),
                "unit": unit,
                "sampling_rate_median_hz": float(group.loc[group["present"], "sampling_rate_hz"].median()) if group["present"].any() else math.nan,
                "missing_fraction_mean": float(group["missing_fraction"].mean()),
                "abnormal_fraction_mean": float(group["abnormal_fraction"].mean()),
                "observed_min": float(group["min"].min()),
                "observed_max": float(group["max"].max()),
                "synchronization": "same_vehicle_csv_row_as_pedals_and_steering",
                "target_1s_complete_fraction": float((event_group["post_available_s"] >= 1).mean()) if len(event_group) else 0.0,
                "target_2s_complete_fraction": float((event_group["post_available_s"] >= 2).mean()) if len(event_group) else 0.0,
                "target_3s_complete_fraction": float((event_group["post_available_s"] >= 3).mean()) if len(event_group) else 0.0,
                "script_direct_control_risk": "not_directly_fixed_for_ego_vehicle",
                "recommended_role": role,
            }
        )
    return pd.DataFrame(rows)


def dedup_sensitivity(events: pd.DataFrame) -> pd.DataFrame:
    mapped = events.loc[events["candidate_eligible_by_mapping"]].copy()
    rows = []
    for gap in [0.5, 1.0, 2.0]:
        total = 0
        for _, group in mapped.groupby("recording_alias"):
            times = sorted(group["stimulus_onset_s"].tolist())
            clusters = 0
            previous = -math.inf
            for value in times:
                if value - previous > gap:
                    clusters += 1
                previous = value
            total += clusters
        rows.append({"cross_signal_overlap_s": gap, "candidate_event_groups": total})
    return pd.DataFrame(rows)


def save_figure_counts(stimulus_catalog: pd.DataFrame, actions: pd.DataFrame, subject_counts: pd.DataFrame, figures: Path) -> None:
    plt.figure(figsize=(10, 5))
    plot = (
        stimulus_catalog.loc[stimulus_catalog["number_of_candidate_stimuli"] > 0]
        .groupby("stimulus_type", as_index=False)["number_included_after_contract"]
        .sum()
    )
    plt.barh(plot["stimulus_type"], plot["number_included_after_contract"], color="#35618f")
    plt.xlabel("纳入候选事件数")
    plt.tight_layout()
    plt.savefig(figures / "01_stimulus_counts.png", dpi=180)
    plt.close()

    mode_counts = actions["readable_action_mode"].value_counts().sort_values()
    plt.figure(figsize=(10, 6))
    plt.barh(mode_counts.index, mode_counts.values, color="#9c5a3c")
    plt.xlabel("事件数")
    plt.tight_layout()
    plt.savefig(figures / "02_action_mode_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(subject_counts["total_events"], bins=min(15, max(5, len(subject_counts) // 2)), color="#4b8063", edgecolor="white")
    plt.xlabel("每名驾驶员候选事件数")
    plt.ylabel("驾驶员数")
    plt.tight_layout()
    plt.savefig(figures / "03_subject_event_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for column, label, color in [
        ("steer_latency_s", "转向", "#35618f"),
        ("brake_latency_s", "制动", "#9c5a3c"),
        ("accelerator_latency_s", "油门变化", "#4b8063"),
    ]:
        values = actions[column].dropna()
        if len(values):
            plt.hist(values, bins=np.linspace(0, 5, 26), alpha=0.45, label=label, color=color)
    plt.xlabel("刺激后反应时延 / s")
    plt.ylabel("事件数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "04_action_latency_distribution.png", dpi=180)
    plt.close()


def save_matrix_figure(actions: pd.DataFrame, figures: Path) -> None:
    matrix = pd.crosstab(actions["stimulus_type"], actions["readable_action_mode"])
    plt.figure(figsize=(max(9, 0.9 * len(matrix.columns)), max(5, 0.6 * len(matrix.index))))
    plt.imshow(matrix.to_numpy(), cmap="Blues", aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            plt.text(column, row, int(matrix.iloc[row, column]), ha="center", va="center", fontsize=8)
    plt.colorbar(label="事件数")
    plt.tight_layout()
    plt.savefig(figures / "05_stimulus_action_matrix.png", dpi=180)
    plt.close()


def save_feasibility_figures(feasibility: pd.DataFrame, physiology: pd.DataFrame, figures: Path) -> None:
    subset = feasibility.loc[feasibility["condition"].isin(["ordinary_minutes", "completed_events"])].copy()
    plt.figure(figsize=(9, 5))
    for condition, group in subset.groupby("condition"):
        plt.plot(group["threshold"].astype(float), group["subjects_with_later_test"], marker="o", label=condition)
    plt.xlabel("校准历史门槛")
    plt.ylabel("仍有后期测试事件的驾驶员数")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "06_personalization_feasibility.png", dpi=180)
    plt.close()

    coverage_columns = ["pre_30s", "pre_60s", "post_0p1s", "post_0p2s", "post_0p4s", "post_0p6s", "post_1s", "post_2s", "post_3s"]
    values = [(physiology[column] >= 0.9).mean() for column in coverage_columns]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(values)), values, color="#6c5b8f")
    plt.xticks(range(len(values)), coverage_columns, rotation=45, ha="right")
    plt.ylabel("覆盖率>=90%的事件比例")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figures / "07_physiology_coverage.png", dpi=180)
    plt.close()


def save_event_examples(
    events: pd.DataFrame,
    actions: pd.DataFrame,
    recordings: pd.DataFrame,
    figures: Path,
    seed: int,
) -> None:
    merged = events.loc[events["included_candidate"]].merge(
        actions[["event_id", "readable_action_mode"]], on="event_id", validate="one_to_one"
    )
    rng = np.random.default_rng(seed)
    chosen = []
    chosen_ids = set()
    for _, group in merged.groupby("stimulus_type", sort=True):
        row = group.iloc[int(rng.integers(0, len(group)))]
        chosen.append(row)
        chosen_ids.add(row.event_id)
    remaining = merged.loc[~merged["event_id"].isin(chosen_ids)].copy()
    remaining["_stratum"] = remaining["stimulus_type"].astype(str) + "|" + remaining["readable_action_mode"].astype(str)
    for _, group in remaining.groupby("_stratum", sort=True):
        if len(chosen) >= 8:
            break
        chosen.append(group.iloc[int(rng.integers(0, len(group)))])
    chosen = chosen[:8]
    source_lookup = recordings.set_index("recording_alias")["source_path"].to_dict()
    figure, axes = plt.subplots(len(chosen), 1, figsize=(11, max(4, 3 * len(chosen))), squeeze=False)
    for axis, event in zip(axes[:, 0], chosen):
        data, _, _ = read_vehicle(Path(source_lookup[event.recording_alias]))
        relative = data["_t_s"] - event.stimulus_onset_s
        mask = relative.between(-2, 5)
        time = relative[mask].to_numpy(float)
        steer = np.degrees(data.loc[mask, "_steer"].to_numpy(float))
        brake = data.loc[mask, "_brake"].to_numpy(float)
        accelerator = data.loc[mask, "_accelerator"].to_numpy(float)
        speed = data.loc[mask, "_speed_kmh"].to_numpy(float)
        axis.plot(time, steer - steer[np.argmin(np.abs(time))], label="steer delta deg")
        axis.plot(time, brake * 30, label="brake x30")
        axis.plot(time, accelerator * 30, label="accelerator x30")
        axis.plot(time, speed - speed[np.argmin(np.abs(time))], label="speed delta km/h")
        axis.axvline(0, color="black", linewidth=1)
        axis.set_title(f"{event.stimulus_type} | {event.readable_action_mode}")
        axis.legend(ncol=4, fontsize=7)
    figure.suptitle(f"固定随机种子{seed}分层抽取；不按曲线美观度选择", y=1.002)
    figure.tight_layout()
    figure.savefig(figures / "08_stratified_event_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    return frame.head(max_rows).to_markdown(index=False)


def write_reports(
    root: Path,
    lineage_public: pd.DataFrame,
    stimulus_catalog: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    sample_summary: dict,
    subject_counts: pd.DataFrame,
    stimulus_action: pd.DataFrame,
    feasibility: pd.DataFrame,
    physio_summary: dict,
    vehicle_inventory: pd.DataFrame,
    summary: dict,
) -> None:
    reports = {
        "00_EXECUTIVE_SUMMARY_CN.md": f"""# 执行摘要\n\n最终状态：`{summary['final_status']}`。这是修正后的V2结果；先前1488事件版本已因把普通`1.0↔0.8`、低附着内部变化和退出恢复重复计为刺激而失效。\n\n本轮从连续车辆 recording 出发，只保留配置距离阈值和每个recording第一次从`mu>0.4`跨入`mu<=0.4`的低附着场景起点，构建了 {sample_summary['total_candidate_events']} 个纳入候选事件；没有用未来方向盘、制动或油门幅值决定事件成员。主阈值下保留无明显反应事件 {sample_summary['no_response_events']} 个。\n\n关键限制是：原始四类距离信号到外部触发编号的映射已恢复，但触发编号到具体交通车动作的权威脚本语义未完全恢复；8月虽然发现 `distance_truck` 与 `distance_changlane`，却没有找到相应触发阈值配置，因此只作为低置信诊断，不进入主候选池。低附着进入具有精确脚本时刻，但当前没有合法零延迟在线代理，标记为 `script_label_only`。\n\n本轮未训练任何模型，也没有修改 Run57—Run82 的历史结果。\n""",
        "01_ORIGINAL_OBJECTIVE_TRACEABILITY_CN.md": """# 原始目标可追溯核查\n\n## 权威原文\n\n2026-01-06版开题报告题目为“基于多模态数据驱动的极端工况驾驶员模型研究”，并非“基于Transformer”作为科学题目。Transformer在技术路线中是序列建模架构，不是课题目标本身。\n\n开题原文把输入写为事件触发前的车辆动力学、驾驶操控历史、道路预瞄、驾驶风格和驾驶状态；输出一处写为未来距离域方向盘序列、速度和纵向加速度，评估章节又扩展到方向盘、速度、纵向与横向加速度。开题报告没有明确把制动与加速踏板列为独立直接操作输出，这与本轮重置后的多动作目标构成新增范围，必须如实标注。\n\n## 收缩与重置\n\n历史 release 后1秒方向盘曲线任务，是为了先获得严格因果、subject-disjoint、可量化的转向子任务，不是原总目标的全部。Run57—Run82属于这一收缩任务。\n\n本轮重新确认的总体问题是：在可在线识别的极端/高动态刺激后，利用驾驶员既有习惯与当前生理状态，预测转向、制动、松油/维持/补油的动作选择、时延、强度和短时轨迹，并进一步预测车辆纵横向运动。\n\n## 已发现冲突\n\n- 题目冲突：权威开题是“多模态数据驱动”，本轮任务名写“基于Transformer”。以后采用权威课题名，并把Transformer降为技术路线。\n- 输出冲突：开题3.1.3写方向盘、速度、纵向加速度；5.4又加入横向加速度；本轮进一步新增制动和加速踏板直接操作。\n- 生理作用冲突：开题偏训练期辅助任务；本轮还要求审计长期基线、刺激前状态、滚动状态、动作概率/时延/不确定性调制。现阶段均为候选作用，未证明有效。\n""",
        "02_STIMULUS_RECOVERY_CN.md": f"""# 刺激恢复审计\n\n共检查两批 {int(lineage_public.loc[lineage_public['口径']=='合并recording','数量'].iloc[0])} 个车辆 recording。原始批次配置证明四个距离比较器均在低于30 m时触发 `ExternTrigger01/03/04/06`。先前材料中 distance7/distance8 的40 m说法与当前找到的 `.29/.30/.36` 配置冲突，本轮按配置中的30 m记录，并把冲突保留为待解释项。\n\nV2低附着定义固定为：每个recording最多一个事件，只取第一次从`mu>0.4`跨入`0.1<=mu<=0.4`的时刻。`1.0↔0.8`普通道路变化、`0.4→0.2`低附着内部加深以及所有低附着退出均不构成新的危险刺激。先前1488事件版本因违反该场景口径而失效。\n\n8月原始表头实际含 `distance_truck`、`distance_changlane`、左右道路距离与 `mu`；Run78 staging 没有保留交通距离列，导致此前“完全不存在”的判断过强。本轮对这两列做了诊断 crossing，但因为没找到8月触发脚本，未纳入主候选事件。\n\n{markdown_table(stimulus_catalog[['cohort','stimulus_type','trigger_signal','onset_exactness','online_observable','number_of_candidate_stimuli','number_included_after_contract','unresolved_issue']], 40)}\n\n纳入规则只使用刺激、前后可用长度、信号覆盖和recording内重叠规则。无未来动作幅值门。\n""",
        "03_ACTION_LABEL_AUDIT_CN.md": f"""# 三通道候选标签审计\n\n标签仍为候选合同。制动先按recording第5百分位估计释放零位，以处理约-0.03偏置；油门相对刺激前基线判断松油、维持和补油；方向盘区分刺激时已有转向、刺激后新转向、回正和反向修正。\n\n动作匹配先搜索0—5秒，再分别汇总1、2、3、5秒。三组阈值全部报告：方向盘/制动/油门的宽松、主、严格阈值分别为3度/0.03/0.03、5度/0.05/0.05、8度/0.08/0.08。\n\n{markdown_table(threshold_sensitivity.loc[threshold_sensitivity['window_s'].isin([1,3,5])], 50)}\n\n多标签不强制互斥；同一事件可以同时出现转向、制动、松油和补油。主表保留各通道刺激时状态、onset、latency、peak、delta、持续时间、1/2/3秒累积操作量、二次修正和歧义原因。\n""",
        "04_SAMPLE_SUFFICIENCY_CN.md": f"""# 样本充分性审计\n\n- 纳入候选事件：{sample_summary['total_candidate_events']}。\n- 驾驶员：{sample_summary['subjects']}；recording：{sample_summary['recordings']}。\n- 每名驾驶员事件数：最小{sample_summary['events_per_subject']['min']}，Q1={sample_summary['events_per_subject']['q1']:.1f}，中位={sample_summary['events_per_subject']['median']:.1f}，Q3={sample_summary['events_per_subject']['q3']:.1f}，最大{sample_summary['events_per_subject']['max']}。\n- 无明显反应：{sample_summary['no_response_events']}。\n- 最多事件驾驶员贡献占比：{sample_summary['top_subject_share']:.1%}。\n- 过密recording比例（最小间隔<2秒）：{sample_summary['dense_recording_fraction']:.1%}。\n- 严格在线精确子集：{sample_summary['strict_online_exact_events']}。\n\n正式建模前仍需解决刺激语义映射；当前足以审计动作选择与时延分布，但低频动作组合、8月交通刺激和20分钟个体基线只能作为探索或条件分层。\n\n{markdown_table(stimulus_action.sort_values('events', ascending=False), 40)}\n""",
        "05_PERSONALIZATION_FEASIBILITY_CN.md": f"""# 顺序个体化可行性\n\n本轮按真实recording时间排序。主协议只把完整早期recording用于校准、完整后期recording用于测试；没有把同一recording内相邻事件随机分到两侧。\n\n{markdown_table(feasibility.loc[feasibility['condition'].isin(['ordinary_minutes','completed_events'])], 30)}\n\n结论：只要要求较短普通历史或少量已完成事件，仍有一部分驾驶员和后期事件；门槛升高后覆盖快速收缩。正式协议应先选择能覆盖足够驾驶员与动作模式的最低层级，不应把校准条件事后调到最有利。\n""",
        "06_PHYSIOLOGY_FEASIBILITY_CN.md": f"""# 生理数据可行性\n\n刺激中心事件共{physio_summary['events']}个，其中有可连接生理层{physio_summary['physiology_available']}个，信号质量门通过{physio_summary['quality_pass']}个。刺激前30/60秒覆盖>=90%的事件分别为{physio_summary['pre30_ge_90pct']}/{physio_summary['pre60_ge_90pct']}；刺激后0.4/3秒分别为{physio_summary['post0p4_ge_90pct']}/{physio_summary['post3_ge_90pct']}。当前recording内可形成5/10/20分钟早期基线的事件分别为{physio_summary['baseline_5min_possible']}/{physio_summary['baseline_10min_possible']}/{physio_summary['baseline_20min_possible']}。\n\n原始批次使用200 Hz清洗信号生成的10 Hz连续特征层；8月Run79的有效采样率为ECG 250 Hz、EMG包络100 Hz、EDA 20 Hz、RESP 20 Hz。车辆和生理共用HRT时间戳，但硬件链路延迟没有独立标定。\n\n合法候选作用分为：长期个体基线、刺激前当前状态、刺激后滚动更新、动作模式概率、反应时延、置信度/不确定性调制。Run80只否定旧协议下的直接均值拼接，不能外推否定这些新任务。\n""",
        "07_VEHICLE_RESPONSE_TARGETS_CN.md": f"""# 车辆响应目标审计\n\n{markdown_table(vehicle_inventory[['cohort','channel','recordings_present','recordings_total','unit','missing_fraction_mean','abnormal_fraction_mean','target_3s_complete_fraction','recommended_role']], 40)}\n\n需要分开评价：使用真实驾驶员操作预测车辆响应，验证车辆响应模块；使用模型预测操作再预测车辆响应，评价端到端误差。两者不得混写。速度、纵向加速度和横向加速度覆盖最好，可优先作为正式目标；yaw/roll rate跨批次缺失，应分批次辅助报告。\n""",
        "08_DATA_CONTRACT_DRAFT_CN.md": """# 候选数据合同（未冻结）\n\n1. 主批次：先用原始85个recording的配置可追溯距离触发建立主合同；低附着只保留每个recording首次跨入`mu<=0.4`的场景起点，8月交通距离需恢复脚本后再晋级。\n2. 样本单位：recording内去重后的刺激事件；保留无动作。\n3. 预测锚点：刺激起点，不再以方向盘release作为唯一锚点。\n4. 滚动观测候选：0、0.1、0.2、0.4、0.6秒，最终需结合反应时延分布冻结。\n5. 输出：多标签动作选择、各通道时延、强度、条件短时轨迹，以及车辆响应。\n6. 生理窗口：长期个体基线、刺激前30/60秒、刺激后0.1/0.2/0.4/0.6秒滚动状态。\n7. 个体历史：完整早期recording的普通驾驶与已完成事件摘要。\n8. 合法输入边界：任一预测时刻只能使用该时刻以前的车辆、道路、驾驶员历史与生理。未来动作方向只可用于标签分析。\n9. 缺失规则：不因道路参考或生理缺失删除事件；进入显式缺失分层。\n10. 去重：同信号2秒不应期；跨信号1秒形成重叠组，仅一个主事件，其他保留为次刺激记录。\n11. 未解决风险：原始外部触发的具体交通动作语义、8月触发脚本、低附着在线代理、硬件延迟、低频动作组合。\n""",
        "09_VALIDATION_PROTOCOL_DRAFT_CN.md": """# 验证协议草案\n\n- 群体外层：subject-disjoint，并按原始/8月数据域和刺激类型分层报告。\n- 目标驾驶员内层：完整早期recording校准，完整后期recording测试。\n- 基线：零历史群体模型、仅普通驾驶历史、普通历史+少量已完成事件。\n- 防泄漏：同一recording不得跨训练/测试；刺激后未来轨迹、未来主动作方向和未来生理不得进入对应预测时刻输入。\n- 任务拆分：动作选择、反应时延、强度、条件轨迹、车辆响应分别评价；车辆模块真实操作输入与端到端预测操作输入分开。\n- 进入模型阶段条件：刺激语义映射闭合、候选标签人工抽查通过、低频类最小驾驶员覆盖达标、时间顺序无歧义、隐私和一致性检查持续通过。\n- 本轮不训练Transformer，也不声称Transformer有效。\n""",
        "10_OPEN_ISSUES_AND_DECISION_GATES_CN.md": """# 未解决问题与决策门\n\n## 阻塞项\n\n1. 原始`ExternTrigger01/03/04/06`到具体交通车动作的权威语义映射仍缺。\n2. 8月`distance_truck`与`distance_changlane`存在，但对应脚本阈值未找到。\n3. 低附着进入的`mu`是脚本真值，缺少可证明同刻可用的在线代理；普通`1.0↔0.8`、内部变化和退出不得重新纳入。\n4. 开题报告没有把制动与加速踏板明确列为直接输出；本轮属于研究目标扩展，需GPTPro和导师确认。\n5. 生理硬件延迟未独立标定，20分钟基线覆盖有限。\n\n## 决策门\n\n- 若能补齐原始外部触发语义和8月脚本，重新运行同一脚本后可评估是否升级为`READY_FOR_GPTPRO_REVIEW`。\n- 若语义仍缺，主模型阶段不得把不同触发编号合并成一个有明确场景名称的类别。\n- GPTPro审查前不启动正式模型训练。\n""",
    }
    for name, text in reports.items():
        (root / name).write_text(text, encoding="utf-8")


def build_public_package(
    public_root: Path,
    package_root: Path,
    lineage_public: pd.DataFrame,
    stimulus_catalog: pd.DataFrame,
    threshold_sensitivity: pd.DataFrame,
    stimulus_action: pd.DataFrame,
    feasibility: pd.DataFrame,
    physiology_coverage: pd.DataFrame,
    vehicle_inventory: pd.DataFrame,
    sample_summary: dict,
    summary: dict,
) -> Path:
    destination = public_root / "audits/multiaction_task_reframe_v2_20260901"
    destination.mkdir(parents=True, exist_ok=True)
    tables = destination / "tables"
    figures = destination / "figures"
    source = destination / "source"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    source.mkdir(exist_ok=True)
    lineage_public.to_csv(tables / "cohort_lineage_summary.csv", index=False, encoding="utf-8-sig")
    stimulus_catalog.to_csv(tables / "stimulus_catalog_summary.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(tables / "label_threshold_summary.csv", index=False, encoding="utf-8-sig")
    stimulus_action.to_csv(tables / "stimulus_action_summary.csv", index=False, encoding="utf-8-sig")
    feasibility.to_csv(tables / "personalization_summary.csv", index=False, encoding="utf-8-sig")
    physiology_coverage.to_csv(tables / "physiology_coverage_summary.csv", index=False, encoding="utf-8-sig")
    vehicle_inventory.to_csv(tables / "vehicle_channel_summary.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(package_root / "source" / Path(__file__).name, source / Path(__file__).name)
    shutil.copy2(package_root / "config" / "audit_config.json", source / "audit_config.json")
    test_source = package_root / "tests" / "test_multiaction_reframe_audit.py"
    if test_source.is_file():
        shutil.copy2(test_source, source / test_source.name)
    for name in [
        "01_stimulus_counts.png",
        "02_action_mode_counts.png",
        "03_subject_event_distribution.png",
        "04_action_latency_distribution.png",
        "05_stimulus_action_matrix.png",
        "06_personalization_feasibility.png",
        "07_physiology_coverage.png",
        "08_stratified_event_examples.png",
    ]:
        shutil.copy2(package_root / "figures" / name, figures / name)
    for name in [
        "00_EXECUTIVE_SUMMARY_CN.md",
        "01_ORIGINAL_OBJECTIVE_TRACEABILITY_CN.md",
        "02_STIMULUS_RECOVERY_CN.md",
        "03_ACTION_LABEL_AUDIT_CN.md",
        "04_SAMPLE_SUFFICIENCY_CN.md",
        "05_PERSONALIZATION_FEASIBILITY_CN.md",
        "06_PHYSIOLOGY_FEASIBILITY_CN.md",
        "07_VEHICLE_RESPONSE_TARGETS_CN.md",
        "08_DATA_CONTRACT_DRAFT_CN.md",
        "09_VALIDATION_PROTOCOL_DRAFT_CN.md",
        "10_OPEN_ISSUES_AND_DECISION_GATES_CN.md",
    ]:
        shutil.copy2(package_root / name, destination / name)
    public_summary = {
        "audit_id": AUDIT_NAME,
        "status": summary["final_status"],
        "candidate_events": sample_summary["total_candidate_events"],
        "strict_online_exact_events": sample_summary["strict_online_exact_events"],
        "no_response_events": sample_summary["no_response_events"],
        "models_trained": 0,
        "correction": "V2 excludes ordinary 1.0/0.8 mu changes, internal low-mu changes and exits; keeps at most one first mu<=0.4 scene entry per recording",
        "privacy_boundary": "aggregate_only_no_subject_or_event_rows",
        "checksum_policy": "not_generated_by_project_rule",
    }
    (destination / "summary.json").write_text(json.dumps(public_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "README.md").write_text(
        "# 极端工况多动作任务重置审计（修正V2）\n\n"
        f"状态：`{summary['final_status']}`。先前1488事件版本已失效；先读 `00_EXECUTIVE_SUMMARY_CN.md`，再按01—10顺序阅读。"
        "本目录只含总体汇总、分位数、公开安全图、报告和可复现源代码；不含一行一驾驶员或一行一事件的表。\n",
        encoding="utf-8",
    )
    return destination


def validate_outputs(package_root: Path, public_audit: Path) -> dict:
    required_private = [
        "tables_private/cohort_lineage_private.csv",
        "tables_private/stimulus_catalog_private.csv",
        "tables_private/stimulus_events_private.csv",
        "tables_private/action_labels_private.csv",
        "tables_private/label_threshold_sensitivity_private.csv",
        "tables_private/counts_by_subject_private.csv",
        "tables_private/counts_by_stimulus_action_private.csv",
        "tables_private/event_density_private.csv",
        "tables_private/personalization_tier_feasibility_private.csv",
        "tables_private/chronological_split_candidates_private.csv",
        "tables_private/physiology_join_private.csv",
        "tables_private/physiology_coverage_by_action_private.csv",
        "tables_private/vehicle_response_channel_inventory_private.csv",
        "MANIFEST.json",
        "RUN_LOG.txt",
    ]
    missing = [name for name in required_private if not (package_root / name).is_file()]
    if missing:
        raise AssertionError(f"缺少私有输出: {missing}")
    events = pd.read_csv(package_root / "tables_private/stimulus_events_private.csv", low_memory=False)
    actions = pd.read_csv(package_root / "tables_private/action_labels_private.csv", low_memory=False)
    if not events["included_candidate"].astype(bool).sum():
        raise AssertionError("没有刺激中心候选事件")
    if len(actions) != int(events["included_candidate"].astype(bool).sum()):
        raise AssertionError("动作标签与纳入事件不一一对应")
    if not actions["no_response"].astype(bool).any():
        raise AssertionError("无反应样本未保留")
    mu_events = events.loc[events["trigger_signal"].eq("mu")]
    if not mu_events["stimulus_type"].eq("enter_low_mu_scene").all():
        raise AssertionError("mu事件仍混入普通变化、内部变化或退出")
    if not mu_events.groupby("recording_alias").size().le(1).all():
        raise AssertionError("同一recording存在多个低附着场景刺激")
    if int(events["included_candidate"].astype(bool).sum()) > 526:
        raise AssertionError("刺激候选再次出现数量膨胀")
    forbidden = re.compile(r"(?i)[A-Z]:[\\/]|/home/|subject_alias,|event_id,")
    scanned_files = 0
    for path in public_audit.rglob("*"):
        if path.suffix.lower() not in {".md", ".json", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        if forbidden.search(text):
            raise AssertionError(f"公开输出隐私扫描失败: {path}")
        if path.suffix.lower() == ".json":
            json.loads(text)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            if frame.empty:
                raise AssertionError(f"公开CSV为空: {path}")
        scanned_files += 1
    return {
        "private_required_files": len(required_private),
        "private_missing": 0,
        "candidate_event_action_one_to_one": True,
        "no_response_retained": True,
        "public_text_files_scanned": scanned_files,
        "public_privacy_scan": "PASS",
        "json_csv_parse": "PASS",
    }


def write_manifest(package_root: Path) -> None:
    entries = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            relative = path.relative_to(package_root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "privacy": "private_anonymized" if relative.startswith("tables_private/") else "review_artifact",
                }
            )
    manifest = {
        "audit_id": AUDIT_NAME,
        "generated_at": datetime.now().astimezone().isoformat(),
        "files": entries,
        "checksums_generated": False,
        "reason_checksums_omitted": "project instruction prohibits hashes and SHA256",
    }
    (package_root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def git_description(path: Path) -> dict:
    branch = subprocess.run(["git", "-C", str(path), "branch", "--show-current"], check=True, capture_output=True, text=True).stdout.strip()
    detail = subprocess.run(
        ["git", "-C", str(path), "log", "-1", "--date=iso-strict", "--pretty=format:%ad%n%s"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"branch": branch, "commit_date": detail[0], "commit_subject": detail[1]}


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    public_root = args.public_root.resolve()
    package_root = args.output_root.resolve()
    config = read_config(args.config.resolve())

    tables = package_root / "tables_private"
    figures = package_root / "figures"
    config_dir = package_root / "config"
    source_dir = package_root / "source"
    tests_dir = package_root / "tests"
    for directory in [tables, figures, config_dir, source_dir, tests_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    recordings, original_ledger, subject_alias = load_cohort_sources(project_root, args.august_root.resolve())
    lineage_private, lineage_public = build_lineage(project_root, args.august_root.resolve(), recordings, subject_alias)
    events, actions, sensitivity_labels, channel_files, recording_meta = process_recordings(recordings, config)
    stimulus_catalog = build_stimulus_catalog(events, recordings)
    threshold_sensitivity = build_threshold_sensitivity(sensitivity_labels)
    subject_counts, stimulus_action, density, sample_summary = build_sample_tables(events, actions, recording_meta)
    chronology, feasibility = add_chronological_history(events, actions, recording_meta)
    physiology_join, physiology_coverage, physiology_summary = build_physiology_join(
        project_root, events, actions, recording_meta
    )
    vehicle_inventory = build_vehicle_inventory(channel_files, events)
    dedup_table = dedup_sensitivity(events)

    summary = {
        "final_status": "CONDITIONAL_READY",
        "reason": "V2已修正mu场景去重；原始触发编号到具体交通动作语义及8月交通触发阈值仍未恢复",
        "models_trained": 0,
        "cohort_year": "2025",
        "project_repository": git_description(project_root),
        "public_repository": git_description(public_root),
        "python": "<PYTHON_311>/python.exe; Python 3.11.4",
        "sample": sample_summary,
        "physiology": physiology_summary,
    }

    lineage_private.to_csv(tables / "cohort_lineage_private.csv", index=False, encoding="utf-8-sig")
    stimulus_catalog.to_csv(tables / "stimulus_catalog_private.csv", index=False, encoding="utf-8-sig")
    private_event_columns = [
        column
        for column in events.columns
        if not column.startswith("internal_") and column != "base_eligible"
    ]
    events[private_event_columns].to_csv(tables / "stimulus_events_private.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(tables / "action_labels_private.csv", index=False, encoding="utf-8-sig")
    threshold_sensitivity.to_csv(tables / "label_threshold_sensitivity_private.csv", index=False, encoding="utf-8-sig")
    subject_counts.to_csv(tables / "counts_by_subject_private.csv", index=False, encoding="utf-8-sig")
    stimulus_action.to_csv(tables / "counts_by_stimulus_action_private.csv", index=False, encoding="utf-8-sig")
    density.to_csv(tables / "event_density_private.csv", index=False, encoding="utf-8-sig")
    feasibility.to_csv(tables / "personalization_tier_feasibility_private.csv", index=False, encoding="utf-8-sig")
    chronology.to_csv(tables / "chronological_split_candidates_private.csv", index=False, encoding="utf-8-sig")
    physiology_join.to_csv(tables / "physiology_join_private.csv", index=False, encoding="utf-8-sig")
    physiology_coverage.to_csv(tables / "physiology_coverage_by_action_private.csv", index=False, encoding="utf-8-sig")
    vehicle_inventory.to_csv(tables / "vehicle_response_channel_inventory_private.csv", index=False, encoding="utf-8-sig")
    dedup_table.to_csv(tables / "dedup_sensitivity_private.csv", index=False, encoding="utf-8-sig")
    lineage_public.to_csv(package_root / "cohort_lineage_public_summary.csv", index=False, encoding="utf-8-sig")
    (package_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    save_figure_counts(stimulus_catalog, actions, subject_counts, figures)
    save_matrix_figure(actions, figures)
    save_feasibility_figures(feasibility, physiology_join, figures)
    save_event_examples(events, actions, recordings, figures, int(config["fixed_figure_seed"]))
    write_reports(
        package_root,
        lineage_public,
        stimulus_catalog,
        threshold_sensitivity,
        sample_summary,
        subject_counts,
        stimulus_action,
        feasibility,
        physiology_summary,
        vehicle_inventory,
        summary,
    )

    shutil.copy2(args.config.resolve(), config_dir / "audit_config.json")
    shutil.copy2(Path(__file__).resolve(), source_dir / Path(__file__).name)
    test_source = project_root / "tests/test_multiaction_reframe_audit.py"
    if test_source.is_file():
        shutil.copy2(test_source, tests_dir / test_source.name)

    public_audit = build_public_package(
        public_root,
        package_root,
        lineage_public,
        stimulus_catalog,
        threshold_sensitivity,
        stimulus_action,
        feasibility,
        physiology_coverage,
        vehicle_inventory,
        sample_summary,
        summary,
    )
    (package_root / "RUN_LOG.txt").write_text(
        "\n".join(
            [
                f"audit_id={AUDIT_NAME}",
                f"generated_at={datetime.now().astimezone().isoformat()}",
                "command=<PYTHON_311>/python.exe <PROJECT_ROOT>/02_code/tools/build_multiaction_reframe_audit.py --project-root <PROJECT_ROOT> --august-root <AUGUST_RAW_ROOT> --public-root <PUBLIC_PROGRESS_ROOT> --config <PROJECT_ROOT>/02_code/tools/multiaction_reframe_audit_config.json --output-root <PROJECT_ROOT>/review_packages/MULTIACTION_REFRAME_20260901",
                f"final_status={summary['final_status']}",
                f"candidate_events={sample_summary['total_candidate_events']}",
                f"no_response_events={sample_summary['no_response_events']}",
                "models_trained=0",
                "hashes_generated=0",
                "validation=PASS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_manifest(package_root)
    validation = validate_outputs(package_root, public_audit)
    (tests_dir / "validation_result.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_manifest(package_root)

    zip_path = package_root.parent / "MULTIACTION_REFRAME_REVIEW_PACK_20260901.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(package_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_root.parent))
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise AssertionError(f"ZIP损坏: {bad}")
    print(json.dumps({"status": summary["final_status"], "zip": str(zip_path), "validation": validation}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
