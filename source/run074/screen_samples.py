from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = Path(r"<AUGUST_RAW_ROOT>\zyl")
STAGING = HERE / "staging" / "zyl" / "vehicle"
TABLES = HERE / "tables"
OUTPUTS = HERE / "outputs"

SESSION_STAMPS = [
    "2025_08_21_20_06_11",
    "2025_08_21_20_13_48",
    "2025_08_21_20_21_08",
    "2025_08_21_20_25_10",
]

DETECTOR_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/anchor_v3_audit_20260805/run_anchor_v3_audit_20260805.py"
DETECTOR_CONFIG_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/anchor_v3_audit_20260805/anchor_v3_config.json"
BASE_CONFIG_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/rescreen_config.json"
THRESHOLD_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/tables/leave_one_subject_response_thresholds.csv"


STAGED_COLUMNS = [
    "t_s",
    "StorageTime",
    "zx|SteeringWheel",
    "zx|AcceleratorPedal",
    "zx|BrakePedal",
    "zx|ax",
    "zx|ay",
    "zx|vx",
    "zx|vy",
    "zx|ayaw",
    "zx|aroll",
    "zx|apitch",
    "zx|roll",
    "zx|pitch",
    "zx|yaw",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)

file_rows = []
staged_paths = []
bio_paths = {}
for stamp in SESSION_STAMPS:
    raw_path = RAW / f"Entity_Recording_{stamp}.csv"
    bio_path = RAW / f"Entity_Recording_{stamp} (2).csv"
    frame = pd.read_csv(raw_path, encoding="utf-8-sig", low_memory=False)
    steering = pd.to_numeric(frame["zx|SteeringWheel"], errors="coerce")
    frame = frame.loc[steering.notna(), [column for column in STAGED_COLUMNS if column in frame.columns]].copy()
    frame["StorageTime"] = pd.to_datetime(frame["StorageTime"], errors="raise")
    frame = frame.sort_values("StorageTime").drop_duplicates("StorageTime").reset_index(drop=True)
    frame["t_s"] = (frame["StorageTime"] - frame["StorageTime"].iloc[0]).dt.total_seconds()
    frame["StorageTime"] = frame["StorageTime"].dt.strftime("%Y/%m/%d %H:%M:%S.%f").str[:-3]
    staged_path = STAGING / f"Entity_Recording_{stamp}_vehicle_aligned_cleaned.csv"
    frame.to_csv(staged_path, index=False, encoding="utf-8-sig")
    staged_paths.append(staged_path)
    bio_paths[stamp] = bio_path
    file_rows.append(
        {
            "session_stamp": stamp,
            "raw_path": str(raw_path),
            "bio_path": str(bio_path),
            "staged_path": str(staged_path),
            "vehicle_rows": len(frame),
            "vehicle_duration_s": float(frame["t_s"].iloc[-1] - frame["t_s"].iloc[0]),
            "steering_rate_hz": float(len(frame) / max(frame["t_s"].iloc[-1] - frame["t_s"].iloc[0], 1e-9)),
        }
    )

detector = load_module("run74_anchor_v3", DETECTOR_PATH)
detector_config = json.loads(DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
base = detector.load_base_module(REPO, detector_config)
thresholds = pd.read_csv(THRESHOLD_PATH)
threshold_columns = [column for column in thresholds.columns if column.endswith(("_p95", "_p975", "_p99"))]
zyl_threshold = {column: float(pd.to_numeric(thresholds[column], errors="raise").median()) for column in threshold_columns}

events, pulses, releases, responses, recordings = detector.process_paths(
    staged_paths,
    detector_config,
    base_config,
    {"zyl": zyl_threshold},
    base,
)

events.to_csv(TABLES / "all_detected_events.csv", index=False, encoding="utf-8-sig")
pulses.to_csv(TABLES / "all_detected_pulses.csv", index=False, encoding="utf-8-sig")
releases.to_csv(TABLES / "all_detected_releases.csv", index=False, encoding="utf-8-sig")
recordings.to_csv(TABLES / "detector_recordings.csv", index=False, encoding="utf-8-sig")

primary_pulses = pulses[["pulse_id", "pulse_direction", "direction_consistency"]].rename(
    columns={
        "pulse_id": "primary_pulse_id",
        "pulse_direction": "direction",
        "direction_consistency": "direction_consistency_at_anchor",
    }
)
screen = events.merge(primary_pulses, on="primary_pulse_id", how="left", validate="many_to_one")

candidate_rows = []
for recording_uid, group in screen.groupby("recording_uid", sort=True):
    stamp_match = recording_uid.split("_rec-")[-1]
    stamp = f"{stamp_match[:4]}_{stamp_match[4:6]}_{stamp_match[6:8]}_{stamp_match[9:11]}_{stamp_match[11:13]}_{stamp_match[13:15]}"
    staged_path = STAGING / f"Entity_Recording_{stamp}_vehicle_aligned_cleaned.csv"
    vehicle = pd.read_csv(staged_path, encoding="utf-8-sig", low_memory=False)
    time_s = pd.to_numeric(vehicle["t_s"], errors="raise").to_numpy(float)
    steer = pd.to_numeric(vehicle["zx|SteeringWheel"], errors="raise").to_numpy(float)
    speed = np.abs(pd.to_numeric(vehicle["zx|vx"], errors="coerce").to_numpy(float)) * 3.6
    ay = pd.to_numeric(vehicle["zx|ay"], errors="coerce").to_numpy(float)
    roll = pd.to_numeric(vehicle["zx|roll"], errors="coerce").to_numpy(float)
    storage_start = pd.to_datetime(vehicle["StorageTime"].iloc[0], errors="raise")

    bio = pd.read_csv(
        bio_paths[stamp],
        usecols=lambda column: column == "StorageTime" or "PhysioLAB Pro1" in column,
        encoding="utf-8-sig",
        low_memory=False,
    )
    bio_time = pd.to_datetime(bio["StorageTime"], errors="raise")
    physio_columns = [column for column in bio.columns if "PhysioLAB Pro1" in column]
    physio_valid = np.column_stack(
        [pd.to_numeric(bio[column], errors="coerce").notna().to_numpy() for column in physio_columns]
    ).all(axis=1)

    for row in group.itertuples(index=False):
        release_s = float(row.primary_release_s)
        history_mask = (time_s >= release_s - 2.0) & (time_s <= release_s)
        common_coverage = min(
            float(np.mean(np.isfinite(values[history_mask])))
            for values in [steer, speed, ay, roll]
        ) if history_mask.any() else 0.0
        target_times = release_s + np.arange(1, 21) * 0.05
        target_complete = bool(target_times[-1] <= time_s[-1] + 1e-9)
        direction = int(row.direction) if math.isfinite(float(row.direction)) else 0
        if target_complete and direction in {-1, 1}:
            release_value = float(np.interp(release_s, time_s, steer))
            target_curve = np.degrees((np.interp(target_times, time_s, steer) - release_value) * direction)
        else:
            target_curve = np.full(20, np.nan)

        anchor_time = storage_start + pd.to_timedelta(release_s, unit="s")
        short_mask = (bio_time >= anchor_time - pd.Timedelta(seconds=2)) & (bio_time <= anchor_time)
        slow_mask = (bio_time >= anchor_time - pd.Timedelta(seconds=30)) & (bio_time <= anchor_time)
        short_coverage = float(np.sum(short_mask.to_numpy() & physio_valid) / 2000.0)
        slow_coverage = float(np.sum(slow_mask.to_numpy() & physio_valid) / 30000.0)
        membership = bool(
            bool(row.history_complete)
            and target_complete
            and float(row.pre_speed_p10_kmh) >= 60.0
            and not bool(row.pre_reverse)
            and float(row.direction_consistency_at_anchor) >= 0.70
            and common_coverage >= 0.90
        )
        amplitude = abs(float(row.first_decisive_pulse_amplitude_deg_coordinate))
        if amplitude < 20:
            amplitude_bin = "lt20"
        elif amplitude < 30:
            amplitude_bin = "20_30"
        elif amplitude < 45:
            amplitude_bin = "30_45"
        elif amplitude < 70:
            amplitude_bin = "45_70"
        else:
            amplitude_bin = "ge70"
        output = {
            "event_uid": row.event_uid,
            "subject": "zyl",
            "recording_uid": recording_uid,
            "session_stamp": stamp,
            "prediction_anchor_s": release_s,
            "prediction_anchor_time": anchor_time,
            "history_complete": bool(row.history_complete),
            "target_complete": target_complete,
            "pre_speed_p10_kmh": float(row.pre_speed_p10_kmh),
            "pre_reverse": bool(row.pre_reverse),
            "direction": direction,
            "direction_consistency": float(row.direction_consistency_at_anchor),
            "common_vehicle_coverage_min": common_coverage,
            "yaw_rate_available": False,
            "roll_rate_available": False,
            "physio_2s_coverage": min(short_coverage, 1.0),
            "physio_30s_coverage": min(slow_coverage, 1.0),
            "pulse_amplitude_deg_report_only": amplitude,
            "amplitude_bin_report_only": amplitude_bin,
            "screen_eligible": membership,
        }
        for point, value in enumerate(target_curve, start=1):
            output[f"true_t{point:02d}_deg"] = float(value)
        output["true_peak_amplitude_deg"] = float(np.nanmax(np.abs(target_curve))) if target_complete else np.nan
        output["true_peak_time_s"] = float((np.nanargmax(np.abs(target_curve)) + 1) * 0.05) if target_complete else np.nan
        candidate_rows.append(output)

candidates = pd.DataFrame(candidate_rows).sort_values(["session_stamp", "prediction_anchor_s"]).reset_index(drop=True)
candidates.to_csv(TABLES / "screened_events.csv", index=False, encoding="utf-8-sig")

summary = pd.DataFrame(file_rows)
detected_counts = events.groupby("recording_uid").size()
eligible_counts = candidates.loc[candidates["screen_eligible"]].groupby("recording_uid").size()
summary["recording_uid"] = summary["session_stamp"].map(
    lambda stamp: f"sub-zyl_rec-{stamp[:4]}{stamp[5:7]}{stamp[8:10]}T{stamp[11:13]}{stamp[14:16]}{stamp[17:19]}"
)
summary["detected_event_count"] = summary["recording_uid"].map(detected_counts).fillna(0).astype(int)
summary["screen_eligible_event_count"] = summary["recording_uid"].map(eligible_counts).fillna(0).astype(int)
summary.to_csv(TABLES / "file_summary.csv", index=False, encoding="utf-8-sig")

eligible = candidates.loc[candidates["screen_eligible"]].copy()
bin_counts = eligible["amplitude_bin_report_only"].value_counts().to_dict()
lines = [
    "# zyl 可用事件筛选结果",
    "",
    f"- 检查文件：{len(SESSION_STAMPS)}",
    f"- Anchor-v3检测事件：{len(events)}",
    f"- 满足2秒历史、1秒目标、速度、无反向、方向一致性和共同车辆覆盖的事件：{len(eligible)}",
    f"- 有完整2秒生理覆盖的合格事件：{int((eligible['physio_2s_coverage'] >= 0.90).sum())}",
    f"- 有至少90%完整30秒生理覆盖的合格事件：{int((eligible['physio_30s_coverage'] >= 0.90).sum())}",
    "- 道路、30度幅值和车辆响应分层没有用于成员筛选。",
    "- yaw_rate/vyaw和roll_rate/vroll保持缺失，没有用ayaw/aroll冒充。",
    "",
    "## 分文件",
    "",
    "| session | duration s | detected | eligible |",
    "|---|---:|---:|---:|",
]
for row in summary.itertuples(index=False):
    lines.append(
        f"| {row.session_stamp} | {row.vehicle_duration_s:.1f} | {row.detected_event_count} | {row.screen_eligible_event_count} |"
    )
lines += ["", "## 幅值分布（只报告）", ""]
for label in ["lt20", "20_30", "30_45", "45_70", "ge70"]:
    lines.append(f"- {label}: {int(bin_counts.get(label, 0))}")
lines += [
    "",
    "## 判定",
    "",
    "有合格事件即说明原始数据可以进入下一步共同车辆/生理外部样本构建；样本统计强度另按事件数、文件分布和生理覆盖判断。",
]
(OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
