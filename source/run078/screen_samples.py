from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW_ROOT = Path(r"<AUGUST_RAW_ROOT>\zssy_zx\zssy_zx")
RUN = HERE / "run_1"
TABLES = RUN / "tables"
OUTPUTS = RUN / "outputs"
STAGING = RUN / "staging"

DETECTOR_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/anchor_v3_audit_20260805/run_anchor_v3_audit_20260805.py"
DETECTOR_CONFIG_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/anchor_v3_audit_20260805/anchor_v3_config.json"
BASE_CONFIG_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/rescreen_config.json"
THRESHOLD_PATH = REPO / "05_rebuild_from_raw_20260511/01_audit/highway_emergency_episode_rescreen_20260801/tables/leave_one_subject_response_thresholds.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return next(csv.reader(handle))


def field(columns: list[str], suffix: str) -> str | None:
    suffix = "|" + suffix.lower()
    candidates = [column for column in columns if column.lower().endswith(suffix)]
    return candidates[0] if candidates else None


def stamp_from_name(name: str) -> str | None:
    match = re.search(r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", name)
    return match.group(1) if match else None


TABLES.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)
STAGING.mkdir(parents=True, exist_ok=True)

all_files = list(RAW_ROOT.glob("*/*.csv"))
file_headers = {path: header(path) for path in all_files}
vehicle_candidates = []
for path, columns in file_headers.items():
    steer_column = field(columns, "SteeringWheel")
    stamp = stamp_from_name(path.name)
    if steer_column is None or stamp is None:
        continue
    data = pd.read_csv(path, usecols=["StorageTime", steer_column], encoding="utf-8-sig", low_memory=False)
    count = int(pd.to_numeric(data[steer_column], errors="coerce").notna().sum())
    if count < 1000:
        continue
    exact_plain = path.name == f"Entity_Recording_{stamp}.csv"
    rank = 0 if exact_plain else (1 if "(zx)" in path.name or "TCPNetworkDataStreamzx" in path.name else 2)
    vehicle_candidates.append(
        {
            "subject": path.parent.name.lower(),
            "stamp": stamp,
            "path": path,
            "steer_column": steer_column,
            "steering_rows": count,
            "rank": rank,
        }
    )

chosen = []
for (_, _), group in pd.DataFrame(vehicle_candidates).groupby(["subject", "stamp"], sort=True):
    row = group.sort_values(["steering_rows", "rank"], ascending=[False, True]).iloc[0]
    chosen.append(row.to_dict())

physio_by_subject_stamp = {}
for path, columns in file_headers.items():
    stamp = stamp_from_name(path.name)
    if stamp and any("PhysioLAB Pro1" in column for column in columns):
        key = (path.parent.name.lower(), stamp)
        current = physio_by_subject_stamp.get(key)
        if current is None or path.stat().st_size > current.stat().st_size:
            physio_by_subject_stamp[key] = path

staged_paths = []
file_rows = []
for item in chosen:
    subject = item["subject"]
    stamp = item["stamp"]
    source = Path(item["path"])
    columns = file_headers[source]
    mapping = {
        "steer": field(columns, "SteeringWheel"),
        "accelerator": field(columns, "AcceleratorPedal"),
        "brake": field(columns, "BrakePedal"),
        "ax": field(columns, "ax"),
        "ay": field(columns, "ay"),
        "vx": field(columns, "vx"),
        "vy": field(columns, "vy"),
        "roll": field(columns, "roll"),
        "pitch": field(columns, "pitch"),
        "yaw": field(columns, "yaw"),
        "x": field(columns, "x") or field(columns, "X"),
        "y": field(columns, "y") or field(columns, "Y"),
        "curvature": field(columns, "lanecurvatureXY"),
        "lateral": field(columns, "LateralDistance") or field(columns, "lateraldistance"),
        "speed_reported": field(columns, "v_kmh") or field(columns, "v_km/h"),
    }
    usecols = ["StorageTime", *sorted({column for column in mapping.values() if column})]
    raw = pd.read_csv(source, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    steer = pd.to_numeric(raw[mapping["steer"]], errors="coerce")
    raw = raw.loc[steer.notna()].copy()
    raw["StorageTime"] = pd.to_datetime(raw["StorageTime"], errors="raise")
    raw = raw.sort_values("StorageTime").drop_duplicates("StorageTime").reset_index(drop=True)
    staged = pd.DataFrame()
    staged["StorageTime"] = raw["StorageTime"]
    staged["t_s"] = (raw["StorageTime"] - raw["StorageTime"].iloc[0]).dt.total_seconds()
    output_columns = {
        "steer": "zx|SteeringWheel",
        "accelerator": "zx|AcceleratorPedal",
        "brake": "zx|BrakePedal",
        "ax": "zx|ax",
        "ay": "zx|ay",
        "vx": "zx|vx",
        "vy": "zx|vy",
        "roll": "zx|roll",
        "pitch": "zx|pitch",
        "yaw": "zx|yaw",
        "x": "zx|x",
        "y": "zx|y",
        "curvature": "zx1|lanecurvatureXY",
        "lateral": "zx1|lateraldistance",
        "speed_reported": "zx1|v_km/h",
    }
    for key, output_column in output_columns.items():
        staged[output_column] = pd.to_numeric(raw[mapping[key]], errors="coerce") if mapping[key] else np.nan
    staged["ref_nn_ok"] = staged["zx1|lanecurvatureXY"].notna() & staged["zx1|lateraldistance"].notna()
    staged["StorageTime"] = staged["StorageTime"].dt.strftime("%Y/%m/%d %H:%M:%S.%f").str[:-3]
    subject_dir = STAGING / subject / "vehicle"
    subject_dir.mkdir(parents=True, exist_ok=True)
    staged_path = subject_dir / f"Entity_Recording_{stamp}_vehicle_aligned_cleaned.csv"
    staged.to_csv(staged_path, index=False, encoding="utf-8-sig")
    staged_paths.append(staged_path)
    duration = float(staged["t_s"].iloc[-1] - staged["t_s"].iloc[0])
    physio_path = physio_by_subject_stamp.get((subject, stamp))
    file_rows.append(
        {
            "subject": subject,
            "session_stamp": stamp,
            "source_path": str(source),
            "staged_path": str(staged_path),
            "physio_path": str(physio_path) if physio_path else "",
            "vehicle_rows": len(staged),
            "duration_s": duration,
            "vehicle_hz": len(staged) / duration,
            "physio_file_available": bool(physio_path),
            "curvature_available": bool(staged["zx1|lanecurvatureXY"].notna().mean() >= 0.9),
            "lateral_available": bool(staged["zx1|lateraldistance"].notna().mean() >= 0.9),
        }
    )

detector = load_module("run76_anchor_v3", DETECTOR_PATH)
detector_config = json.loads(DETECTOR_CONFIG_PATH.read_text(encoding="utf-8"))
base_config = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
base = detector.load_base_module(REPO, detector_config)
thresholds = pd.read_csv(THRESHOLD_PATH)
threshold_columns = [column for column in thresholds.columns if column.endswith(("_p95", "_p975", "_p99"))]
population_thresholds = {column: float(pd.to_numeric(thresholds[column], errors="raise").median()) for column in threshold_columns}
threshold_by_subject = {subject: population_thresholds for subject in sorted({item["subject"] for item in chosen})}

events, pulses, releases, responses, detector_recordings = detector.process_paths(
    staged_paths, detector_config, base_config, threshold_by_subject, base
)
events.to_csv(TABLES / "all_detected_events.csv", index=False, encoding="utf-8-sig")
pulses.to_csv(TABLES / "all_detected_pulses.csv", index=False, encoding="utf-8-sig")
detector_recordings.to_csv(TABLES / "detector_recordings.csv", index=False, encoding="utf-8-sig")

primary = pulses[["pulse_id", "pulse_direction", "direction_consistency"]].rename(
    columns={"pulse_id": "primary_pulse_id", "pulse_direction": "direction", "direction_consistency": "direction_consistency_at_anchor"}
)
screen = events.merge(primary, on="primary_pulse_id", how="left", validate="many_to_one")
files = pd.DataFrame(file_rows)
file_index = files.set_index(["subject", "session_stamp"])

candidate_rows = []
bad_physio_timestamp_rows = 0
for recording_uid, group in screen.groupby("recording_uid", sort=True):
    subject = str(group.iloc[0]["subject"])
    compact = recording_uid.split("_rec-")[-1]
    stamp = f"{compact[:4]}_{compact[4:6]}_{compact[6:8]}_{compact[9:11]}_{compact[11:13]}_{compact[13:15]}"
    info = file_index.loc[(subject, stamp)]
    vehicle = pd.read_csv(info["staged_path"], encoding="utf-8-sig", low_memory=False)
    time_s = pd.to_numeric(vehicle["t_s"], errors="raise").to_numpy(float)
    steer = pd.to_numeric(vehicle["zx|SteeringWheel"], errors="raise").to_numpy(float)
    reported = pd.to_numeric(vehicle["zx1|v_km/h"], errors="coerce").to_numpy(float)
    vx = pd.to_numeric(vehicle["zx|vx"], errors="coerce").to_numpy(float)
    speed = np.where(np.isfinite(reported), reported, np.abs(vx) * 3.6)
    ay = pd.to_numeric(vehicle["zx|ay"], errors="coerce").to_numpy(float)
    roll = pd.to_numeric(vehicle["zx|roll"], errors="coerce").to_numpy(float)
    storage_start = pd.to_datetime(vehicle["StorageTime"].iloc[0], errors="raise")

    physio_time = None
    physio_valid = None
    if info["physio_path"]:
        physio = pd.read_csv(
            info["physio_path"], usecols=lambda column: column == "StorageTime" or "PhysioLAB Pro1" in column,
            encoding="utf-8-sig", low_memory=False
        )
        physio_columns = [column for column in physio.columns if "PhysioLAB Pro1" in column]
        if len(physio_columns) == 4:
            physio_time = pd.to_datetime(physio["StorageTime"], format="mixed", errors="coerce")
            valid_time = physio_time.notna().to_numpy()
            bad_physio_timestamp_rows += int((~valid_time).sum())
            physio_valid = np.column_stack(
                [pd.to_numeric(physio[column], errors="coerce").notna().to_numpy() for column in physio_columns]
            ).all(axis=1) & valid_time

    for row in group.itertuples(index=False):
        release = float(row.primary_release_s)
        history = (time_s >= release - 2.0) & (time_s <= release)
        common_coverage = min(float(np.mean(np.isfinite(values[history]))) for values in [steer, speed, ay, roll]) if history.any() else 0.0
        target_times = release + np.arange(1, 21) * 0.05
        target_complete = bool(target_times[-1] <= time_s[-1] + 1e-9)
        direction = int(row.direction) if math.isfinite(float(row.direction)) else 0
        target = np.full(20, np.nan)
        if target_complete and direction in {-1, 1}:
            anchor_value = float(np.interp(release, time_s, steer))
            target = np.degrees((np.interp(target_times, time_s, steer) - anchor_value) * direction)
        anchor_time = storage_start + pd.to_timedelta(release, unit="s")
        physio_2s = 0.0
        physio_30s = 0.0
        if physio_time is not None and physio_valid is not None:
            short = (physio_time >= anchor_time - pd.Timedelta(seconds=2)) & (physio_time <= anchor_time)
            slow = (physio_time >= anchor_time - pd.Timedelta(seconds=30)) & (physio_time <= anchor_time)
            physio_2s = min(float(np.sum(short.to_numpy() & physio_valid) / 2000.0), 1.0)
            physio_30s = min(float(np.sum(slow.to_numpy() & physio_valid) / 30000.0), 1.0)
        eligible = bool(
            bool(row.history_complete) and target_complete and float(row.pre_speed_p10_kmh) >= 60.0
            and not bool(row.pre_reverse) and float(row.direction_consistency_at_anchor) >= 0.70
            and common_coverage >= 0.90
        )
        output = {
            "event_uid": row.event_uid,
            "subject": subject,
            "recording_uid": recording_uid,
            "session_stamp": stamp,
            "source_path": info["staged_path"],
            "prediction_anchor_s": release,
            "prediction_anchor_time": anchor_time,
            "direction": direction,
            "direction_consistency": float(row.direction_consistency_at_anchor),
            "pre_speed_p10_kmh": float(row.pre_speed_p10_kmh),
            "pre_reverse": bool(row.pre_reverse),
            "history_complete": bool(row.history_complete),
            "target_complete": target_complete,
            "common_vehicle_coverage_min": common_coverage,
            "road_available": bool(info["curvature_available"] and info["lateral_available"]),
            "physio_2s_coverage": physio_2s,
            "physio_30s_coverage": physio_30s,
            "pulse_amplitude_deg_report_only": abs(float(row.first_decisive_pulse_amplitude_deg_coordinate)),
            "screen_eligible": eligible,
        }
        for point, value in enumerate(target, start=1):
            output[f"true_t{point:02d}_deg"] = float(value)
        candidate_rows.append(output)

screened = pd.DataFrame(candidate_rows).sort_values(["subject", "session_stamp", "prediction_anchor_s"]).reset_index(drop=True)
screened.to_csv(TABLES / "screened_events.csv", index=False, encoding="utf-8-sig")
eligible = screened.loc[screened["screen_eligible"]].copy()

detected_counts = events.groupby(["subject", "recording_uid"]).size()
eligible_counts = eligible.groupby(["subject", "recording_uid"]).size()
files["recording_uid"] = files.apply(
    lambda row: f"sub-{row.subject}_rec-{row.session_stamp[:4]}{row.session_stamp[5:7]}{row.session_stamp[8:10]}T{row.session_stamp[11:13]}{row.session_stamp[14:16]}{row.session_stamp[17:19]}", axis=1
)
files["detected_event_count"] = [int(detected_counts.get((row.subject, row.recording_uid), 0)) for row in files.itertuples()]
files["eligible_event_count"] = [int(eligible_counts.get((row.subject, row.recording_uid), 0)) for row in files.itertuples()]
files.to_csv(TABLES / "file_summary.csv", index=False, encoding="utf-8-sig")

subject_summary = files.groupby("subject").agg(
    vehicle_files=("recording_uid", "size"),
    detected_events=("detected_event_count", "sum"),
    eligible_events=("eligible_event_count", "sum"),
    physio_files=("physio_file_available", "sum"),
).reset_index()
physio_events = eligible.groupby("subject").agg(
    eligible_events_with_physio_2s=("physio_2s_coverage", lambda values: int((values >= 0.90).sum())),
    eligible_events_with_physio_30s=("physio_30s_coverage", lambda values: int((values >= 0.90).sum())),
).reset_index()
subject_summary = subject_summary.merge(physio_events, on="subject", how="left").fillna(0)
subject_summary.to_csv(TABLES / "subject_summary.csv", index=False, encoding="utf-8-sig")

result = {
    "subject_directories": len(list(RAW_ROOT.glob("*"))),
    "subjects_with_vehicle_files": int((subject_summary["vehicle_files"] > 0).sum()),
    "subjects_with_eligible_events": int((subject_summary["eligible_events"] > 0).sum()),
    "selected_vehicle_files": len(files),
    "detected_events": len(events),
    "eligible_events": len(eligible),
    "eligible_events_with_physio_2s": int((eligible["physio_2s_coverage"] >= 0.90).sum()),
    "eligible_events_with_physio_30s": int((eligible["physio_30s_coverage"] >= 0.90).sum()),
    "bad_physio_timestamp_rows": bad_physio_timestamp_rows,
}
(OUTPUTS / "screening_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

lines = [
    "# 2025年8月被试批量样本筛选",
    "",
    f"- 被试目录：{result['subject_directories']}",
    f"- 有车辆文件的被试：{result['subjects_with_vehicle_files']}",
    f"- 有合格事件的被试：{result['subjects_with_eligible_events']}",
    f"- 去重后车辆文件：{result['selected_vehicle_files']}",
    f"- Anchor-v3检测事件：{result['detected_events']}",
    f"- 核心筛选合格事件：{result['eligible_events']}",
    f"- 合格事件中2秒生理完整：{result['eligible_events_with_physio_2s']}",
    f"- 合格事件中30秒生理完整：{result['eligible_events_with_physio_30s']}",
    f"- 原始生理坏时间戳行：{result['bad_physio_timestamp_rows']}",
    "",
    "## 分被试",
    "",
    "| subject | vehicle files | detected | eligible | physio 2s | physio 30s |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in subject_summary.itertuples(index=False):
    lines.append(
        f"| {row.subject} | {int(row.vehicle_files)} | {int(row.detected_events)} | {int(row.eligible_events)} | "
        f"{int(row.eligible_events_with_physio_2s)} | {int(row.eligible_events_with_physio_30s)} |"
    )
(OUTPUTS / "SCREENING_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
