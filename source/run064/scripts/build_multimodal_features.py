from __future__ import annotations

"""为 Run64 构造预测起点前的生理“状态”和既往驾驶“风格”特征。

设计边界
--------
1. 生理特征只读取 prediction_anchor_s 及以前的原始 1000 Hz 通道。
2. 当前事件窗和历史参考窗都在同一 recording 内；不会跨 recording 借值。
3. 驾驶风格只汇总当前 recording 之前已经完成的 session；第一段 session 明确缺失。
4. 本脚本只构造输入，不读取任何模型结果，也不按未来曲线筛选特征。

这里刻意不用既有整段 ``filtfilt`` 派生列。ECG/RESP 的滤波只在已经完整获得的
预测起点前窗口内部执行，因此不会读取 prediction_anchor_s 之后的样本。
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
PFULL_PATH = (
    ROOT
    / "05_rebuild_from_raw_20260511"
    / "03_baselines"
    / "run57_a_full_release_population_causal_baseline_20260827"
    / "run_1"
    / "tables"
    / "pfull_event_manifest.csv"
)
STYLE_PATH = ROOT / "04_project_logs" / "reports" / "style_probe_artifacts" / "prior_session_style_vectors.csv"
RAW_PHYSIO_ROOT = ROOT / "01_datasets" / "数据预处理" / "原始生理数据"
OUT_PATH = RUN_DIR / "tables" / "multimodal_features.csv"
SUMMARY_PATH = RUN_DIR / "outputs" / "feature_build_summary.json"

EPS = 1e-12


PHYSIO_FEATURES = [
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
]


# 风格只用当前 session 之前的历史。为控制自由度，首轮固定 15 个生理/驾驶含义清楚
# 的统计量，再加 log1p(prior_session_count)；不做结果后特征筛选。
STYLE_SOURCE_FEATURES = [
    "steer_abs_mean__median",
    "steer_abs_p90__median",
    "steer_rate_abs_mean__median",
    "steer_rate_abs_p90__median",
    "brake_usage_ratio__median",
    "hard_brake_ratio__median",
    "throttle_usage_ratio__median",
    "hard_accel_ratio__median",
    "speed_mean__median",
    "speed_std__median",
    "ax_abs_mean__median",
    "ay_abs_mean__median",
    "yaw_rate_abs_mean__median",
    "lane_offset_abs_mean__median",
    "lane_offset_std__median",
]
STYLE_FEATURES = [f"style_{x}" for x in STYLE_SOURCE_FEATURES] + ["style_log1p_prior_session_count"]


def _as_bool(v: object) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _mad_scale(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _robust_z(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=float)
    reference = reference[np.isfinite(reference)]
    if reference.size < 3 or not np.isfinite(value):
        return float("nan")
    scale = _mad_scale(reference)
    if not np.isfinite(scale) or scale <= EPS:
        return float("nan")
    return float((value - np.median(reference)) / scale)


def _linear_slope(x: np.ndarray, fs: float) -> float:
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < max(5, int(0.5 * fs)):
        return float("nan")
    t = np.arange(x.size, dtype=float) / fs
    t = t[ok]
    y = x[ok]
    t = t - t.mean()
    denom = float(np.dot(t, t))
    if denom <= EPS:
        return float("nan")
    return float(np.dot(t, y - y.mean()) / denom)


def _fill_small_missing(x: np.ndarray) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    fraction = float(valid.mean()) if x.size else 0.0
    if valid.sum() < 3:
        return np.full_like(x, np.nan), fraction
    if not valid.all():
        idx = np.arange(x.size)
        x = np.interp(idx, idx[valid], x[valid])
    return x, fraction


def _block_values(x: np.ndarray, fs: float, block_s: float, fn) -> np.ndarray:
    block = max(1, int(round(block_s * fs)))
    values: list[float] = []
    for start in range(0, max(0, x.size - block + 1), block):
        value = fn(x[start : start + block])
        if np.isfinite(value):
            values.append(float(value))
    return np.asarray(values, dtype=float)


def _log_rms(x: np.ndarray) -> float:
    x, valid = _fill_small_missing(x)
    if valid < 0.8 or not np.isfinite(x).any():
        return float("nan")
    centered = x - np.median(x)
    return float(np.log1p(np.sqrt(np.mean(centered * centered))))


def _bandpass(x: np.ndarray, low: float, high: float, fs: float, order: int = 2) -> np.ndarray:
    x, valid = _fill_small_missing(x)
    if valid < 0.8 or x.size < int(4 * fs):
        return np.full_like(x, np.nan)
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, x)
    except ValueError:
        return np.full_like(x, np.nan)


def _ecg_beats(ecg: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, float]:
    clean, valid_fraction = _fill_small_missing(ecg)
    if valid_fraction < 0.8:
        return np.asarray([]), np.asarray([]), valid_fraction
    filtered = _bandpass(clean, 5.0, 25.0, fs, order=2)
    if not np.isfinite(filtered).any() or np.nanstd(filtered) <= EPS:
        return np.asarray([]), np.asarray([]), 0.0
    prominence = max(0.35 * float(np.nanstd(filtered)), EPS)
    distance = int(round(0.30 * fs))
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for signal in (filtered, -filtered):
        peaks, _ = find_peaks(signal, distance=distance, prominence=prominence)
        times = peaks.astype(float) / fs
        rr = np.diff(times)
        plausible = rr[(rr >= 0.30) & (rr <= 2.00)]
        candidates.append((peaks, plausible))
    peaks, rr = max(candidates, key=lambda item: item[1].size)
    return peaks.astype(float) / fs, rr, valid_fraction


def _heart_features(ecg30: np.ndarray, ecg15: np.ndarray, ecg_base: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    peak_t, rr, valid_fraction = _ecg_beats(ecg30, fs)
    _, base_rr, _ = _ecg_beats(ecg_base, fs)
    if rr.size < 8:
        return float("nan"), float("nan"), float("nan"), 0.0
    inst_hr = 60.0 / rr
    base_hr = 60.0 / base_rr if base_rr.size >= 8 else np.asarray([])
    hr = float(np.median(inst_hr))
    hr_z = _robust_z(hr, base_hr)
    rmssd_ms = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0) if rr.size >= 3 else float("nan")
    log_rmssd = float(np.log1p(rmssd_ms)) if np.isfinite(rmssd_ms) else float("nan")

    peak15, rr15, _ = _ecg_beats(ecg15, fs)
    if rr15.size >= 4 and peak15.size >= 5:
        hr15 = 60.0 / rr15
        hr_times = (peak15[1 : 1 + hr15.size] + peak15[: hr15.size]) / 2.0
        centered = hr_times - hr_times.mean()
        denom = float(np.dot(centered, centered))
        hr_slope = float(np.dot(centered, hr15 - hr15.mean()) / denom) if denom > EPS else float("nan")
    else:
        hr_slope = float("nan")
    return hr_z, hr_slope, log_rmssd, valid_fraction


def _resp_stats(resp: np.ndarray, fs: float) -> tuple[float, float, float, float]:
    clean, valid_fraction = _fill_small_missing(resp)
    if valid_fraction < 0.8 or np.nanstd(clean) <= EPS:
        return float("nan"), float("nan"), float("nan"), 0.0
    filtered = _bandpass(clean, 0.05, 0.50, fs, order=2)
    if not np.isfinite(filtered).any():
        return float("nan"), float("nan"), float("nan"), 0.0
    prominence = max(0.20 * float(np.nanstd(filtered)), EPS)
    peaks, _ = find_peaks(filtered, distance=int(round(2.0 * fs)), prominence=prominence)
    intervals = np.diff(peaks.astype(float) / fs)
    intervals = intervals[(intervals >= 2.0) & (intervals <= 10.0)]
    rate = float(60.0 / np.median(intervals)) if intervals.size >= 2 else float("nan")
    irregularity = float(np.std(intervals) / np.mean(intervals)) if intervals.size >= 3 else float("nan")
    amplitude = float(np.nanpercentile(filtered, 90) - np.nanpercentile(filtered, 10))
    return rate, amplitude, irregularity, valid_fraction


def _resp_amplitude_reference(resp_base: np.ndarray, fs: float) -> np.ndarray:
    return _block_values(resp_base, fs, 15.0, lambda x: _resp_stats(x, fs)[1])


def _segment(x: np.ndarray, t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    left = int(np.searchsorted(t, lo, side="left"))
    right = int(np.searchsorted(t, hi, side="right"))
    return np.asarray(x[left:right], dtype=float)


def _feature_row(
    row: pd.Series,
    signals: dict[str, np.ndarray],
    times: dict[str, np.ndarray],
    signal_fs: dict[str, float],
) -> dict[str, float]:
    result = {name: float("nan") for name in PHYSIO_FEATURES}
    anchor = float(row["primary_release_s"]) - float(row["physio_sync_offset_s"])
    result["physio_anchor_s"] = anchor

    baseline_lo = max(0.0, anchor - 120.0)
    baseline_hi = anchor - 30.0
    baseline_duration = max(0.0, baseline_hi - baseline_lo)
    result["physio_baseline_duration_s"] = baseline_duration

    emg2 = _segment(signals["EMG"], times["EMG"], anchor - 2.0, anchor)
    eda30 = _segment(signals["EDA"], times["EDA"], anchor - 30.0, anchor)
    eda15 = _segment(signals["EDA"], times["EDA"], anchor - 15.0, anchor)
    ecg30 = _segment(signals["ECG"], times["ECG"], anchor - 30.0, anchor)
    ecg15 = _segment(signals["ECG"], times["ECG"], anchor - 15.0, anchor)
    resp30 = _segment(signals["RESP"], times["RESP"], anchor - 30.0, anchor)

    emg_ok = _as_bool(row.get("physio_emg_recording_usable", False))
    eda_ok = _as_bool(row.get("physio_eda_recording_usable", False))
    ecg_ok = _as_bool(row.get("physio_hr_recording_usable", False))
    resp_ok = _as_bool(row.get("physio_resp_recording_usable", False))

    expected = {
        "emg": max(1, int(round(2.0 * signal_fs["EMG"]))),
        "eda": max(1, int(round(30.0 * signal_fs["EDA"]))),
        "ecg": max(1, int(round(30.0 * signal_fs["ECG"]))),
        "resp": max(1, int(round(30.0 * signal_fs["RESP"]))),
    }
    result["phys_emg_valid_fraction_2s"] = min(1.0, np.isfinite(emg2).sum() / expected["emg"]) if emg_ok else 0.0
    result["phys_eda_valid_fraction_30s"] = min(1.0, np.isfinite(eda30).sum() / expected["eda"]) if eda_ok else 0.0
    result["phys_ecg_valid_fraction_30s"] = min(1.0, np.isfinite(ecg30).sum() / expected["ecg"]) if ecg_ok else 0.0
    result["phys_resp_valid_fraction_30s"] = min(1.0, np.isfinite(resp30).sum() / expected["resp"]) if resp_ok else 0.0

    if baseline_duration < 30.0:
        return result

    emg_base = _segment(signals["EMG"], times["EMG"], baseline_lo, baseline_hi)
    eda_base = _segment(signals["EDA"], times["EDA"], baseline_lo, baseline_hi)
    ecg_base = _segment(signals["ECG"], times["ECG"], baseline_lo, baseline_hi)
    resp_base = _segment(signals["RESP"], times["RESP"], baseline_lo, baseline_hi)

    if emg_ok and result["phys_emg_valid_fraction_2s"] >= 0.8:
        current_log_rms = _log_rms(emg2)
        base_log_rms = _block_values(emg_base, signal_fs["EMG"], 2.0, _log_rms)
        result["phys_emg_log_rms_z_2s"] = _robust_z(current_log_rms, base_log_rms)
        emg_clean, _ = _fill_small_missing(emg2)
        emg_envelope = np.abs(emg_clean - np.nanmedian(emg_clean))
        result["phys_emg_envelope_slope_2s"] = _linear_slope(emg_envelope, signal_fs["EMG"])
        base_clean, base_valid = _fill_small_missing(emg_base)
        if base_valid >= 0.8 and np.nanstd(base_clean) > EPS:
            base_env = np.abs(base_clean - np.nanmedian(base_clean))
            threshold = float(np.nanpercentile(base_env, 90))
            result["phys_emg_burst_fraction_2s"] = float(np.mean(emg_envelope > threshold))

    if eda_ok and result["phys_eda_valid_fraction_30s"] >= 0.8:
        cur, cur_valid = _fill_small_missing(eda30)
        base, base_valid = _fill_small_missing(eda_base)
        if cur_valid >= 0.8 and base_valid >= 0.8 and np.nanstd(base) > EPS:
            result["phys_eda_tonic_z_30s"] = _robust_z(float(np.nanmedian(cur)), base)
            smooth30 = uniform_filter1d(cur, size=max(3, int(round(0.5 * signal_fs["EDA"]))), mode="nearest")
            result["phys_eda_phasic_slope_30s"] = _linear_slope(smooth30, signal_fs["EDA"])
        cur15, valid15 = _fill_small_missing(eda15)
        if valid15 >= 0.8 and np.nanstd(cur15) > EPS:
            smooth15 = uniform_filter1d(cur15, size=max(3, int(round(0.5 * signal_fs["EDA"]))), mode="nearest")
            positive_change = np.maximum(np.diff(smooth15), 0.0)
            result["phys_eda_phasic_area_per_s_15s"] = float(np.sum(positive_change) / 15.0)

    if ecg_ok and result["phys_ecg_valid_fraction_30s"] >= 0.8:
        hr_z, hr_slope, log_rmssd, ecg_valid = _heart_features(
            ecg30, ecg15, ecg_base, signal_fs["ECG"]
        )
        result["phys_hr_z_30s"] = hr_z
        result["phys_hr_slope_15s"] = hr_slope
        result["phys_log_rmssd_30s"] = log_rmssd
        if ecg_valid < 0.8 or not np.isfinite(log_rmssd):
            result["phys_ecg_valid_fraction_30s"] = 0.0

    if resp_ok and result["phys_resp_valid_fraction_30s"] >= 0.8:
        rate, amplitude, irregularity, resp_valid = _resp_stats(resp30, signal_fs["RESP"])
        amplitude_reference = _resp_amplitude_reference(resp_base, signal_fs["RESP"])
        result["phys_resp_rate_30s"] = rate
        result["phys_resp_amplitude_z_30s"] = _robust_z(amplitude, amplitude_reference)
        result["phys_resp_interval_irregularity_30s"] = irregularity
        if resp_valid < 0.8 or not np.isfinite(rate):
            result["phys_resp_valid_fraction_30s"] = 0.0

    return result


def _raw_path(subject: str, session_stamp: str) -> Path:
    return RAW_PHYSIO_ROOT / subject / f"Entity_Recording_{session_stamp}_physio.csv"


def _load_raw(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float], float]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    id_col = "ID" if "ID" in header else None
    channel_map: dict[str, str] = {}
    suffixes = {"ECG": "|CH1-ECG", "EMG": "|CH2-EMG", "EDA": "|CH3-EDA", "RESP": "|CH4-RESP"}
    for key, suffix in suffixes.items():
        matches = [c for c in header if suffix.lower() in c.lower()]
        if not matches:
            raise ValueError(f"missing {suffix} in {path}")
        channel_map[key] = matches[0]
    storage_col = "StorageTime" if "StorageTime" in header else None
    usecols = ([id_col] if id_col else []) + ([storage_col] if storage_col else []) + list(channel_map.values())
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    # PhysioLAB 的原始采样率不是所有记录都等于旧注释中的 1000 Hz。以首末
    # StorageTime 和行数逐 recording 估计，可正确识别本批数据常见的约1600 Hz。
    fs = float("nan")
    if storage_col and len(df) >= 2:
        first_time = pd.to_datetime(df[storage_col].iloc[0], errors="coerce")
        last_time = pd.to_datetime(df[storage_col].iloc[-1], errors="coerce")
        duration = (last_time - first_time).total_seconds() if pd.notna(first_time) and pd.notna(last_time) else float("nan")
        if np.isfinite(duration) and duration > 1.0:
            fs = float((len(df) - 1) / duration)
    if not np.isfinite(fs) or fs < 100.0 or fs > 5000.0:
        fs = 1600.0
    if id_col:
        ids = pd.to_numeric(df[id_col], errors="coerce").to_numpy(float)
        first = ids[np.isfinite(ids)][0]
        t = (ids - first) / fs
    else:
        t = np.arange(len(df), dtype=float) / fs
    # PhysioLAB 的一行代表总线消息；约1600行/s中，每个生理通道通常只有约
    # 1000个真实采样，其余行为调度空值。必须按通道压缩到各自真实时间轴，不能
    # 把调度空值误判成37.5%的生理缺失。
    signals: dict[str, np.ndarray] = {}
    times: dict[str, np.ndarray] = {}
    signal_fs: dict[str, float] = {}
    duration = float(t[-1] - t[0]) if len(t) >= 2 else float("nan")
    for key, col in channel_map.items():
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(t)
        signals[key] = values[valid]
        times[key] = t[valid]
        if np.isfinite(duration) and duration > 1.0:
            signal_fs[key] = float(max(1, valid.sum() - 1) / duration)
        else:
            signal_fs[key] = 1000.0
    return times, signals, signal_fs, fs


def _style_table(pfull: pd.DataFrame) -> pd.DataFrame:
    style = pd.read_csv(STYLE_PATH)
    style = style.copy()
    keep = ["session_id", "prior_session_count"] + STYLE_SOURCE_FEATURES
    missing = [c for c in keep if c not in style.columns]
    if missing:
        raise ValueError(f"style table missing columns: {missing}")
    style = style[keep]
    pfull_key = pfull[["event_uid", "source_path"]].copy()
    pfull_key["session_id"] = pfull_key["source_path"].map(lambda x: Path(str(x)).stem)
    joined = pfull_key.merge(style, on="session_id", how="left", validate="many_to_one")
    if joined["prior_session_count"].isna().any():
        bad = joined.loc[joined["prior_session_count"].isna(), "session_id"].drop_duplicates().tolist()
        raise ValueError(f"style sessions not found: {bad[:10]}")
    result = joined[["event_uid"]].copy()
    for source, target in zip(STYLE_SOURCE_FEATURES, STYLE_FEATURES[:-1]):
        result[target] = pd.to_numeric(joined[source], errors="coerce")
    result["style_log1p_prior_session_count"] = np.log1p(joined["prior_session_count"].astype(float))
    result["style_prior_session_count"] = joined["prior_session_count"].astype(int)
    result["style_available"] = joined["prior_session_count"].astype(int) > 0
    return result


def main() -> int:
    started = time.time()
    print("[Run64 feature build] 目标：构造预测起点前固定16维生理状态与固定16维既往驾驶风格。")
    pfull = pd.read_csv(PFULL_PATH)
    if len(pfull) != 2323 or pfull["event_uid"].nunique() != 2323:
        raise ValueError("P_full coverage mismatch")

    style = _style_table(pfull)
    rows: list[dict[str, object]] = []
    grouped = list(pfull.groupby(["subject", "session_stamp"], sort=True, dropna=False))
    for group_index, ((subject, session_stamp), events) in enumerate(grouped, start=1):
        path = _raw_path(str(subject), str(session_stamp))
        has_file = path.exists()
        has_any_physio = bool(events["physio_available"].map(_as_bool).any())
        print(f"[{group_index:02d}/{len(grouped):02d}] {subject} {session_stamp}: events={len(events)} raw={has_file}")
        if has_file and has_any_physio:
            times, signals, signal_fs, stream_fs = _load_raw(path)
            duration = max((x[-1] for x in times.values() if len(x)), default=float("nan"))
            print(
                f"           stream_fs={stream_fs:.3f} Hz channel_fs="
                + ",".join(f"{k}:{signal_fs[k]:.3f}" for k in sorted(signal_fs))
                + f" duration={duration:.3f}s"
            )
        else:
            signals = {key: np.asarray([], dtype=float) for key in ("ECG", "EMG", "EDA", "RESP")}
            times = {key: np.asarray([], dtype=float) for key in signals}
            signal_fs = {key: 1000.0 for key in signals}

        for _, event in events.iterrows():
            out: dict[str, object] = {
                "event_uid": event["event_uid"],
                "subject": event["subject"],
                "recording_uid": event["recording_uid"],
                "outer_fold": int(event["outer_fold"]),
                "session_stamp": event["session_stamp"],
                "raw_physio_path": str(path) if has_file else "",
                "physio_source_available": bool(has_file and _as_bool(event["physio_available"])),
            }
            if out["physio_source_available"] and np.isfinite(event.get("physio_sync_offset_s", np.nan)):
                out.update(_feature_row(event, signals, times, signal_fs))
            else:
                out.update({name: float("nan") for name in PHYSIO_FEATURES})
                out["physio_anchor_s"] = float("nan")
                out["physio_baseline_duration_s"] = 0.0
                for name in PHYSIO_FEATURES[-4:]:
                    out[name] = 0.0
            rows.append(out)

    phys = pd.DataFrame(rows)
    merged = phys.merge(style, on="event_uid", how="left", validate="one_to_one")
    if len(merged) != 2323 or merged["event_uid"].nunique() != 2323:
        raise ValueError("final feature coverage mismatch")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    summary = {
        "status": "ok",
        "rows": int(len(merged)),
        "subjects": int(merged["subject"].nunique()),
        "recordings": int(merged["recording_uid"].nunique()),
        "physio_source_available": int(merged["physio_source_available"].sum()),
        "physio_baseline_at_least_30s": int((merged["physio_baseline_duration_s"] >= 30.0).sum()),
        "style_available": int(merged["style_available"].sum()),
        "physio_feature_nonmissing": {name: int(merged[name].notna().sum()) for name in PHYSIO_FEATURES},
        "style_feature_nonmissing": {name: int(merged[name].notna().sum()) for name in STYLE_FEATURES},
        "output": str(OUT_PATH),
        "elapsed_seconds": float(time.time() - started),
        "evidence_boundary": (
            "Inputs stop at prediction_anchor_s. The raw source is the archived 1000 Hz physiology stream; "
            "filters are applied only inside past-only windows. Historical reference is not labelled as a calm/rest baseline."
        ),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
