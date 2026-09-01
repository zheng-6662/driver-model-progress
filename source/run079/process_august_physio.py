from __future__ import annotations

"""处理2025年8月全部四通道PhysioLAB记录，并构造事件前生理特征。

本脚本只读取原始CSV。连续输出与事件特征均写入Run79，不覆盖D盘原始文件。
PhysioLAB文件可能是一行一个总线消息：总行率约1000/1600/1720 Hz，但每个
通道的真实有效采样率约1000 Hz。因此必须先按通道提取非空时间戳，再恢复各自
的规则时间轴，不能先按整表行号插值。

连续输出使用单向IIR滤波。这样保存后的任意时刻只依赖该时刻及以前的样本，
后续从连续层截取prediction anchor前缀时不会混入事件后的波形。
"""

import json
import math
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, iirnotch, sosfilt, sosfilt_zi, tf2sos


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
RAW_ROOT = Path(CONFIG["source_root"])
EVENT_PATH = Path(CONFIG["event_table"])
RUN = HERE / "run_1"
TABLES = RUN / "tables"
OUTPUTS = RUN / "outputs"
FIGURES = RUN / "figures"
QC_FIGURES = FIGURES / "qc_examples"
PROCESSED = RUN / "processed"
LOGS = RUN / "logs"

NATIVE_FS = int(CONFIG["native_grid_hz"])
WARMUP_S = float(CONFIG["filter_warmup_s"])
QUALITY = CONFIG["quality"]
WINDOWS = CONFIG["event_windows_s"]
EPS = 1e-12

FEATURE_COLUMNS = [
    "clean_phys_emg_log_rms_z_2s",
    "clean_phys_emg_envelope_slope_2s",
    "clean_phys_emg_burst_fraction_2s",
    "clean_phys_eda_tonic_z_30s",
    "clean_phys_eda_phasic_area_per_s_15s",
    "clean_phys_eda_phasic_slope_30s",
    "clean_phys_hr_z_30s",
    "clean_phys_hr_slope_15s",
    "clean_phys_log_rmssd_30s",
    "clean_phys_resp_rate_30s",
    "clean_phys_resp_amplitude_z_30s",
    "clean_phys_resp_interval_irregularity_30s",
    "clean_phys_emg_valid_fraction_2s",
    "clean_phys_eda_valid_fraction_30s",
    "clean_phys_ecg_valid_fraction_30s",
    "clean_phys_resp_valid_fraction_30s",
]


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def stamp_from_name(name: str) -> str | None:
    match = re.search(r"Entity_Recording_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", name)
    return match.group(1) if match else None


def channel_map(columns: list[str]) -> dict[str, str] | None:
    mapping: dict[str, str] = {}
    for name, spec in CONFIG["channels"].items():
        suffix = str(spec["suffix"]).lower()
        matches = [column for column in columns if column.lower().endswith(suffix)]
        if len(matches) != 1:
            return None
        mapping[name] = matches[0]
    return mapping


def build_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    """盘点所有四通道生理CSV；不以车辆事件是否合格限制处理范围。"""
    rows: list[dict[str, object]] = []
    subject_dirs = sorted(path for path in RAW_ROOT.iterdir() if path.is_dir())
    for path in sorted(RAW_ROOT.glob("*/*.csv")):
        columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
        mapping = channel_map(columns)
        stamp = stamp_from_name(path.name)
        if mapping is None or stamp is None:
            continue
        rows.append(
            {
                "subject": path.parent.name.lower(),
                "session_stamp": stamp,
                "source_path": str(path),
                "source_size_bytes": path.stat().st_size,
                "ecg_column": mapping["ECG"],
                "emg_column": mapping["EMG"],
                "eda_column": mapping["EDA"],
                "resp_column": mapping["RESP"],
            }
        )
    inventory = pd.DataFrame(rows).sort_values(["subject", "session_stamp"]).reset_index(drop=True)
    duplicate = inventory.duplicated(["subject", "session_stamp"], keep=False)
    if duplicate.any():
        raise ValueError("存在重复subject+session_stamp，必须先明确来源后再处理")
    counts = inventory.groupby("subject").size()
    subject_inventory = pd.DataFrame(
        {
            "subject": [path.name.lower() for path in subject_dirs],
            "physio_recordings": [int(counts.get(path.name.lower(), 0)) for path in subject_dirs],
        }
    )
    subject_inventory["source_type"] = np.select(
        [
            subject_inventory["physio_recordings"] > 0,
            subject_inventory["subject"].eq("rjy"),
            subject_inventory["subject"].eq("zxy"),
        ],
        ["four_channel_physio", "eye_only", "eeg_only"],
        default="other_non_physio",
    )
    return inventory, subject_inventory


def regularize_channel(
    timestamp_ns: np.ndarray,
    values: np.ndarray,
    fs: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, float]]:
    """把单个通道的有效总线消息恢复成规则1000 Hz时间轴。"""
    valid = np.isfinite(values) & (timestamp_ns > 0)
    times = timestamp_ns[valid].astype(np.int64)
    signal = values[valid].astype(float)
    if times.size < fs * 5:
        raise ValueError("通道有效数据不足5秒")
    order = np.argsort(times, kind="stable")
    times = times[order]
    signal = signal[order]
    unique = np.r_[True, np.diff(times) > 0]
    times = times[unique]
    signal = signal[unique]
    start_ns = int(times[0])
    relative_s = (times - start_ns) / 1e9
    duration_s = float(relative_s[-1])
    if duration_s <= 5.0:
        raise ValueError("通道持续时间不足5秒")
    grid = np.arange(int(math.floor(duration_s * fs)) + 1, dtype=float) / fs
    interpolated = np.interp(grid, relative_s, signal)

    # 规则网格点只有在附近存在真实采样时才标为有效；长缺口虽可插值，但不会
    # 被误当成真实观测。2.5 ms容差覆盖正常的约1000 Hz轻微抖动。
    right = np.searchsorted(relative_s, grid, side="left")
    left = np.clip(right - 1, 0, relative_s.size - 1)
    right = np.clip(right, 0, relative_s.size - 1)
    nearest = np.minimum(np.abs(grid - relative_s[left]), np.abs(relative_s[right] - grid))
    mask = nearest <= 0.0025
    gaps = np.diff(relative_s)
    meta = {
        "estimated_fs_hz": float((times.size - 1) / duration_s),
        "duration_s": duration_s,
        "coverage_fraction": float(min(1.0, (times.size - 1) / (duration_s * fs))),
        "max_gap_s": float(gaps.max()) if gaps.size else float("nan"),
        "raw_p01": float(np.percentile(signal, 1)),
        "raw_p50": float(np.percentile(signal, 50)),
        "raw_p99": float(np.percentile(signal, 99)),
        "raw_std": float(np.std(signal)),
    }
    return interpolated, mask, start_ns, meta


def causal_sos(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    initial = sosfilt_zi(sos) * float(signal[0])
    output, _ = sosfilt(sos, signal, zi=initial)
    return output


def causal_notch(signal: np.ndarray, fs: float, frequency: float) -> np.ndarray:
    b, a = iirnotch(frequency, 30.0, fs=fs)
    return causal_sos(signal, tf2sos(b, a))


def causal_bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return causal_sos(signal, sos)


def causal_lowpass(signal: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    return causal_sos(signal, sos)


def downsample(signal: np.ndarray, mask: np.ndarray, input_fs: int, output_fs: int) -> tuple[np.ndarray, np.ndarray]:
    if input_fs % output_fs != 0:
        raise ValueError(f"采样率必须整除: {input_fs} -> {output_fs}")
    factor = input_fs // output_fs
    blocks = len(signal) // factor
    if blocks < output_fs * 5:
        raise ValueError("降采样后持续时间不足5秒")
    signal_out = signal[: blocks * factor : factor].astype(np.float32)
    mask_out = mask[: blocks * factor].reshape(blocks, factor).mean(axis=1) >= 0.8
    mask_out[: int(math.ceil(WARMUP_S * output_fs))] = False
    return signal_out, mask_out.astype(np.uint8)


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan"), float("nan")
    center = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - center)))
    return center, scale


def robust_z(value: float, reference: np.ndarray) -> float:
    center, scale = robust_scale(reference)
    if not np.isfinite(value) or not np.isfinite(scale) or scale <= EPS:
        return float("nan")
    return float((value - center) / scale)


def linear_slope(values: np.ndarray, fs: float) -> float:
    finite = np.isfinite(values)
    if finite.sum() < max(5, int(0.5 * fs)):
        return float("nan")
    t = np.arange(values.size, dtype=float)[finite] / fs
    y = values[finite]
    t -= t.mean()
    denominator = float(np.dot(t, t))
    return float(np.dot(t, y - y.mean()) / denominator) if denominator > EPS else float("nan")


def ecg_beats(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    if signal.size < int(4 * fs) or np.std(signal) <= EPS:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    prominence = max(0.35 * float(np.std(signal)), EPS)
    distance = max(1, int(round(0.30 * fs)))
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for polarity in (signal, -signal):
        peaks, _ = find_peaks(polarity, distance=distance, prominence=prominence)
        peak_times = peaks.astype(float) / fs
        rr = np.diff(peak_times)
        plausible = rr[(rr >= 0.30) & (rr <= 2.00)]
        candidates.append((peak_times, plausible))
    return max(candidates, key=lambda item: item[1].size)


def resp_stats(signal: np.ndarray, fs: float) -> tuple[float, float, float, int]:
    if signal.size < int(15 * fs) or np.std(signal) <= EPS:
        return float("nan"), float("nan"), float("nan"), 0
    prominence = max(0.20 * float(np.std(signal)), EPS)
    peaks, _ = find_peaks(signal, distance=max(1, int(round(2.0 * fs))), prominence=prominence)
    intervals = np.diff(peaks.astype(float) / fs)
    intervals = intervals[(intervals >= 2.0) & (intervals <= 10.0)]
    rate = float(60.0 / np.median(intervals)) if intervals.size >= 2 else float("nan")
    irregularity = float(np.std(intervals) / np.mean(intervals)) if intervals.size >= 3 else float("nan")
    amplitude = float(np.percentile(signal, 90) - np.percentile(signal, 10))
    return rate, amplitude, irregularity, int(peaks.size)


def valid_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return values[mask.astype(bool) & np.isfinite(values)]


def block_values(values: np.ndarray, mask: np.ndarray, fs: float, block_s: float, function) -> np.ndarray:
    block = max(1, int(round(block_s * fs)))
    output: list[float] = []
    for start in range(0, values.size - block + 1, block):
        part_mask = mask[start : start + block].astype(bool)
        if part_mask.mean() < float(QUALITY["minimum_event_window_coverage"]):
            continue
        value = function(values[start : start + block][part_mask])
        if np.isfinite(value):
            output.append(float(value))
    return np.asarray(output, dtype=float)


def slice_absolute(
    values: np.ndarray,
    mask: np.ndarray,
    start_ns: int,
    fs: float,
    lo_ns: int,
    hi_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = max(0, int(math.floor((lo_ns - start_ns) / 1e9 * fs)))
    right = min(values.size, int(math.ceil((hi_ns - start_ns) / 1e9 * fs)))
    if right <= left:
        return np.asarray([], dtype=float), np.asarray([], dtype=np.uint8)
    return values[left:right].astype(float), mask[left:right].astype(np.uint8)


def window(
    channel: dict[str, object],
    value_name: str,
    anchor_ns: int,
    start_before_s: float,
    end_before_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    lo_ns = anchor_ns - int(round(start_before_s * 1e9))
    hi_ns = anchor_ns - int(round(end_before_s * 1e9))
    return slice_absolute(
        np.asarray(channel[value_name]),
        np.asarray(channel["mask"]),
        int(channel["start_ns"]),
        float(channel["fs"]),
        lo_ns,
        hi_ns,
    )


def fixed_window_coverage(mask: np.ndarray, fs: float, duration_s: float) -> float:
    expected = max(1, int(round(fs * duration_s)))
    return float(min(1.0, mask.astype(bool).sum() / expected))


def history_usable(values: np.ndarray, mask: np.ndarray, fs: float) -> bool:
    duration = values.size / fs
    return bool(
        duration >= float(WINDOWS["minimum_history_reference"])
        and mask.size > 0
        and mask.astype(bool).mean() >= float(QUALITY["minimum_event_window_coverage"])
    )


def build_event_feature(event: pd.Series, channels: dict[str, dict[str, object]]) -> dict[str, object]:
    anchor = pd.Timestamp(event["prediction_anchor_time"])
    anchor_ns = int(anchor.value)
    output: dict[str, object] = {
        "event_uid": event["event_uid"],
        "subject": event["subject"],
        "recording_uid": event["recording_uid"],
        "session_stamp": event["session_stamp"],
        "prediction_anchor_time": event["prediction_anchor_time"],
        "physio_source_available": True,
        "physio_processing_version": CONFIG["run_id"],
    }
    output.update({name: float("nan") for name in FEATURE_COLUMNS})

    emg = channels["EMG"]
    eda = channels["EDA"]
    ecg = channels["ECG"]
    resp = channels["RESP"]

    emg2, emg2_mask = window(emg, "envelope", anchor_ns, 2.0, 0.0)
    eda30, eda30_mask = window(eda, "clean", anchor_ns, 30.0, 0.0)
    eda15_phasic, eda15_mask = window(eda, "phasic", anchor_ns, 15.0, 0.0)
    eda30_phasic, _ = window(eda, "phasic", anchor_ns, 30.0, 0.0)
    eda30_tonic, _ = window(eda, "tonic", anchor_ns, 30.0, 0.0)
    ecg30, ecg30_mask = window(ecg, "clean", anchor_ns, 30.0, 0.0)
    ecg15, ecg15_mask = window(ecg, "clean", anchor_ns, 15.0, 0.0)
    resp30, resp30_mask = window(resp, "clean", anchor_ns, 30.0, 0.0)

    output["clean_phys_emg_valid_fraction_2s"] = fixed_window_coverage(emg2_mask, float(emg["fs"]), 2.0)
    output["clean_phys_eda_valid_fraction_30s"] = fixed_window_coverage(eda30_mask, float(eda["fs"]), 30.0)
    output["clean_phys_ecg_valid_fraction_30s"] = fixed_window_coverage(ecg30_mask, float(ecg["fs"]), 30.0)
    output["clean_phys_resp_valid_fraction_30s"] = fixed_window_coverage(resp30_mask, float(resp["fs"]), 30.0)

    history_start = float(WINDOWS["history_start"])
    history_end = float(WINDOWS["history_end"])
    emg_base, emg_base_mask = window(emg, "envelope", anchor_ns, history_start, history_end)
    eda_base_tonic, eda_base_mask = window(eda, "tonic", anchor_ns, history_start, history_end)
    ecg_base, ecg_base_mask = window(ecg, "clean", anchor_ns, history_start, history_end)
    resp_base, resp_base_mask = window(resp, "clean", anchor_ns, history_start, history_end)
    baseline_durations = [
        emg_base.size / float(emg["fs"]),
        eda_base_tonic.size / float(eda["fs"]),
        ecg_base.size / float(ecg["fs"]),
        resp_base.size / float(resp["fs"]),
    ]
    output["physio_history_reference_duration_s"] = float(min(baseline_durations))
    emg_history_ok = history_usable(emg_base, emg_base_mask, float(emg["fs"]))
    eda_history_ok = history_usable(eda_base_tonic, eda_base_mask, float(eda["fs"]))
    ecg_history_ok = history_usable(ecg_base, ecg_base_mask, float(ecg["fs"]))
    resp_history_ok = history_usable(resp_base, resp_base_mask, float(resp["fs"]))
    output["physio_long_baseline_available"] = all(
        [emg_history_ok, eda_history_ok, ecg_history_ok, resp_history_ok]
    )

    minimum_coverage = float(QUALITY["minimum_event_window_coverage"])
    if output["clean_phys_emg_valid_fraction_2s"] >= minimum_coverage:
        emg_current = valid_values(emg2, emg2_mask)
        current_log_rms = float(np.log1p(np.sqrt(np.mean(emg_current * emg_current))))
        base_log_rms = block_values(
            emg_base,
            emg_base_mask,
            float(emg["fs"]),
            2.0,
            lambda values: float(np.log1p(np.sqrt(np.mean(values * values)))),
        )
        if emg_history_ok:
            output["clean_phys_emg_log_rms_z_2s"] = robust_z(current_log_rms, base_log_rms)
        output["clean_phys_emg_envelope_slope_2s"] = linear_slope(emg_current, float(emg["fs"]))
        threshold_source = valid_values(emg_base, emg_base_mask)
        if threshold_source.size < int(30 * float(emg["fs"])* minimum_coverage):
            local, local_mask = window(emg, "envelope", anchor_ns, 30.0, 2.0)
            threshold_source = valid_values(local, local_mask)
        if threshold_source.size:
            threshold = float(np.percentile(threshold_source, 90))
            output["clean_phys_emg_burst_fraction_2s"] = float(np.mean(emg_current > threshold))

    if output["clean_phys_eda_valid_fraction_30s"] >= minimum_coverage:
        tonic_current = valid_values(eda30_tonic, eda30_mask)
        tonic_base = valid_values(eda_base_tonic, eda_base_mask)
        if eda_history_ok:
            output["clean_phys_eda_tonic_z_30s"] = robust_z(float(np.median(tonic_current)), tonic_base)
        phasic15 = valid_values(eda15_phasic, eda15_mask)
        output["clean_phys_eda_phasic_area_per_s_15s"] = float(
            np.sum(np.maximum(phasic15, 0.0)) / float(eda["fs"]) / 15.0
        )
        output["clean_phys_eda_phasic_slope_30s"] = linear_slope(
            valid_values(eda30_phasic, eda30_mask), float(eda["fs"])
        )

    if output["clean_phys_ecg_valid_fraction_30s"] >= minimum_coverage:
        current_ecg = valid_values(ecg30, ecg30_mask)
        peak_times, rr = ecg_beats(current_ecg, float(ecg["fs"]))
        base_ecg = valid_values(ecg_base, ecg_base_mask)
        _, base_rr = ecg_beats(base_ecg, float(ecg["fs"]))
        if rr.size >= 8:
            current_hr = float(np.median(60.0 / rr))
            base_hr = 60.0 / base_rr if base_rr.size >= 8 else np.asarray([], dtype=float)
            if ecg_history_ok:
                output["clean_phys_hr_z_30s"] = robust_z(current_hr, base_hr)
            output["clean_phys_log_rmssd_30s"] = float(
                np.log1p(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)
            )
        current15 = valid_values(ecg15, ecg15_mask)
        peak15, rr15 = ecg_beats(current15, float(ecg["fs"]))
        if rr15.size >= 4 and peak15.size >= 5:
            hr15 = 60.0 / rr15
            hr_times = (peak15[1 : 1 + hr15.size] + peak15[: hr15.size]) / 2.0
            centered = hr_times - hr_times.mean()
            denominator = float(np.dot(centered, centered))
            if denominator > EPS:
                output["clean_phys_hr_slope_15s"] = float(
                    np.dot(centered, hr15 - hr15.mean()) / denominator
                )

    if output["clean_phys_resp_valid_fraction_30s"] >= minimum_coverage:
        current_resp = valid_values(resp30, resp30_mask)
        rate, amplitude, irregularity, _ = resp_stats(current_resp, float(resp["fs"]))
        output["clean_phys_resp_rate_30s"] = rate
        output["clean_phys_resp_interval_irregularity_30s"] = irregularity
        base_amplitudes = block_values(
            resp_base,
            resp_base_mask,
            float(resp["fs"]),
            15.0,
            lambda values: float(np.percentile(values, 90) - np.percentile(values, 10)),
        )
        if resp_history_ok:
            output["clean_phys_resp_amplitude_z_30s"] = robust_z(amplitude, base_amplitudes)
    return output


def normalized_for_plot(values: np.ndarray) -> np.ndarray:
    center, scale = robust_scale(values)
    if not np.isfinite(scale) or scale <= EPS:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= EPS:
        return np.zeros_like(values)
    return np.clip((values - center) / scale, -6.0, 6.0)


def plot_example(subject: str, stamp: str, native: dict[str, np.ndarray], channels: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
    specifications = [
        ("ECG", "clean"),
        ("EMG", "filtered"),
        ("EDA", "clean"),
        ("RESP", "clean"),
    ]
    for axis, (name, processed_name) in zip(axes, specifications):
        raw = native[name]
        channel = channels[name]
        raw_lo = min(int(10 * NATIVE_FS), max(0, raw.size // 4))
        raw_hi = min(raw.size, raw_lo + int(20 * NATIVE_FS))
        raw_t = np.arange(raw_hi - raw_lo) / NATIVE_FS
        raw_y = normalized_for_plot(raw[raw_lo:raw_hi])
        step = max(1, NATIVE_FS // 200)
        axis.plot(raw_t[::step], raw_y[::step], color="#9E9E9E", linewidth=0.7, label="raw")
        processed = np.asarray(channel[processed_name])
        fs = float(channel["fs"])
        proc_lo = min(int(10 * fs), max(0, processed.size // 4))
        proc_hi = min(processed.size, proc_lo + int(20 * fs))
        proc_t = np.arange(proc_hi - proc_lo) / fs
        axis.plot(proc_t, normalized_for_plot(processed[proc_lo:proc_hi]), color="#1565C0", linewidth=0.9, label="processed")
        if name == "EMG":
            envelope = np.asarray(channel["envelope"])
            envelope_fs = float(channel["envelope_fs"])
            env_lo = min(int(10 * envelope_fs), max(0, envelope.size // 4))
            env_hi = min(envelope.size, env_lo + int(20 * envelope_fs))
            axis.plot(
                np.arange(env_hi - env_lo) / envelope_fs,
                normalized_for_plot(envelope[env_lo:env_hi]),
                color="#D95F02",
                linewidth=1.0,
                label="envelope",
            )
        if name == "EDA":
            tonic = np.asarray(channel["tonic"])
            axis.plot(proc_t, normalized_for_plot(tonic[proc_lo:proc_hi]), color="#2E7D32", linewidth=1.0, label="tonic")
        axis.set_title(name)
        axis.set_ylabel("robust z")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("seconds within displayed segment")
    fig.suptitle(f"{subject} {stamp}: raw vs processed physiology", y=0.995)
    fig.tight_layout()
    fig.savefig(QC_FIGURES / f"{subject}_{stamp}_filter_check.png", dpi=150)
    plt.close(fig)


def process_recording(row: pd.Series, make_plot: bool) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    path = Path(str(row["source_path"]))
    columns = {
        "ECG": row["ecg_column"],
        "EMG": row["emg_column"],
        "EDA": row["eda_column"],
        "RESP": row["resp_column"],
    }
    raw = pd.read_csv(
        path,
        usecols=["StorageTime", *columns.values()],
        encoding="utf-8-sig",
        low_memory=False,
    )
    timestamps = pd.to_datetime(raw["StorageTime"], format="mixed", errors="coerce")
    valid_time = timestamps.notna().to_numpy()
    timestamp_ns = timestamps.astype("int64").to_numpy()
    timestamp_ns[~valid_time] = -1

    native: dict[str, np.ndarray] = {}
    native_mask: dict[str, np.ndarray] = {}
    starts: dict[str, int] = {}
    raw_meta: dict[str, dict[str, float]] = {}
    for name, column in columns.items():
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
        native[name], native_mask[name], starts[name], raw_meta[name] = regularize_channel(
            timestamp_ns,
            values,
            NATIVE_FS,
        )

    ecg_spec = CONFIG["channels"]["ECG"]
    ecg_filtered = causal_notch(native["ECG"], NATIVE_FS, float(ecg_spec["notch_hz"]))
    ecg_filtered = causal_bandpass(ecg_filtered, NATIVE_FS, *map(float, ecg_spec["band_hz"]))
    ecg_clean, ecg_out_mask = downsample(
        ecg_filtered, native_mask["ECG"], NATIVE_FS, int(ecg_spec["output_hz"])
    )

    emg_spec = CONFIG["channels"]["EMG"]
    emg_filtered_native = causal_notch(native["EMG"], NATIVE_FS, float(emg_spec["notch_hz"]))
    emg_filtered_native = causal_bandpass(
        emg_filtered_native, NATIVE_FS, *map(float, emg_spec["band_hz"])
    )
    emg_envelope_native = causal_lowpass(
        np.abs(emg_filtered_native), NATIVE_FS, float(emg_spec["envelope_lowpass_hz"]), order=2
    )
    emg_filtered, emg_out_mask = downsample(
        emg_filtered_native, native_mask["EMG"], NATIVE_FS, int(emg_spec["output_hz"])
    )
    emg_envelope, emg_envelope_mask = downsample(
        emg_envelope_native,
        native_mask["EMG"],
        NATIVE_FS,
        int(emg_spec["envelope_output_hz"]),
    )

    eda_spec = CONFIG["channels"]["EDA"]
    eda_clean_native = causal_lowpass(native["EDA"], NATIVE_FS, float(eda_spec["lowpass_hz"]))
    eda_tonic_native = causal_lowpass(
        eda_clean_native, NATIVE_FS, float(eda_spec["tonic_lowpass_hz"]), order=2
    )
    eda_phasic_native = eda_clean_native - eda_tonic_native
    eda_clean, eda_out_mask = downsample(
        eda_clean_native, native_mask["EDA"], NATIVE_FS, int(eda_spec["output_hz"])
    )
    eda_tonic, _ = downsample(
        eda_tonic_native, native_mask["EDA"], NATIVE_FS, int(eda_spec["output_hz"])
    )
    eda_phasic, _ = downsample(
        eda_phasic_native, native_mask["EDA"], NATIVE_FS, int(eda_spec["output_hz"])
    )

    resp_spec = CONFIG["channels"]["RESP"]
    resp_filtered_native = causal_bandpass(
        native["RESP"], NATIVE_FS, *map(float, resp_spec["band_hz"])
    )
    resp_clean, resp_out_mask = downsample(
        resp_filtered_native, native_mask["RESP"], NATIVE_FS, int(resp_spec["output_hz"])
    )

    channels: dict[str, dict[str, object]] = {
        "ECG": {
            "clean": ecg_clean,
            "mask": ecg_out_mask,
            "start_ns": starts["ECG"],
            "fs": int(ecg_spec["output_hz"]),
        },
        "EMG": {
            "filtered": emg_filtered,
            "envelope": emg_envelope,
            "mask": emg_envelope_mask,
            "start_ns": starts["EMG"],
            "fs": int(emg_spec["envelope_output_hz"]),
            "filtered_fs": int(emg_spec["output_hz"]),
            "envelope_fs": int(emg_spec["envelope_output_hz"]),
            "filtered_mask": emg_out_mask,
        },
        "EDA": {
            "clean": eda_clean,
            "tonic": eda_tonic,
            "phasic": eda_phasic,
            "mask": eda_out_mask,
            "start_ns": starts["EDA"],
            "fs": int(eda_spec["output_hz"]),
        },
        "RESP": {
            "clean": resp_clean,
            "mask": resp_out_mask,
            "start_ns": starts["RESP"],
            "fs": int(resp_spec["output_hz"]),
        },
    }

    output_path = PROCESSED / str(row["subject"]) / f"{row['session_stamp']}_physio_cleaned.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        subject=np.asarray([str(row["subject"])]),
        session_stamp=np.asarray([str(row["session_stamp"])]),
        source_path=np.asarray([str(path)]),
        processing_version=np.asarray([CONFIG["run_id"]]),
        ecg_start_time_ns=np.asarray([starts["ECG"]], dtype=np.int64),
        ecg_fs_hz=np.asarray([channels["ECG"]["fs"]], dtype=float),
        ecg_clean=ecg_clean,
        ecg_valid_mask=ecg_out_mask,
        emg_start_time_ns=np.asarray([starts["EMG"]], dtype=np.int64),
        emg_filtered_fs_hz=np.asarray([channels["EMG"]["filtered_fs"]], dtype=float),
        emg_filtered=emg_filtered,
        emg_filtered_valid_mask=emg_out_mask,
        emg_envelope_fs_hz=np.asarray([channels["EMG"]["envelope_fs"]], dtype=float),
        emg_envelope=emg_envelope,
        emg_envelope_valid_mask=emg_envelope_mask,
        eda_start_time_ns=np.asarray([starts["EDA"]], dtype=np.int64),
        eda_fs_hz=np.asarray([channels["EDA"]["fs"]], dtype=float),
        eda_clean=eda_clean,
        eda_tonic=eda_tonic,
        eda_phasic=eda_phasic,
        eda_valid_mask=eda_out_mask,
        resp_start_time_ns=np.asarray([starts["RESP"]], dtype=np.int64),
        resp_fs_hz=np.asarray([channels["RESP"]["fs"]], dtype=float),
        resp_clean=resp_clean,
        resp_valid_mask=resp_out_mask,
    )

    peak_times, rr = ecg_beats(ecg_clean[ecg_out_mask.astype(bool)], float(channels["ECG"]["fs"]))
    resp_rate, _, _, resp_peaks = resp_stats(
        resp_clean[resp_out_mask.astype(bool)], float(channels["RESP"]["fs"])
    )
    quality_row: dict[str, object] = {
        "subject": row["subject"],
        "session_stamp": row["session_stamp"],
        "source_path": str(path),
        "processed_path": str(output_path),
        "source_rows": len(raw),
        "bad_timestamp_rows": int((~valid_time).sum()),
        "source_size_bytes": path.stat().st_size,
        "ecg_detected_beats": int(peak_times.size),
        "ecg_median_hr_bpm": float(np.median(60.0 / rr)) if rr.size else float("nan"),
        "resp_detected_peaks": resp_peaks,
        "resp_rate_bpm": resp_rate,
    }
    for name in ["ECG", "EMG", "EDA", "RESP"]:
        meta = raw_meta[name]
        prefix = name.lower()
        for key, value in meta.items():
            quality_row[f"{prefix}_{key}"] = value
        quality_row[f"{prefix}_usable"] = bool(
            meta["duration_s"] >= float(QUALITY["minimum_recording_duration_s"])
            and meta["coverage_fraction"] >= float(QUALITY["minimum_channel_coverage"])
            and meta["max_gap_s"] <= float(QUALITY["maximum_gap_s"])
            and meta["raw_std"] > EPS
        )
    if make_plot:
        plot_example(str(row["subject"]), str(row["session_stamp"]), native, channels)
    return quality_row, channels


def feature_coverage_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in FEATURE_COLUMNS:
        nonmissing = features[column].notna()
        if column.endswith("valid_fraction_2s") or column.endswith("valid_fraction_30s"):
            informative = features[column].fillna(0.0) >= float(QUALITY["minimum_event_window_coverage"])
        else:
            informative = nonmissing
        rows.append(
            {
                "feature": column,
                "nonmissing_events": int(nonmissing.sum()),
                "nonmissing_fraction": float(nonmissing.mean()),
                "usable_or_nonmissing_events": int(informative.sum()),
                "coverage_fraction": float(informative.mean()),
            }
        )
    return pd.DataFrame(rows)


def summary_figures(quality: pd.DataFrame, feature_coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    data = [quality[f"{name}_estimated_fs_hz"].to_numpy(float) for name in ["ecg", "emg", "eda", "resp"]]
    ax.boxplot(data, tick_labels=["ECG", "EMG", "EDA", "RESP"], showfliers=True)
    ax.axhline(1000.0, color="#D95F02", linestyle="--", linewidth=1.0)
    ax.set_ylabel("estimated channel sampling rate (Hz)")
    ax.set_title("August physiology: channel-native sampling rates")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_1_channel_sampling_rates.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    channels = ["ecg", "emg", "eda", "resp"]
    values = [int(quality[f"{name}_usable"].sum()) for name in channels]
    ax.bar([name.upper() for name in channels], values, color=["#1565C0", "#D95F02", "#2E7D32", "#6A1B9A"])
    ax.set_ylim(0, len(quality) * 1.08)
    ax.set_ylabel("usable recordings")
    ax.set_title(f"Channel quality coverage (n={len(quality)})")
    for i, value in enumerate(values):
        ax.text(i, value + len(quality) * 0.015, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_2_channel_quality_coverage.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = feature_coverage.sort_values("coverage_fraction")
    labels = ordered["feature"].str.replace("clean_phys_", "", regex=False)
    ax.barh(labels, ordered["coverage_fraction"], color="#1976D2")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("non-missing event fraction")
    ax.set_title("Processed physiology feature coverage on 275 events")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "Figure_3_event_feature_coverage.png", dpi=180)
    plt.close(fig)


def main() -> int:
    started = time.time()
    for directory in [TABLES, OUTPUTS, FIGURES, QC_FIGURES, PROCESSED, LOGS]:
        directory.mkdir(parents=True, exist_ok=True)

    inventory, subject_inventory = build_inventory()
    inventory.to_csv(TABLES / "source_inventory.csv", index=False, encoding="utf-8-sig")
    subject_inventory.to_csv(TABLES / "subject_inventory.csv", index=False, encoding="utf-8-sig")
    print(
        f"[Run79] 四通道生理recording={len(inventory)}，被试={inventory['subject'].nunique()}，"
        f"原始体积={inventory['source_size_bytes'].sum() / 1024**3:.3f} GB",
        flush=True,
    )

    events = pd.read_csv(EVENT_PATH, encoding="utf-8-sig", low_memory=False)
    events = events.loc[events["screen_eligible"].map(as_bool)].copy().reset_index(drop=True)
    if events["event_uid"].duplicated().any():
        raise ValueError("事件表event_uid重复")
    event_groups = {
        (str(subject), str(stamp)): group.copy()
        for (subject, stamp), group in events.groupby(["subject", "session_stamp"], sort=False)
    }

    quality_rows: list[dict[str, object]] = []
    event_feature_rows: list[dict[str, object]] = []
    plotted_subjects: set[str] = set()
    for index, row in inventory.iterrows():
        subject = str(row["subject"])
        stamp = str(row["session_stamp"])
        make_plot = subject not in plotted_subjects
        print(f"[{index + 1:03d}/{len(inventory):03d}] {subject} {stamp}", flush=True)
        quality_row, channels = process_recording(row, make_plot=make_plot)
        quality_rows.append(quality_row)
        if make_plot:
            plotted_subjects.add(subject)
        for _, event in event_groups.get((subject, stamp), pd.DataFrame()).iterrows():
            event_feature_rows.append(build_event_feature(event, channels))

    quality = pd.DataFrame(quality_rows).sort_values(["subject", "session_stamp"]).reset_index(drop=True)
    if len(quality) != len(inventory):
        raise ValueError("处理后的recording数量与输入不一致")
    quality.to_csv(TABLES / "recording_quality.csv", index=False, encoding="utf-8-sig")

    base = events[
        ["event_uid", "subject", "recording_uid", "session_stamp", "prediction_anchor_time"]
    ].copy()
    built = pd.DataFrame(event_feature_rows)
    if len(built):
        if built["event_uid"].duplicated().any():
            raise ValueError("构造后的事件生理特征重复")
        features = base.merge(
            built.drop(columns=["subject", "recording_uid", "session_stamp", "prediction_anchor_time"]),
            on="event_uid",
            how="left",
            validate="one_to_one",
        )
    else:
        features = base.copy()
    features["physio_source_available"] = features["physio_source_available"].map(as_bool)
    features["physio_processing_version"] = features.get("physio_processing_version", CONFIG["run_id"]).fillna(CONFIG["run_id"])
    for column in FEATURE_COLUMNS:
        if column not in features:
            features[column] = np.nan
    for column in [
        "clean_phys_emg_valid_fraction_2s",
        "clean_phys_eda_valid_fraction_30s",
        "clean_phys_ecg_valid_fraction_30s",
        "clean_phys_resp_valid_fraction_30s",
    ]:
        features.loc[~features["physio_source_available"], column] = 0.0
    features["physio_long_baseline_available"] = features["physio_long_baseline_available"].map(as_bool)
    features["physio_history_reference_duration_s"] = features.get(
        "physio_history_reference_duration_s", 0.0
    ).fillna(0.0)
    if len(features) != len(events) or features["event_uid"].nunique() != len(events):
        raise ValueError("事件特征表覆盖不完整")
    features.to_csv(TABLES / "event_physio_features.csv", index=False, encoding="utf-8-sig")
    feature_coverage = feature_coverage_table(features)
    feature_coverage.to_csv(TABLES / "event_feature_coverage.csv", index=False, encoding="utf-8-sig")

    processed_paths = sorted(PROCESSED.glob("*/*_physio_cleaned.npz"))
    if len(processed_paths) != len(inventory):
        raise ValueError("连续处理文件数量与输入不一致")
    required_arrays = {
        "ecg_clean",
        "emg_filtered",
        "emg_envelope",
        "eda_clean",
        "eda_tonic",
        "eda_phasic",
        "resp_clean",
    }
    for path in processed_paths:
        with np.load(path, allow_pickle=False) as data:
            if not required_arrays.issubset(data.files):
                raise ValueError(f"连续处理文件字段不完整: {path}")
            for name in required_arrays:
                if not np.isfinite(data[name]).all():
                    raise ValueError(f"连续处理文件含非有限值: {path} {name}")

    summary_figures(quality, feature_coverage)
    channel_usable = {
        name.upper(): int(quality[f"{name}_usable"].sum()) for name in ["ecg", "emg", "eda", "resp"]
    }
    summary = {
        "run_id": CONFIG["run_id"],
        "status": "PREPROCESSING_COMPLETE",
        "source_subject_directories": int(len(subject_inventory)),
        "physio_subjects": int(inventory["subject"].nunique()),
        "physio_recordings": int(len(inventory)),
        "source_size_gb": float(inventory["source_size_bytes"].sum() / 1024**3),
        "processed_recordings": int(len(processed_paths)),
        "channel_usable_recordings": channel_usable,
        "event_rows": int(len(features)),
        "events_with_processed_source": int(features["physio_source_available"].sum()),
        "events_with_long_baseline": int(features["physio_long_baseline_available"].sum()),
        "bad_timestamp_rows": int(quality["bad_timestamp_rows"].sum()),
        "qc_example_figures": int(len(list(QC_FIGURES.glob("*.png")))),
        "elapsed_seconds": float(time.time() - started),
        "evidence_boundary": CONFIG["evidence_boundary"],
    }
    (OUTPUTS / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    subject_counts = inventory.groupby("subject").size().sort_index()
    lines = [
        "# 2025年8月全量生理数据预处理",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 四通道生理被试：{summary['physio_subjects']}；recording：{summary['physio_recordings']}。",
        f"- 原始体积：{summary['source_size_gb']:.3f} GB；处理后连续文件：{summary['processed_recordings']}。",
        f"- 当前事件表：{summary['event_rows']}条；可连接处理后生理源：{summary['events_with_processed_source']}条。",
        f"- 具有至少30秒历史参考：{summary['events_with_long_baseline']}条。",
        f"- 原始坏时间戳：{summary['bad_timestamp_rows']}行。",
        "",
        "## 通道质量",
        "",
        "| 通道 | usable recording / 全部 |",
        "|---|---:|",
    ]
    for channel, value in channel_usable.items():
        lines.append(f"| {channel} | {value}/{len(inventory)} |")
    lines += [
        "",
        "## 不可用于当前30秒窗口的recording",
        "",
        "| subject | session_stamp | duration_s | 主要原因 |",
        "|---|---|---:|---|",
    ]
    unusable = quality.loc[
        ~quality[["ecg_usable", "emg_usable", "eda_usable", "resp_usable"]].all(axis=1)
    ]
    for row in unusable.itertuples(index=False):
        if float(row.ecg_duration_s) < float(QUALITY["minimum_recording_duration_s"]):
            reason = "持续时间不足30秒"
        else:
            reason = "时间戳稀疏，通道覆盖和最大间隔不合格"
        lines.append(
            f"| {row.subject} | {row.session_stamp} | {float(row.ecg_duration_s):.3f} | {reason} |"
        )
    coverage_index = feature_coverage.set_index("feature")["usable_or_nonmissing_events"]
    lines += [
        "",
        "## 275条事件的处理后特征覆盖",
        "",
        f"- 2秒EMG包络/爆发：{int(coverage_index['clean_phys_emg_envelope_slope_2s'])}/275。",
        f"- 15/30秒EDA、HRV与心率趋势：{int(coverage_index['clean_phys_log_rmssd_30s'])}/275。",
        f"- 至少30秒个体历史参考的EMG/EDA/HR z特征：{int(coverage_index['clean_phys_hr_z_30s'])}/275。",
        f"- 可稳定计算的呼吸率：{int(coverage_index['clean_phys_resp_rate_30s'])}/275。",
        f"- 需要多个15秒历史块的呼吸幅值z：{int(coverage_index['clean_phys_resp_amplitude_z_30s'])}/275。",
    ]
    lines += [
        "",
        "## 分被试recording",
        "",
        "| subject | recording |",
        "|---|---:|",
    ]
    for subject, count in subject_counts.items():
        lines.append(f"| {subject} | {int(count)} |")
    lines += [
        "",
        "## 证据边界",
        "",
        "本轮只完成生理信号预处理、质量统计和事件前特征构造，不训练模型，也不声称生理数据已经带来预测收益。",
        "rjy只有眼动数据，zxy是EEG而非四通道PhysioLAB，均未伪装成生理recording。",
    ]
    (OUTPUTS / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
