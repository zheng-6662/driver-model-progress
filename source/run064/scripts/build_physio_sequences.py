from __future__ import annotations

"""构造预测起点前30秒、10 Hz、4通道的因果生理序列缓存。

每个0.1秒bin分别计算：ECG局部RMS、EMG局部RMS、EDA均值、RESP均值；再用
同一recording的 `[anchor-120s, anchor-30s]` 历史bin中位数/MAD做稳健标准化。
没有至少30秒历史或通道不可用时，该通道全零并由mask显式标记。
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_feature_builder", RUN_DIR / "scripts" / "build_multimodal_features.py")
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

OUT = RUN_DIR / "cache" / "physio_sequence_10hz.npz"
SUMMARY = RUN_DIR / "outputs" / "physio_sequence_build_summary.json"
CHANNELS = ("ECG", "EMG", "EDA", "RESP")
N_STEPS = 300


def bin_series(values, times, lo, hi, n_steps, mode):
    edges = np.linspace(lo, hi, n_steps + 1)
    out = np.full(n_steps, np.nan, dtype=float)
    left = np.searchsorted(times, edges[:-1], side="left")
    right = np.searchsorted(times, edges[1:], side="left")
    for i, (a, b) in enumerate(zip(left, right)):
        x = values[a:b]
        x = x[np.isfinite(x)]
        if x.size < 5:
            continue
        if mode == "rms":
            x = x - np.median(x)
            out[i] = np.sqrt(np.mean(x * x))
        else:
            out[i] = np.mean(x)
    return out


def robust_normalize(current, reference):
    ref = reference[np.isfinite(reference)]
    if ref.size < 300:
        return np.full_like(current, np.nan)
    med = np.median(ref)
    scale = 1.4826 * np.median(np.abs(ref - med))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.full_like(current, np.nan)
    return (current - med) / scale


def main():
    started = time.time()
    pfull = pd.read_csv(builder.PFULL_PATH)
    seq = np.zeros((len(pfull), len(CHANNELS), N_STEPS), dtype=np.float32)
    mask = np.zeros((len(pfull), len(CHANNELS)), dtype=np.float32)
    baseline_duration = np.zeros(len(pfull), dtype=np.float32)
    uid_to_row = {uid: i for i, uid in enumerate(pfull["event_uid"].astype(str))}
    usable_col = {
        "ECG": "physio_hr_recording_usable",
        "EMG": "physio_emg_recording_usable",
        "EDA": "physio_eda_recording_usable",
        "RESP": "physio_resp_recording_usable",
    }
    mode = {"ECG": "rms", "EMG": "rms", "EDA": "mean", "RESP": "mean"}
    groups = list(pfull.groupby(["subject", "session_stamp"], sort=True))
    for gi, ((subject, stamp), events) in enumerate(groups, start=1):
        path = builder._raw_path(str(subject), str(stamp))
        has_source = path.exists() and bool(events["physio_available"].map(builder._as_bool).any())
        print(f"[{gi:02d}/{len(groups):02d}] {subject} {stamp} events={len(events)} raw={path.exists()}")
        if not has_source:
            continue
        times, signals, _, _ = builder._load_raw(path)
        for _, event in events.iterrows():
            out_idx = uid_to_row[str(event["event_uid"])]
            if not builder._as_bool(event["physio_available"]) or not np.isfinite(event["physio_sync_offset_s"]):
                continue
            anchor = float(event["primary_release_s"]) - float(event["physio_sync_offset_s"])
            base_lo = max(0.0, anchor - 120.0)
            base_hi = anchor - 30.0
            base_duration = max(0.0, base_hi - base_lo)
            baseline_duration[out_idx] = base_duration
            if base_duration < 30.0:
                continue
            base_steps = max(300, int(round(base_duration * 10.0)))
            for ci, channel in enumerate(CHANNELS):
                if not builder._as_bool(event[usable_col[channel]]):
                    continue
                current = bin_series(
                    signals[channel], times[channel], anchor - 30.0, anchor, N_STEPS, mode[channel]
                )
                reference = bin_series(
                    signals[channel], times[channel], base_lo, base_hi, base_steps, mode[channel]
                )
                normalized = robust_normalize(current, reference)
                valid_fraction = float(np.isfinite(normalized).mean())
                if valid_fraction < 0.8:
                    continue
                normalized = np.nan_to_num(normalized, nan=0.0, posinf=8.0, neginf=-8.0)
                seq[out_idx, ci] = np.clip(normalized, -8.0, 8.0).astype(np.float32)
                mask[out_idx, ci] = 1.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        event_uid=pfull["event_uid"].astype(str).to_numpy(),
        sequence=seq,
        channel_mask=mask,
        baseline_duration_s=baseline_duration,
        channel_names=np.asarray(CHANNELS),
        hz=np.asarray([10.0]),
    )
    summary = {
        "status": "ok",
        "shape": list(seq.shape),
        "channel_valid_events": {c: int(mask[:, i].sum()) for i, c in enumerate(CHANNELS)},
        "all_four_valid_events": int(np.all(mask > 0, axis=1).sum()),
        "any_valid_events": int(np.any(mask > 0, axis=1).sum()),
        "baseline_at_least_30s": int((baseline_duration >= 30.0).sum()),
        "output": str(OUT),
        "elapsed_seconds": float(time.time() - started),
        "boundary": "All sequence samples are at or before prediction_anchor_s; history reference is not labelled calm/rest.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

