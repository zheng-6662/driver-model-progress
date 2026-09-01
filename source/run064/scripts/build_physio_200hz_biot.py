from __future__ import annotations

"""构造BIOT使用的预测起点前30秒、4通道、200 Hz缓存。"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
SPEC = importlib.util.spec_from_file_location("run64_builder_biot", RUN_DIR / "scripts" / "build_multimodal_features.py")
builder = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)

OUT = RUN_DIR / "cache" / "physio_prefix_200hz_biot.npz"
SUMMARY = RUN_DIR / "outputs" / "physio_200hz_biot_summary.json"
CHANNELS = ("ECG", "EMG", "EDA", "RESP")
N = 6000


def main():
    started = time.time()
    pfull = pd.read_csv(builder.PFULL_PATH)
    seq = np.zeros((len(pfull), 4, N), dtype=np.float32)
    mask = np.zeros((len(pfull), 4), dtype=np.float32)
    uid_to_row = {uid: i for i, uid in enumerate(pfull["event_uid"].astype(str))}
    usable_col = {
        "ECG": "physio_hr_recording_usable",
        "EMG": "physio_emg_recording_usable",
        "EDA": "physio_eda_recording_usable",
        "RESP": "physio_resp_recording_usable",
    }
    groups = list(pfull.groupby(["subject", "session_stamp"], sort=True))
    for gi, ((subject, stamp), events) in enumerate(groups, start=1):
        path = builder._raw_path(str(subject), str(stamp))
        print(f"[{gi:02d}/{len(groups):02d}] {subject} {stamp} events={len(events)} raw={path.exists()}")
        if not path.exists() or not bool(events["physio_available"].map(builder._as_bool).any()):
            continue
        times, signals, _, _ = builder._load_raw(path)
        for _, event in events.iterrows():
            oi = uid_to_row[str(event["event_uid"])]
            if not builder._as_bool(event["physio_available"]) or not np.isfinite(event["physio_sync_offset_s"]):
                continue
            anchor = float(event["primary_release_s"]) - float(event["physio_sync_offset_s"])
            grid = np.linspace(anchor - 30.0, anchor, N, endpoint=False)
            for ci, channel in enumerate(CHANNELS):
                if not builder._as_bool(event[usable_col[channel]]):
                    continue
                t = times[channel]
                x = signals[channel]
                if len(t) < 10 or grid[0] < t[0] or grid[-1] > t[-1]:
                    continue
                y = np.interp(grid, t, x)
                center = float(np.median(y))
                scale = 1.4826 * float(np.median(np.abs(y - center)))
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = float(np.std(y))
                if not np.isfinite(scale) or scale <= 1e-12:
                    continue
                seq[oi, ci] = np.clip((y - center) / scale, -8.0, 8.0).astype(np.float32)
                mask[oi, ci] = 1.0
    np.savez_compressed(
        OUT,
        event_uid=pfull["event_uid"].astype(str).to_numpy(),
        sequence=seq,
        channel_mask=mask,
        channel_names=np.asarray(CHANNELS),
        hz=np.asarray([200.0]),
    )
    summary = {
        "status": "ok",
        "shape": list(seq.shape),
        "channel_valid_events": {c: int(mask[:, i].sum()) for i, c in enumerate(CHANNELS)},
        "all_four_valid_events": int(np.all(mask > 0, axis=1).sum()),
        "any_valid_events": int(np.any(mask > 0, axis=1).sum()),
        "output": str(OUT),
        "elapsed_seconds": float(time.time() - started),
        "boundary": "Every sample is at or before prediction_anchor_s; per-event robust scaling uses the same past window only.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

