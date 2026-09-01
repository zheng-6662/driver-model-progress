from __future__ import annotations

"""构造训练期特权教师使用的事件后0–5秒生理序列。

该缓存禁止进入部署学生或当前事件测试输入。所有通道仍以预测起点前
`[anchor-120, anchor-30]` 历史参考做稳健标准化；事件后数据只承担训练期LUPI教师。
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


def load_local(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_local("run64_feature_builder_post", RUN_DIR / "scripts" / "build_multimodal_features.py")
seq_builder = load_local("run64_seq_builder_post", RUN_DIR / "scripts" / "build_physio_sequences.py")

OUT = RUN_DIR / "cache" / "physio_post_sequence_10hz_teacher_only.npz"
SUMMARY = RUN_DIR / "outputs" / "post_physio_sequence_build_summary.json"
CHANNELS = ("ECG", "EMG", "EDA", "RESP")
N_STEPS = 50


def main():
    started = time.time()
    pfull = pd.read_csv(builder.PFULL_PATH)
    seq = np.zeros((len(pfull), 4, N_STEPS), dtype=np.float32)
    mask = np.zeros((len(pfull), 4), dtype=np.float32)
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
        print(f"[{gi:02d}/{len(groups):02d}] {subject} {stamp} events={len(events)} raw={path.exists()}")
        if not path.exists() or not bool(events["physio_available"].map(builder._as_bool).any()):
            continue
        times, signals, _, _ = builder._load_raw(path)
        for _, event in events.iterrows():
            oi = uid_to_row[str(event["event_uid"])]
            if not builder._as_bool(event["physio_available"]) or not np.isfinite(event["physio_sync_offset_s"]):
                continue
            anchor = float(event["primary_release_s"]) - float(event["physio_sync_offset_s"])
            base_lo = max(0.0, anchor - 120.0)
            base_hi = anchor - 30.0
            base_duration = max(0.0, base_hi - base_lo)
            if base_duration < 30.0:
                continue
            base_steps = max(300, int(round(base_duration * 10.0)))
            for ci, channel in enumerate(CHANNELS):
                if not builder._as_bool(event[usable_col[channel]]):
                    continue
                post = seq_builder.bin_series(
                    signals[channel], times[channel], anchor, anchor + 5.0, N_STEPS, mode[channel]
                )
                reference = seq_builder.bin_series(
                    signals[channel], times[channel], base_lo, base_hi, base_steps, mode[channel]
                )
                normalized = seq_builder.robust_normalize(post, reference)
                if np.isfinite(normalized).mean() < 0.8:
                    continue
                seq[oi, ci] = np.clip(np.nan_to_num(normalized, nan=0.0), -8.0, 8.0).astype(np.float32)
                mask[oi, ci] = 1.0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        event_uid=pfull["event_uid"].astype(str).to_numpy(),
        post_sequence=seq,
        channel_mask=mask,
        channel_names=np.asarray(CHANNELS),
        hz=np.asarray([10.0]),
        window_s=np.asarray([0.0, 5.0]),
    )
    summary = {
        "status": "ok",
        "shape": list(seq.shape),
        "channel_valid_events": {c: int(mask[:, i].sum()) for i, c in enumerate(CHANNELS)},
        "all_four_valid_events": int(np.all(mask > 0, axis=1).sum()),
        "any_valid_events": int(np.any(mask > 0, axis=1).sum()),
        "teacher_only": True,
        "forbidden_at_deployment": True,
        "output": str(OUT),
        "elapsed_seconds": float(time.time() - started),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

