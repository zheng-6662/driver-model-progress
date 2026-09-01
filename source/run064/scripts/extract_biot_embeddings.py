from __future__ import annotations

"""用NeurIPS 2023 BIOT外部预训练编码器提取预测起点前生理嵌入。

只复用其频谱patch投影和Transformer；由于本项目四通道不是EEG montage，前四个
channel token被确定性替换为18个预训练token的均值，避免把ECG/EMG/EDA/RESP
错误冒充FP1-F7等特定EEG导联。编码器全部冻结。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
BIOT_ROOT = ROOT / "04_project_logs" / "research" / "github_scan_20260830" / "BIOT"
INPUT = RUN_DIR / "cache" / "physio_prefix_200hz_biot.npz"
OUTPUT = RUN_DIR / "cache" / "biot_pretrained_embeddings.npz"
SUMMARY = RUN_DIR / "outputs" / "biot_embedding_summary.json"
CHECKPOINT = BIOT_ROOT / "pretrained-models" / "EEG-SHHS+PREST-18-channels.ckpt"


def main():
    started = time.time()
    sys.path.insert(0, str(BIOT_ROOT))
    from model.biot import BIOTEncoder

    cache = np.load(INPUT, allow_pickle=True)
    event_uid = cache["event_uid"].astype(str)
    sequence = cache["sequence"].astype(np.float32)
    mask = cache["channel_mask"].astype(np.float32)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BIOTEncoder(
        emb_size=256,
        heads=8,
        depth=4,
        n_channels=18,
        n_fft=200,
        hop_length=100,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    with torch.no_grad():
        common_token = model.channel_tokens.weight.mean(dim=0, keepdim=True)
        model.channel_tokens.weight[:4].copy_(common_token.repeat(4, 1))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    embeddings = np.full((len(event_uid), 256), np.nan, dtype=np.float32)
    batch_size = 16
    print(f"[BIOT embedding] device={device} events={len(event_uid)} batch={batch_size}")
    with torch.inference_mode():
        for start in range(0, len(event_uid), batch_size):
            end = min(len(event_uid), start + batch_size)
            x = torch.from_numpy(sequence[start:end]).to(device)
            y = model(x).cpu().numpy().astype(np.float32)
            valid = np.any(mask[start:end] > 0, axis=1)
            embeddings[start:end][valid] = y[valid]
            if start % 256 == 0:
                print(f"  {start}/{len(event_uid)}")
    np.savez_compressed(
        OUTPUT,
        event_uid=event_uid,
        embedding=embeddings,
        channel_mask=mask,
        checkpoint=np.asarray([str(CHECKPOINT)]),
        embedding_dim=np.asarray([256]),
    )
    summary = {
        "status": "ok",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "shape": list(embeddings.shape),
        "valid_events": int(np.isfinite(embeddings).all(axis=1).sum()),
        "checkpoint": str(CHECKPOINT),
        "external_repo": "https://github.com/ycq091044/BIOT",
        "external_commit": "d138e32634e52ae9fa6ec98ac9c4087b14ca869a",
        "channel_token_adaptation": "first four tokens replaced by the mean of 18 pretrained EEG channel tokens; encoder frozen",
        "output": str(OUTPUT),
        "elapsed_seconds": float(time.time() - started),
        "boundary": "Fixed external transform; all input samples stop at prediction_anchor_s. EEG classification performance is not steering evidence.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

