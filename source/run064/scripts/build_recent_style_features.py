from __future__ import annotations

"""构造当前事件前 `[anchor-65s, anchor-5s]` 的近期驾驶行为/风格代理。

它不同于此前的 prior-session 静态风格：第一session也可用，且严格截止预测起点前5秒。
输出保留速度、道路曲率与有效率，后续可用同容量控制判断它是否只是场景代理。
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run64_physio_style_regret_distillation_20260829"
PFULL = (
    ROOT
    / "05_rebuild_from_raw_20260511"
    / "03_baselines"
    / "run57_a_full_release_population_causal_baseline_20260827"
    / "run_1"
    / "tables"
    / "pfull_event_manifest.csv"
)
OUT = RUN_DIR / "tables" / "recent_style_features.csv"
SUMMARY = RUN_DIR / "outputs" / "recent_style_feature_summary.json"


COLS = {
    "time": "t_s",
    "steer": "zx|SteeringWheel",
    "brake": "zx|BrakePedal",
    "throttle": "zx|AcceleratorPedal",
    "speed": "zx1|v_km/h",
    "ax": "zx|ax",
    "ay": "zx|ay",
    "yaw_rate": "zx|vyaw",
    "lane_offset": "zx1|lateraldistance",
    "curvature": "zx1|lanecurvatureXY",
}


def safe_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.mean(x)), float(np.std(x)), float(np.percentile(x, 90))


def main():
    started = time.time()
    pfull = pd.read_csv(PFULL)
    rows = []
    groups = list(pfull.groupby("source_path", sort=True))
    for gi, (source_path, events) in enumerate(groups, start=1):
        path = Path(str(source_path))
        print(f"[{gi:02d}/{len(groups):02d}] {path.name} events={len(events)}")
        frame = pd.read_csv(path, usecols=list(COLS.values()), low_memory=False)
        data = {k: pd.to_numeric(frame[v], errors="coerce").to_numpy(float) for k, v in COLS.items()}
        t = data["time"]
        steer = data["steer"]
        dt = np.gradient(t)
        steer_rate = np.gradient(steer) / np.where(np.abs(dt) > 1e-9, dt, np.nan)
        for _, event in events.iterrows():
            anchor = float(event["primary_release_s"])
            use = (t >= anchor - 65.0) & (t <= anchor - 5.0)
            expected = max(1, int(round(60.0 / max(float(np.nanmedian(np.diff(t))), 1e-3))))
            valid_fraction = min(1.0, float(np.isfinite(t[use]).sum() / expected))
            out = {
                "event_uid": event["event_uid"],
                "subject": event["subject"],
                "recording_uid": event["recording_uid"],
                "outer_fold": int(event["outer_fold"]),
                "recent_style_valid_fraction": valid_fraction,
                "recent_style_window_end_minus_anchor_s": -5.0,
            }
            if valid_fraction < 0.8:
                rows.append(out)
                continue
            steer_abs = np.abs(steer[use])
            rate_abs = np.abs(steer_rate[use])
            brake = data["brake"][use]
            throttle = data["throttle"][use]
            speed = data["speed"][use]
            ax = np.abs(data["ax"][use])
            ay = np.abs(data["ay"][use])
            yaw = np.abs(data["yaw_rate"][use])
            lane = np.abs(data["lane_offset"][use])
            curvature = np.abs(data["curvature"][use])
            out.update(
                {
                    "recent_steer_abs_mean": safe_stats(steer_abs)[0],
                    "recent_steer_abs_std": safe_stats(steer_abs)[1],
                    "recent_steer_abs_p90": safe_stats(steer_abs)[2],
                    "recent_steer_rate_abs_mean": safe_stats(rate_abs)[0],
                    "recent_steer_rate_abs_p90": safe_stats(rate_abs)[2],
                    "recent_brake_usage_ratio": float(np.nanmean(brake > 1e-6)),
                    "recent_brake_mean_nonzero": float(np.nanmean(brake[brake > 1e-6])) if np.any(brake > 1e-6) else 0.0,
                    "recent_hard_brake_ratio": float(np.nanmean(brake > 0.5)),
                    "recent_throttle_usage_ratio": float(np.nanmean(throttle > 1e-6)),
                    "recent_throttle_mean_nonzero": float(np.nanmean(throttle[throttle > 1e-6])) if np.any(throttle > 1e-6) else 0.0,
                    "recent_hard_accel_ratio": float(np.nanmean(throttle > 0.5)),
                    "recent_speed_mean": safe_stats(speed)[0],
                    "recent_speed_std": safe_stats(speed)[1],
                    "recent_ax_abs_mean": safe_stats(ax)[0],
                    "recent_ay_abs_mean": safe_stats(ay)[0],
                    "recent_yaw_rate_abs_mean": safe_stats(yaw)[0],
                    "recent_lane_offset_abs_mean": safe_stats(lane)[0],
                    "recent_lane_offset_std": safe_stats(lane)[1],
                    "recent_curvature_abs_mean": safe_stats(curvature)[0],
                    "recent_curvature_abs_p90": safe_stats(curvature)[2],
                }
            )
            rows.append(out)
    result = pd.DataFrame(rows)
    if len(result) != 2323 or result["event_uid"].nunique() != 2323:
        raise ValueError("recent style coverage mismatch")
    result.to_csv(OUT, index=False, encoding="utf-8-sig")
    value_cols = [c for c in result if c.startswith("recent_") and c not in {"recent_style_valid_fraction", "recent_style_window_end_minus_anchor_s"}]
    summary = {
        "status": "ok",
        "rows": len(result),
        "valid_events": int((result["recent_style_valid_fraction"] >= 0.8).sum()),
        "feature_columns": value_cols,
        "nonmissing": {c: int(result[c].notna().sum()) for c in value_cols},
        "output": str(OUT),
        "elapsed_seconds": float(time.time() - started),
        "boundary": "All source samples end at prediction_anchor_s-5s.",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

