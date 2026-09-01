from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

RUN57 = REPO / "05_rebuild_from_raw_20260511/03_baselines/run57_a_full_release_population_causal_baseline_20260827/run_1"
RUN76 = REPO / "05_rebuild_from_raw_20260511/03_baselines/run76_august_subject_augmented_training_20260831/run_1"
RUN78 = REPO / "05_rebuild_from_raw_20260511/03_baselines/run78_august_subject_rescreen_20260831/run_1"
RUN80 = REPO / "05_rebuild_from_raw_20260511/03_baselines/run80_clean_physio_ab_20260831/run_1"
CAUSAL_PATH = REPO / "05_rebuild_from_raw_20260511/03_baselines/run56_frozen848_input_fidelity_ladder_20260827/causal_preprocess.py"

OLD_MANIFEST = RUN57 / "tables/pfull_event_manifest.csv"
OLD_CACHE = RUN57 / "tables/causal_input_cache.npz"
NEW_EVENTS = RUN78 / "tables/screened_events.csv"
RUN76_FEATURES = RUN76 / "tables/new_event_features.csv"
RUN80_FEATURES = RUN80 / "tables/vehicle_features_134.csv"

COMMON_CHANNELS = (
    "steer_smooth",
    "steer_rate",
    "speed_kmh",
    "ay",
    "roll",
    "curvature",
    "lateral_distance",
)
OLD_CHANNEL_INDEX = np.asarray([0, 1, 2, 3, 6, 7, 8], dtype=int)


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    if specification.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    specification.loader.exec_module(module)
    return module


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_new_sequence(events: pd.DataFrame, causal) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    sequence = np.full((len(events), len(COMMON_CHANNELS), 101), np.nan, dtype=np.float32)
    summary = np.full((len(events), 172), np.nan, dtype=np.float64)
    lookup = {str(uid): index for index, uid in enumerate(events["event_uid"].astype(str))}
    maximum_support_minus_release = -np.inf

    for source_path, group in events.groupby("source_path", sort=True):
        raw = causal.load_raw_recording(Path(str(source_path)))
        times = raw["time"]
        smooth, rate, support, _, _ = causal.causal_endpoint_savgol(raw["steer_raw"], times, 0.1)
        arrays = {
            "steer_smooth": smooth,
            "steer_rate": rate,
            "speed_kmh": raw["speed_kmh"],
            "ay": raw["ay"],
            "yaw_rate": raw["yaw_rate"],
            "roll_rate": raw["roll_rate"],
            "roll": raw["roll"],
            "curvature": raw["curvature"],
            "lateral_distance": raw["lateral_distance"],
        }
        for event in group.itertuples(index=False):
            index = lookup[str(event.event_uid)]
            release = float(event.prediction_anchor_s)
            direction = float(event.direction)
            road_available = as_bool(event.road_available)
            sequence_grid = np.linspace(release - 2.0, release, 101)
            summary_grid = np.linspace(release - 2.0, release, 401)
            blocks: dict[str, np.ndarray] = {}
            common_index = {channel: i for i, channel in enumerate(COMMON_CHANNELS)}
            for channel in causal.CHANNELS:
                if channel in {"yaw_rate", "roll_rate"}:
                    blocks[channel] = np.zeros_like(summary_grid)
                    continue
                if channel in {"curvature", "lateral_distance"} and not road_available:
                    sequence[index, common_index[channel]] = np.nan
                    blocks[channel] = np.zeros_like(summary_grid)
                    continue
                values = arrays[channel]
                source_support = support if channel in {"steer_smooth", "steer_rate"} else None
                seq_values, seq_support = causal.causal_hold(times, values, sequence_grid, source_support)
                sum_values, sum_support = causal.causal_hold(times, values, summary_grid, source_support)
                maximum_support_minus_release = max(
                    maximum_support_minus_release,
                    float(np.max(seq_support) - release),
                    float(np.max(sum_support) - release),
                )
                if channel in COMMON_CHANNELS:
                    if channel in causal.DIRECTIONAL_CHANNELS:
                        seq_values = seq_values * direction
                    sequence[index, common_index[channel]] = seq_values.astype(np.float32)
                blocks[channel] = sum_values
            summary[index, :171] = causal.summary_features_from_200hz_blocks(
                summary_grid, blocks, release, direction, road_available
            )
            summary[index, 171] = 0.0 if road_available else 1.0
    audit = {"maximum_raw_support_minus_release_s": float(maximum_support_minus_release)}
    return sequence, summary, audit


def assign_new_subject_folds(metadata: pd.DataFrame, old_subject_fold: dict[str, int]) -> dict[str, int]:
    fold_counts = {
        fold: int(metadata.loc[metadata["outer_fold"].eq(fold), "event_uid"].size)
        for fold in range(1, 6)
    }
    assignment = dict(old_subject_fold)
    new_only = metadata.loc[metadata["outer_fold"].isna()].groupby("subject").size().sort_values(ascending=False)
    if len(new_only) != 20:
        raise ValueError(f"真正新增被试应为20，实际为{len(new_only)}")
    new_subject_slots = {fold: 0 for fold in range(1, 6)}
    for subject, count in new_only.items():
        candidates = [fold for fold in range(1, 6) if new_subject_slots[fold] < 4]
        fold = min(
            candidates,
            key=lambda candidate: (fold_counts[candidate], new_subject_slots[candidate], candidate),
        )
        assignment[str(subject)] = int(fold)
        fold_counts[fold] += int(count)
        new_subject_slots[fold] += 1
    if any(count != 4 for count in new_subject_slots.values()):
        raise ValueError(f"新增被试未按每折4人分配: {new_subject_slots}")
    return assignment


def build_combined_cache(output_path: Path, anchor_path: Path) -> dict[str, object]:
    causal = load_module("run82_causal", CAUSAL_PATH)
    old = pd.read_csv(OLD_MANIFEST, encoding="utf-8-sig", low_memory=False)
    new = pd.read_csv(NEW_EVENTS, encoding="utf-8-sig", low_memory=False)
    new = new.loc[new["screen_eligible"].map(as_bool)].copy().reset_index(drop=True)
    if len(old) != 2323 or len(new) != 275:
        raise ValueError("事件数量必须为2323+275")

    with np.load(OLD_CACHE, allow_pickle=False) as archive:
        old_sequence_9 = np.asarray(archive["sequence"], dtype=np.float32)
        old_summary_172 = np.asarray(archive["summary"], dtype=float)
        feature_names = archive["feature_names"].astype(str)
    old_sequence = old_sequence_9[:, OLD_CHANNEL_INDEX]
    remove_summary = np.asarray(
        ["__yaw_rate__" in name or "__roll_rate__" in name for name in feature_names], dtype=bool
    )
    old_summary = old_summary_172[:, ~remove_summary]

    new_sequence, new_summary_172, support_audit = build_new_sequence(new, causal)
    new_summary = new_summary_172[:, ~remove_summary]
    if old_sequence.shape != (2323, 7, 101) or new_sequence.shape != (275, 7, 101):
        raise ValueError("共同序列维度错误")
    if old_summary.shape != (2323, 134) or new_summary.shape != (275, 134):
        raise ValueError("共同摘要维度错误")
    if support_audit["maximum_raw_support_minus_release_s"] > 1e-12:
        raise ValueError("八月序列使用了release后原始支持")

    reference_all = pd.read_csv(RUN80_FEATURES, encoding="utf-8-sig", low_memory=False)
    reference_148 = pd.read_csv(RUN76_FEATURES, encoding="utf-8-sig", low_memory=False)
    kept_names = feature_names[~remove_summary].tolist()
    current = pd.DataFrame(new_summary, columns=kept_names)
    current.insert(0, "event_uid", new["event_uid"].astype(str).to_numpy())

    joined_all = reference_all[["event_uid", *kept_names]].merge(
        current, on="event_uid", suffixes=("_reference", "_current"), validate="one_to_one"
    )
    joined_148 = reference_148[["event_uid", *kept_names]].merge(
        current, on="event_uid", suffixes=("_reference", "_current"), validate="one_to_one"
    )

    def maximum_difference(table: pd.DataFrame) -> float:
        differences = []
        for name in kept_names:
            left = table[f"{name}_reference"].to_numpy(float)
            right = table[f"{name}_current"].to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)):
                return float("inf")
            finite = np.isfinite(left) & np.isfinite(right)
            if finite.any():
                differences.append(float(np.max(np.abs(left[finite] - right[finite]))))
        return max(differences, default=0.0)

    diff_all = maximum_difference(joined_all)
    diff_148 = maximum_difference(joined_148)
    anchor = {
        "status": "PASS" if diff_all <= 1e-12 and diff_148 <= 1e-12 else "FAIL",
        "new_events": len(new),
        "public_148_events": len(joined_148),
        "summary_feature_count": len(kept_names),
        "max_abs_diff_all_275": diff_all,
        "max_abs_diff_public_148": diff_148,
        **support_audit,
        "sequence_anchor_boundary": (
            "Run76 did not persist 101-point August sequences. Run82 therefore uses the identical Run57 causal_hold "
            "path and requires exact reproduction of the persisted 134D summaries for public 148 and all 275 events."
        ),
    }
    anchor_path.write_text(json.dumps(anchor, ensure_ascii=False, indent=2), encoding="utf-8")
    if anchor["status"] != "PASS":
        raise ValueError(f"输入锚点失败: {anchor}")

    old_meta = pd.DataFrame(
        {
            "event_uid": old["event_uid"].astype(str),
            "subject": old["subject"].astype(str),
            "recording_uid": old["recording_uid"].astype(str),
            "domain": "original",
            "outer_fold": old["outer_fold"].astype(int),
            "amplitude_deg": pd.to_numeric(old["causal_pulse_amplitude_deg_at_release"], errors="raise"),
        }
    )
    new_meta = pd.DataFrame(
        {
            "event_uid": new["event_uid"].astype(str),
            "subject": new["subject"].astype(str),
            "recording_uid": new["recording_uid"].astype(str),
            "domain": "august",
            "outer_fold": np.nan,
            "amplitude_deg": pd.to_numeric(new["pulse_amplitude_deg_report_only"], errors="raise"),
        }
    )
    old_subject_fold = old_meta.drop_duplicates("subject").set_index("subject")["outer_fold"].astype(int).to_dict()
    for subject, fold in old_subject_fold.items():
        new_meta.loc[new_meta["subject"].eq(subject), "outer_fold"] = int(fold)
    combined_meta = pd.concat([old_meta, new_meta], ignore_index=True)
    assignment = assign_new_subject_folds(combined_meta, old_subject_fold)
    combined_meta["outer_fold"] = combined_meta["subject"].map(assignment).astype(int)
    if combined_meta.groupby("subject")["outer_fold"].nunique().max() != 1:
        raise ValueError("同一被试跨外折")

    old_truth = old[[f"target_t{point:02d}_deg" for point in range(1, 21)]].to_numpy(float)
    new_truth = new[[f"true_t{point:02d}_deg" for point in range(1, 21)]].to_numpy(float)
    truth = np.vstack([old_truth, new_truth]).astype(np.float32)
    sequence_7 = np.concatenate([old_sequence, new_sequence], axis=0)
    summary_134 = np.vstack([old_summary, new_summary]).astype(np.float32)
    road_mask_old = old_summary[:, -1].astype(np.float32)
    road_mask_new = new_summary[:, -1].astype(np.float32)
    road_mask = np.concatenate([road_mask_old, road_mask_new])[:, None, None]
    road_mask_sequence = np.broadcast_to(road_mask, (len(combined_meta), 1, 101)).copy()
    sequence_8 = np.concatenate([sequence_7, road_mask_sequence], axis=1).astype(np.float32)
    if len(combined_meta) != 2598 or combined_meta["subject"].nunique() != 38:
        raise ValueError((len(combined_meta), combined_meta["subject"].nunique()))
    if combined_meta["event_uid"].nunique() != 2598:
        raise ValueError("合并event_uid不唯一")
    if not np.isfinite(truth).all():
        raise ValueError("目标不完整")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        event_uid=np.asarray(combined_meta["event_uid"].astype(str).tolist(), dtype=str),
        subject=np.asarray(combined_meta["subject"].astype(str).tolist(), dtype=str),
        recording_uid=np.asarray(combined_meta["recording_uid"].astype(str).tolist(), dtype=str),
        domain=np.asarray(combined_meta["domain"].astype(str).tolist(), dtype=str),
        outer_fold=combined_meta["outer_fold"].to_numpy(np.int16),
        amplitude_deg=combined_meta["amplitude_deg"].to_numpy(np.float32),
        sequence=sequence_8,
        summary=summary_134,
        truth=truth,
        channel_names=np.asarray([*COMMON_CHANNELS, "road_missing_mask"]),
        summary_feature_names=np.asarray(kept_names),
    )
    return {
        "events": len(combined_meta),
        "subjects": int(combined_meta["subject"].nunique()),
        "recordings": int(combined_meta["recording_uid"].nunique()),
        "fold_event_counts": {
            str(fold): int(combined_meta["outer_fold"].eq(fold).sum()) for fold in range(1, 6)
        },
        "fold_subject_counts": {
            str(fold): int(combined_meta.loc[combined_meta["outer_fold"].eq(fold), "subject"].nunique())
            for fold in range(1, 6)
        },
        "anchor": anchor,
        "cache_path": str(output_path),
    }


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}
