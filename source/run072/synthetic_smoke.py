from __future__ import annotations

"""Run72隔离合成smoke：只验证代码路径，不代表任何真实实验结果。"""

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import experiment as run72


def make_synthetic() -> tuple[run72.DataBundle, list[run72.NestedContext], dict]:
    config = copy.deepcopy(run72.load_config())
    config["support_gates"] = {
        "eeg_global": {"minimum_subjects": 8, "minimum_events": 40},
        "eeg_each_outer_context": {"minimum_subjects": 6, "minimum_events": 30},
        "real_shift_common_minimum_rate_each_direction": 0.8,
        "trait_global": {"minimum_subjects": 8, "minimum_events": 40},
        "trait_each_outer_context": {"minimum_subjects": 6, "minimum_events": 30},
        "joint_global": {"minimum_subjects": 8, "minimum_events": 40},
        "joint_each_outer_context": {"minimum_subjects": 6, "minimum_events": 30},
    }
    config["pair_gates"]["improved_subjects_required"] = 8
    # 人工小样本的ordinary/road分配不承担科学harm含义；放宽这里只为穿透测试条件分支。
    # 正式config和候选内部harm保护仍固定为0.02°，不会被此深拷贝修改。
    config["pair_gates"]["harm_worst_context_max_deg"] = 1.0
    config["model"]["harm_tolerance_deg"] = 1.0
    for pair in config["pair_gates"]["required"]:
        pair["minimum_gain_deg"] = 0.01

    rng = np.random.default_rng(20260830)
    subjects = [f"s{i:02d}" for i in range(18)]
    rows = []
    e_values = []
    t_values = []
    state_values = []
    nuisance_values = []
    prior_counts = []
    for subject_index, subject in enumerate(subjects):
        outer_fold = subject_index % 5 + 1
        subject_state = 1.4 * np.sin((subject_index + 1) * 1.7)
        for event_index in range(8):
            # 每个state offset都配对+n/-n，使E与T的独立难度严格对称。
            latent_state = subject_state + (-1.5, -1.5, -0.5, -0.5, 0.5, 0.5, 1.5, 1.5)[event_index]
            nuisance = (-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0)[event_index]
            # 两个独立模态是同一latent state的互补带噪测量；平均可消去相反号nuisance。
            event_e = latent_state + nuisance
            event_t = latent_state - nuisance
            prior = (1, 1, 2, 2, 3, 3, 4, 4)[event_index]
            rows.append(
                {
                    "event_uid": f"{subject}::event-{event_index:02d}",
                    "subject": subject,
                    "recording_uid": f"{subject}::recording",
                    "outer_fold": outer_fold,
                    "episode_id": f"{subject}::episode-{event_index // 2}",
                    "road_reference_stratum": "road_reference_missing" if event_index % 2 == 1 else "road_reference_valid",
                }
            )
            e_values.append(event_e)
            t_values.append(event_t)
            state_values.append(latent_state)
            nuisance_values.append(nuisance)
            prior_counts.append(prior)
    metadata = pd.DataFrame(rows)
    e = np.asarray(e_values, dtype=float)
    t = np.asarray(t_values, dtype=float)
    latent_state = np.asarray(state_values, dtype=float)
    nuisance = np.asarray(nuisance_values, dtype=float)
    n = len(metadata)
    x = np.linspace(0.05, 1.0, 20)
    basis_state = 12.0 * np.sin(np.pi * x)
    basis_interaction = 2.5 * np.maximum(x - 0.55, 0.0)
    truth = latent_state[:, None] * basis_state + (e * t)[:, None] * basis_interaction
    truth += rng.normal(0.0, 0.02, size=truth.shape)
    for column, values in zip(run72.TARGET_COLUMNS, truth.T):
        metadata[column] = values

    summary = np.zeros((n, 172), dtype=float)
    main46 = np.zeros((n, 46), dtype=float)
    shift46 = np.zeros((n, 46), dtype=float)
    quality11 = np.zeros((n, 11), dtype=float)
    trait15 = np.zeros((n, 15), dtype=float)
    main46[:, 0] = e
    shift46[:, 0] = rng.permutation(e)
    trait15[:, 0] = t
    active = np.ones(n, dtype=bool)
    data = run72.DataBundle(
        metadata=metadata,
        truth=truth,
        summary172=summary,
        main46=main46,
        shift46=shift46,
        quality11=quality11,
        main_active=active.copy(),
        shifted_active=active.copy(),
        trait15=trait15,
        trait_active=active.copy(),
        prior_count=np.asarray(prior_counts, dtype=int),
        signal_feature_names=tuple(f"synthetic_signal_{i:02d}" for i in range(46)),
        quality_feature_names=tuple(f"synthetic_quality_{i:02d}" for i in range(11)),
        trait_feature_names=tuple(config["trait"]["feature_columns"]),
        provenance={
            "synthetic": True,
            "truth_is_artificial": True,
            "scientific_result": False,
            "outer_test_opened": False,
        },
    )

    contexts: list[run72.NestedContext] = []
    subject_array = metadata["subject"].astype(str).to_numpy()
    outer_array = metadata["outer_fold"].astype(int).to_numpy()
    for outer_fold in range(1, 6):
        train_subjects = np.asarray(sorted(np.unique(subject_array[outer_array != outer_fold])))
        subject_to_meta = {subject: position % 3 + 1 for position, subject in enumerate(train_subjects)}
        for meta_fold in (1, 2, 3):
            validation_subjects = {subject for subject, fold in subject_to_meta.items() if fold == meta_fold}
            validation = np.flatnonzero(
                (outer_array != outer_fold) & np.isin(subject_array, list(validation_subjects))
            )
            fit = np.flatnonzero(
                (outer_array != outer_fold) & ~np.isin(subject_array, list(validation_subjects))
            )
            fit_three = np.stack(
                [
                    np.full((len(fit), 20), -0.05),
                    np.zeros((len(fit), 20)),
                    np.full((len(fit), 20), 0.05),
                ],
                axis=1,
            )
            validation_three = np.stack(
                [
                    np.full((len(validation), 20), -0.05),
                    np.zeros((len(validation), 20)),
                    np.full((len(validation), 20), 0.05),
                ],
                axis=1,
            )
            contexts.append(
                run72.NestedContext(
                    outer_fold=outer_fold,
                    meta_fold=meta_fold,
                    fit_indices=fit,
                    validation_indices=validation,
                    fit_reference=np.mean(fit_three, axis=1),
                    validation_reference=np.mean(validation_three, axis=1),
                )
            )
    return data, contexts, config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run72 isolated synthetic smoke")
    parser.add_argument("--out-dir", type=Path, default=run72.RUN_DIR / "smoke")
    args = parser.parse_args()
    data, contexts, config = make_synthetic()
    decision = run72.run_pipeline(data, contexts, config, args.out_dir, synthetic_smoke=True)
    if decision["status"] != "synthetic_smoke_pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
