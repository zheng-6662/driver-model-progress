"""Run37 支持门通过后的未来分叉、负对照和统计推断。"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from audit_core import (
    CHANNELS,
    channel_balanced_distances,
    fit_robust_transform,
    matching_candidates,
    stable_unique_order,
)


METRICS = (
    "curve_rmse_deg",
    "peak_amplitude_abs_diff_deg",
    "tail5_rmse_deg",
    "tail_level_abs_diff_deg",
    "peak_time_abs_diff_s",
    "post_peak_drop_disagreement",
    "head5_rmse_deg",
    "endpoint_abs_diff_deg",
)


def post_peak_drop_flag(curve_aligned_deg: np.ndarray) -> bool:
    curve = np.concatenate(([0.0], np.asarray(curve_aligned_deg, dtype=float)))
    peak = int(np.argmax(curve))
    first = peak + 2
    return bool(first < len(curve) and curve[peak] - np.min(curve[first:]) >= 3.0)


def outcome_arrays(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    future = np.rad2deg(
        (
            np.asarray(bundle["target_absolute"], dtype=float)
            - np.asarray(bundle["release_value"], dtype=float)[:, None]
        )
        * np.asarray(bundle["direction"], dtype=float)[:, None]
    )
    peak_index = np.argmax(np.abs(future), axis=1)
    peak_amplitude = np.take_along_axis(
        np.abs(future), peak_index[:, None], axis=1
    )[:, 0]
    return {
        "future_aligned_deg": future,
        "peak_index": peak_index,
        "peak_amplitude_deg": peak_amplitude,
        "tail_level_deg": np.mean(future[:, -5:], axis=1),
        "drop_flag": np.asarray([post_peak_drop_flag(row) for row in future], dtype=bool),
    }


def pairwise_metric_matrices(outcomes: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    future = outcomes["future_aligned_deg"]
    difference = future[:, None, :] - future[None, :, :]
    peak = outcomes["peak_amplitude_deg"]
    peak_index = outcomes["peak_index"]
    tail_level = outcomes["tail_level_deg"]
    drop = outcomes["drop_flag"]
    return {
        "curve_rmse_deg": np.sqrt(np.mean(difference**2, axis=2)),
        "peak_amplitude_abs_diff_deg": np.abs(peak[:, None] - peak[None, :]),
        "tail5_rmse_deg": np.sqrt(np.mean(difference[:, :, -5:] ** 2, axis=2)),
        "tail_level_abs_diff_deg": np.abs(tail_level[:, None] - tail_level[None, :]),
        "peak_time_abs_diff_s": np.abs(peak_index[:, None] - peak_index[None, :]) * 0.05,
        "post_peak_drop_disagreement": (drop[:, None] != drop[None, :]).astype(float),
        "head5_rmse_deg": np.sqrt(np.mean(difference[:, :, :5] ** 2, axis=2)),
        "endpoint_abs_diff_deg": np.abs(future[:, None, -1] - future[None, :, -1]),
    }


def pairwise_mean(indices: list[int], matrix: np.ndarray) -> float:
    if len(indices) < 2:
        return float("nan")
    pairs = list(combinations(indices, 2))
    return float(np.mean([matrix[left, right] for left, right in pairs]))


def query_neighbor_mean(query_index: int, indices: list[int], matrix: np.ndarray) -> float:
    if not indices:
        return float("nan")
    return float(np.mean([matrix[query_index, neighbor] for neighbor in indices]))


def _matched_random_set(
    pool: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    rng: np.random.Generator,
    k: int = 5,
) -> list[int]:
    by_subject: dict[str, list[int]] = {}
    for index in pool:
        by_subject.setdefault(str(subjects[index]), []).append(int(index))
    available_subjects = np.asarray(sorted(by_subject), dtype=object)
    if len(available_subjects) < k:
        return []
    chosen_subjects = rng.choice(available_subjects, size=k, replace=False)
    selected: list[int] = []
    used_recordings: set[str] = set()
    for subject in chosen_subjects:
        choices = list(by_subject[str(subject)])
        rng.shuffle(choices)
        selected_index = next(
            (index for index in choices if str(recordings[index]) not in used_recordings),
            None,
        )
        if selected_index is None:
            return []
        selected.append(int(selected_index))
        used_recordings.add(str(recordings[selected_index]))
    return selected


def _permuted_ranking_set(
    pool: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    rng: np.random.Generator,
    k: int = 5,
) -> list[int]:
    selected: list[int] = []
    used_subjects: set[str] = set()
    used_recordings: set[str] = set()
    for index in rng.permutation(pool):
        subject = str(subjects[index])
        recording = str(recordings[index])
        if subject in used_subjects or recording in used_recordings:
            continue
        selected.append(int(index))
        used_subjects.add(subject)
        used_recordings.add(recording)
        if len(selected) == k:
            break
    return selected


def _batch_matched_random_sets(
    pool: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
    k: int = 5,
) -> np.ndarray:
    """按被试均匀抽样，再在被试内均匀抽一个事件；批量保持固定次数。"""

    by_subject = {
        subject: np.asarray(pool[subjects[pool] == subject], dtype=int)
        for subject in sorted(np.unique(subjects[pool]))
    }
    subject_names = list(by_subject)
    if len(subject_names) < k:
        raise RuntimeError("匹配随机池少于5个被试")
    # recording_uid在当前合同中从属于唯一subject；因此不同subject自动满足不同recording。
    recording_subject_count = (
        pd.DataFrame({"recording": recordings[pool], "subject": subjects[pool]})
        .groupby("recording")["subject"]
        .nunique()
    )
    if not recording_subject_count.eq(1).all():
        raise RuntimeError("同一recording跨subject，批量随机约束不成立")
    random_keys = rng.random((repetitions, len(subject_names)))
    chosen_subject_positions = np.argpartition(random_keys, kth=k - 1, axis=1)[:, :k]
    event_choice = np.empty((repetitions, len(subject_names)), dtype=int)
    for position, name in enumerate(subject_names):
        events = by_subject[name]
        event_choice[:, position] = events[
            rng.integers(0, len(events), size=repetitions)
        ]
    return np.take_along_axis(event_choice, chosen_subject_positions, axis=1)


def _batch_permuted_ranking_sets(
    pool: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    rng: np.random.Generator,
    repetitions: int,
    k: int = 5,
) -> np.ndarray:
    result = np.empty((repetitions, k), dtype=int)
    for repetition in range(repetitions):
        selected = _permuted_ranking_set(pool, subjects, recordings, rng, k=k)
        if len(selected) != k:
            raise RuntimeError("距离排序置换无法形成5个合法邻居")
        result[repetition] = selected
    return result


def _batch_pairwise_means(sets: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    left = np.asarray([pair[0] for pair in combinations(range(sets.shape[1]), 2)], dtype=int)
    right = np.asarray([pair[1] for pair in combinations(range(sets.shape[1]), 2)], dtype=int)
    return matrix[sets[:, left], sets[:, right]].mean(axis=1)


def _unrestricted_neighbors(
    query_index: int,
    z: np.ndarray,
    active: np.ndarray,
    bundle: dict[str, np.ndarray],
) -> list[int]:
    all_indices = np.arange(len(z), dtype=int)
    base = all_indices[all_indices != query_index]
    pool, _ = matching_candidates(query_index, base, bundle, minimum_unique=5)
    distance, _ = channel_balanced_distances(z[query_index], z[pool], active)
    uid = bundle["event_uid"].astype(str)
    order = sorted(
        range(len(pool)), key=lambda position: (float(distance[position]), uid[pool[position]])
    )
    return [int(pool[position]) for position in order[:5]]


def run_outcome_audit(
    bundle: dict[str, np.ndarray],
    support: pd.DataFrame,
    main_edges: pd.DataFrame,
    baseline: pd.DataFrame,
    random_repetitions: int = 2000,
    seed: int = 20260812,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    outcomes = outcome_arrays(bundle)
    matrices = pairwise_metric_matrices(outcomes)
    x = np.asarray(bundle["summary171"], dtype=float)
    folds = bundle["outer_fold"].astype(int)
    subjects = bundle["subject"].astype(str)
    recordings = bundle["recording_uid"].astype(str)
    uid = bundle["event_uid"].astype(str)
    main_edges = main_edges[~main_edges["is_pseudo_reference"]].copy()
    neighbor_map = {
        int(query): group.sort_values("neighbor_rank")["neighbor_index"].astype(int).tolist()
        for query, group in main_edges.groupby("query_index", sort=False)
    }
    baseline_error = baseline.set_index("event_uid")["event_curve_mae_deg"].to_dict()
    support_by_index = support.set_index("query_index")
    rng = np.random.default_rng(seed)
    query_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    drop_rows: list[dict[str, Any]] = []

    for fold in range(1, 6):
        train = np.where(folds != fold)[0]
        test = np.where(folds == fold)[0]
        transform = fit_robust_transform(x[train])
        z = transform.apply(x)
        for number, query_index in enumerate(test, start=1):
            if number == 1 or number % 25 == 0 or number == len(test):
                print(f"[outcome fold {fold}] {number}/{len(test)}", flush=True)
            main = neighbor_map[int(query_index)]
            support_row = support_by_index.loc[int(query_index)]
            base_candidates = train
            pool, match_level = matching_candidates(
                int(query_index), base_candidates, bundle, minimum_unique=5
            )
            distance, _ = channel_balanced_distances(
                z[query_index], z[pool], transform.active
            )
            unique_order = stable_unique_order(
                pool, distance, subjects, recordings, uid, limit=None
            )
            if main != unique_order[:5]:
                raise RuntimeError(f"阶段C邻居与锁定阶段A不一致：{uid[query_index]}")

            observed = {metric: pairwise_mean(main, matrix) for metric, matrix in matrices.items()}
            random_sets = _batch_matched_random_sets(
                pool,
                subjects,
                recordings,
                rng,
                repetitions=random_repetitions,
                k=5,
            )
            permuted_sets = _batch_permuted_ranking_sets(
                pool,
                subjects,
                recordings,
                rng,
                repetitions=random_repetitions,
                k=5,
            )
            random_values = {
                metric: _batch_pairwise_means(random_sets, matrix)
                for metric, matrix in matrices.items()
            }
            rank_permutation_values = {
                metric: _batch_pairwise_means(permuted_sets, matrix)
                for metric, matrix in matrices.items()
            }

            unrestricted = _unrestricted_neighbors(
                int(query_index), z, transform.active, bundle
            )
            row: dict[str, Any] = {
                "event_uid": uid[query_index],
                "query_index": int(query_index),
                "subject": subjects[query_index],
                "recording_uid": recordings[query_index],
                "outer_fold": int(fold),
                "tier": str(bundle["tier"][query_index]),
                "road_stratum": str(bundle["road_stratum"][query_index]),
                "release_phase": str(bundle["release_phase"][query_index]),
                "response_stratum": str(bundle["response_stratum"][query_index]),
                "support_class": str(support_row["support_class"]),
                "match_level": match_level,
                "nested_blend_oof_mae_deg": float(baseline_error[uid[query_index]]),
                "true_peak_amplitude_deg": float(outcomes["peak_amplitude_deg"][query_index]),
                "true_peak_time_s": float((outcomes["peak_index"][query_index] + 1) * 0.05),
                "true_post_peak_drop": bool(outcomes["drop_flag"][query_index]),
            }
            for metric in METRICS:
                random_mean = float(np.mean(random_values[metric]))
                rank_mean = float(np.mean(rank_permutation_values[metric]))
                unrestricted_value = pairwise_mean(unrestricted, matrices[metric])
                row[f"top5_{metric}"] = observed[metric]
                row[f"random_mean_{metric}"] = random_mean
                row[f"ratio_{metric}"] = observed[metric] / max(random_mean, 1e-12)
                row[f"unrestricted_top5_{metric}"] = unrestricted_value
                control_rows.extend(
                    [
                        {
                            "event_uid": uid[query_index],
                            "query_index": int(query_index),
                            "control": "matched_random_subject_recording_unique",
                            "metric": metric,
                            "observed_value": observed[metric],
                            "control_mean": random_mean,
                            "control_p025": float(np.quantile(random_values[metric], 0.025)),
                            "control_p975": float(np.quantile(random_values[metric], 0.975)),
                            "ratio_observed_over_control": observed[metric]
                            / max(random_mean, 1e-12),
                        },
                        {
                            "event_uid": uid[query_index],
                            "query_index": int(query_index),
                            "control": "distance_ranking_permutation",
                            "metric": metric,
                            "observed_value": observed[metric],
                            "control_mean": rank_mean,
                            "control_p025": float(
                                np.quantile(rank_permutation_values[metric], 0.025)
                            ),
                            "control_p975": float(
                                np.quantile(rank_permutation_values[metric], 0.975)
                            ),
                            "ratio_observed_over_control": observed[metric]
                            / max(rank_mean, 1e-12),
                        },
                        {
                            "event_uid": uid[query_index],
                            "query_index": int(query_index),
                            "control": "pseudo_repeat_unrestricted",
                            "metric": metric,
                            "observed_value": observed[metric],
                            "control_mean": unrestricted_value,
                            "control_p025": np.nan,
                            "control_p975": np.nan,
                            "ratio_observed_over_control": observed[metric]
                            / max(unrestricted_value, 1e-12),
                        },
                    ]
                )
            for k in (1, 3, 5):
                chosen = unique_order[:k]
                gradient_rows.append(
                    {
                        "event_uid": uid[query_index],
                        "query_index": int(query_index),
                        "comparison": f"query_to_top{k}",
                        "neighbor_count": k,
                        "curve_rmse_deg": query_neighbor_mean(
                            int(query_index), chosen, matrices["curve_rmse_deg"]
                        ),
                    }
                )
            group_candidates = {
                "rank_1_5": unique_order[:5],
                "rank_6_10": unique_order[5:10],
                "middle_5": unique_order[
                    max(0, len(unique_order) // 2 - 2) : max(0, len(unique_order) // 2 - 2)
                    + 5
                ],
                "farthest_5": unique_order[-5:],
            }
            for group_name, selected in group_candidates.items():
                gradient_rows.append(
                    {
                        "event_uid": uid[query_index],
                        "query_index": int(query_index),
                        "comparison": group_name,
                        "neighbor_count": len(selected),
                        **{
                            metric: pairwise_mean(selected, matrix)
                            for metric, matrix in matrices.items()
                        },
                    }
                )

            for left_rank, right_rank in combinations(range(5), 2):
                left, right = main[left_rank], main[right_rank]
                edge_rows.append(
                    {
                        "query_event_uid": uid[query_index],
                        "query_index": int(query_index),
                        "query_subject": subjects[query_index],
                        "left_neighbor_event_uid": uid[left],
                        "left_neighbor_subject": subjects[left],
                        "right_neighbor_event_uid": uid[right],
                        "right_neighbor_subject": subjects[right],
                        **{
                            metric: float(matrix[left, right])
                            for metric, matrix in matrices.items()
                        },
                    }
                )
            drop_rows.append(
                {
                    "event_uid": uid[query_index],
                    "query_index": int(query_index),
                    "subject": subjects[query_index],
                    "support_class": str(support_row["support_class"]),
                    "true_post_peak_drop": bool(outcomes["drop_flag"][query_index]),
                    "neighbor_positive_count": int(np.sum(outcomes["drop_flag"][main])),
                    "neighbor_mixed": bool(
                        0 < int(np.sum(outcomes["drop_flag"][main])) < len(main)
                    ),
                    "neighbor_pair_disagreement_rate": observed[
                        "post_peak_drop_disagreement"
                    ],
                    "random_pair_disagreement_rate": float(
                        np.mean(random_values["post_peak_drop_disagreement"])
                    ),
                }
            )
            query_rows.append(row)

    return (
        pd.DataFrame(query_rows),
        pd.DataFrame(edge_rows),
        pd.DataFrame(control_rows),
        pd.DataFrame(gradient_rows),
        pd.DataFrame(drop_rows),
        matrices,
    )


def subject_cluster_bootstrap(
    per_query: pd.DataFrame,
    ratio_column: str,
    repetitions: int = 10000,
    seed: int = 20260812,
) -> tuple[dict[str, float], np.ndarray]:
    subject_values = (
        per_query.groupby("subject", sort=True)[ratio_column].mean().to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed + sum(ord(char) for char in ratio_column))
    indices = rng.integers(0, len(subject_values), size=(repetitions, len(subject_values)))
    boot = subject_values[indices].mean(axis=1)
    return (
        {
            "estimate": float(np.mean(subject_values)),
            "ci_lower": float(np.quantile(boot, 0.025)),
            "ci_upper": float(np.quantile(boot, 0.975)),
            "subject_count": int(len(subject_values)),
            "repetitions": int(repetitions),
        },
        boot,
    )


def hierarchical_bootstrap(
    per_query: pd.DataFrame,
    ratio_column: str,
    repetitions: int = 10000,
    seed: int = 20260812,
) -> dict[str, float]:
    subjects = sorted(per_query["subject"].unique())
    grouped = {
        subject: {
            recording: group[ratio_column].to_numpy(dtype=float)
            for recording, group in per_query[per_query["subject"].eq(subject)].groupby(
                "recording_uid", sort=True
            )
        }
        for subject in subjects
    }
    rng = np.random.default_rng(seed + 1000 + sum(ord(char) for char in ratio_column))
    boot = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        drawn_subjects = rng.choice(subjects, size=len(subjects), replace=True)
        subject_means: list[float] = []
        for subject in drawn_subjects:
            recordings = list(grouped[str(subject)])
            drawn_recordings = rng.choice(recordings, size=len(recordings), replace=True)
            recording_means: list[float] = []
            for recording in drawn_recordings:
                values = grouped[str(subject)][str(recording)]
                recording_means.append(float(np.mean(rng.choice(values, size=len(values), replace=True))))
            subject_means.append(float(np.mean(recording_means)))
        boot[repetition] = float(np.mean(subject_means))
    return {
        "estimate": float(
            per_query.groupby("subject", sort=True)[ratio_column].mean().mean()
        ),
        "ci_lower": float(np.quantile(boot, 0.025)),
        "ci_upper": float(np.quantile(boot, 0.975)),
        "repetitions": int(repetitions),
    }


def two_way_query_neighbor_cluster_bootstrap(
    per_query: pd.DataFrame,
    main_edges: pd.DataFrame,
    ratio_column: str,
    repetitions: int = 10000,
    seed: int = 20260812,
) -> dict[str, float]:
    """查询被试与邻居被试两个维度独立重采样的敏感性区间。"""

    edges = main_edges[~main_edges["is_pseudo_reference"]].copy()
    query_indices = per_query["query_index"].to_numpy(dtype=int)
    query_subject = per_query["subject"].astype(str).to_numpy()
    values = per_query[ratio_column].to_numpy(dtype=float)
    subjects = sorted(per_query["subject"].astype(str).unique())
    subject_position = {subject: index for index, subject in enumerate(subjects)}
    query_subject_position = np.asarray(
        [subject_position[subject] for subject in query_subject], dtype=int
    )
    neighbor_map = (
        edges.groupby("query_index", sort=False)["neighbor_subject"]
        .apply(lambda group: group.astype(str).tolist())
        .to_dict()
    )
    neighbor_positions = np.asarray(
        [
            [subject_position[subject] for subject in neighbor_map[int(query)]]
            for query in query_indices
        ],
        dtype=int,
    )
    if neighbor_positions.shape != (len(per_query), 5):
        raise RuntimeError("双向聚类bootstrap要求每个查询恰有5个邻居被试")
    rng = np.random.default_rng(seed + 3000 + sum(ord(char) for char in ratio_column))
    estimates = np.empty(repetitions, dtype=float)
    probability = np.full(len(subjects), 1.0 / len(subjects))
    subject_masks = [query_subject_position == index for index in range(len(subjects))]
    for repetition in range(repetitions):
        query_counts = rng.multinomial(len(subjects), probability)
        neighbor_counts = rng.multinomial(len(subjects), probability)
        neighbor_weight = neighbor_counts[neighbor_positions].mean(axis=1)
        subject_estimates: list[float] = []
        subject_weights: list[float] = []
        for subject_index, mask in enumerate(subject_masks):
            if query_counts[subject_index] == 0:
                continue
            weights = neighbor_weight[mask]
            if float(np.sum(weights)) <= 0:
                continue
            subject_estimates.append(float(np.average(values[mask], weights=weights)))
            subject_weights.append(float(query_counts[subject_index]))
        estimates[repetition] = float(
            np.average(subject_estimates, weights=subject_weights)
        )
    return {
        "estimate": float(
            per_query.groupby("subject", sort=True)[ratio_column].mean().mean()
        ),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "repetitions": int(repetitions),
        "cluster_dimensions": "query_subject_x_neighbor_subject",
    }


def subject_block_permutation_indices(
    subjects: np.ndarray,
    event_uid: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    unique_subjects = sorted(np.unique(subjects.astype(str)))
    recipient_blocks = [
        np.asarray(
            sorted(
                np.where(subjects.astype(str) == subject)[0],
                key=lambda index: str(event_uid[index]),
            ),
            dtype=int,
        )
        for subject in unique_subjects
    ]
    source_order = rng.permutation(len(recipient_blocks))
    mapping = np.empty(len(subjects), dtype=int)
    for recipient, source_position in zip(recipient_blocks, source_order, strict=True):
        donor = recipient_blocks[int(source_position)]
        # 不同被试事件数不同。按供体块内分位位置确定性匹配，使一个接收被试
        # 的全部未来都来自同一个供体被试；20点曲线始终整行移动，不逐点打乱。
        donor_positions = np.floor(
            (np.arange(len(recipient), dtype=float) + 0.5) * len(donor) / len(recipient)
        ).astype(int)
        donor_positions = np.clip(donor_positions, 0, len(donor) - 1)
        mapping[recipient] = donor[donor_positions]
    return mapping


def permutation_tests(
    per_query: pd.DataFrame,
    edge_table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    bundle: dict[str, np.ndarray],
    repetitions: int = 5000,
    seed: int = 20260812,
) -> pd.DataFrame:
    subjects = bundle["subject"].astype(str)
    uid = bundle["event_uid"].astype(str)
    query_subject = per_query.set_index("query_index")["subject"].to_dict()
    edge_query = edge_table["query_index"].to_numpy(dtype=int)
    left = np.asarray(
        [int(np.where(uid == value)[0][0]) for value in edge_table["left_neighbor_event_uid"]],
        dtype=int,
    )
    right = np.asarray(
        [int(np.where(uid == value)[0][0]) for value in edge_table["right_neighbor_event_uid"]],
        dtype=int,
    )
    unique_queries = per_query["query_index"].to_numpy(dtype=int)
    per_query_denominators = {
        metric: per_query.set_index("query_index")[f"random_mean_{metric}"].to_dict()
        for metric in METRICS
    }
    macro_denominators = {
        metric: float(
            per_query.groupby("subject", sort=True)[f"random_mean_{metric}"].mean().mean()
        )
        for metric in METRICS
    }
    observed = {}
    for metric in METRICS:
        if metric == "post_peak_drop_disagreement":
            numerator = float(
                per_query.groupby("subject", sort=True)[f"top5_{metric}"].mean().mean()
            )
            observed[metric] = numerator / max(macro_denominators[metric], 1e-12)
        else:
            observed[metric] = float(
                per_query.groupby("subject", sort=True)[f"ratio_{metric}"].mean().mean()
            )
    query_positions = {
        query: np.where(edge_query == query)[0] for query in unique_queries
    }
    rng = np.random.default_rng(seed + 5000)
    null = {metric: np.empty(repetitions, dtype=float) for metric in METRICS}
    for repetition in range(repetitions):
        mapping = subject_block_permutation_indices(subjects, uid, rng)
        for metric, matrix in matrices.items():
            edge_values = matrix[mapping[left], mapping[right]]
            per_query_value = {
                query: float(np.mean(edge_values[positions]))
                for query, positions in query_positions.items()
            }
            subject_values: dict[str, list[float]] = {}
            for query, value in per_query_value.items():
                subject_values.setdefault(str(query_subject[query]), []).append(value)
            if metric == "post_peak_drop_disagreement":
                numerator = float(
                    np.mean([np.mean(values) for values in subject_values.values()])
                )
                null[metric][repetition] = numerator / max(
                    macro_denominators[metric], 1e-12
                )
            else:
                ratio_by_subject: dict[str, list[float]] = {}
                for query, value in per_query_value.items():
                    ratio_by_subject.setdefault(str(query_subject[query]), []).append(
                        value
                        / max(float(per_query_denominators[metric][query]), 1e-12)
                    )
                null[metric][repetition] = float(
                    np.mean([np.mean(values) for values in ratio_by_subject.values()])
                )
    rows: list[dict[str, Any]] = []
    raw_p: dict[str, float] = {}
    for metric in METRICS:
        center = float(np.median(null[metric]))
        p_value = float(
            (1 + np.sum(np.abs(null[metric] - center) >= abs(observed[metric] - center)))
            / (repetitions + 1)
        )
        raw_p[metric] = p_value
    # Holm校正；主终点仍保留未校正双侧p。
    secondary = [metric for metric in METRICS if metric != "curve_rmse_deg"]
    ordered = sorted(secondary, key=lambda metric: raw_p[metric])
    adjusted: dict[str, float] = {"curve_rmse_deg": raw_p["curve_rmse_deg"]}
    running = 0.0
    for rank, metric in enumerate(ordered):
        value = min(1.0, raw_p[metric] * (len(secondary) - rank))
        running = max(running, value)
        adjusted[metric] = running
    for metric in METRICS:
        rows.append(
            {
                "metric": metric,
                "observed_ratio": observed[metric],
                "null_median_ratio": float(np.median(null[metric])),
                "null_p005": float(np.quantile(null[metric], 0.005)),
                "null_p025": float(np.quantile(null[metric], 0.025)),
                "null_p975": float(np.quantile(null[metric], 0.975)),
                "null_p995": float(np.quantile(null[metric], 0.995)),
                "two_sided_p": raw_p[metric],
                "holm_adjusted_p": adjusted[metric],
                "repetitions": repetitions,
                "permutation_unit": "future_curve_subject_blocks",
            }
        )
    return pd.DataFrame(rows)
