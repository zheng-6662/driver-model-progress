from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.ensemble import ExtraTreesRegressor

import data as data_module
from model import (
    LGRS,
    PlainRawTCN,
    RoleTCN,
    curve_loss,
    fit_absolute_scaler,
    parameter_count,
    prepare_prefix,
    relation_loss,
    transform_absolute,
)


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def inner_subject_split(
    subjects: np.ndarray,
    outer_train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    table = pd.DataFrame({"index": outer_train, "subject": subjects[outer_train]})
    counts = table.groupby("subject").size()
    rng = np.random.default_rng(seed)
    ordered = counts.index.to_numpy().copy()
    rng.shuffle(ordered)
    target = int(round(len(outer_train) * 0.20))
    validation_subjects: list[str] = []
    current = 0
    for subject in ordered:
        if current >= target and len(validation_subjects) >= 3:
            break
        validation_subjects.append(str(subject))
        current += int(counts.loc[subject])
    validation_mask = np.isin(subjects[outer_train], validation_subjects)
    fit = outer_train[~validation_mask]
    validation = outer_train[validation_mask]
    if set(subjects[fit]) & set(subjects[validation]):
        raise ValueError("inner被试交叉")
    if len(np.unique(subjects[validation])) < 3:
        raise ValueError("inner验证被试不足")
    return fit, validation


def sampling_weights(subjects: np.ndarray, domains: np.ndarray, indices: np.ndarray) -> np.ndarray:
    table = pd.DataFrame(
        {
            "subject": subjects[indices].astype(str),
            "domain": domains[indices].astype(str),
        }
    )
    domain_subjects = table.groupby("domain")["subject"].nunique().to_dict()
    pair_counts = table.groupby(["domain", "subject"])["subject"].transform("size").to_numpy(float)
    values = np.asarray(
        [1.0 / domain_subjects[domain] for domain in table["domain"]], dtype=float
    ) / pair_counts
    values /= values.mean()
    return values.astype(np.float64)


def subject_macro_mae(
    prediction: np.ndarray,
    truth: np.ndarray,
    subjects: np.ndarray,
) -> float:
    mae = np.mean(np.abs(prediction - truth), axis=1)
    return float(pd.DataFrame({"subject": subjects, "mae": mae}).groupby("subject")["mae"].mean().mean())


def make_loader(
    sequence: np.ndarray,
    absolute: np.ndarray,
    response_mask: np.ndarray,
    truth: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(sequence[indices]),
        torch.from_numpy(absolute[indices]),
        torch.from_numpy(response_mask[indices]),
        torch.from_numpy(truth[indices]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(indices),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)


@torch.no_grad()
def predict(
    model: nn.Module,
    sequence: np.ndarray,
    absolute: np.ndarray,
    response_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, bool]:
    model.eval()
    output = []
    finite_relation = True
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        result = model(
            torch.from_numpy(sequence[batch]).cuda(),
            torch.from_numpy(absolute[batch]).cuda(),
            torch.from_numpy(response_mask[batch]).cuda(),
        )
        prediction = result["prediction"]
        output.append(prediction.cpu().numpy())
        if "gains" in result:
            finite_relation = finite_relation and bool(torch.isfinite(result["gains"]).all())
            finite_relation = finite_relation and bool(torch.isfinite(result["expected_response"]).all())
    return np.vstack(output), finite_relation


def train_smoke_model(
    model_name: str,
    sequence: np.ndarray,
    absolute: np.ndarray,
    response_mask: np.ndarray,
    truth: np.ndarray,
    subjects: np.ndarray,
    domains: np.ndarray,
    fit: np.ndarray,
    validation: np.ndarray,
    config: dict[str, object],
    seed: int,
    relation_weight: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    set_seed(seed)
    if model_name == "Role_TCN":
        model: nn.Module = RoleTCN(CONFIG["model"])
    else:
        model = LGRS(CONFIG["model"], CONFIG["lag_samples"])
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    weights = sampling_weights(subjects, domains, fit)
    loader = make_loader(
        sequence,
        absolute,
        response_mask,
        truth,
        fit,
        weights,
        int(config["batch_size"]),
        seed,
    )
    trace_rows: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, int(config["smoke_epochs"]) + 1):
        model.train()
        epoch_curve = 0.0
        epoch_relation = 0.0
        batches = 0
        for batch_sequence, batch_absolute, batch_mask, batch_truth in loader:
            batch_sequence = batch_sequence.cuda(non_blocking=True)
            batch_absolute = batch_absolute.cuda(non_blocking=True)
            batch_mask = batch_mask.cuda(non_blocking=True)
            batch_truth = batch_truth.cuda(non_blocking=True)
            result = model(batch_sequence, batch_absolute, batch_mask)
            base_loss = curve_loss(
                result["prediction"],
                batch_truth,
                float(config["curve_diff_loss_weight"]),
            )
            relation = relation_loss(result) if "expected_response" in result else torch.zeros((), device="cuda")
            loss = base_loss + relation_weight * relation
            if not torch.isfinite(loss):
                raise ValueError(f"{model_name}出现非有限loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            epoch_curve += float(base_loss.detach().cpu())
            epoch_relation += float(relation.detach().cpu())
            batches += 1
        validation_prediction, relation_finite = predict(
            model,
            sequence,
            absolute,
            response_mask,
            validation,
            int(config["batch_size"]),
        )
        validation_macro = subject_macro_mae(
            validation_prediction,
            truth[validation],
            subjects[validation],
        )
        trace_rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_curve_loss": epoch_curve / batches,
                "train_relation_loss": epoch_relation / batches,
                "validation_subject_macro_mae_deg": validation_macro,
                "relation_outputs_finite": relation_finite,
            }
        )
        print(
            f"{model_name} epoch {epoch}/{config['smoke_epochs']}: "
            f"curve={epoch_curve / batches:.4f} relation={epoch_relation / batches:.4f} "
            f"val_macro={validation_macro:.4f}",
            flush=True,
        )
    trace = pd.DataFrame(trace_rows)
    summary = {
        "model": model_name,
        "parameter_count": parameter_count(model),
        "last_validation_subject_macro_mae_deg": float(trace.iloc[-1]["validation_subject_macro_mae_deg"]),
        "minimum_validation_subject_macro_mae_deg": float(trace["validation_subject_macro_mae_deg"].min()),
        "relation_outputs_all_finite": bool(trace["relation_outputs_finite"].all()),
        "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }
    return trace, summary


def create_model(model_name: str) -> nn.Module:
    if model_name == "Plain_Raw_TCN":
        return PlainRawTCN(CONFIG["model"])
    if model_name == "Role_TCN":
        return RoleTCN(CONFIG["model"])
    if model_name in {"LGRS", "LGRS_lambda0"}:
        return LGRS(CONFIG["model"], CONFIG["lag_samples"])
    raise ValueError(model_name)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    training: dict[str, object],
    relation_weight: float,
) -> tuple[float, float]:
    model.train()
    curve_total = 0.0
    relation_total = 0.0
    batches = 0
    for batch_sequence, batch_absolute, batch_mask, batch_truth in loader:
        batch_sequence = batch_sequence.cuda(non_blocking=True)
        batch_absolute = batch_absolute.cuda(non_blocking=True)
        batch_mask = batch_mask.cuda(non_blocking=True)
        batch_truth = batch_truth.cuda(non_blocking=True)
        result = model(batch_sequence, batch_absolute, batch_mask)
        base = curve_loss(
            result["prediction"],
            batch_truth,
            float(training["curve_diff_loss_weight"]),
        )
        relation = relation_loss(result) if "expected_response" in result else torch.zeros((), device="cuda")
        loss = base + relation_weight * relation
        if not torch.isfinite(loss):
            raise ValueError("正式训练出现非有限loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
        optimizer.step()
        curve_total += float(base.detach().cpu())
        relation_total += float(relation.detach().cpu())
        batches += 1
    return curve_total / batches, relation_total / batches


def train_full_neural_model(
    model_name: str,
    sequence: np.ndarray,
    absolute_raw: np.ndarray,
    response_mask: np.ndarray,
    truth: np.ndarray,
    subjects: np.ndarray,
    domains: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    fold: int,
    seed: int,
    training: dict[str, object],
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    relation_weight = (
        float(training["relation_loss_weight"])
        if model_name == "LGRS"
        else 0.0
    )
    fit, validation = inner_subject_split(subjects, outer_train, seed + fold * 1000)
    select_center, select_scale = fit_absolute_scaler(absolute_raw, fit)
    select_absolute = transform_absolute(absolute_raw, select_center, select_scale)
    set_seed(seed + fold * 100000)
    select_model = create_model(model_name).cuda()
    optimizer = torch.optim.AdamW(
        select_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loader = make_loader(
        sequence,
        select_absolute,
        response_mask,
        truth,
        fit,
        sampling_weights(subjects, domains, fit),
        int(training["batch_size"]),
        seed + fold * 100000,
    )
    best_epoch = 0
    best_metric = float("inf")
    stale = 0
    trace_rows: list[dict[str, object]] = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        curve_value, relation_value = train_one_epoch(
            select_model, loader, optimizer, training, relation_weight
        )
        validation_prediction, finite_relation = predict(
            select_model,
            sequence,
            select_absolute,
            response_mask,
            validation,
            int(training["batch_size"]),
        )
        validation_metric = subject_macro_mae(
            validation_prediction, truth[validation], subjects[validation]
        )
        trace_rows.append(
            {
                "model": model_name,
                "outer_fold": fold,
                "seed": seed,
                "epoch": epoch,
                "train_curve_loss": curve_value,
                "train_relation_loss": relation_value,
                "validation_subject_macro_mae_deg": validation_metric,
                "relation_outputs_finite": finite_relation,
            }
        )
        if validation_metric < best_metric - 1e-6:
            best_metric = validation_metric
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"select {model_name} fold={fold} seed={seed} epoch={epoch} "
                f"val_macro={validation_metric:.4f} best={best_metric:.4f}@{best_epoch}",
                flush=True,
            )
        if stale >= int(training["patience"]):
            break
    if best_epoch <= 0:
        raise ValueError("未选出best epoch")
    del select_model
    torch.cuda.empty_cache()

    full_center, full_scale = fit_absolute_scaler(absolute_raw, outer_train)
    full_absolute = transform_absolute(absolute_raw, full_center, full_scale)
    set_seed(seed + fold * 100000)
    final_model = create_model(model_name).cuda()
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    final_loader = make_loader(
        sequence,
        full_absolute,
        response_mask,
        truth,
        outer_train,
        sampling_weights(subjects, domains, outer_train),
        int(training["batch_size"]),
        seed + fold * 100000,
    )
    torch.cuda.reset_peak_memory_stats()
    for _ in range(best_epoch):
        train_one_epoch(final_model, final_loader, final_optimizer, training, relation_weight)
    test_prediction, finite_relation = predict(
        final_model,
        sequence,
        full_absolute,
        response_mask,
        outer_test,
        int(training["batch_size"]),
    )
    lag_rows: list[dict[str, object]] = []
    if model_name == "LGRS":
        base_metric = subject_macro_mae(test_prediction, truth[outer_test], subjects[outer_test])
        for shift in [-4, -2, 2, 4]:
            shifted = sequence.copy()
            shifted[:, 0:2] = np.roll(shifted[:, 0:2], shift=shift, axis=2)
            shifted_prediction, _ = predict(
                final_model,
                shifted,
                full_absolute,
                response_mask,
                outer_test,
                int(training["batch_size"]),
            )
            shifted_metric = subject_macro_mae(
                shifted_prediction, truth[outer_test], subjects[outer_test]
            )
            lag_rows.append(
                {
                    "outer_fold": fold,
                    "seed": seed,
                    "shift_samples": shift,
                    "shift_ms": shift * 20,
                    "base_subject_macro_mae_deg": base_metric,
                    "shifted_subject_macro_mae_deg": shifted_metric,
                    "mae_change_deg": shifted_metric - base_metric,
                }
            )
    summary = {
        "model": model_name,
        "outer_fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_inner_validation_subject_macro_mae_deg": best_metric,
        "parameter_count": parameter_count(final_model),
        "relation_outputs_finite": finite_relation,
        "gpu_peak_memory_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "fit_subjects": int(len(np.unique(subjects[fit]))),
        "validation_subjects": int(len(np.unique(subjects[validation]))),
        "outer_train_subjects": int(len(np.unique(subjects[outer_train]))),
        "outer_test_subjects": int(len(np.unique(subjects[outer_test]))),
    }
    del final_model
    torch.cuda.empty_cache()
    return test_prediction, pd.DataFrame(trace_rows), summary, lag_rows


def fill_summary_from_training(
    train_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(train_x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return (
        np.where(np.isfinite(train_x), train_x, median),
        np.where(np.isfinite(test_x), test_x, median),
    )


def event_metric_table(truth: np.ndarray, prediction: np.ndarray) -> pd.DataFrame:
    error = np.abs(prediction - truth)
    return pd.DataFrame(
        {
            "curve_mae_deg": error.mean(axis=1),
            "head5_mae_deg": error[:, :5].mean(axis=1),
            "tail5_mae_deg": error[:, -5:].mean(axis=1),
            "endpoint_mae_deg": error[:, -1],
            "peak_time_mae_s": np.abs(
                (np.argmax(np.abs(prediction), axis=1) - np.argmax(np.abs(truth), axis=1)) * 0.05
            ),
        }
    )


def paired_gate(
    comparator: str,
    candidate: str,
    metrics: dict[str, pd.DataFrame],
    subjects: np.ndarray,
    folds: np.ndarray,
    amplitude_bins: np.ndarray,
    required_gain: float,
) -> dict[str, object]:
    subject_frame = pd.DataFrame(
        {
            "subject": subjects,
            "fold": folds,
            "comparator": metrics[comparator]["curve_mae_deg"],
            "candidate": metrics[candidate]["curve_mae_deg"],
        }
    ).groupby(["subject", "fold"], as_index=False).mean()
    subject_frame["improvement"] = subject_frame["comparator"] - subject_frame["candidate"]
    values = subject_frame["improvement"].to_numpy(float)
    rng = np.random.default_rng(20260831)
    draws = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(2000)]
    )
    fold_gain = subject_frame.groupby("fold")["improvement"].mean()
    amplitude_changes = {}
    amplitude_pass = True
    for label in ["20_30", "30_45", "45_70", "ge70"]:
        mask = amplitude_bins == label
        per_subject = []
        for subject in np.unique(subjects[mask]):
            selected = mask & (subjects == subject)
            comp = float(metrics[comparator].loc[selected, "curve_mae_deg"].mean())
            cand = float(metrics[candidate].loc[selected, "curve_mae_deg"].mean())
            amplitude = float(np.median(CACHE_FOR_GATE["amplitude_deg"][selected]))
            per_subject.append((comp / amplitude, cand / amplitude))
        comp_value = float(np.mean([item[0] for item in per_subject]))
        cand_value = float(np.mean([item[1] for item in per_subject]))
        change = cand_value - comp_value
        amplitude_changes[label] = {
            "comparator": comp_value,
            "candidate": cand_value,
            "change": change,
        }
        amplitude_pass = amplitude_pass and change <= float(CONFIG["gates"]["maximum_amplitude_relative_regression"])
    gates = {
        "subject_macro_gain": float(values.mean()) >= required_gain,
        "bootstrap_ci_lower": float(np.quantile(draws, 0.025)) > 0.0,
        "positive_outer_folds": int((fold_gain > 0).sum()) >= int(CONFIG["gates"]["positive_outer_folds_min"]),
        "amplitude_protection": bool(amplitude_pass),
    }
    return {
        "comparison": f"{candidate}_vs_{comparator}",
        "subject_macro_gain_deg": float(values.mean()),
        "bootstrap_ci_lower_deg": float(np.quantile(draws, 0.025)),
        "bootstrap_ci_upper_deg": float(np.quantile(draws, 0.975)),
        "positive_outer_fold_count": int((fold_gain > 0).sum()),
        "improved_subject_count": int((values > 0).sum()),
        "harmed_subject_count": int((values < 0).sum()),
        "amplitude_changes": amplitude_changes,
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "subject_detail": subject_frame.to_dict(orient="records"),
    }


CACHE_FOR_GATE: dict[str, np.ndarray] = {}


def run_full_experiment(
    run: Path,
    tables: Path,
    outputs: Path,
    cache_summary: dict[str, object],
    cache: dict[str, np.ndarray],
    sequence: np.ndarray,
    absolute_raw: np.ndarray,
    response_mask: np.ndarray,
    truth: np.ndarray,
) -> int:
    global CACHE_FOR_GATE
    CACHE_FOR_GATE = cache
    subjects = cache["subject"].astype(str)
    domains = cache["domain"].astype(str)
    folds = cache["outer_fold"].astype(int)
    summary_x = cache["summary"].astype(np.float32)
    training = CONFIG["training"]
    model_names = ["Plain_Raw_TCN", "Role_TCN", "LGRS", "LGRS_lambda0"]
    seed_predictions = {
        model_name: [] for model_name in model_names
    }
    trace_tables = []
    training_rows = []
    lag_rows: list[dict[str, object]] = []
    for model_name in model_names:
        print(f"正式模型：{model_name}", flush=True)
        for seed in [int(value) for value in training["seeds"]]:
            prediction = np.full_like(truth, np.nan)
            for fold in range(1, 6):
                outer_train = np.flatnonzero(folds != fold)
                outer_test = np.flatnonzero(folds == fold)
                test_prediction, trace, train_summary, model_lag_rows = train_full_neural_model(
                    model_name,
                    sequence,
                    absolute_raw,
                    response_mask,
                    truth,
                    subjects,
                    domains,
                    outer_train,
                    outer_test,
                    fold,
                    seed,
                    training,
                )
                prediction[outer_test] = test_prediction
                trace_tables.append(trace)
                training_rows.append(train_summary)
                lag_rows.extend(model_lag_rows)
                print(
                    f"outer完成 model={model_name} seed={seed} fold={fold} "
                    f"test_macro={subject_macro_mae(test_prediction, truth[outer_test], subjects[outer_test]):.4f}",
                    flush=True,
                )
            if not np.isfinite(prediction).all():
                raise ValueError(f"{model_name} seed={seed} OOF不完整")
            seed_predictions[model_name].append(prediction)

    predictions = {
        model_name: np.mean(np.stack(values, axis=0), axis=0)
        for model_name, values in seed_predictions.items()
    }
    extra_prediction = np.full_like(truth, np.nan)
    for fold in range(1, 6):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        train_x, test_x = fill_summary_from_training(summary_x[train], summary_x[test])
        weights = sampling_weights(subjects, domains, train)
        model = ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=20260831 + fold,
            n_jobs=8,
        )
        model.fit(train_x, truth[train], sample_weight=weights)
        extra_prediction[test] = model.predict(test_x)
    predictions["ExtraTrees_134D"] = extra_prediction

    pd.concat(trace_tables, ignore_index=True).to_csv(
        tables / "full_training_trace.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        tables / "full_training_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(lag_rows).to_csv(
        tables / "lag_perturbation.csv", index=False, encoding="utf-8-sig"
    )

    metrics = {name: event_metric_table(truth, prediction) for name, prediction in predictions.items()}
    aggregate_rows = []
    for name, table in metrics.items():
        aggregate_rows.append(
            {
                "model": name,
                "subject_macro_curve_mae_deg": subject_macro_mae(
                    predictions[name], truth, subjects
                ),
                "subject_macro_head5_mae_deg": float(
                    pd.DataFrame({"subject": subjects, "value": table["head5_mae_deg"]})
                    .groupby("subject")["value"].mean().mean()
                ),
                "subject_macro_tail5_mae_deg": float(
                    pd.DataFrame({"subject": subjects, "value": table["tail5_mae_deg"]})
                    .groupby("subject")["value"].mean().mean()
                ),
                "subject_macro_endpoint_mae_deg": float(
                    pd.DataFrame({"subject": subjects, "value": table["endpoint_mae_deg"]})
                    .groupby("subject")["value"].mean().mean()
                ),
                "subject_macro_peak_time_mae_s": float(
                    pd.DataFrame({"subject": subjects, "value": table["peak_time_mae_s"]})
                    .groupby("subject")["value"].mean().mean()
                ),
                "pooled_curve_mae_deg_reference": float(table["curve_mae_deg"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(tables / "aggregate_metrics.csv", index=False, encoding="utf-8-sig")

    amplitude = cache["amplitude_deg"].astype(float)
    amplitude_bins = np.asarray(
        pd.cut(
            amplitude,
            [0, 20, 30, 45, 70, np.inf],
            labels=["lt20", "20_30", "30_45", "45_70", "ge70"],
            right=False,
        ).astype(str)
    )
    lgrs_role = paired_gate(
        "Role_TCN",
        "LGRS",
        metrics,
        subjects,
        folds,
        amplitude_bins,
        float(CONFIG["gates"]["lgrs_vs_role_tcn_subject_macro_gain_min_deg"]),
    )
    lgrs_extra = paired_gate(
        "ExtraTrees_134D",
        "LGRS",
        metrics,
        subjects,
        folds,
        amplitude_bins,
        float(CONFIG["gates"]["lgrs_vs_extratrees_subject_macro_gain_min_deg"]),
    )
    status = (
        "LGRS_EFFECTIVE"
        if lgrs_role["all_gates_pass"] and lgrs_extra["all_gates_pass"]
        else "RAW_SEQUENCE_USEFUL_LGRS_NOT_PROVEN"
        if aggregate.set_index("model").loc["Role_TCN", "subject_macro_curve_mae_deg"]
        < aggregate.set_index("model").loc["ExtraTrees_134D", "subject_macro_curve_mae_deg"]
        else "LGRS_NOT_EFFECTIVE"
    )
    decision = {
        "run_id": CONFIG["run_id"],
        "status": status,
        "events": len(subjects),
        "subjects": int(len(np.unique(subjects))),
        "cache": cache_summary,
        "comparisons": {
            "LGRS_vs_Role_TCN": {key: value for key, value in lgrs_role.items() if key != "subject_detail"},
            "LGRS_vs_ExtraTrees": {key: value for key, value in lgrs_extra.items() if key != "subject_detail"},
        },
        "evidence_boundary": CONFIG["evidence_boundary"],
    }
    (outputs / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subject_detail = pd.concat(
        [
            pd.DataFrame(lgrs_role["subject_detail"]).assign(comparison="LGRS_vs_Role_TCN"),
            pd.DataFrame(lgrs_extra["subject_detail"]).assign(comparison="LGRS_vs_ExtraTrees"),
        ],
        ignore_index=True,
    )
    subject_detail.to_csv(tables / "paired_subject_improvements.csv", index=False, encoding="utf-8-sig")

    metadata_output = pd.DataFrame(
        {
            "event_uid": cache["event_uid"].astype(str),
            "subject": subjects,
            "recording_uid": cache["recording_uid"].astype(str),
            "domain": domains,
            "outer_fold": folds,
            "amplitude_bin": amplitude_bins,
        }
    )
    prediction_columns: dict[str, np.ndarray] = {}
    for point in range(20):
        prediction_columns[f"true_t{point + 1:02d}_deg"] = truth[:, point]
        for name, prediction in predictions.items():
            prediction_columns[f"{name}_pred_t{point + 1:02d}_deg"] = prediction[:, point]
    for name, table in metrics.items():
        prediction_columns[f"{name}_curve_mae_deg"] = table["curve_mae_deg"].to_numpy(float)
    output = pd.concat([metadata_output, pd.DataFrame(prediction_columns)], axis=1)
    predictions_dir = run / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_dir / "per_event_predictions.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# Run82 LGRS正式5折×3seed结果",
        "",
        f"- 状态：`{status}`",
        f"- 事件：{len(subjects)}；被试：{len(np.unique(subjects))}。",
        "",
        "| model | subject-macro MAE° | head5° | tail5° | endpoint° | peak-time s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.subject_macro_curve_mae_deg:.4f} | "
            f"{row.subject_macro_head5_mae_deg:.4f} | {row.subject_macro_tail5_mae_deg:.4f} | "
            f"{row.subject_macro_endpoint_mae_deg:.4f} | {row.subject_macro_peak_time_mae_s:.4f} |"
        )
    for comparison in [lgrs_role, lgrs_extra]:
        lines += [
            "",
            f"## {comparison['comparison']}",
            "",
            f"- subject-macro gain：{comparison['subject_macro_gain_deg']:+.4f}°。",
            f"- 95%CI：[{comparison['bootstrap_ci_lower_deg']:+.4f}, {comparison['bootstrap_ci_upper_deg']:+.4f}]°。",
            f"- 正向折：{comparison['positive_outer_fold_count']}/5。",
            f"- 改善/退化被试：{comparison['improved_subject_count']}/{comparison['harmed_subject_count']}。",
            f"- 四门全过：`{comparison['all_gates_pass']}`。",
        ]
    lines += [
        "",
        "## 证据边界",
        "",
        "本轮是combined 38-subject developmental OOF。若LGRS不超过参数配平Role-TCN，必须关闭LGRS方法线；不能只因超过树模型就宣称关系瓶颈成立。",
    ]
    (outputs / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "final_info.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n".join(lines), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    arguments = parser.parse_args()
    run = HERE / arguments.out_dir
    tables = run / "tables"
    outputs = run / "outputs"
    figures = run / "figures"
    cache_dir = run / "cache"
    logs = run / "logs"
    for directory in [tables, outputs, figures, cache_dir, logs]:
        directory.mkdir(parents=True, exist_ok=True)

    print("实验目的：先验证2598事件共同序列锚点与LGRS单折实现，不做正式外折结论。", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Run82 smoke要求CUDA可用")
    started = time.time()
    cache_path = cache_dir / "combined_common_sequence.npz"
    anchor_path = outputs / "input_anchor.json"
    cache_summary = data_module.build_combined_cache(cache_path, anchor_path)
    cache = data_module.load_cache(cache_path)
    sequence_raw = cache["sequence"].astype(np.float32)
    truth = cache["truth"].astype(np.float32)
    subjects = cache["subject"].astype(str)
    domains = cache["domain"].astype(str)
    outer_fold = cache["outer_fold"].astype(int)
    sequence, absolute_raw, response_mask = prepare_prefix(sequence_raw)
    if sequence.shape != (2598, 8, 101) or absolute_raw.shape != (2598, 22):
        raise ValueError((sequence.shape, absolute_raw.shape))

    if CONFIG["stage"] == "full_oof":
        return run_full_experiment(
            run,
            tables,
            outputs,
            cache_summary,
            cache,
            sequence,
            absolute_raw,
            response_mask,
            truth,
        )

    smoke_fold = int(CONFIG["smoke"]["outer_fold"])
    outer_train = np.flatnonzero(outer_fold != smoke_fold)
    fit, validation = inner_subject_split(subjects, outer_train, int(CONFIG["smoke"]["seed"]))
    center, scale = fit_absolute_scaler(absolute_raw, fit)
    absolute = transform_absolute(absolute_raw, center, scale)

    role = RoleTCN(CONFIG["model"])
    lgrs = LGRS(CONFIG["model"], CONFIG["lag_samples"])
    role_parameters = parameter_count(role)
    lgrs_parameters = parameter_count(lgrs)
    parameter_difference = abs(role_parameters - lgrs_parameters) / role_parameters
    lag_test = torch.arange(5, dtype=torch.float32).view(1, 1, 5)
    lagged_two = LGRS.lagged(lag_test, 2).flatten().tolist()
    lag_padding_pass = lagged_two == [0.0, 0.0, 0.0, 1.0, 2.0]
    parameter_pass = parameter_difference <= float(CONFIG["smoke"]["parameter_difference_fraction_max"])
    if not parameter_pass:
        raise ValueError(
            f"参数量差超过5%: role={role_parameters} lgrs={lgrs_parameters} diff={parameter_difference:.4f}"
        )
    if not lag_padding_pass:
        raise ValueError(f"lag padding错误: {lagged_two}")

    training = CONFIG["training"]
    traces = []
    summaries = []
    for model_name, relation_weight in [
        ("Role_TCN", 0.0),
        ("LGRS", float(training["relation_loss_weight"])),
        ("LGRS_lambda0", 0.0),
    ]:
        trace, summary = train_smoke_model(
            model_name,
            sequence,
            absolute,
            response_mask,
            truth,
            subjects,
            domains,
            fit,
            validation,
            training,
            int(CONFIG["smoke"]["seed"]),
            relation_weight,
        )
        traces.append(trace)
        summaries.append(summary)
    trace_table = pd.concat(traces, ignore_index=True)
    summary_table = pd.DataFrame(summaries)
    trace_table.to_csv(tables / "smoke_training_trace.csv", index=False, encoding="utf-8-sig")
    summary_table.to_csv(tables / "smoke_model_summary.csv", index=False, encoding="utf-8-sig")

    smoke_pass = bool(
        cache_summary["anchor"]["status"] == "PASS"
        and parameter_pass
        and lag_padding_pass
        and summary_table["relation_outputs_all_finite"].all()
        and np.isfinite(summary_table["minimum_validation_subject_macro_mae_deg"]).all()
    )
    final_info = {
        "run_id": CONFIG["run_id"],
        "stage": CONFIG["stage"],
        "status": "SMOKE_PASS" if smoke_pass else "SMOKE_FAIL",
        "cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cache": cache_summary,
        "tensor_shapes": {
            "sequence": list(sequence.shape),
            "absolute": list(absolute.shape),
            "response_mask": list(response_mask.shape),
            "truth": list(truth.shape),
        },
        "smoke_split": {
            "outer_fold_reserved": smoke_fold,
            "fit_events": len(fit),
            "fit_subjects": int(len(np.unique(subjects[fit]))),
            "validation_events": len(validation),
            "validation_subjects": int(len(np.unique(subjects[validation]))),
        },
        "parameter_audit": {
            "Role_TCN": role_parameters,
            "LGRS": lgrs_parameters,
            "difference_fraction": parameter_difference,
            "maximum_allowed": float(CONFIG["smoke"]["parameter_difference_fraction_max"]),
            "pass": parameter_pass,
        },
        "lag_padding": {"lag_2_result": lagged_two, "pass": lag_padding_pass},
        "models": summaries,
        "elapsed_seconds": float(time.time() - started),
        "evidence_boundary": (
            "This is an implementation smoke on outer-training data only. Validation metrics are not a scientific "
            "model comparison and do not authorize claims before full 5-fold x 3-seed OOF."
        ),
    }
    (run / "final_info.json").write_text(json.dumps(final_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (outputs / "smoke_result.json").write_text(json.dumps(final_info, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Run82 输入锚点与LGRS单折smoke",
        "",
        f"- 状态：`{final_info['status']}`",
        f"- CUDA：{final_info['cuda_device']}；PyTorch {final_info['torch_version']}。",
        f"- 合并事件：{cache_summary['events']}；被试：{cache_summary['subjects']}。",
        f"- 输入shape：{final_info['tensor_shapes']['sequence']}；绝对尺度：{final_info['tensor_shapes']['absolute']}。",
        f"- 公共148摘要锚点最大差：{cache_summary['anchor']['max_abs_diff_public_148']}。",
        f"- 全275摘要锚点最大差：{cache_summary['anchor']['max_abs_diff_all_275']}。",
        f"- Role-TCN参数：{role_parameters}；LGRS参数：{lgrs_parameters}；差异：{parameter_difference:.3%}。",
        f"- lag padding：{lagged_two}，通过={lag_padding_pass}。",
        "",
        "| model | min inner-val subject-macro MAE° | last MAE° | peak GPU MB | relation finite |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.minimum_validation_subject_macro_mae_deg:.4f} | "
            f"{row.last_validation_subject_macro_mae_deg:.4f} | {row.gpu_peak_memory_mb:.1f} | "
            f"{row.relation_outputs_all_finite} |"
        )
    lines += [
        "",
        "本轮只验证实现边界。inner validation数值不得解释为正式模型优劣，也不允许据此改超参数。",
    ]
    (outputs / "RESULT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
