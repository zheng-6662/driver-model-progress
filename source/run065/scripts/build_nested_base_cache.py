from __future__ import annotations

"""为Run65生成严格双层subject-disjoint基专家缓存。

Run63现有inner预测对Run65的meta-validation行本身是干净的；但其他meta-fit行的
基预测曾使用meta-validation被试训练。这里永久排除当前meta-validation被试，再在
剩余meta-fit被试内做2折base cross-fit，重新生成训练残差标签。
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"<PROJECT_ROOT>")
RUN63_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run63_protected_residual_and_soft_gating_20260829"
RUN65_DIR = ROOT / "05_rebuild_from_raw_20260511" / "03_baselines" / "run65_multimodal_residual_distillation_20260830"
SPEC = importlib.util.spec_from_file_location("run63_nested_source", RUN63_DIR / "experiment.py")
source = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = source
SPEC.loader.exec_module(source)

OUT_DIR = RUN65_DIR / "cache" / "nested_base"
SUMMARY_PATH = RUN65_DIR / "outputs" / "nested_base_cache_summary.json"
TABLE_PATH = RUN65_DIR / "tables" / "nested_base_cache_manifest.csv"
SUBFOLDS = 2
SEED = 20260830


def load_run63_context(outer_fold: int):
    path = RUN63_DIR / "run_1" / "cache" / f"outer_{outer_fold}_inner_oof.npz"
    with np.load(path, allow_pickle=False) as z:
        return (
            z["indices"].astype(int),
            z["inner_fold"].astype(int),
            z["base_predictions"].astype(float),
        )


def main():
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = source.load_config()
    data = source.load_inputs(config)
    expected_assignments, _ = source.build_inner_assignment(data.metadata, config)
    rows = []
    for outer_fold in range(1, 6):
        indices, inner_fold, existing_predictions = load_run63_context(outer_fold)
        if not np.array_equal(expected_assignments[outer_fold][indices], inner_fold):
            raise RuntimeError(f"outer {outer_fold}: Run63 context assignment mismatch")
        outer_test_subjects = set(
            data.metadata.loc[data.metadata["outer_fold"].astype(int).eq(outer_fold), "subject"].astype(str)
        )
        for meta_fold in (1, 2, 3):
            path = OUT_DIR / f"outer_{outer_fold}_meta_{meta_fold}.npz"
            if path.exists():
                print(f"[nested cache] existing outer={outer_fold} meta={meta_fold}")
                with np.load(path, allow_pickle=False) as z:
                    rows.append(
                        {
                            "outer_fold": outer_fold,
                            "meta_fold": meta_fold,
                            "fit_events": int(len(z["fit_indices"])),
                            "validation_events": int(len(z["validation_indices"])),
                            "fit_subjects": int(len(np.unique(data.metadata.iloc[z["fit_indices"]]["subject"]))),
                            "validation_subjects": int(len(np.unique(data.metadata.iloc[z["validation_indices"]]["subject"]))),
                            "cache_path": str(path),
                            "reused": True,
                        }
                    )
                continue
            val_local = inner_fold == meta_fold
            fit_local = ~val_local
            validation_indices = indices[val_local]
            fit_indices = indices[fit_local]
            validation_predictions = existing_predictions[val_local]
            fit_predictions = np.full((len(fit_indices), 3, 20), np.nan, dtype=float)
            fit_subjects = np.asarray(sorted(data.metadata.iloc[fit_indices]["subject"].astype(str).unique()))
            val_subjects = set(data.metadata.iloc[validation_indices]["subject"].astype(str))
            if set(fit_subjects) & val_subjects or outer_test_subjects & (set(fit_subjects) | val_subjects):
                raise RuntimeError("nested subject split overlap")
            rng = np.random.default_rng(SEED + outer_fold * 100 + meta_fold)
            permuted = rng.permutation(fit_subjects)
            subject_to_subfold = {str(s): int(i % SUBFOLDS) + 1 for i, s in enumerate(permuted)}
            global_to_local = {int(g): i for i, g in enumerate(fit_indices)}
            print(
                f"[nested cache] outer={outer_fold} meta={meta_fold} fit={len(fit_indices)} "
                f"val={len(validation_indices)} subjects={len(fit_subjects)}/{len(val_subjects)}",
                flush=True,
            )
            for subfold in range(1, SUBFOLDS + 1):
                subval_subjects = [s for s, f in subject_to_subfold.items() if f == subfold]
                subval = fit_indices[
                    data.metadata.iloc[fit_indices]["subject"].astype(str).isin(subval_subjects).to_numpy()
                ]
                basefit = fit_indices[
                    ~data.metadata.iloc[fit_indices]["subject"].astype(str).isin(subval_subjects).to_numpy()
                ]
                if set(data.metadata.iloc[basefit]["subject"].astype(str)) & set(
                    data.metadata.iloc[subval]["subject"].astype(str)
                ):
                    raise RuntimeError("nested base fit/subval subject overlap")
                filled, _ = source.fit_imputation(data.summary, basefit)
                weights = source.training_weights(data.metadata, basefit)
                seed_fold = meta_fold * 10 + subfold
                m2 = source.fit_extra_trees(
                    filled, data.truth, basefit, subval, weights, config, outer_fold, seed_fold
                )
                m3 = source.fit_lgbm_points(
                    filled, data.truth, basefit, subval, weights, config, outer_fold, seed_fold
                )
                m4 = source.fit_hist_points(
                    filled, data.truth, basefit, subval, weights, config, outer_fold, seed_fold
                )
                local = np.asarray([global_to_local[int(g)] for g in subval], dtype=int)
                fit_predictions[local, 0] = m2
                fit_predictions[local, 1] = m3
                fit_predictions[local, 2] = m4
                print(
                    f"  subfold={subfold} basefit={len(basefit)} subval={len(subval)} complete",
                    flush=True,
                )
            if not np.isfinite(fit_predictions).all() or not np.isfinite(validation_predictions).all():
                raise RuntimeError("nested base prediction coverage failed")
            np.savez_compressed(
                path,
                outer_fold=np.asarray([outer_fold], dtype=np.int16),
                meta_fold=np.asarray([meta_fold], dtype=np.int8),
                fit_indices=fit_indices.astype(np.int32),
                validation_indices=validation_indices.astype(np.int32),
                fit_base_predictions=fit_predictions.astype(np.float64),
                validation_base_predictions=validation_predictions.astype(np.float64),
                fit_subjects=np.asarray(sorted(fit_subjects)),
                validation_subjects=np.asarray(sorted(val_subjects)),
                subfold_count=np.asarray([SUBFOLDS], dtype=np.int8),
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "meta_fold": meta_fold,
                    "fit_events": len(fit_indices),
                    "validation_events": len(validation_indices),
                    "fit_subjects": len(fit_subjects),
                    "validation_subjects": len(val_subjects),
                    "cache_path": str(path),
                    "reused": False,
                }
            )
    table = pd.DataFrame(rows).sort_values(["outer_fold", "meta_fold"])
    table.to_csv(TABLE_PATH, index=False, encoding="utf-8-sig")
    summary = {
        "status": "ok",
        "contexts": len(table),
        "subfold_count_within_meta_fit": SUBFOLDS,
        "fit_events_total_across_contexts": int(table["fit_events"].sum()),
        "validation_events_total_across_contexts": int(table["validation_events"].sum()),
        "elapsed_seconds": float(time.time() - started),
        "cache_dir": str(OUT_DIR),
        "boundary": (
            "For every outer/meta context, meta-validation subjects are permanently excluded from all newly fitted base models. "
            "Meta-fit residual labels use subject-disjoint base cross-fitting."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

