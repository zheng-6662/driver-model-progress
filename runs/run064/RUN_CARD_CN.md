# Run64 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run64_physio_style_regret_distillation_20260829`

## One-line conclusion

生理、驾驶风格、TCN、FiLM、BIOT等训练侧筛选没有稳定独立增量。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`floor_ratio_by_support_class.csv`](artifacts/floor_ratio_by_support_class.csv)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)

## Published figures

- [`Figure_1_model_comparison.png`](figures/Figure_1_model_comparison.png)

## Source code and contracts

- [`experiment.py`](../../source/run064/experiment.py)
- [`RESULT_CN.md`](../../source/run064/RESULT_CN.md)
- [`build_multimodal_features.py`](../../source/run064/scripts/build_multimodal_features.py)
- [`build_physio_200hz_biot.py`](../../source/run064/scripts/build_physio_200hz_biot.py)
- [`build_physio_sequences.py`](../../source/run064/scripts/build_physio_sequences.py)
- [`build_post_physio_sequences.py`](../../source/run064/scripts/build_post_physio_sequences.py)
- [`build_recent_style_features.py`](../../source/run064/scripts/build_recent_style_features.py)
- [`extract_biot_embeddings.py`](../../source/run064/scripts/extract_biot_embeddings.py)
- [`screen_biot_inner.py`](../../source/run064/scripts/screen_biot_inner.py)
- [`screen_inner_candidates.py`](../../source/run064/scripts/screen_inner_candidates.py)
- [`screen_inner_residual_teacher.py`](../../source/run064/scripts/screen_inner_residual_teacher.py)
- [`screen_inner_uncertainty.py`](../../source/run064/scripts/screen_inner_uncertainty.py)
- [`screen_post_teacher.py`](../../source/run064/scripts/screen_post_teacher.py)
- [`screen_recent_style.py`](../../source/run064/scripts/screen_recent_style.py)
- [`screen_structured_targets.py`](../../source/run064/scripts/screen_structured_targets.py)
- [`screen_tcn_inner.py`](../../source/run064/scripts/screen_tcn_inner.py)
- [`screen_warm_memory.py`](../../source/run064/scripts/screen_warm_memory.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
