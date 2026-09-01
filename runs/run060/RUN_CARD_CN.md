# Run60 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run60_gbm_vs_noise_floor_20260828`

## One-line conclusion

LightGBM/HistGBM在同一静态摘要上没有稳定超过ExtraTrees。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`decision.json`](artifacts/decision.json)
- [`floor_ratio_by_support_class.csv`](artifacts/floor_ratio_by_support_class.csv)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)

## Published figures

- 本轮未发布图片。

## Source code and contracts

- [`validate_results.py`](../../source/run060/validate_results.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
