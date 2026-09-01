# Run75 Research Card

## Status

- Machine/research status: `ablation`
- Original local directory: `run75_remove_yaw_roll_rate_ablation_20260831`

## One-line conclusion

去除vyaw/vroll后整体MAE影响很小，但尾部/个别被试有差异。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)

## Published figures

- 本轮未发布图片。

## Source code and contracts

- [`run_ablation.py`](../../source/run075/run_ablation.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
