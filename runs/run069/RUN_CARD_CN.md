# Run69 Research Card

## Status

- Machine/research status: `changed_estimand_guardrail_fail`
- Original local directory: `run69_rolling_vehicle_outer_oof_20260830`

## One-line conclusion

t0+0.4s滚动车辆更新显著改善尾部，但改变估计目标且道路缺失保护门失败。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)

## Published figures

- 本轮未发布图片。

## Source code and contracts

- [`CONTRACT_CN.md`](../../source/run069/CONTRACT_CN.md)
- [`README_CN.md`](../../source/run069/README_CN.md)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
