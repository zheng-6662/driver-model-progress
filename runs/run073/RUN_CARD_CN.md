# Run73 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run73_eye_tracking_increment_20260830`

## One-line conclusion

原18人眼动增量未超过车辆基线。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)

## Published figures

- [`Figure_2_amplitude_relative_mae.png`](figures/Figure_2_amplitude_relative_mae.png)

## Source code and contracts

- [`config.json`](../../source/run073/config.json)
- [`notes.txt`](../../source/run073/notes.txt)
- [`plot.py`](../../source/run073/plot.py)
- [`post_result_analysis.py`](../../source/run073/post_result_analysis.py)
- [`validate.py`](../../source/run073/validate.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
