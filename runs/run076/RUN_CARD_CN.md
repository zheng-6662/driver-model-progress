# Run76 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run76_august_subject_augmented_training_20260831`

## One-line conclusion

直接把早期148条八月事件加入旧训练使原18人MAE退化约0.35度。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)
- [`screening_result.json`](artifacts/screening_result.json)

## Published figures

- [`Figure_2_amplitude_relative_mae.png`](figures/Figure_2_amplitude_relative_mae.png)

## Source code and contracts

- [`experiment.py`](../../source/run076/experiment.py)
- [`notes.txt`](../../source/run076/notes.txt)
- [`screen_samples.py`](../../source/run076/screen_samples.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
