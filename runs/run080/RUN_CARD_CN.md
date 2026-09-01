# Run80 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run80_clean_physio_ab_20260831`

## One-line conclusion

正式清洗生理16维仍未超过车辆或旧生理特征。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)

## Published figures

- [`Figure_1_model_comparison.png`](figures/Figure_1_model_comparison.png)
- [`Figure_3_amplitude_relative_mae.png`](figures/Figure_3_amplitude_relative_mae.png)

## Source code and contracts

- [`config.json`](../../source/run080/config.json)
- [`experiment.py`](../../source/run080/experiment.py)
- [`notes.txt`](../../source/run080/notes.txt)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
