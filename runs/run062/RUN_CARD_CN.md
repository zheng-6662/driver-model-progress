# Run62 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run62_amplitude_shape_factorized_20260829`

## One-line conclusion

幅值—形状因子化和8个控制相位标量没有形成有效增量。

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

- [`Figure_01_factorized_model_comparison.png`](figures/Figure_01_factorized_model_comparison.png)

## Source code and contracts

- [`findings.md`](../../source/run062/findings.md)
- [`validate_results.py`](../../source/run062/validate_results.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
