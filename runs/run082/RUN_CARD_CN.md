# Run82 Research Card

## Status

- Machine/research status: `mechanism_yes_model_no_go`
- Original local directory: `run82_lgrs_sequence_model_20260831`

## One-line conclusion

LGRS稳定超过Role-TCN，但显著落后ExtraTrees；机制增量成立，主模型晋级失败。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`additional_mechanism_comparisons.csv`](artifacts/additional_mechanism_comparisons.csv)
- [`aggregate_metrics.csv`](artifacts/aggregate_metrics.csv)
- [`decision.json`](artifacts/decision.json)
- [`lag_perturbation.csv`](artifacts/lag_perturbation.csv)
- [`metrics_by_domain.csv`](artifacts/metrics_by_domain.csv)

## Published figures

- [`Figure_1_model_comparison.png`](figures/Figure_1_model_comparison.png)
- [`Figure_3_domain_comparison.png`](figures/Figure_3_domain_comparison.png)
- [`Figure_4_lag_perturbation.png`](figures/Figure_4_lag_perturbation.png)

## Source code and contracts

- [`config.json`](../../source/run082/config.json)
- [`data.py`](../../source/run082/data.py)
- [`experiment.py`](../../source/run082/experiment.py)
- [`model.py`](../../source/run082/model.py)
- [`notes.txt`](../../source/run082/notes.txt)
- [`plot.py`](../../source/run082/plot.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
