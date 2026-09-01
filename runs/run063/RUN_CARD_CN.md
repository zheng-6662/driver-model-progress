# Run63 Research Card

## Status

- Machine/research status: `no_go`
- Original local directory: `run63_protected_residual_and_soft_gating_20260829`

## One-line conclusion

低秩残差和软门控存在小信号，但没有达到冻结晋级门。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`floor_ratio_by_support_class.csv`](artifacts/floor_ratio_by_support_class.csv)
- [`paired_bootstrap_subject.csv`](artifacts/paired_bootstrap_subject.csv)
- [`relative_mae_by_amplitude_bin.csv`](artifacts/relative_mae_by_amplitude_bin.csv)

## Published figures

- [`Figure_1_model_comparison.png`](figures/Figure_1_model_comparison.png)

## Source code and contracts

- [`LITERATURE_REQUEST_CN.md`](../../source/run063/LITERATURE_REQUEST_CN.md)
- [`notes.txt`](../../source/run063/notes.txt)
- [`plot.py`](../../source/run063/plot.py)
- [`post_result_harm_tolerance_audit.py`](../../source/run063/post_result_harm_tolerance_audit.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
