# Run65 Research Card

## Status

- Machine/research status: `training_side_no_go`
- Original local directory: `run65_multimodal_residual_distillation_20260830`

## One-line conclusion

多模态教师/学生和残差蒸馏训练侧有局部信号，但没有通过进入outer的成对门。

## Reading order

1. [Full Chinese result](RESULT_CN.md)
2. Review the aggregate artifacts below
3. Read the selected source files if judging implementation details

## Published aggregate artifacts

- [`decision.json`](artifacts/decision.json)

## Published figures

- 本轮未发布图片。

## Source code and contracts

- [`experiment.py`](../../source/run065/experiment.py)
- [`experiment_v2.py`](../../source/run065/experiment_v2.py)
- [`build_nested_base_cache.py`](../../source/run065/scripts/build_nested_base_cache.py)

## Evidence boundary

This public card intentionally excludes raw vehicle/physiology/EEG/eye data, caches, checkpoints, local logs, and per-event predictions. Use the full result file for the run-specific scientific boundary. A no-go result must not be reopened without a genuinely different hypothesis.
