# Run71 原始 EEG 因果状态缓存

本目录是只追加、无模型训练的原始 EEG 特征缓存。正式定义见 `CONTRACT_CN.md`，机器可读冻结项见 `config.json`。

固定执行顺序：

1. `<PYTHON_ENV>\python.exe build_cache.py --mode smoke`
2. 检查 `smoke/decision.json` 必须为 `pass`。
3. `<PYTHON_ENV>\python.exe build_cache.py --mode full`

脚本拒绝覆盖任何同名产物。若需要重新试验，应创建新的 Run 目录，不能删除或改写本目录已经形成的证据。

正式输出：

- `cache/eeg_state_main_shifted.npz`：2323 条事件；Unicode `event_uid`；主窗/移位窗各 46 维；主窗 11 维质量；活跃标志、时间支持与源清单 provenance。
- `tables/event_coverage.csv`：逐事件可用性、ROI 覆盖、端点滞后、warmup、因果支持和不活跃原因。
- `tables/recording_audit.csv`：逐录制时间对齐、fs、断点、滤波和处理状态。
- `outputs/decision.json`：机器裁决和完整计数。
- `RESULT_CN.md`：中文结果与边界。

注意：`primary_release_s` 只作为 P_full 宽表中的遗留列名读取；Run71 新产物统一称 `prediction_anchor_s`（预测起点）。

