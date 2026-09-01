# 第一版输出已由 corrected_run_2 取代

根目录中的第一版 `inventory_files.csv`、`inventory_sessions.csv`、`external_subject_summary.csv`、`RESULT_CN.md`、`decision.json` 保留不删，以遵守 append-only 约束，但**不得用于结论**。

复核发现 5 个 raw 文件含明显损坏的 `StorageTime`，包括年份 `3994`。第一版使用原始极值计算 duration，导致少数 session 的 union duration 和 longitudinal-style 前序关系被单个坏时间戳污染。

纠正版没有删除原始时间证据，而是在 file 表中同时保存：

- `storage_time_raw_min/max`：原始字符串极值；
- `storage_time_implausible_count`：与 recording 日期不一致或时间格式非法的数量；
- `storage_time_min/max`：与 recording 日期一致且时间格式合法的 plausible 极值；
- `duration_s` 与所有 effective rate：只使用 plausible 极值计算。

权威输出位于：

- `corrected_run_2/inventory_files.csv`
- `corrected_run_2/inventory_sessions.csv`
- `corrected_run_2/external_subject_summary.csv`
- `corrected_run_2/RESULT_CN.md`
- `corrected_run_2/decision.json`

纠正版 session span 范围为约 61.143–483.839 s，不再存在数十亿秒的伪时长。
