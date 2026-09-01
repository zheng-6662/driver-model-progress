---
name: 新进度记录协议默认规则
description: 2026-07 起进度默认记录到 05_rebuild_from_raw_20260511\00_project_notes 看板；更早的 reports/progress 结构和 project_progress_master.md 均已停更，只作历史。
type: feedback
originSessionId: c9b5819d-a339-47f7-a8fd-7c3ffdc6b8da
modified: 2026-07-26T10:57:11.379Z
---
本项目进度记录地址经历了两次迁移，当前有效的是第三代：

1. （已停用）`04_project_logs\reports\project_progress_master.md` 旧总档流水。
2. （已停用，停更于 2026-05-12）`04_project_logs\reports\progress\` 的 daily / decision_log / experiment_registry / hub 结构。
3. （当前）`05_rebuild_from_raw_20260511\00_project_notes\` 看板体系，见 [[project-log-location-rebuild-20260511]]：
   - `PROJECT_STATUS_CN.md` 状态看板（倒序追加，最新在顶部）
   - `TASK_QUEUE_CN.md` 任务队列
   - `ARTIFACT_INDEX_CN.md` 产物索引
   - `SERVER_RUNS_CN.md` 服务器运行记录
   - 单实验详情写在 `03_baselines\vNNN_*\reports\`，用户摘要写在 `09_reports\*_user_summary_cn.md`

**Why:** 用户于 2026-07-26 明确确认 `05_rebuild_from_raw_20260511\00_project_notes` 是目前最新的项目日志地址；此前两代结构都已实际停更。

**How to apply:** 完成实质工作后，把状态更新以倒序追加方式写入 `PROJECT_STATUS_CN.md` 顶部（沿用其现有小节格式：当前阶段/当前完成/最近一次结果/当前判断/当前最大风险/下一步/用户可优先查看），并同步 `TASK_QUEUE_CN.md`。不要再写回前两代日志；人工记录重点写专业结论、白话解释和判断，不重复手抄指标文件已有内容。
