---
name: project-log-location-rebuild-20260511
description: 2026-05-11 起项目主线与日志搬到 05_rebuild_from_raw_20260511，旧 progress 日志与 hub 已停更
metadata:
  node_type: memory
  type: project
  originSessionId: b1ccef57-31dc-4ab5-ba99-4827b92bfa51
  modified: 2026-07-26T11:37:35.033Z
---

自 2026-05-11 起，项目主线切换为"从原始数据重建"，工作区与日志入口移到：

- 根目录：`<PROJECT_ROOT>\05_rebuild_from_raw_20260511\`
- 状态看板：`00_project_notes\PROJECT_STATUS_CN.md`（倒序追加，最新在顶部）
- 任务队列：`00_project_notes\TASK_QUEUE_CN.md`
- 产物索引：`00_project_notes\ARTIFACT_INDEX_CN.md`
- 服务器运行：`00_project_notes\SERVER_RUNS_CN.md`
- 用户摘要：`09_reports\*_user_summary_cn.md`
- 单实验报告：`03_baselines\vNNN_*_YYYYMMDD\reports\*_report_cn.md`（含 guardrail JSON）

旧日志（`04_project_logs\reports\progress\` daily/decision_log 和 project_progress_hub.md）停更于 2026-05-12/04-23，仅作历史。

注意：看板文件本身也可能滞后于实验目录（2026-07 时看板曾停在 5-26，实为 v 系列到 v348/0708、R 系列到 R379/0716）。查最新状态时应同时看 `03_baselines` 下最大 vNNN 和 RNNN 编号目录——编号有 v 系列（锚点+2s 曲线线）和 R 系列（R349 起，含 rolling-0.2s next-1s 在线滚动线）两套。实验多在 AutoDL 服务器跑（`ssh -p 55060 <REDACTED_EMAIL>`，平时关机连不上），v2xx/v3xx/R 系列训练脚本多数只在服务器，本地缺失时可在 `00_project_notes/reports_20260706/gptpro_review_packages/*/05_manifest/package_manifest.csv` 按哈希找脚本副本；本地 git 提交也可能滞后。2026-07-26 起新增 R380（本地跑的不确定性门控，strict pass）。
