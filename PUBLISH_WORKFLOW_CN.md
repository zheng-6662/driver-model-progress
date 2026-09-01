<!-- STIMULUS_CENTERED_MULTIACTION_V2 -->
# 公开进展发布流程

## 总体目标保护

顶层文档中的 `STIMULUS_CENTERED_MULTIACTION_V2` 标记表示当前正式叙事是修正后的“刺激中心、个体化、多动作驾驶员反应”。主项目发布脚本检测到该标记后，不覆盖这些人工维护的顶层事实文件；Run卡、聚合结果和公开源代码仍可正常更新。

## 当前多动作审计再生成

使用主项目中的只读审计脚本：

```powershell
<PYTHON_311>/python.exe <PROJECT_ROOT>/02_code/tools/build_multiaction_reframe_audit.py `
  --project-root <PROJECT_ROOT> `
  --august-root <AUGUST_RAW_ROOT> `
  --public-root <PUBLIC_PROGRESS_ROOT> `
  --config <PROJECT_ROOT>/02_code/tools/multiaction_reframe_audit_config.json `
  --output-root <PROJECT_ROOT>/review_packages/MULTIACTION_REFRAME_20260901
```

该脚本只读原始数据，不训练模型；私有事件级和驾驶员级表只写入主项目审查包，公开目录只写聚合表、分位数、报告和公开安全图。

## 历史Run发布

```powershell
<PYTHON_311>/python.exe <PROJECT_ROOT>/02_code/tools/publish_gptpro_progress.py --update
```

发布器更新 Run 卡、聚合产物和 `RUN_INDEX.md`。它不会自动执行 `git add`、`git commit` 或 `git push`。

## 推送前检查

- 所有Markdown相对链接可解析；
- JSON和CSV可解析；
- 不含本机根路径、访问凭据、原始文件清单；
- 不含一行一事件、一行一驾驶员或匿名ID对应行；
- 不含原始车辆、生理、EEG、眼动、视频、checkpoint或逐事件预测；
- 本轮顶层目标仍是刺激中心多动作任务；
- Run57—Run82仍被标注为历史release转向子任务；
- 公开审计隐私检查与完整性检查通过。

## Git提交边界

只显式添加本轮公开文件，禁止使用 `git add -A`。主项目脏工作区与公开仓库保持独立，私有审查包不得加入公开仓库。
