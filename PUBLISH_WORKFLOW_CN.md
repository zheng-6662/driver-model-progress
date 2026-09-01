# 后续每轮实验发布流程

## 目标

每个新Run完成后，把GPTPro真正需要的项目状态、结果卡、聚合数字、关键图片和源代码更新到本仓库；不上传原始数据、缓存、checkpoint、日志或逐事件预测。

## 本地生成器

主项目中的生成脚本：

```text
<PROJECT_ROOT>/02_code/tools/publish_gptpro_progress.py
```

首次创建后，后续必须显式使用：

```powershell
<PYTHON_311>/python.exe <PROJECT_ROOT>/02_code/tools/publish_gptpro_progress.py --update
```

生成器只更新公开副本，不执行 `git add`、`git commit` 或 `git push`。

## 新Run登记

每个新Run完成后，在生成脚本中增加：

1. `RUN_SPECS`：本地Run目录和正式结果相对路径；
2. `RUN_SUMMARIES`：状态与一句话结论；
3. 如有必要，更新 `GPTPRO_CONTEXT_CN.md` 的当前最强模型、最新结果和关闭方向。

## 每次推送前检查

必须确认：

- `RUN_CARD_CN.md` 与 `RESULT_CN.md` 存在；
- Markdown相对链接可解析；
- JSON可解析；
- 不含本机绝对路径；
- 不含访问凭据；
- 不含原始车辆/生理/EEG/眼动数据；
- 不含 `predictions/`、`cache/`、`logs/`、checkpoint；
- 不含逐事件或逐被试身份表；
- 公开仓库总增量合理。

## 提交顺序

```powershell
git status --short
git add <明确选择的公开文件>
git commit -m "Add RunXX GPTPro research update"
git push origin main
```

禁止在原始项目脏工作区执行 `git add -A`。公开仓库与原项目仓库必须保持独立。

## GPTPro固定入口

每次更新后，GPTPro仍从同一个文件开始：

```text
GPTPRO_CONTEXT_CN.md
```

随后读取：

```text
CURRENT_STATUS_CN.md
RUN_INDEX.md
REQUEST_TO_GPTPRO_CN.md
```

