---
name: 运行代码固定用 predict2 和 GPU
description: 用户明确要求后续只要运行代码，就调用 conda 环境 predict2 的 python，并优先使用 GPU，不要走 CPU。
type: feedback
originSessionId: af2308aa-8c25-4ac5-a4c7-ee67647fd7bd
modified: 2026-07-26T15:25:46.627Z
---
在本项目中，只要需要运行代码，就应调用 conda 环境 `predict_2` 的 Python，并使用 GPU，不要走 CPU。

**Why:** 用户说明该环境已经配重好，直接使用能避免环境漂移和依赖不一致；模型相关执行默认应走 GPU。

**How to apply:** 环境实名是 `predict_2`（带下划线，不是 `predict2`），解释器在 `<PYTHON_ENV>\python.exe`，conda 在 `D:\ProgramData\anaconda3\Scripts\conda.exe`。`conda` 不在 PATH 里，Bash/PowerShell 里直接敲 `conda run -n predict2 ...` 会失败——用绝对路径调解释器，或 `& "D:\ProgramData\anaconda3\Scripts\conda.exe" run -n predict_2 python ...`。同机还有 `gate0_reprocess_20260710`（Gate 0 专用隔离环境）与 `predict_1`。只有在环境不可用或用户明确要求 CPU 时才例外。
