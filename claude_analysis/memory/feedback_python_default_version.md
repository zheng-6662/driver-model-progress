---
name: prefer-predict2-conda-environment-and-gpu-for-model-runs
description: "Prefer running Python work in the conda predict2 environment, and prefer GPU execution for model programs in this project."
metadata:
  node_type: memory
  type: feedback
  originSessionId: 71952f25-eac7-403e-8913-9d0b24f9702e
  modified: 2026-07-26T15:42:34.038Z
---

Prefer using the conda `predict2` environment whenever Claude or Codex needs Python in this project, and prefer GPU execution when running model programs.

**Why:** The user has already installed the needed libraries in `predict2`, so using that environment avoids repeated package installation and reduces environment friction. For model runs, GPU is the preferred default.

**How to apply:** When I run Python commands, prefer `conda run -n predict2 python ...` or another explicit `predict2` activation path. When preparing Codex handoffs or command suggestions, default to `predict2`. If the task involves training, evaluation, or other model execution, prefer the GPU path/config by default, unless the user explicitly asks for CPU or the environment/script constraints require otherwise.

**Verified on 2026-07-26 (local Windows machine):** `conda` is NOT on PATH in Claude's shells, and the environment's real name is `predict_2` (underscore), not `predict2`. The working invocation is:

`& "<PYTHON_ENV>\python.exe" script.py`

Other envs present: clip_dg, gate0_reprocess_20260710, test1. On the AutoDL server the name may differ — this note is about the local machine only.
