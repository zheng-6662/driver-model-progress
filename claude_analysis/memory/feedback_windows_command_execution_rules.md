---
name: prefer stable windows command execution rules for Claude and Codex
description: Default to safer Windows command execution patterns in this repo: avoid long inline PowerShell, prefer pwsh -File or predict2 Python, and reduce encoding/quoting issues.
type: feedback
originSessionId: b23d4ec3-976d-4030-a899-183a2a44bd99
---
Prefer treating the following as the default command-execution rules in this project for both Claude work and Codex handoffs: avoid long inline PowerShell `-Command` chains, prefer `pwsh -File` with scripts under `tmp/*.ps1` for multi-step shell logic, use UTF-8-safe PowerShell defaults when scripts are needed, and prefer `conda run -n predict2 python ...` for file-processing or Python-side tasks.

**Why:** The user wants these rules to apply by default so they do not need to repeat them, and this reduces Windows encoding noise, quoting failures, and broken inline command construction.

**How to apply:** When I execute commands myself, default to these rules unless the task clearly needs something else. When preparing Codex handoffs, execution briefs, or command suggestions, explicitly include the same rules instead of assuming Codex shares Claude memory automatically. For complex shell tasks, avoid long inline PowerShell and prefer checked script files or Python in `predict2`.
