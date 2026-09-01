---
name: Codex CLI 先查本地路径
description: 在本项目中涉及 Codex 协作时，默认优先走 D:\ClaudeCode\codex-bridge 下的本地桥接入口，而不是先假设 PATH 中存在 codex。
type: feedback
originSessionId: cfb89fc1-5f57-4867-8adc-67a62bebf77d
---
在本项目中涉及 Codex 协作时，默认优先走 `D:\ClaudeCode\codex-bridge` 下的本地桥接入口，而不是先假设 PATH 中存在 `codex`。

**Why:** 用户希望后续会话里 Claude 直接通过已验证可用的 codex-bridge 调用 Codex，避免再次因为 PATH 中没有 `codex` 而误判为不可用。
**How to apply:** 以后只要用户提到让 Claude 调 Codex、本地 Codex、CLI 调用 Codex，在本项目里默认优先使用 `D:\ClaudeCode\codex-bridge\codex.exe`、`D:\ClaudeCode\codex-bridge\invoke-codex.ps1` 或 `D:\ClaudeCode\codex-bridge\run-codex.cmd`；只有当用户明确要求检查其他安装方式时，才再去查 PATH 或其他位置。
