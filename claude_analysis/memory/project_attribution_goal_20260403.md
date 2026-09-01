---
name: Attribution 归因分析的最终目标
description: 当前归因分析阶段的最终目的不是描述性统计，而是诊断 conditioned v2 tail 退化根因并给出可操作的模型改进方向。
type: project
---

当前工作阶段是 conditioned v2 vs baseline 的 tail 预测退化归因分析。

**最终目标**不是停在"哪些样本好/差"的描述层面，而是要：
1. 定位 conditioned 信号在哪些场景帮了倒忙，诊断出机制（是 anchor 定义问题、信号注入方式问题、还是场景适配问题）
2. 给出可操作的下一步模型改进方向（改 anchor、改 conditioning 方式、对特定场景降权/关闭 conditioning，等等）

**Why:** 模型 tail RMSE 整体改善 56%，但快反应样本恶化、boundary_shift 普遍变大。只看均值会掩盖这些关键退化模式，需要精细归因才能决定改什么。

**How to apply:** 所有归因分析产出（脚本、表格、切片统计）都应当最终指向"改什么、怎么改"，而不是"看起来怎样"。
