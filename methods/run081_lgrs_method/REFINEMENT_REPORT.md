# Refinement Report

**Problem：** 改造当前树模型的整体思考结构和输入表征，使release前原始车辆序列的控制相位与跨通道时滞能够进入完整曲线预测。  
**Initial Approach：** 轻量时序编码、幅值/形状结构、生理与驾驶风格条件融合。  
**Date：** 2026-08-31  
**Rounds：** 3 / 5  
**Final Score：** 9.2 / 10  
**Final Verdict：** READY

## Problem Anchor

在车辆失控/近失稳代理事件的release时刻，只使用当时及以前的信息，跨被试预测未来1秒20点完整方向盘响应曲线。必须补足的是静态摘要丢失的控制相位与驾驶指令—车辆响应关系，不是单纯扩大网络。

## Output Files

- Review summary：`REVIEW_SUMMARY.md`
- Final proposal：`FINAL_PROPOSAL.md`
- Round 1 review：`round-1-review.md`
- Round 1 full refinement：`round-1-refinement.md`
- Round 2 review：`round-2-review.md`
- Round 2 full refinement：`round-2-refinement.md`
- Round 3 final review：`round-3-review.md`
- Score history：`score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.2 | 5.9 | 6.3 | 8.1 | 7.9 | 6.7 | 6.1 | 7.1 | REVISE |
| 2 | 9.6 | 8.5 | 8.6 | 8.5 | 9.0 | 8.6 | 8.0 | 8.7 | REVISE |
| 3 | 9.8 | 9.1 | 9.2 | 8.9 | 9.3 | 9.2 | 9.0 | 9.2 | READY |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|---:|---|---|---|
| 1 | 换名TCN风险、采样率合同错误、贡献过宽 | fixed-lag/rank-2瓶颈、50Hz/101点、删除physio/style | 核心方向成立 |
| 2 | 损失空间和生成头仍有实现自由度 | masked `Huber(r_hat,r_tilde)`、单线性gain/bias、148事件逐值锚定 | 实现合同闭合 |
| 3 | 最终drift/复杂性检查 | 不再增加组件 | READY |

## Final Proposal Snapshot

- 共同7通道、release前2秒、50Hz/101点。
- 固定lag `{0,20,40,80,120,160,240 ms}`。
- 事件条件rank-2 gain估计expected ay/roll。
- 下游只能看到command/context和relation residual，不直接看到raw response。
- 主对照是参数配平Role-TCN，不是只和ExtraTrees比较。
- 首轮无生理、风格、自监督、200Hz、结构化输出或release后teacher。

## Method Evolution Highlights

1. 最重要的简化：physiology/style从首轮彻底删除，防止三个贡献并行。
2. 最重要的机制升级：自由关系卷积变为fixed-lag low-rank gain operator。
3. 最重要的可证伪性：Role-TCN matched control与硬stop rule。

## Pushback / Drift Log

| Round | Reviewer / Temptation | Author Response | Outcome |
|---:|---|---|---|
| 1 | 使用更现代的MOMENT/PatchTST/自监督 | 当前瓶颈是跨通道关系，通用预训练增加自由度 | 拒绝进入首轮 |
| 1 | 同时加入physio/style | 既有Run64–Run80未支持；车辆核心先过门 | 从主方法删除 |
| 2 | 重建200Hz可能更精细 | 会混淆采样率与方法；50Hz分辨率已为20ms | 固定50Hz |

## Remaining Weaknesses

- LGRS只是READY proposal，不是已验证结果。
- 数据事件数增加，但独立主体仍只有潜在38位；统计功效有限。
- 若LGRS只超过ExtraTrees而不超过Role-TCN，不能声称LGRS方法贡献。
- 首轮方案有意不解决生理和驾驶风格；它们只能在车辆核心成功后另立课题。

## Raw Reviewer Responses

完整原始文本分别保存在：

- `round-1-review.md`
- `round-2-review.md`
- `round-3-review.md`

## Next Steps

1. 进入实现前实验计划：先做148公共事件101点逐值锚定。
2. 单折smoke：tensor、lag索引、padding、参数量差≤5%。
3. 先跑Role-TCN/LGRS/LGRS-λ0，再补ExtraTrees/Plain Raw-TCN。
4. 严格执行stop rule，不在外折后改结构。

