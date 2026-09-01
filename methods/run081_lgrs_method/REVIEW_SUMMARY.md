# Review Summary

**Problem：** release时刻只用历史信息，跨被试预测未来1秒20点完整转向曲线。  
**Initial Approach：** 从静态树模型转向原始时序关系表征，并考虑生理/风格条件融合。  
**Date：** 2026-08-31  
**Rounds：** 3 / 5  
**Final Score：** 9.2 / 10  
**Final Verdict：** READY

## Problem Anchor

- 底线任务始终保持为release-time完整曲线预测。
- 关键瓶颈始终是134/172维摘要丢失控制相位和跨通道指令—响应时滞。
- 没有引入release后观察、subject ID或测试真值。

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | Simplified / Modernized | Solved? | Remaining Risk |
|---:|---|---|---|---|
| 1 | CRSE可能只是换名TCN；误写200Hz/401点；physio/style和数据claim过多 | 改为真实50Hz/101点；自由卷积收紧为fixed-lag low-rank gain；删除首轮physio/style与第二claim | partial | 关系损失和生成头仍有实现分叉 |
| 2 | `L_rel`空间不清；gain/bias head未写死；八月序列对齐需冻结 | 固定normalized response监督；单层线性gain/bias；Run57/Run76同函数101点锚定 | yes | 只剩工程实现与真实实验结果 |
| 3 | 检查最终歧义、drift和复杂性 | 不再加模块；READY | yes | LGRS必须真实超过参数配平Role-TCN |

## Overall Evolution

- 从“多尺度TCN+关系卷积+生理/style FiLM”收缩为一个固定时滞、rank-2 gain的LGRS瓶颈。
- 从错误的200Hz/401点回到现有50Hz/101点缓存，避免采样率成为混杂变量。
- 把plain Role-TCN设为参数配平主对照，确保实验能判断relation bottleneck本身。
- 把生理、风格、自监督、基础模型和数据平衡全部移出主贡献。
- 将失败条件写死：LGRS不稳定超过Role-TCN就关闭，不换backbone挽救。

## Final Status

- Anchor status：preserved。
- Focus status：tight；单一LGRS贡献。
- Modernity status：appropriately frontier-aware；拒绝不必要基础模型堆叠。
- Strongest parts：fixed-lag physical interface、low-rank conditional gain、response residual bottleneck、matched control、hard stop rule。
- Remaining weakness：尚无实验结果；论文价值完全依赖LGRS vs Role-TCN的稳定优势。

