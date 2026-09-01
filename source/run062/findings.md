# Run62 Research Findings

## Tested claim

幅值—归一化形状因子化、8个 release 前控制相位特征，以及固定25%保守融合，能否在 P_full=2323 的被试互斥五折上超过冻结 B_all3。

## Result

- 表示能力通过：8节点 PCHIP 的 oracle 重构平均 MAE `0.3453°`、P95 `1.0237°`。
- 学习后的因子化模型失败：B_all3 `12.6151°`，F `15.5134°`，被试bootstrap差值区间整体为负。
- 8个相位特征无增量：R `15.5772°`，相对F变化 `-0.0638°`，区间跨0。
- 固定25%融合未保护主指标：R_blend25 `12.7534°`，相对B_all3变化 `-0.1383°`，区间跨0。
- 峰值误差和 r<0 比例有局部移动，但不足以支持完整曲线改进或保护性改进。

## Verdict

`NO_EFFECTIVE_INCREMENT_FINAL`；二次 result-to-claim 判断为 `claim_supported=no`、`confidence=high`。

## Route

- 不在 Run62 合同上重试、调节点、调融合权重或扩ExtraTrees容量。
- 相位门失败，后续 TCN 形状头不授权。
- 若未来研究峰值/终点取舍，必须另立新目标和新合同，不能称作 Run62 的延续。
