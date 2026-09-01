# Run63：受保护低秩残差与可靠性软门控

## 结论先行

- 科学状态：NO_EFFECTIVE_INCREMENT_FINAL。
- 选中模型：None。
- 有效增量候选：[]。
- 达到 dense floor_ratio <= 2.01 的候选：[]。
- 本轮是同一 P_full=2323 上的发展性 OOF，不是独立确认。

## 主指标：逐幅值档 subject-macro relative MAE

| 模型 | 10_20 | 20_30 | 30_45 | 45_70 | ge70 |
|---|---:|---:|---:|---:|---:|
| B_all3 | 0.4843 | 0.3377 | 0.3512 | 0.3435 | 0.3133 |
| R_lowrank_residual | 0.4820 | 0.3370 | 0.3495 | 0.3412 | 0.3043 |
| G_soft_expert_gate | 0.4868 | 0.3364 | 0.3485 | 0.3433 | 0.3071 |

release幅值只在所有OOF预测完成后用于报告；未进入模型输入、权重、候选选择或事件路由。

## floor_ratio：三个支持域

| 模型 | dense_overlap | borderline | out_of_support |
|---|---:|---:|---:|
| B_all3 | 2.0816 | 2.1564 | 3.5635 |
| R_lowrank_residual | 2.0813 | 2.1309 | 3.4670 |
| G_soft_expert_gate | 2.0739 | 2.1346 | 3.5105 |

## 三条正式信息门与保护门

| 候选 | dense改善 | 宏MAE改善° | dense bootstrap 95%CI | 去最大贡献被试后 | 幅值保护 | 信息门 | 部署门 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R_lowrank_residual | +0.0003 | +0.0853 | [-0.0050, +0.0062] | -0.0019 | True | False | False |
| G_soft_expert_gate | +0.0077 | +0.0880 | [+0.0015, +0.0145] | +0.0055 | True | False | False |

每个候选必须同时通过：dense改善至少0.02、subject-macro MAE改善至少0.05°、被试bootstrap下界大于0、去掉最大贡献被试后仍改善、五个幅值档均不退化超过0.01。

## 绝对MAE参考与样本伤害

| 模型 | subject-macro MAE° | pooled MAE° |
|---|---:|---:|
| B_all3 | 12.6151 | 13.1846 |
| R_lowrank_residual | 12.5298 | 13.0936 |
| G_soft_expert_gate | 12.5272 | 13.0928 |

| 候选 | 事件恶化比例 | 恶化被试数 | 改善被试数 | 最差被试MAE退化° |
|---|---:|---:|---:|---:|
| R_lowrank_residual | 0.190 | 0 | 8 | 0.0000 |
| G_soft_expert_gate | 0.351 | 2 | 12 | 0.0226 |

被试伤害按与逐事件相同的 `1e-12°` 数值容差复核。冻结生产表 `harm_summary.csv` 直接按 `<0` 计数，把精确回退产生的约 `1e-16°` 浮点差记成了恶化；结果后只读复核见 `tables/harm_summary_tolerance_audit.csv` 与 `audit/harm_tolerance_audit.json`。该修正不改变任何预测、bootstrap、门值或 `decision.json`。

## 各外折训练侧选择

| 分支 | outer fold | 选中候选 |
|---|---:|---|
| G_soft_expert_gate | 1 | G_ridge_t01_s025 |
| G_soft_expert_gate | 2 | G_ridge_t03_s025 |
| G_soft_expert_gate | 3 | G_ridge_t03_s025 |
| G_soft_expert_gate | 4 | G00_fallback |
| G_soft_expert_gate | 5 | G_ridge_t01_s025 |
| R_lowrank_residual | 1 | R00_fallback |
| R_lowrank_residual | 2 | R00_fallback |
| R_lowrank_residual | 3 | R_ridge_k5_a025 |
| R_lowrank_residual | 4 | R00_fallback |
| R_lowrank_residual | 5 | R_ridge_k5_a025 |

## 训练侧残差几何

- inner meta-fit 残差PCA累计解释率中位数：K1=0.835、K2=0.957、K3=0.981、K5=0.995。
- PCA方向和裁剪边界每次只在当前meta-fit事件上估计；残差均值明确不加入最终修正。
- 低秩只说明误差几何可压缩，不证明残差系数能由预测起点输入识别。

## 完整性

- P_full=2323、18被试、85 recordings；外折计数352/471/435/539/526未改变。
- 每个外折训练集按被试划成3个inner folds；每个训练事件恰好获得一次M2/M3/M4 inner OOF预测。
- outer测试被试不进入该outer上下文的任何inner fit、PCA、缺失中位数、残差头、损失头或候选选择。
- M2/M3/M4基模型超参数固定自Run57/Run60既有训练侧选择；Run63未搜索基模型家族。
- alpha=0与等权门控是精确B_all3回退；外折结果打开后不调rank、alpha、温度、收缩或门槛。
- 未读、未跑、未改任何verify_*.py；P0门记录为EXTERNAL。

## 证据边界

Run59 floor_ratio分母是d5→0 OLS截距的高斯RMSE→MAE近似，不是因果下限或理论不可约噪声。Run63是在同一2323事件集上根据既有结果设计的发展性二层组合，不能当作独立确认。模型分歧、残差PCA和门控损失都不构成理论不可预测性证明。

运行时长：47.1分钟。
