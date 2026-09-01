# Run61：道路预览残差修正诊断

## 最终状态与主结论

- 最终科学状态：`DIAGNOSTIC_ONLY`。
- 该状态由卡 E 的 `diagnostic_only=true` 硬覆盖；无论下列数值门结果如何，本轮都不得声称道路预览构成有效增量或可部署增量。
- 下列 M5/M6 结果仅用于诊断车辆侧已高度隐含弯道条件下，冻结道路编码与残差修正器呈现的关联结构。

### 主指标与双门（仅诊断审计）

| 模型 | subject-macro MAE°（参考） | dense floor_ratio | 相对B_all3改善 | 信息门a-d | 部署门e-f |
|---|---:|---:|---:|---:|---:|
| B_all3 | 12.6151 | 2.0816 | 0.0000 | — | — |
| M5_preview | 12.7743 | 2.1296 | -0.0480 | False | False |
| M6_preview_only | 12.8560 | 2.1220 | -0.0404 | False | False |

- 卡 E 的 OOF AUC：`0.9742`；判读档：`vehicle_side_already_implies_curve_diagnostic_only`。
- 忽略 E 的前提失败、只机械读取冻结门时，M5 的状态会是 `NO_EFFECTIVE_INCREMENT_FINAL`；这不是本轮科学结论。

## floor_ratio：三个支持域

| 模型 | dense_overlap | borderline | out_of_support |
|---|---:|---:|---:|
| B_all3 | 2.0816 | 2.1564 | 3.5635 |
| M5_preview | 2.1296 | 2.1480 | 3.4851 |
| M6_preview_only | 2.1220 | 2.1862 | 3.5970 |

dense 使用 Run59 `dense_overlap/curve` 的固定分母；borderline 与 out_of_support 使用 Run59 `all/curve` 的固定分母。本轮只读这些分母，没有重算地板。

## 信息门 a–d 与部署门 e–f

- `a_dense_floor_ratio_improvement_at_least_0_02`：`False`。
- `b_subject_paired_bootstrap_95ci_lower_above_zero`：`False`。
- `c_leave_top_contributing_subject_improvement_still_positive`：`False`。
- `d_curve_improvement_greater_than_straight`：`True`。
- `e_dense_floor_ratio_at_most_2_01`：`False`。
- `f_no_amplitude_bin_regression_over_0_01`：`False`。

- 去掉贡献最大的被试后，M5 dense floor_ratio 改善为 `-0.0530`。
- 弯道 dense 改善为 `+0.1742`，直路 dense 改善为 `-0.0638`。
- M5 距离部署阈值 2.01 尚差 `0.1196`。

这些门值只用于记录。如果 E 卡已判 diagnostic_only，它们不能升级成增量主张。

## 被试级配对 bootstrap

| 候选 | dense改善估计 | 95% CI | CI下界>0 | 去掉最大贡献被试后改善 |
|---|---:|---:|---:|---:|
| M5_preview | -0.0480 | [-0.0714, -0.0255] | False | -0.0530 |
| M6_preview_only | -0.0404 | [-0.0653, -0.0181] | False | -0.0454 |

bootstrap 单位是被试，固定 2000 次，seed=20261469。

## 逐幅值档相对 MAE 与保护门

| 模型 | 10_20 | 20_30 | 30_45 | 45_70 | ge70 |
|---|---:|---:|---:|---:|---:|
| B_all3 | 0.4843 | 0.3377 | 0.3512 | 0.3435 | 0.3133 |
| M5_preview | 0.5099 | 0.3463 | 0.3552 | 0.3435 | 0.3148 |
| M6_preview_only | 0.5169 | 0.3481 | 0.3556 | 0.3491 | 0.3109 |

幅值保护检查覆盖全部五档；任一档相对 B_all3 退化超过 0.01 即门 f 失败。ge70 的 96 个事件/14 名被试脆弱性已登记，但没有因此修改阈值。

## 道路预览可用率与结构性偏斜

- 全部 P_full 锚点的可用率是 `0.7176`，分子/分母为 `1667/2323`。
- Run59 map 报告的约 `0.753` 来自 `1609/2136` 真值可用子集；它与 `0.7176=1667/2323` 不是同一个分母，不能混用。
- 可用率在外折、被试与支持域间存在结构性偏斜；定位还含公里级 SetupPoint 别名错误证据。因此预览不是无误差真值，且没有删掉任何不可用事件。

## 弯道与直路诊断

门 d 只在 preview_available=1 且 dense_overlap 的事件上比较弯道与直路 subject-macro floor_ratio 改善；弯道标签沿用卡 E/F 的冻结定义。

## curve3 模块（单独报告）

| 模型 | 事件数 | 被试数 | subject-macro MAE°（参考） | subject-macro floor_ratio | 相对 B_all3 改善 |
|---|---:|---:|---:|---:|---:|
| B_all3 | 134 | 17 | 10.1720 | 1.7805 | +0.0000 |
| M5_preview | 134 | 17 | 10.6969 | 1.8481 | -0.0675 |
| M6_preview_only | 134 | 17 | 10.3806 | 1.8132 | -0.0327 |

本段只描述该模块的 OOF 误差；由于 E 卡已把整轮判为诊断轮，这里不得转写为增量结论。

## ZD 模块（单独报告）

| 模型 | 事件数 | 被试数 | subject-macro MAE°（参考） | subject-macro floor_ratio | 相对 B_all3 改善 |
|---|---:|---:|---:|---:|---:|
| B_all3 | 63 | 12 | 11.5144 | 1.8662 | +0.0000 |
| M5_preview | 63 | 12 | 12.2329 | 1.9737 | -0.1075 |
| M6_preview_only | 63 | 12 | 11.6174 | 1.8784 | -0.0122 |

本段只描述该模块的 OOF 误差；由于 E 卡已把整轮判为诊断轮，这里不得转写为增量结论。

## 模型、选参与缺失处理完整性

- M5 是 B_all3 的 20 点残差修正器，输入为 172 维车辆摘要加实际 31 维预览编码；M6 使用同一残差目标但只输入 31 维预览编码。
- 31 维预览编码由 15 个语义特征、1 个 preview_available 和原样保留的 15 个缺失指示组成；没有把 15 个指示丢掉。
- 每次内层选择和 outer-final fit 都只使用相应 fit 行计算 nanmedian；非 fit 行从未参与中位数。
- 两个模型都只使用 4 个冻结候选：learning_rate∈{0.05,0.10} × max_leaf_nodes∈{15,31}；固定参数沿 Run60 M4，early_stopping=False。
- 内层种子是 20260829+outer_fold；任一被试进入内验的折数不超过 2；outer test 不参与选参。
- 选中的 fold×model 组合共 `10` 个；完整候选成绩见 `tables/inner_candidate_metrics.csv`。
- ExtraTrees 没有重训，B_all3 是 Run60 三份已落盘 OOF 的零训练平均；地板分母没有重算；搜索空间没有扩展或重试。
- 禁止字段没有进入成员判定、权重、模型输入、选参、早停或删样。release 幅值只在 OOF 完成后用于正式相对误差报告。

## 时间点结果

20 个 release+0.05…+1.00s 时间点、五个幅值档、三个模型的 subject-macro 相对 MAE 已完整写入 `tables/relative_mae_by_timepoint.csv`；没有据此回头选参。

## 证据边界

`floor_ratio` 的分母是 Run59 OLS d5→0 截距经过高斯 RMSE→MAE 近似后的数值。它不是因果下限，不是理论不可约噪声，也不证明模型能够达到该数值；这里只作为跨轮可比较的尺度无关参照。

更关键的是，卡 E 已证明车辆侧特征高度隐含 is_curve，故本轮是诊断轮；M5/M6 的任何门值都不能被写成道路预览提供了新的独立信息。

运行时长：`50.8` 分钟。
