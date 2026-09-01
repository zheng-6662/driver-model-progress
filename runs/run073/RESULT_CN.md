# Run73：18位被试眼动数值增量实验

## 结论先行

- 科学状态：`NO_EFFECTIVE_EYE_TRACKING_INCREMENT_FINAL`
- 眼动事件时序增量成立：`False`
- P_full：2323事件、18被试；未加入zyl；视频未进入模型。
- 眼动匹配recording：68/85；眼动可用事件：1857/2323。

## 主指标：subject-macro完整曲线MAE

| 模型 | subject-macro MAE° | head5 MAE° | tail5 MAE° | endpoint MAE° | peak-time MAE s | pooled MAE°参考 |
|---|---:|---:|---:|---:|---:|---:|
| B_all3 | 12.6151 | 4.0721 | 19.0607 | 19.6330 | 0.1989 | 13.1846 |
| B_all3_eye_true | 12.6246 | 4.0835 | 19.0625 | 19.6363 | 0.1997 | 13.2002 |
| B_all3_eye_shift_control | 12.6404 | 4.0802 | 19.0883 | 19.6511 | 0.1993 | 13.2149 |

## 被试配对比较

| 比较 | 改善° | 95%CI | 正向外折 | 改善被试 | 去最大贡献被试后° | 事件恶化比例 |
|---|---:|---:|---:|---:|---:|---:|
| B_all3_eye_true_vs_B_all3 | -0.0094 | [-0.0330, +0.0135] | 1/5 | 9/18 | -0.0166 | 0.500 |
| B_all3_eye_true_vs_shift_control | +0.0158 | [-0.0160, +0.0448] | 4/5 | 13/18 | +0.0065 | 0.476 |
| shift_control_vs_B_all3 | -0.0253 | [-0.0403, -0.0085] | 1/5 | 3/18 | -0.0303 | 0.512 |

## >=20°幅值档 subject-macro relative MAE

| 模型 | 20_30 | 30_45 | 45_70 | ge70 |
|---|---:|---:|---:|---:|
| B_all3 | 0.3374 | 0.3479 | 0.3427 | 0.3156 |
| B_all3_eye_true | 0.3382 | 0.3476 | 0.3420 | 0.3177 |
| B_all3_eye_shift_control | 0.3397 | 0.3479 | 0.3425 | 0.3169 |

## 正式门

- a_true_vs_B_all3_gain_at_least_0_05_deg: `False`
- b_true_vs_B_all3_bootstrap_ci_lower_above_zero: `False`
- c_true_vs_shift_gain_at_least_0_02_deg: `False`
- d_true_vs_shift_bootstrap_ci_lower_above_zero: `False`
- e_positive_outer_folds_at_least_4: `False`
- f_leave_top_subject_improvement_positive: `False`
- g_no_ge20_amplitude_relative_regression_over_0_01: `True`

## 解释边界

本轮只是在同一P_full=2323和同一18位被试上的发展性OOF。眼动视频没有进入模型，zyl没有进入任何fit或test。
真实眼动只有同时优于B_all3和同recording错位对照，才能解释为预测起点前眼动状态的事件时序增量；否则只能说明稳定被试/设备/recording背景或模型容量效应。
所有眼动窗口严格截止预测起点；缺眼动事件没有删除，填补中位数只来自当前外折训练被试。绝对度数和pooled只作参考，逐幅值档subject-macro relative MAE仍是主要报告口径。
