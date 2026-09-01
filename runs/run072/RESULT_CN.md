# Run72 受保护 EEG 状态与历史驾驶风格筛查

## 当前裁决

- 状态：`independent_gate_failed_no_additive`。
- 这是同一 P_full 人口的发展性训练侧证据：`same_population_developmental=true`。
- 不是外部确认：`external_confirmation=false`；fresh strict subjects=`1`。
- outer-test 从未打开：`outer_test_opened=false`，outer-test预测行=`0`。
- 不删除任何事件；pooled event 指标只作诊断。

## 支持度

| context                  |   outer_fold |   eeg_common_events |   eeg_common_subjects |   common_minimum_directional_rate |   trait_events |   trait_subjects |   joint_events |   joint_subjects |
|:-------------------------|-------------:|--------------------:|----------------------:|----------------------------------:|---------------:|-----------------:|---------------:|-----------------:|
| global_unique_population |            0 |                2145 |                    18 |                            0.9830 |           1920 |               16 |           1765 |               16 |
| outer_training_context   |            1 |                1799 |                    16 |                            0.9831 |           1600 |               14 |           1450 |               14 |
| outer_training_context   |            2 |                1681 |                    14 |                            0.9813 |           1555 |               13 |           1405 |               13 |
| outer_training_context   |            3 |                1711 |                    14 |                            0.9794 |           1570 |               12 |           1416 |               12 |
| outer_training_context   |            4 |                1623 |                    14 |                            0.9872 |           1434 |               13 |           1293 |               13 |
| outer_training_context   |            5 |                1766 |                    14 |                            0.9844 |           1521 |               12 |           1496 |               12 |

支持门：`{"eeg_global_support": true, "eeg_each_outer_context_support": true, "real_shift_common_rate": true, "eeg_support_pass": true, "trait_global_support": true, "trait_each_outer_context_support": true, "trait_support_pass": true, "joint_global_support": true, "joint_each_outer_context_support": true, "joint_support_pass": true}`

## 成对门禁

| family   | comparison             |   required_gain_deg |   subject_macro_gain_deg |   positive_outer_contexts |   bootstrap_ci_lower_deg |   leave_top_subject_gain_deg |   improved_subjects |   worst_context_guardrail_regression_deg | gate_pass   |
|:---------|:-----------------------|--------------------:|-------------------------:|--------------------------:|-------------------------:|-----------------------------:|--------------------:|-----------------------------------------:|:------------|
| eeg      | E_real_minus_B         |              0.1000 |                   0.0000 |                         0 |                   0.0000 |                       0.0000 |                   0 |                                   0.0000 | False       |
| eeg      | E_real_minus_Q         |              0.0500 |                   0.0000 |                         0 |                   0.0000 |                       0.0000 |                   0 |                                   0.0000 | False       |
| eeg      | E_real_minus_C_E       |              0.0500 |                   0.0050 |                         1 |                  -0.0067 |                      -0.0009 |                   4 |                                   0.0228 | False       |
| eeg      | E_real_minus_shift     |              0.0500 |                   0.0000 |                         0 |                   0.0000 |                       0.0000 |                   0 |                                   0.0000 | False       |
| trait    | T_trait_minus_B        |              0.1000 |                  -0.0026 |                         0 |                  -0.0087 |                      -0.0033 |                   1 |                                   0.0307 | False       |
| trait    | T_trait_minus_C_T      |              0.0500 |                  -0.0033 |                         0 |                  -0.0098 |                      -0.0041 |                   2 |                                   0.0307 | False       |
| trait    | T_trait_minus_permuted |              0.0500 |                  -0.0026 |                         0 |                  -0.0087 |                      -0.0033 |                   1 |                                   0.0307 | False       |

## 条件执行边界

- 独立 EEG 门通过：`False`。
- 独立 trait 门通过：`False`。
- A_additive 只有两类独立门都通过才实现：`False`。
- 固定 4D interaction 只有 A_additive 再通过才实现：`False`。

## 诚实边界

Run71 输入只接受逐事件独立的固定物理频谱/协方差特征；主 EEG 路径不允许全局切空间、pyriemann、PCA 或未来目标变换。trait 只使用 15 个先前会话中位数特征；`style_available` 仅作 active，prior-session count 仅作置乱分层，被试 ID 不进入模型。普通样本在每个 outer×meta context 内只由 meta-fit B_all3 预测绝对峰值的被试等权加权中位数冻结，不读取真值、risk 或道路字段；道路参考缺失另列报告，尾部固定为第16--20点。
