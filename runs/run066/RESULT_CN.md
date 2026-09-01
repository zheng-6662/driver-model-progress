# Run66 生理—车辆耦合适配器：训练侧结果

- 是否进入 outer-test：**False**
- 本文件只包含 outer-train/meta-validation；正式 outer-test 未读取。
- 缺失生理事件保留，并逐点精确回退到 V_vehicle。

## 成对结果

| comparison                       | required_gate   |   positive_outer_context_count |   mean_outer_gain_deg |   subject_macro_gain_deg |   bootstrap_ci_lower_deg |   bootstrap_ci_upper_deg |   leave_top_subject_gain_deg |   subject_improved_count |   subject_worsened_count |   subject_count | pair_gate_pass   |
|:---------------------------------|:----------------|-------------------------------:|----------------------:|-------------------------:|-------------------------:|-------------------------:|-----------------------------:|-------------------------:|-------------------------:|----------------:|:-----------------|
| coupling_minus_B                 | True            |                              2 |               -0.0487 |                  -0.0491 |                  -0.2101 |                   0.0463 |                      -0.0595 |                        9 |                        9 |              18 | False            |
| coupling_minus_V                 | True            |                              0 |               -0.0641 |                  -0.0646 |                  -0.2082 |                   0.0126 |                      -0.0730 |                        6 |                        3 |              18 | False            |
| coupling_minus_shifted_physio    | True            |                              2 |               -0.0662 |                  -0.0667 |                  -0.2084 |                   0.0088 |                      -0.0736 |                        8 |                        7 |              18 | False            |
| coupling_minus_quality_only      | False           |                              0 |               -0.0641 |                  -0.0646 |                  -0.2071 |                   0.0131 |                      -0.0730 |                        6 |                        3 |              18 | False            |
| coupling_minus_vehicle_x_vehicle | False           |                              0 |               -0.0650 |                  -0.0655 |                  -0.2026 |                   0.0083 |                      -0.0726 |                        5 |                        6 |              18 | False            |

## 保护分层

| reference   | stratum                |   mean_outer_subject_macro_regression_deg |   maximum_outer_subject_macro_regression_deg | harm_gate_pass   |
|:------------|:-----------------------|------------------------------------------:|---------------------------------------------:|:-----------------|
| B_all3      | ordinary_causal        |                                    0.0623 |                                       0.1365 | False            |
| B_all3      | road_reference_missing |                                   -0.0839 |                                       0.0175 | True             |
| B_all3      | tail_points_16_20      |                                    0.0625 |                                       0.1943 | False            |
| V_vehicle   | ordinary_causal        |                                    0.0730 |                                       0.1503 | False            |
| V_vehicle   | road_reference_missing |                                   -0.0122 |                                       0.0000 | True             |
| V_vehicle   | tail_points_16_20      |                                    0.1089 |                                       0.2216 | False            |

## 证据边界

这是同一 P_full 上的发展性训练侧筛选，不是独立确认。只有全部冻结门通过，才允许另行决定是否打开 outer-test。
