# Run68 训练侧区间校准结果

- 生理区间尺度门：**NO-GO**
- rolling-V 均值中心未改变。
- outer-test 未打开。
- 当前仍是同一 P_full 上的 training-side meta-validation。

## 四臂摘要

| arm      |   subject_macro_simultaneous_coverage |   subject_macro_active_coverage |   subject_macro_ordinary_coverage |   subject_macro_road_missing_coverage |   subject_macro_mean_width_deg |   subject_macro_interval_score |   subject_macro_ordinary_interval_score |   subject_macro_road_missing_interval_score |   subject_macro_lead_interval_score |   subject_macro_tail_interval_score |   subject_macro_selective_tail_mae_80_deg |   subject_macro_risk_coverage_auc |
|:---------|--------------------------------------:|--------------------------------:|----------------------------------:|--------------------------------------:|-------------------------------:|-------------------------------:|----------------------------------------:|--------------------------------------------:|------------------------------------:|------------------------------------:|------------------------------------------:|----------------------------------:|
| U_V      |                              0.834598 |                        0.834661 |                          0.879541 |                              0.816957 |                      53.962880 |                      61.332204 |                               46.985588 |                                   77.214759 |                           21.313791 |                          100.126084 |                                  8.393660 |                          8.585427 |
| U_VP     |                              0.836273 |                        0.836720 |                          0.878472 |                              0.820881 |                      55.491009 |                      62.134085 |                               46.969159 |                                   77.263854 |                           21.610020 |                          101.457714 |                                  8.368435 |                          8.542193 |
| U_VQ     |                              0.836054 |                        0.836349 |                          0.880365 |                              0.834103 |                      54.167823 |                      61.448105 |                               47.179054 |                                   77.502569 |                           21.377912 |                          100.285729 |                                  8.405130 |                          8.593657 |
| U_Vshift |                              0.835204 |                        0.835795 |                          0.877659 |                              0.822358 |                      54.939589 |                      61.979539 |                               47.400617 |                                   78.935606 |                           21.545870 |                          101.177319 |                                  8.423823 |                          8.605950 |

## 硬门

| gate_group       | gate                                          |        value | threshold   | pass   |
|:-----------------|:----------------------------------------------|-------------:|:------------|:-------|
| coverage         | full_subject_macro_coverage_band              |  0.836273    | [0.77,0.83] | False  |
| coverage         | outer_context_coverage_band_count             |  5           | >=4/5       | True   |
| coverage         | ordinary_coverage                             |  0.878472    | >=0.75      | True   |
| coverage         | road_missing_coverage                         |  0.820881    | >=0.75      | True   |
| coverage         | coverage_deficit_vs_U_V                       | -0.0016753   | <=0.01      | True   |
| efficiency       | width_improvement_vs_U_V                      | -0.0283182   | >=0.02      | False  |
| efficiency       | interval_score_improvement_vs_U_V             | -0.0130744   | >=0.02      | False  |
| efficiency       | outer_width_count_vs_U_V                      |  0           | >=4/5       | False  |
| efficiency       | outer_interval_score_count_vs_U_V             |  0           | >=4/5       | False  |
| efficiency       | bootstrap_width_ratio_upper_vs_U_V            |  1.05364     | <1.0        | False  |
| efficiency       | subjects_width_narrower_vs_U_V                |  7           | >=12/18     | False  |
| efficiency       | subjects_interval_score_better_vs_U_V         |  6           | >=12/18     | False  |
| protection       | worst_stratum_interval_score_harm_vs_U_V      |  0.0206777   | <=0.02      | False  |
| selective_risk   | selective_tail_mae_improvement_vs_U_V         |  0.00300517  | >=0.02      | False  |
| selective_risk   | risk_coverage_auc_improvement_vs_U_V          |  0.00503571  | >=0.02      | False  |
| selective_risk   | outer_selective_count_vs_U_V                  |  0           | >=4/5       | False  |
| selective_risk   | outer_risk_auc_count_vs_U_V                   |  0           | >=4/5       | False  |
| selective_risk   | selective_leave_top_vs_U_V                    |  0.00202889  | >0          | True   |
| selective_risk   | risk_auc_leave_top_vs_U_V                     |  0.0214405   | >0          | True   |
| coverage         | coverage_deficit_vs_U_VQ                      | -0.000218935 | <=0.01      | True   |
| efficiency       | width_improvement_vs_U_VQ                     | -0.0244275   | >=0.02      | False  |
| efficiency       | interval_score_improvement_vs_U_VQ            | -0.0111636   | >=0.02      | False  |
| efficiency       | outer_width_count_vs_U_VQ                     |  0           | >=4/5       | False  |
| efficiency       | outer_interval_score_count_vs_U_VQ            |  0           | >=4/5       | False  |
| efficiency       | bootstrap_width_ratio_upper_vs_U_VQ           |  1.04869     | <1.0        | False  |
| efficiency       | subjects_width_narrower_vs_U_VQ               |  6           | >=12/18     | False  |
| efficiency       | subjects_interval_score_better_vs_U_VQ        |  4           | >=12/18     | False  |
| protection       | worst_stratum_interval_score_harm_vs_U_VQ     |  0.0147433   | <=0.02      | True   |
| selective_risk   | selective_tail_mae_improvement_vs_U_VQ        |  0.00436578  | >=0.02      | False  |
| selective_risk   | risk_coverage_auc_improvement_vs_U_VQ         |  0.00598852  | >=0.02      | False  |
| selective_risk   | outer_selective_count_vs_U_VQ                 |  0           | >=4/5       | False  |
| selective_risk   | outer_risk_auc_count_vs_U_VQ                  |  0           | >=4/5       | False  |
| selective_risk   | selective_leave_top_vs_U_VQ                   |  0.0197335   | >0          | True   |
| selective_risk   | risk_auc_leave_top_vs_U_VQ                    |  0.0352285   | >0          | True   |
| coverage         | coverage_deficit_vs_U_Vshift                  | -0.00106954  | <=0.01      | True   |
| efficiency       | width_improvement_vs_U_Vshift                 | -0.0100368   | >=0.02      | False  |
| efficiency       | interval_score_improvement_vs_U_Vshift        | -0.00249349  | >=0.02      | False  |
| efficiency       | outer_width_count_vs_U_Vshift                 |  0           | >=4/5       | False  |
| efficiency       | outer_interval_score_count_vs_U_Vshift        |  0           | >=4/5       | False  |
| efficiency       | bootstrap_width_ratio_upper_vs_U_Vshift       |  1.02607     | <1.0        | False  |
| efficiency       | subjects_width_narrower_vs_U_Vshift           | 10           | >=12/18     | False  |
| efficiency       | subjects_interval_score_better_vs_U_Vshift    |  9           | >=12/18     | False  |
| protection       | worst_stratum_interval_score_harm_vs_U_Vshift |  0.0139764   | <=0.02      | True   |
| selective_risk   | selective_tail_mae_improvement_vs_U_Vshift    |  0.00657512  | >=0.02      | False  |
| selective_risk   | risk_coverage_auc_improvement_vs_U_Vshift     |  0.00740838  | >=0.02      | False  |
| selective_risk   | outer_selective_count_vs_U_Vshift             |  1           | >=4/5       | False  |
| selective_risk   | outer_risk_auc_count_vs_U_Vshift              |  0           | >=4/5       | False  |
| selective_risk   | selective_leave_top_vs_U_Vshift               |  0.0236542   | >0          | True   |
| selective_risk   | risk_auc_leave_top_vs_U_Vshift                |  0.0347951   | >0          | True   |
| identity_latency | center_bit_identity                           |  1           | True        | True   |
| identity_latency | inactive_interval_bit_identity                |  1           | True        | True   |
| identity_latency | u_vp_latency_p95_ms                           |  0.98336     | <50.0       | True   |
