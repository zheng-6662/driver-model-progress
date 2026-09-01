# Run69 t0+0.4 changed-estimand 正式 outer OOF

是否通过全部门：**False**

## 总体指标

| model        |   pooled_event_mae_deg |   subject_macro_mae_deg |
|:-------------|-----------------------:|------------------------:|
| initial_tail |                17.6809 |                 16.8164 |
| pre_only     |                17.5857 |                 16.7187 |
| rolling_V    |                10.3557 |                  9.8219 |

## 配对被试 bootstrap

| comparison                   |   positive_outer_fold_count |   subject_macro_gain_deg |   bootstrap_ci_lower_deg |   bootstrap_ci_upper_deg |   subject_improved_count |   subject_worsened_count |   subject_count | pair_gate_pass   |
|:-----------------------------|----------------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------:|-------------------------:|----------------:|:-----------------|
| rolling_V_minus_initial_tail |                           5 |                   6.9944 |                   6.1790 |                   7.8817 |                       18 |                        0 |              18 | True             |
| rolling_V_minus_pre_only     |                           5 |                   6.8968 |                   6.1567 |                   7.7096 |                       18 |                        0 |              18 | True             |

本轮只预测点9-20；不报告 floor-ratio；physiology/style/context/KD 均未使用。

## 保护门与最终裁决

两项配对门均通过：rolling-V相对initial-tail改善 `6.9944°`，相对pre-only改善 `6.8968°`；均为5/5 outer正、18/18被试改善，bootstrap 95%CI下界分别为 `6.1790°` 和 `6.1567°`。覆盖和延迟门也通过，推理P95为 `0.7144 ms`。

但保护门失败。Fold 3 `road_reference_missing` 的subject-macro误差：

- 相对initial-tail退化 `+2.5772°`；
- 相对pre-only退化 `+3.8639°`。

冻结上限是 `+0.02°`，因此最终必须保持 `advance=false`，不能把总体强改善写成满足道路缺失层保护的部署模型。

## 证据边界

- 估计目标已改变：在 `t0+0.4 s`、点1–8已观察后，只预测点9–20；不是t0时完整20点任务。
- 结果来自单一冻结seed配置、同一P_full近失稳代理事件总体。
- 正式完整性审计为 `WARN`：真值、原始MAE、结果数字与泄漏边界通过；警告来自内置coverage审计部分自证、changed-estimand/single-seed范围和运行前顶层文档漂移。
- Run69未使用physiology、style、behavior context或KD，也不能为这些模态提供正证据。
