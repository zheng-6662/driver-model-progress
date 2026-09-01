# 刺激恢复审计

共检查两批 221 个车辆 recording。原始批次配置证明四个距离比较器均在低于30 m时触发 `ExternTrigger01/03/04/06`。先前材料中 distance7/distance8 的40 m说法与当前找到的 `.29/.30/.36` 配置冲突，本轮按配置中的30 m记录，并把冲突保留为待解释项。

V2低附着定义固定为：每个recording最多一个事件，只取第一次从`mu>0.4`跨入`0.1<=mu<=0.4`的时刻。`1.0↔0.8`普通道路变化、`0.4→0.2`低附着内部加深以及所有低附着退出均不构成新的危险刺激。先前1488事件版本因违反该场景口径而失效。

8月原始表头实际含 `distance_truck`、`distance_changlane`、左右道路距离与 `mu`；Run78 staging 没有保留交通距离列，导致此前“完全不存在”的判断过强。本轮对这两列做了诊断 crossing，但因为没找到8月触发脚本，未纳入主候选事件。

| cohort      | stimulus_type                      | trigger_signal             | onset_exactness                    | online_observable          |   number_of_candidate_stimuli |   number_included_after_contract | unresolved_issue                                                              |
|:------------|:-----------------------------------|:---------------------------|:-----------------------------------|:---------------------------|------------------------------:|---------------------------------:|:------------------------------------------------------------------------------|
| august_2025 | enter_low_mu_scene                 | mu                         | exact_recorded_script_state_change | script_label_only          |                            19 |                                7 | 每个recording只保留首次低附着进入；无合法零延迟在线代理，当前仅可作为脚本标签 |
| original    | distance_threshold_ExternTrigger01 | pointdistance              | exact_configuration_threshold      | yes_with_target_perception |                            80 |                               80 | 外部触发编号对应的具体交通车动作语义尚缺权威脚本映射                          |
| original    | distance_threshold_ExternTrigger03 | distance7                  | exact_configuration_threshold      | yes_with_target_perception |                            74 |                               74 | 外部触发编号对应的具体交通车动作语义尚缺权威脚本映射                          |
| original    | distance_threshold_ExternTrigger04 | distance8                  | exact_configuration_threshold      | yes_with_target_perception |                            71 |                               71 | 外部触发编号对应的具体交通车动作语义尚缺权威脚本映射                          |
| original    | distance_threshold_ExternTrigger06 | pointdistance9             | exact_configuration_threshold      | yes_with_target_perception |                            80 |                               80 | 外部触发编号对应的具体交通车动作语义尚缺权威脚本映射                          |
| original    | enter_low_mu_scene                 | mu                         | exact_recorded_script_state_change | script_label_only          |                            63 |                               63 | 每个recording只保留首次低附着进入；无合法零延迟在线代理，当前仅可作为脚本标签 |
| august_2025 | not_promoted_signal_inventory      | distance_left/right        | not_mapped                         | unknown                    |                             0 |                                0 | 道路边界距离存在，但不是已证实脚本刺激                                        |
| original    | not_promoted_signal_inventory      | curvature/lateral_distance | not_mapped                         | unknown                    |                             0 |                                0 | 道路几何存在，但不能把普通道路变化自动命名为极端刺激                          |
| august_2025 | not_promoted_signal_inventory      | curvature/lateral_distance | not_mapped                         | unknown                    |                             0 |                                0 | 道路几何存在，但不能把普通道路变化自动命名为极端刺激                          |

纳入规则只使用刺激、前后可用长度、信号覆盖和recording内重叠规则。无未来动作幅值门。
