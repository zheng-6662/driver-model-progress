# 样本充分性审计

- 纳入候选事件：1488。
- 驾驶员：28；recording：108。
- 每名驾驶员事件数：最小1，Q1=3.2，中位=16.5，Q3=81.5，最大195。
- 无明显反应：8。
- 最多事件驾驶员贡献占比：13.1%。
- 过密recording比例（最小间隔<2秒）：59.3%。
- 严格在线精确子集：305。

正式建模前仍需解决刺激语义映射；当前足以审计动作选择与时延分布，但低频动作组合、8月交通刺激和20分钟个体基线只能作为探索或条件分层。

| cohort      | stimulus_type                      | readable_action_mode   |   events |   subjects |   recordings |
|:------------|:-----------------------------------|:-----------------------|---------:|-----------:|-------------:|
| original    | exit_lower_mu                      | 转向+制动+补油         |      118 |         15 |           54 |
| original    | exit_lower_mu                      | 转向+补油              |      116 |         15 |           46 |
| original    | enter_lower_mu                     | 转向+松油              |      105 |         14 |           52 |
| original    | exit_lower_mu                      | 转向                   |       98 |         13 |           46 |
| original    | enter_lower_mu                     | 转向+补油              |       92 |         15 |           52 |
| original    | enter_lower_mu                     | 转向                   |       81 |         13 |           42 |
| original    | exit_lower_mu                      | 转向+松油              |       78 |         15 |           46 |
| original    | exit_lower_mu                      | 转向+制动              |       65 |         14 |           41 |
| original    | exit_lower_mu                      | 转向+松油+补油         |       57 |         12 |           37 |
| original    | exit_lower_mu                      | 转向+制动+松油         |       55 |         13 |           36 |
| original    | enter_lower_mu                     | 转向+制动+松油         |       55 |         14 |           34 |
| original    | enter_lower_mu                     | 转向+制动+补油         |       50 |         14 |           32 |
| original    | enter_lower_mu                     | 转向+松油+补油         |       49 |         14 |           31 |
| original    | distance_threshold_ExternTrigger01 | 转向+制动+补油         |       26 |         13 |           26 |
| original    | enter_lower_mu                     | 转向+制动+松油+补油    |       25 |          9 |           19 |
| original    | distance_threshold_ExternTrigger04 | 转向+制动+补油         |       21 |         13 |           21 |
| original    | exit_lower_mu                      | 转向+制动+松油+补油    |       20 |          8 |           14 |
| original    | enter_lower_mu                     | 转向+制动              |       19 |          7 |           14 |
| original    | distance_threshold_ExternTrigger06 | 转向+制动+补油         |       16 |         11 |           16 |
| original    | distance_threshold_ExternTrigger01 | 转向+制动+松油+补油    |       16 |         12 |           16 |
| original    | distance_threshold_ExternTrigger03 | 转向+松油              |       16 |         10 |           16 |
| original    | distance_threshold_ExternTrigger03 | 转向+制动+松油         |       16 |         13 |           16 |
| original    | distance_threshold_ExternTrigger01 | 转向+制动+松油         |       13 |         11 |           13 |
| original    | distance_threshold_ExternTrigger06 | 转向+松油+补油         |       13 |          9 |           13 |
| original    | distance_threshold_ExternTrigger04 | 转向+制动+松油+补油    |       13 |         10 |           13 |
| original    | distance_threshold_ExternTrigger06 | 转向+松油              |       11 |          9 |           11 |
| original    | distance_threshold_ExternTrigger06 | 转向+制动+松油         |       11 |          9 |           11 |
| original    | distance_threshold_ExternTrigger06 | 转向+制动+松油+补油    |       10 |          8 |           10 |
| original    | distance_threshold_ExternTrigger06 | 转向                   |       10 |          5 |           10 |
| original    | distance_threshold_ExternTrigger04 | 转向+补油              |        9 |          7 |            9 |
| original    | distance_threshold_ExternTrigger03 | 转向+松油+补油         |        9 |          5 |            9 |
| original    | distance_threshold_ExternTrigger04 | 转向+松油              |        9 |          7 |            9 |
| august_2025 | exit_lower_mu                      | 转向+制动+补油         |        9 |          5 |            8 |
| original    | distance_threshold_ExternTrigger03 | 转向                   |        9 |          7 |            9 |
| original    | distance_threshold_ExternTrigger03 | 转向+制动              |        8 |          4 |            8 |
| original    | enter_lower_mu                     | 无明显反应             |        8 |          6 |            8 |
| august_2025 | enter_lower_mu                     | 转向+制动+补油         |        7 |          4 |            6 |
| august_2025 | exit_lower_mu                      | 转向+制动              |        7 |          5 |            6 |
| original    | enter_lower_mu                     | 补油                   |        7 |          4 |            7 |
| original    | distance_threshold_ExternTrigger01 | 转向+松油+补油         |        7 |          5 |            7 |
