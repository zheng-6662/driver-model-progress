# 样本充分性审计

- 纳入候选事件：375。
- 驾驶员：24；recording：92。
- 每名驾驶员事件数：最小1，Q1=1.0，中位=17.0，Q3=24.0，最大42。
- 无明显反应：5。
- 最多事件驾驶员贡献占比：11.2%。
- 过密recording比例（最小间隔<2秒）：0.0%。
- 严格在线精确子集：305。

正式建模前仍需解决刺激语义映射；当前足以审计动作选择与时延分布，但低频动作组合、8月交通刺激和20分钟个体基线只能作为探索或条件分层。

| cohort   | stimulus_type                      | readable_action_mode   |   events |   subjects |   recordings |
|:---------|:-----------------------------------|:-----------------------|---------:|-----------:|-------------:|
| original | distance_threshold_ExternTrigger01 | 转向+制动+补油         |       26 |         13 |           26 |
| original | distance_threshold_ExternTrigger04 | 转向+制动+补油         |       21 |         13 |           21 |
| original | distance_threshold_ExternTrigger03 | 转向+制动+松油         |       16 |         13 |           16 |
| original | distance_threshold_ExternTrigger03 | 转向+松油              |       16 |         10 |           16 |
| original | distance_threshold_ExternTrigger01 | 转向+制动+松油+补油    |       16 |         12 |           16 |
| original | distance_threshold_ExternTrigger06 | 转向+制动+补油         |       16 |         11 |           16 |
| original | enter_low_mu_scene                 | 转向                   |       16 |          8 |           16 |
| original | distance_threshold_ExternTrigger01 | 转向+制动+松油         |       13 |         11 |           13 |
| original | distance_threshold_ExternTrigger04 | 转向+制动+松油+补油    |       13 |         10 |           13 |
| original | distance_threshold_ExternTrigger06 | 转向+松油+补油         |       13 |          9 |           13 |
| original | enter_low_mu_scene                 | 转向+松油              |       12 |          9 |           12 |
| original | distance_threshold_ExternTrigger06 | 转向+制动+松油         |       11 |          9 |           11 |
| original | distance_threshold_ExternTrigger06 | 转向+松油              |       11 |          9 |           11 |
| original | distance_threshold_ExternTrigger06 | 转向                   |       10 |          5 |           10 |
| original | distance_threshold_ExternTrigger06 | 转向+制动+松油+补油    |       10 |          8 |           10 |
| original | distance_threshold_ExternTrigger03 | 转向+松油+补油         |        9 |          5 |            9 |
| original | distance_threshold_ExternTrigger03 | 转向                   |        9 |          7 |            9 |
| original | distance_threshold_ExternTrigger04 | 转向+松油              |        9 |          7 |            9 |
| original | distance_threshold_ExternTrigger04 | 转向+补油              |        9 |          7 |            9 |
| original | distance_threshold_ExternTrigger03 | 转向+制动              |        8 |          4 |            8 |
| original | distance_threshold_ExternTrigger01 | 转向+松油+补油         |        7 |          5 |            7 |
| original | enter_low_mu_scene                 | 转向+制动+松油         |        7 |          5 |            7 |
| original | distance_threshold_ExternTrigger04 | 转向+松油+补油         |        6 |          5 |            6 |
| original | distance_threshold_ExternTrigger03 | 转向+制动+松油+补油    |        6 |          5 |            6 |
| original | distance_threshold_ExternTrigger04 | 转向                   |        6 |          5 |            6 |
| original | distance_threshold_ExternTrigger06 | 转向+补油              |        6 |          6 |            6 |
| original | distance_threshold_ExternTrigger01 | 转向+制动              |        6 |          6 |            6 |
| original | distance_threshold_ExternTrigger03 | 转向+制动+补油         |        5 |          4 |            5 |
| original | enter_low_mu_scene                 | 无明显反应             |        5 |          4 |            5 |
| original | distance_threshold_ExternTrigger04 | 转向+制动+松油         |        5 |          4 |            5 |
| original | enter_low_mu_scene                 | 补油                   |        5 |          4 |            5 |
| original | distance_threshold_ExternTrigger03 | 转向+补油              |        5 |          4 |            5 |
| original | distance_threshold_ExternTrigger01 | 转向+松油              |        4 |          2 |            4 |
| original | enter_low_mu_scene                 | 转向+补油              |        4 |          4 |            4 |
| original | enter_low_mu_scene                 | 转向+松油+补油         |        4 |          3 |            4 |
| original | enter_low_mu_scene                 | 松油                   |        4 |          4 |            4 |
| original | distance_threshold_ExternTrigger01 | 转向                   |        4 |          4 |            4 |
| original | enter_low_mu_scene                 | 转向+制动+补油         |        3 |          3 |            3 |
| original | distance_threshold_ExternTrigger06 | 转向+制动              |        3 |          2 |            3 |
| original | distance_threshold_ExternTrigger01 | 转向+补油              |        3 |          3 |            3 |
