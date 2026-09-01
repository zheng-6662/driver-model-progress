# 顺序个体化可行性

本轮按真实recording时间排序。主协议只把完整早期recording用于校准、完整后期recording用于测试；没有把同一recording内相邻事件随机分到两侧。

| protocol                                      | condition        |   threshold |   calibratable_subjects |   subjects_with_later_test |   later_test_events |   median_test_events_per_subject |   action_modes_covered |
|:----------------------------------------------|:-----------------|------------:|------------------------:|---------------------------:|--------------------:|---------------------------------:|-----------------------:|
| complete_early_recordings_to_later_recordings | ordinary_minutes |           2 |                      26 |                         26 |                1224 |                             39.5 |                     15 |
| complete_early_recordings_to_later_recordings | ordinary_minutes |           5 |                      25 |                         25 |                1212 |                             42   |                     15 |
| complete_early_recordings_to_later_recordings | ordinary_minutes |          10 |                      19 |                         19 |                1070 |                             64   |                     15 |
| complete_early_recordings_to_later_recordings | ordinary_minutes |          20 |                      14 |                         14 |                 733 |                             49.5 |                     15 |
| complete_early_recordings_to_later_recordings | completed_events |           0 |                      26 |                         26 |                1228 |                             39.5 |                     15 |
| complete_early_recordings_to_later_recordings | completed_events |           1 |                      21 |                         21 |                1158 |                             57   |                     15 |
| complete_early_recordings_to_later_recordings | completed_events |           3 |                      20 |                         20 |                1115 |                             58.5 |                     15 |
| complete_early_recordings_to_later_recordings | completed_events |           5 |                      19 |                         19 |                1092 |                             57   |                     15 |
| complete_early_recordings_to_later_recordings | completed_events |          10 |                      15 |                         15 |                1033 |                             61   |                     15 |

结论：只要要求较短普通历史或少量已完成事件，仍有一部分驾驶员和后期事件；门槛升高后覆盖快速收缩。正式协议应先选择能覆盖足够驾驶员与动作模式的最低层级，不应把校准条件事后调到最有利。
