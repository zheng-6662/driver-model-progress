# 三通道候选标签审计

标签仍为候选合同。制动先按recording第5百分位估计释放零位，以处理约-0.03偏置；油门相对刺激前基线判断松油、维持和补油；方向盘区分刺激时已有转向、刺激后新转向、回正和反向修正。

动作匹配先搜索0—5秒，再分别汇总1、2、3、5秒。三组阈值全部报告：方向盘/制动/油门的宽松、主、严格阈值分别为3度/0.03/0.03、5度/0.05/0.05、8度/0.08/0.08。

| threshold_set   |   window_s | metric               |   count |   fraction |   median_latency_s |   subjects_with_action |
|:----------------|-----------:|:---------------------|--------:|-----------:|-------------------:|-----------------------:|
| lenient         |          1 | steer_response       |    1260 | 0.846774   |             0.175  |                     27 |
| lenient         |          1 | brake_response       |     326 | 0.219086   |             0.2925 |                     24 |
| lenient         |          1 | accelerator_release  |     510 | 0.342742   |             0.175  |                     23 |
| lenient         |          1 | accelerator_increase |     497 | 0.334005   |             0.125  |                     23 |
| lenient         |          1 | no_clear_response    |      87 | 0.0584677  |           nan      |                     18 |
| lenient         |          3 | steer_response       |    1442 | 0.969086   |             0.22   |                     28 |
| lenient         |          3 | brake_response       |     603 | 0.405242   |             0.85   |                     27 |
| lenient         |          3 | accelerator_release  |     678 | 0.455645   |             0.4    |                     23 |
| lenient         |          3 | accelerator_increase |     717 | 0.481855   |             0.4    |                     23 |
| lenient         |          3 | no_clear_response    |       7 | 0.0047043  |           nan      |                      4 |
| lenient         |          5 | steer_response       |    1472 | 0.989247   |             0.23   |                     28 |
| lenient         |          5 | brake_response       |     659 | 0.442876   |             1.015  |                     27 |
| lenient         |          5 | accelerator_release  |     718 | 0.482527   |             0.4675 |                     23 |
| lenient         |          5 | accelerator_increase |     856 | 0.575269   |             0.625  |                     25 |
| lenient         |          5 | no_clear_response    |       2 | 0.00134409 |           nan      |                      2 |
| primary         |          1 | steer_response       |    1136 | 0.763441   |             0.235  |                     26 |
| primary         |          1 | brake_response       |     297 | 0.199597   |             0.37   |                     23 |
| primary         |          1 | accelerator_release  |     414 | 0.278226   |             0.175  |                     23 |
| primary         |          1 | accelerator_increase |     399 | 0.268145   |             0.145  |                     23 |
| primary         |          1 | no_clear_response    |     182 | 0.122312   |           nan      |                     20 |
| primary         |          3 | steer_response       |    1395 | 0.9375     |             0.34   |                     28 |
| primary         |          3 | brake_response       |     570 | 0.383065   |             0.93   |                     27 |
| primary         |          3 | accelerator_release  |     599 | 0.402554   |             0.605  |                     23 |
| primary         |          3 | accelerator_increase |     609 | 0.409274   |             0.505  |                     23 |
| primary         |          3 | no_clear_response    |      32 | 0.0215054  |           nan      |                     11 |
| primary         |          5 | steer_response       |    1445 | 0.971102   |             0.37   |                     28 |
| primary         |          5 | brake_response       |     631 | 0.424059   |             1.125  |                     27 |
| primary         |          5 | accelerator_release  |     643 | 0.432124   |             0.665  |                     23 |
| primary         |          5 | accelerator_increase |     746 | 0.501344   |             0.82   |                     25 |
| primary         |          5 | no_clear_response    |       8 | 0.00537634 |           nan      |                      6 |
| strict          |          1 | steer_response       |    1011 | 0.679435   |             0.305  |                     25 |
| strict          |          1 | brake_response       |     248 | 0.166667   |             0.4225 |                     22 |
| strict          |          1 | accelerator_release  |     353 | 0.237231   |             0.235  |                     22 |
| strict          |          1 | accelerator_increase |     316 | 0.212366   |             0.175  |                     22 |
| strict          |          1 | no_clear_response    |     283 | 0.190188   |           nan      |                     24 |
| strict          |          3 | steer_response       |    1326 | 0.891129   |             0.485  |                     27 |
| strict          |          3 | brake_response       |     536 | 0.360215   |             1.1125 |                     27 |
| strict          |          3 | accelerator_release  |     537 | 0.360887   |             0.69   |                     23 |
| strict          |          3 | accelerator_increase |     519 | 0.34879    |             0.625  |                     22 |
| strict          |          3 | no_clear_response    |      67 | 0.0450269  |           nan      |                     14 |
| strict          |          5 | steer_response       |    1402 | 0.942204   |             0.53   |                     27 |
| strict          |          5 | brake_response       |     598 | 0.401882   |             1.255  |                     27 |
| strict          |          5 | accelerator_release  |     582 | 0.391129   |             0.775  |                     23 |
| strict          |          5 | accelerator_increase |     658 | 0.442204   |             1.0675 |                     24 |
| strict          |          5 | no_clear_response    |      32 | 0.0215054  |           nan      |                     12 |

多标签不强制互斥；同一事件可以同时出现转向、制动、松油和补油。主表保留各通道刺激时状态、onset、latency、peak、delta、持续时间、1/2/3秒累积操作量、二次修正和歧义原因。
