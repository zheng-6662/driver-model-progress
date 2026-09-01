# 三通道候选标签审计

标签仍为候选合同。制动先按recording第5百分位估计释放零位，以处理约-0.03偏置；油门相对刺激前基线判断松油、维持和补油；方向盘区分刺激时已有转向、刺激后新转向、回正和反向修正。

动作匹配先搜索0—5秒，再分别汇总1、2、3、5秒。三组阈值全部报告：方向盘/制动/油门的宽松、主、严格阈值分别为3度/0.03/0.03、5度/0.05/0.05、8度/0.08/0.08。

| threshold_set   |   window_s | metric               |   count |   fraction |   median_latency_s |   subjects_with_action |
|:----------------|-----------:|:---------------------|--------:|-----------:|-------------------:|-----------------------:|
| lenient         |          1 | steer_response       |     321 | 0.856      |             0.26   |                     20 |
| lenient         |          1 | brake_response       |      81 | 0.216      |             0.385  |                     17 |
| lenient         |          1 | accelerator_release  |     165 | 0.44       |             0.17   |                     22 |
| lenient         |          1 | accelerator_increase |     143 | 0.381333   |             0.1    |                     20 |
| lenient         |          1 | no_clear_response    |      24 | 0.064      |           nan      |                     14 |
| lenient         |          3 | steer_response       |     356 | 0.949333   |             0.2975 |                     21 |
| lenient         |          3 | brake_response       |     180 | 0.48       |             1.1575 |                     21 |
| lenient         |          3 | accelerator_release  |     204 | 0.544      |             0.32   |                     23 |
| lenient         |          3 | accelerator_increase |     193 | 0.514667   |             0.25   |                     21 |
| lenient         |          3 | no_clear_response    |       3 | 0.008      |           nan      |                      2 |
| lenient         |          5 | steer_response       |     365 | 0.973333   |             0.325  |                     23 |
| lenient         |          5 | brake_response       |     197 | 0.525333   |             1.265  |                     21 |
| lenient         |          5 | accelerator_release  |     218 | 0.581333   |             0.4    |                     23 |
| lenient         |          5 | accelerator_increase |     222 | 0.592      |             0.3925 |                     21 |
| lenient         |          5 | no_clear_response    |       2 | 0.00533333 |           nan      |                      2 |
| primary         |          1 | steer_response       |     280 | 0.746667   |             0.3525 |                     19 |
| primary         |          1 | brake_response       |      73 | 0.194667   |             0.385  |                     17 |
| primary         |          1 | accelerator_release  |     144 | 0.384      |             0.185  |                     22 |
| primary         |          1 | accelerator_increase |     122 | 0.325333   |             0.0975 |                     20 |
| primary         |          1 | no_clear_response    |      53 | 0.141333   |           nan      |                     15 |
| primary         |          3 | steer_response       |     342 | 0.912      |             0.4775 |                     21 |
| primary         |          3 | brake_response       |     174 | 0.464      |             1.19   |                     20 |
| primary         |          3 | accelerator_release  |     183 | 0.488      |             0.435  |                     22 |
| primary         |          3 | accelerator_increase |     166 | 0.442667   |             0.315  |                     20 |
| primary         |          3 | no_clear_response    |      11 | 0.0293333  |           nan      |                      8 |
| primary         |          5 | steer_response       |     356 | 0.949333   |             0.5025 |                     23 |
| primary         |          5 | brake_response       |     194 | 0.517333   |             1.31   |                     20 |
| primary         |          5 | accelerator_release  |     198 | 0.528      |             0.54   |                     23 |
| primary         |          5 | accelerator_increase |     192 | 0.512      |             0.58   |                     20 |
| primary         |          5 | no_clear_response    |       5 | 0.0133333  |           nan      |                      4 |
| strict          |          1 | steer_response       |     245 | 0.653333   |             0.49   |                     19 |
| strict          |          1 | brake_response       |      58 | 0.154667   |             0.43   |                     16 |
| strict          |          1 | accelerator_release  |     124 | 0.330667   |             0.2325 |                     22 |
| strict          |          1 | accelerator_increase |     102 | 0.272      |             0.1575 |                     20 |
| strict          |          1 | no_clear_response    |      72 | 0.192      |           nan      |                     18 |
| strict          |          3 | steer_response       |     310 | 0.826667   |             0.5925 |                     20 |
| strict          |          3 | brake_response       |     167 | 0.445333   |             1.27   |                     20 |
| strict          |          3 | accelerator_release  |     170 | 0.453333   |             0.5725 |                     22 |
| strict          |          3 | accelerator_increase |     148 | 0.394667   |             0.45   |                     20 |
| strict          |          3 | no_clear_response    |      26 | 0.0693333  |           nan      |                     11 |
| strict          |          5 | steer_response       |     336 | 0.896      |             0.655  |                     20 |
| strict          |          5 | brake_response       |     185 | 0.493333   |             1.37   |                     20 |
| strict          |          5 | accelerator_release  |     186 | 0.496      |             0.6425 |                     23 |
| strict          |          5 | accelerator_increase |     175 | 0.466667   |             0.69   |                     20 |
| strict          |          5 | no_clear_response    |      14 | 0.0373333  |           nan      |                      8 |

多标签不强制互斥；同一事件可以同时出现转向、制动、松油和补油。主表保留各通道刺激时状态、onset、latency、peak、delta、持续时间、1/2/3秒累积操作量、二次修正和歧义原因。
