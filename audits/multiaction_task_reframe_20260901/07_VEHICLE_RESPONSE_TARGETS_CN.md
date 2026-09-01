# 车辆响应目标审计

| cohort      | channel                   |   recordings_present |   recordings_total | unit   |   missing_fraction_mean |   abnormal_fraction_mean |   target_3s_complete_fraction | recommended_role   |
|:------------|:--------------------------|---------------------:|-------------------:|:-------|------------------------:|-------------------------:|------------------------------:|:-------------------|
| august_2025 | lateral_acceleration      |                  136 |                136 | m/s^2  |                0.370928 |              0           |                             1 | 正式目标           |
| august_2025 | lateral_distance          |                  136 |                136 | m      |                0.370928 |              0.000155766 |                             1 | 辅助目标           |
| august_2025 | lateral_velocity          |                  136 |                136 | m/s    |                0.370928 |              0           |                             1 | 辅助目标           |
| august_2025 | longitudinal_acceleration |                  136 |                136 | m/s^2  |                0.370928 |              0           |                             1 | 正式目标           |
| august_2025 | position_x                |                   34 |                136 | m      |                0.90042  |              0           |                             1 | 解释性目标         |
| august_2025 | position_y                |                   34 |                136 | m      |                0.90042  |              0           |                             1 | 解释性目标         |
| august_2025 | roll                      |                  136 |                136 | rad    |                0.370928 |              0           |                             1 | 辅助目标           |
| august_2025 | roll_rate                 |                   30 |                136 | rad/s  |                0.915143 |              0           |                             1 | 分批次辅助目标     |
| august_2025 | speed                     |                  136 |                136 | km/h   |                0.339256 |              0           |                             1 | 正式目标           |
| august_2025 | yaw_rate                  |                   30 |                136 | rad/s  |                0.915143 |              0           |                             1 | 分批次辅助目标     |
| original    | lateral_acceleration      |                   85 |                 85 | m/s^2  |                0        |              0           |                             1 | 正式目标           |
| original    | lateral_distance          |                   85 |                 85 | m      |                0        |              0.00429597  |                             1 | 辅助目标           |
| original    | lateral_velocity          |                   85 |                 85 | m/s    |                0        |              0           |                             1 | 辅助目标           |
| original    | longitudinal_acceleration |                   85 |                 85 | m/s^2  |                0        |              0           |                             1 | 正式目标           |
| original    | position_x                |                   85 |                 85 | m      |                0        |              0           |                             1 | 解释性目标         |
| original    | position_y                |                   85 |                 85 | m      |                0        |              0           |                             1 | 解释性目标         |
| original    | roll                      |                   85 |                 85 | rad    |                0        |              0           |                             1 | 辅助目标           |
| original    | roll_rate                 |                   85 |                 85 | rad/s  |                0        |              1.81653e-06 |                             1 | 分批次辅助目标     |
| original    | speed                     |                   85 |                 85 | km/h   |                0        |              0           |                             1 | 正式目标           |
| original    | yaw_rate                  |                   85 |                 85 | rad/s  |                0        |              0           |                             1 | 分批次辅助目标     |

需要分开评价：使用真实驾驶员操作预测车辆响应，验证车辆响应模块；使用模型预测操作再预测车辆响应，评价端到端误差。两者不得混写。速度、纵向加速度和横向加速度覆盖最好，可优先作为正式目标；yaw/roll rate跨批次缺失，应分批次辅助报告。
