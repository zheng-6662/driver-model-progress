# Run66 冻结合同：生理—车辆耦合残差适配器

## 1. 状态与停止边界

- 合同编号：`RUN66_PHYSIO_VEHICLE_COUPLING_ADAPTER_V1_20260830`。
- 当前状态：代码与协议已准备；**没有启动正式运行，也没有任何 Run66 数值结果**。
- Run65 v2 在已完成的 outer 1–3 上，`VP−V` 分别为 `+0.0507°、-0.0559°、-0.0690°`。即使 outer 4–5 都为正，也最多达到 3/5，因此原始生理序列路线不可能满足冻结的 4/5 同向门。Run66 不重复该路线。
- Run66 是同一 `P_full=2323、18 被试` 上的发展性训练侧筛选，不是独立确认。
- 本轮只允许读取每个 outer 的训练侧及其 3 个 meta-validation context；不得读取或生成正式 outer-test 预测。
- 若任一冻结输入哈希、P_full 身份、nested subject/recording 边界不一致，立即报错，不允许自动重建、换输入或放宽门。

## 2. 冻结总体、基线与分折

- 总体固定为 Run57 V3 `P_full=2323`；不得删样本、改变成员、用生理可用性改变权重。
- 五个 outer fold 与 18 被试、85 recording 身份完全继承 P_full。
- 每个 outer context 使用 Run65 的三个严格 `nested_base` 缓存：
  - 当前 meta-validation 被试永久不进入该 context 的任何拟合；
  - meta-fit 行的三条基专家曲线来自 meta-fit 内部的 subject-crossfit；
  - B_all3 固定为三条 nested 基专家曲线的逐点算术均值。
- Run66 内部只为选择 trust 做 3 折 subject-disjoint crossfit；imputer、scaler、Ridge 和 trust 全部只读当前 meta-fit。
- 每个被试在自己的正式 outer fold 中永远不参与该 outer context；本轮不打开 outer-test。

## 3. 输入时间边界

### 生理变化

只允许 Run64 已冻结的预测起点前特征：

- `phys_emg_log_rms_z_2s`：预测起点前 2 s 的 EMG log-RMS，相对于同 recording 的 `[anchor-120 s, anchor-30 s]` 历史作稳健标准化；
- `phys_hr_z_30s`：预测起点前 30 s 的 ECG 心率，相对于同 recording 的历史稳健标准化；
- `phys_resp_amplitude_z_30s`：预测起点前 30 s 的呼吸幅值，相对于同 recording 的历史稳健标准化。

这些历史参考不是“平静/静息真值”，只能解释为同 recording 内的因果历史参照。

### 近期车辆需求

只允许 `[prediction_anchor_s-65 s, prediction_anchor_s-5 s]`：

- `recent_steer_rate_abs_mean`；
- `recent_ay_abs_mean`；
- `recent_yaw_rate_abs_mean`；
- `recent_hard_brake_ratio`。

窗口必须固定截止在预测起点前 5 s；不得读取预测起点后车辆量。

## 4. 四个预注册耦合量

候选只能使用以下四维，不做结果后筛选：

1. `emg_x_steer_rate = phys_emg_log_rms_z_2s × recent_steer_rate_abs_mean`；
2. `emg_x_lateral_accel = phys_emg_log_rms_z_2s × recent_ay_abs_mean`；
3. `resp_x_yaw = phys_resp_amplitude_z_30s × recent_yaw_rate_abs_mean`；
4. `ecg_x_braking = phys_hr_z_30s × recent_hard_brake_ratio`。

不允许新增第五个耦合量，不允许看结果后替换通道、窗口或需求量。

## 5. 模型与精确回退

### V_vehicle

- 输入：四个近期车辆需求。
- 输出：未来 20 点 `truth - B_all3` 残差在固定 3 维正交 DCT-II 基上的系数。
- 模型：`Ridge(l2_alpha=100, fit_intercept=True)`；subject-equal sample weight。
- trust 只允许 `{0, 0.10, 0.25}`，由当前 meta-fit 内的 3 折 subject-crossfit 选择；并列时选择较小 trust。
- `trust=0` 必须逐点精确返回 B_all3。

### 四个 delta 臂

`PV_coupling`、`quality_only`、`vehicle_x_vehicle`、`shifted_physio` 都只预测对 cross-fitted V 曲线的 3 维 DCT 残差修正；Ridge、权重、trust 网格和选择协议完全相同。

- 主候选：上面四个生理×车辆耦合量。
- quality-only：`physio_source_available` 及 EMG/ECG/RESP 三个 valid fraction。
- vehicle×vehicle：四个同维车辆交互：steer-rate×ay、steer-rate×yaw、ay×yaw、hard-brake×steer-rate。
- shifted physiology：目标事件保留自身车辆需求和自身 availability/quality mask；生理变化只能取同 recording、同 meta 角色中严格更早的合格事件。无更早 donor 时四个 shadow 生理量置零；禁止循环移位、未来 donor 或跨 recording donor。

### 缺失生理

下列条件全部满足才激活任一 delta 修正：

- physiology source available；
- EMG/ECG/RESP 三个指定变化量有限；
- 三通道 valid fraction 均不低于 0.8；
- recent vehicle valid fraction 不低于 0.8。

不满足时样本仍进入拟合、预测、指标和伤害统计，但四个 delta 臂都必须逐点精确返回 `V_vehicle`，不得删样本或单独路由。

## 6. 预处理和禁止信息

- 中位数填补、标准化、Ridge 与 trust 选择只在当前 fit 行拟合。
- 目标 20 点只用于拟合残差和训练侧 meta-validation 评分；不得作为推理特征。
- 下列字段不得用于模型输入、权重、样本删除、trust 选择或路由：幅值、`amplitude_bin`、`road_reference_stratum`、`road_valid`、`support_class`、风险/响应分层、事件后生理、Run65 结果派生特征。
- 30° 幅值不得作为成员、权重、早停或模型选择条件。
- `road_reference_missing` 事件必须保留，只能作为显式报告/伤害分层。

## 7. 指标与硬门

### 必须比较

- `PV_coupling − B_all3`；
- `PV_coupling − V_vehicle`；
- `PV_coupling − shifted_physio`。

quality-only 与 vehicle×vehicle 是强制报告控制，但不是本合同额外添加的数值硬门。

### 18 被试汇总

每个事件在四个 outer-training context 中出现。先在每个 outer context 内计算逐被试改善，再按被试平均其四次出现，得到 18 个被试值。每个必须比较均需同时满足：

1. 18 被试宏平均改善 `>=0.02°`；
2. 五个 outer context 至少 4 个点估计为正；
3. 18 被试 bootstrap 95% CI 下界 `>0`；
4. 去掉最大贡献被试后改善仍 `>0`；
5. 至少 `12/18` 被试改善。

### 保护分层

主候选相对 B_all3、V_vehicle 都必须检查：

- ordinary：冻结 B_all3 的绝对峰值不高于当前 meta-fit B_all3 峰值中位数。阈值 fit-only，不使用真实未来幅值；
- road-missing：P_full 的 `road_reference_stratum=road_reference_missing`，只作评分分层；
- tail：全部事件的第 16–20 点，即预测起点后 0.80–1.00 s。

六个“参考×分层”组合的最坏 outer-context 被试宏平均 MAE 回归都必须 `<=+0.02°`。

只有全部必须比较门与六个伤害门同时通过，`advance_to_outer` 才允许为 true。即使 true，也只表示“允许另行决定是否打开 outer-test”，不等于已经获得正式泛化结果。

## 8. 运行纪律

- 合成烟测：`F:/python3.11/python.exe experiment.py --smoke-test`。
- 正式训练侧运行：`F:/python3.11/python.exe experiment.py --out_dir=run_1_training_screen`。
- 本次实现阶段只允许运行合成烟测；不得启动正式训练侧运行。
- 输出目录必须不存在；程序拒绝覆盖已有结果。
- 出错不得自动换特征、换阈值、换 alpha、删样本或重建 nested cache。
