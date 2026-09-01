# Run67 冻结合同：t0+0.4 s 异步车辆—生理更新

## 1. 当前状态与研究边界

- 合同编号：`RUN67_ASYNCHRONOUS_PHYSIO_UPDATE_V1_20260830`。
- 当前只实现代码与协议，不运行完整缓存构建，不启动正式训练侧实验。
- Run65 原始序列生理路线已经因 4/5 符号门数学上不可达而停止；Run66 当前也为 no-go。Run67 不复用二者的候选结果来挑特征。
- Run67 回答的是一个新的时序问题：初始预测在 `t0=预测起点` 给出；到 `t0+0.4 s` 已观察前 8 点及同步车辆/生理变化后，只更新第 9–20 点。
- 全部证据仍是同一 `P_full=2323 / 18 被试` 上的 outer-train/meta-validation，不是正式 outer-test 或独立确认。

## 2. 两阶段硬停止

### Stage A：车辆更新

先完成全部 `5 outer × 3 meta` 的 `rolling_V` 训练侧评估。Stage A 未通过时：

- 不加载 physiology cache；
- 不拟合 physiology 模型；
- 不评估 `rolling_VP/VQ/shifted`；
- 立即写 `NO_GO_STAGE_A`；
- 不打开 outer-test。

### Stage B：生理增量

只有 Stage A 全部门和 Stage A 延迟门通过后，才允许加载独立 physiology cache 并运行 Stage B。Stage B 仍只在 meta-validation 上评估。

## 3. 分折与标签隔离

- 外层：继承 P_full 五个 subject-disjoint outer fold。
- meta 层：每个 outer 使用 Run65 已冻结的三个 strict nested context；当前 meta-validation 被试和 recording 不进入拟合。
- inner 层：每个 meta-fit 再固定为 2 折 subject-disjoint crossfit。
- `rolling_V` 的 meta-fit OOF 曲线由这 2 折产生；Stage B 的 physiology target 固定为 `truth_tail - cross_fitted_rolling_V_tail`，不得使用同一行的 in-sample V 预测。
- imputer、scaler、Ridge、trust 选择都只读当前 inner/meta fit。

## 4. 独立车辆缓存

缓存必须从 P_full 的 85 个原始 vehicle recording 独立构建；不得读取 Run64 序列缓存。

### 原始通道

- causal steering rate；
- lateral acceleration `ay`；
- yaw rate；
- brake；
- speed km/h。

方向盘先使用只向后的 0.1 s endpoint Savitzky–Golay；steering rate、ay、yaw 按事件方向对齐。每个事件的原始支持硬截止于 `t0+0.4 s`。

### rolling_V 固定 47 维

1. 前 8 个已观察点相对初始 B_all3 的误差：8；
2. 上述误差的一阶差分：7；
3. 五个 rolling channel 在 `[t0,t0+0.4]` 的 mean/delta/slope：15；
4. 初始 B_all3 第 9–20 点相对已观察第 8 点：12；
5. 五通道有效率：5。

总计严格为 47。前 8 点必须从 raw 的 causal steering 独立重建；P_full `target_t01–08` 只能用于 cache QA 对账，不能直接成为缓存输入。

### pre_only 固定 40 维

- 初始 B_all3 完整 20 点；
- 五通道在 `[t0-0.4,t0]` 的 mean/delta/slope：15；
- 五通道有效率：5。

它只用于控制“多一个 Ridge 头”的容量效应，不读取 t0 后数据。

## 5. 独立生理缓存

生理 builder 只从原始 PhysioLAB recording 解码 ECG、EMG、RESP；代码不得导入或读取：

- `physio_post_sequence_10hz_teacher_only.npz`；
- Run64 post0–5 s `channel_mask`；
- 任何由该 mask 派生的 coverage。

当前 P_full 的 85 个 recording 中，车辆 raw 为 85/85；生理 raw 实际存在 77/85。缺少生理 raw 的事件全部保留并精确回退 rolling_V。

### 同 recording 历史标准化

每个事件、每个通道只用同 recording `[anchor-120 s, anchor-30 s]` 内的 0.4 s bins 计算 median/MAD。不得跨 recording、跨被试或使用 t0 后参考。

### 主 15 维

对 ECG RMS、EMG RMS、RESP mean 各自将 `[0,0.4]` 与 `[-0.4,0]` 分为四个 0.1 s bin。每通道固定：

1. post4 mean；
2. post4 last；
3. post4 slope；
4. post4 mean − pre4 mean；
5. validity。

三通道共 15 维。

### VQ 质量控制 15 维

每通道固定五项：recording usable、baseline duration fraction、baseline valid-bin fraction、pre4 coverage、post4 coverage。共 15 维，与生理状态臂等维。

### shifted 控制 15 维

- pseudo-pre：`[-1.2,-0.8]`；
- pseudo-post：`[-0.8,-0.4]`。

使用同一个历史 median/MAD 和同一 15 维汇总公式。它完全位于 t0 前，不允许循环移位或未来样本。

### 缺失精确身份

仅当三通道 recording usable、15 个主状态有限、三通道 validity 均至少 0.8 时激活 physiology delta。否则 `rolling_VP`、`rolling_VQ`、`rolling_shifted` 都必须逐点与 `rolling_V` 完全相同；不得删样本。

## 6. 模型与 trust

- 所有头均为 `Ridge(alpha=100, fit_intercept=True)`；subject-equal sample weight。
- Stage A trust：`{0,0.1,0.25,0.5,1.0}`。
- Stage B trust：`{0,0.1,0.25}`。
- trust 只由当前 meta-fit 内的 2 折 subject OOF 选择；并列选择较小值。
- `V trust=0` 精确回退 initial_tail；`P trust=0` 或生理缺失精确回退 rolling_V。
- 风格分支当前未认证，`style_branch_enabled=false`；代码遇到 true 必须停止。

## 7. Gate A

必须同时比较：

- `rolling_V − initial_tail`；
- `rolling_V − pre_only`。

每个比较都必须：

- 18 被试宏平均第 9–20 点 MAE 改善 `>=0.20°`；
- 五个 outer context 至少 4 个为正；
- 18 被试 bootstrap 95% CI 下界 `>0`；
- leave-top 后 `>0`；
- 至少 `12/18` 被试改善。

相对 initial_tail 和 pre_only，还需保护：第 9–12 点 lead、causal ordinary、road-reference-missing。每个 outer 的被试宏平均 MAE 回归均不得超过 `+0.02°`。

ordinary 固定为初始 B_all3 tail 绝对峰值不高于当前 meta-fit 中位数；阈值 fit-only，不读取真实未来幅值，也不使用 30°门。

## 8. Gate B

必须同时比较：

- `rolling_VP − rolling_V`；
- `rolling_VP − rolling_VQ`；
- `rolling_VP − rolling_shifted`。

每个比较的 18 被试宏平均改善必须 `>=0.05°`，并满足同样的 4/5、bootstrap、leave-top、12/18 鲁棒性。VP 相对 V 的 lead/ordinary/road 回归不得超过 `+0.02°`；缺失生理精确身份必须通过。

## 9. 延迟门

- 单事件测量：fit-only transform + Ridge predict + trust addition；不含离线 raw I/O/cache build。
- Stage A `rolling_V` P95 必须 `<50 ms`，否则在 Stage B 前停止。
- Stage B `rolling_V + physiology delta` 总 P95 必须 `<50 ms`，否则最终 no-go。

## 10. 禁止项

- 不预测或修改第 1–8 点；输出只覆盖第 9–20 点。
- 不使用 t0+0.4 后 vehicle/physiology。
- 不使用幅值、amplitude bin、road 字段、support、future target 作模型输入、权重、删样本、路由或 trust 选择。
- 不打开 outer-test；即使 Gate A/B 均通过，也只允许另行决定是否执行新的 outer 协议。
- 不改特征、窗口、trust、门限或样本池；不重试搜索。

## 11. Run67b 可选 follow-up（不属于本轮可执行代码）

稳定 prior-session style 已正式 stop/no-go。为避免把第三套特征、四个新控制和独立门强塞进已经冻结的 Stage A→Stage B，近期行为上下文 trust-modulator 单独记录在 `RUN67B_FOLLOWUP_CN.md`：

- 只允许 `[t0-65,t0-5]` 六个近期行为字段，不允许 prior-session traits；
- 只允许在 constant `g0` 附近最多 `±0.10` 调节 initial_tail/rolling_V blend；
- 与 physiology Gate B 完全独立；
- 本轮没有实现、没有运行、不能产生 Run67 结论。

## 12. 运行纪律

- 合成烟测：`F:/python3.11/python.exe experiment.py --smoke-test`。
- 单 recording 车辆提取：`F:/python3.11/python.exe build_vehicle_cache.py --smoke-recording`。
- 单 recording 生理提取：`F:/python3.11/python.exe build_physio_cache.py --smoke-recording`。
- 完整 cache 和正式训练侧运行在本次实现中禁止启动。
- 任一 cache/result 目标已存在时拒绝覆盖。
