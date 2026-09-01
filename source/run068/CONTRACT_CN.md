# Run68 冻结合同：t0+0.4 生理尾段区间校准

## 1. 唯一问题

Run68 不再允许生理修改均值曲线。均值中心逐点固定为 Run67 严格双层 `rolling_V` 的第 9–20 点，只问：在 `t0+0.4 s` 已观测到 ECG/EMG/RESP 的 0–0.4 秒变化后，能否在相同 80% 事件级同时覆盖下，缩窄 rolling-V 尾段区间并改善风险排序。

通过也只能支持“生理改善置信度校准”，不能支持“生理改善均值预测”。

## 2. 输入与重算

- 人口固定 `P_full=2323`、18 被试、85 recording；不删事件。
- 外层和 meta 层完全继承 Run67/Run65 的 subject-disjoint 冻结划分。
- 每个 outer/meta context 必须调用哈希一致的 Run67 `load_nested` 与 `run_stage_a_context`，重新获得 `fit_v_oof` 和 untouched meta-validation `rolling_V`。
- 禁止把 Run67 的结果 CSV 当作拟合输入。
- Run67 车辆、生理 cache、config、experiment、builder 和 P_full 均由 `config.json` 固定 SHA-256。
- Run67 自身仍须复核其 P_full 与 15 个 nested cache 的冻结哈希。

## 3. 中心、分数与模型

中心：

`center = rolling_V[:, t09:t20]`

不得重新选择 rolling-V trust，不得给中心增加 physiology delta。

在当前 meta-fit 的严格 OOF rolling-V 残差上计算：

`sigma_j = median(abs(y_ij - center_ij)) + 1e-6`

`r_i = max_j abs(y_ij - center_ij) / sigma_j`

四个臂：

- `U_V`：Run67 固定 47 维同一时刻车辆输入预测 `log(r_i+eps)`；
- `U_VP`：`U_V` 基础 log-scale 加一个 physiology-15 Ridge delta；
- `U_VQ`：同结构 quality-15 控制；
- `U_Vshift`：同结构 shifted-15 控制。

全部为 `Ridge(alpha=100)`，按被试等总权。delta 逐事件裁剪到 `[-log(2), +log(2)]`，只产生一个事件级尺度，不产生 12 个独立头。

## 4. 严格尺度 OOF 与校准

- meta-fit 被试固定二折；每一折的 scale model 不得读取该折被试。
- 先在 scale-fit subjects 上拟合 `U_V`；delta target 只由同一 scale-fit 内的 vehicle scale 残差构成。
- 合并两折 scale OOF 后，才允许计算 subject-equal 80% 经验分位数。
- `U_V_all` 在全部 scale OOF 行校准，供生理缺失事件回退。
- active common-support 上，`U_V/U_VP/U_VQ/U_Vshift` 使用同一 active mask，并分别在相同 active scale OOF 行校准到 80%。
- 最终 scale model 在完整 meta-fit 上重拟合，校准分位数仍来自 scale OOF；meta-validation 真值不参与任何拟合或校准。
- 由于被试相关和经验校准，不得宣称 distribution-free conformal coverage。

## 5. 缺失精确回退

对 `physiology_active=False` 的事件：

- `U_VP/U_VQ/U_Vshift` 的中心、下界、上界必须与 `U_V_all` 逐点 `np.array_equal`；
- 不得删除 `zx/rjy` 等低覆盖事件；
- 完整 P_full 结果由 active 区间与 inactive vehicle-only 区间组成。

## 6. 固定评价

主覆盖是一个事件 12 点全部被覆盖。必须报告：

- subject-macro 同时覆盖；
- t09–t20 逐点覆盖；
- subject-macro 平均宽度；
- 80% interval score；
- active/full、ordinary、road-missing、lead t09–t12、tail t17–t20；
- 每个 outer context 和每个 subject；
- 由区间宽度直接形成的 80% selective tail MAE 与 retention 60/70/80/90/100% risk-coverage AUC。

禁止另训高误差分类器。AUC 分类不能替代覆盖—宽度门。

## 7. 硬门

`U_VP` 必须同时超过 `U_V/U_VQ/U_Vshift`：

1. 完整 P_full subject-macro 同时覆盖在 `[0.77,0.83]`；至少 4/5 outer 在 `[0.75,0.85]`；相对每个控制覆盖缺口不超过 0.01；ordinary/road-missing 均不低于 0.75。
2. 相对每个控制，subject-macro 宽度和 interval score 均至少改善 2%，至少 4/5 outer 达到 2%；宽度比 bootstrap 95% 上界 `<1`；至少 12/18 被试宽度更窄且至少 12/18 interval score 更好。
3. ordinary、road-missing、lead、tail 任一层 interval score 相对伤害不超过 2%。
4. 80% selective tail MAE 与 risk-coverage AUC 相对每个控制均至少改善 2%，至少 4/5 outer 达到 2%，leave-top-subject 后仍为正。
5. 中心身份、inactive 区间身份全部通过；单事件 scale + interval P95 `<50 ms`。

任一项失败即 `FINAL_NO_GO_PHYSIOLOGY_MEAN_AND_UNCERTAINTY_STOP`。不得改覆盖率、窗口、特征、模型、Ridge alpha 或分数定义重试。

## 8. 当前实施边界

本目录本次只实现配置、代码、合同、绘图和纯合成 smoke。未运行 5×3 训练侧实验，未读取 outer-test，也没有结果数值。

