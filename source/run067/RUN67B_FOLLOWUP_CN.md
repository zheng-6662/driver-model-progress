# Run67b 可选后续：近期行为上下文 trust-modulator

## 状态

这是为了保持 Run67 核心实现稳定而单独记录的 follow-up。它目前：

- 不在 `experiment.py` 中实现；
- 不属于 Stage A 或 physiology Gate B；
- 不运行；
- 不产生任何结果；
- 在真正实现前必须重新冻结成独立实验编号和输入哈希。

稳定 prior-session style 已正式 stop/no-go，因此 Run67b 禁止使用任何跨 session 驾驶风格或长期人格特征。

## 固定问题

在 Run67 Stage A 已获得 initial_tail 与 rolling_V 后，先只在 meta-fit 内选择一个 constant blend：

`constant_prediction = (1-g0) * initial_tail + g0 * rolling_V`

未来实现时 `g0` 候选预注册为 `0.0,0.1,...,1.0`，只通过当前 meta-fit inner subject-OOF 选择，并列取更小值。

行为上下文模型只允许在 `g0` 周围小幅调节：

`g = clip(g0 + 0.10 * tanh(Ridge1000(context)), 0, 1)`

因此任何事件相对 g0 的调整绝对值不得超过 0.10；上下文缺失时逐事件精确返回 g0。

## 六个预注册字段

全部来自 `[t0-65 s,t0-5 s]`，不读取 t0 后信息：

1. `recent_steer_rate_abs_mean`；
2. `recent_ay_abs_mean`；
3. `recent_yaw_rate_abs_mean`；
4. `recent_hard_brake_ratio`；
5. `recent_speed_std`；
6. `recent_lane_offset_std`。

不允许结果后换字段、加第七维或引入 prior-session traits。

## 四个控制

1. constant `g0`；
2. vehicle-prediction-disagreement-only：只用 initial_tail 与 rolling_V 的预测分歧，不读六维行为上下文；
3. same-recording earlier nonoverlap shifted context：只能取同 recording 内严格更早、窗口不重叠的六维上下文；禁止循环移位和未来 donor；
4. availability-only：与主模型等覆盖，只读六维字段可用性/coverage。

## 独立硬门

主 modulator 相对四个控制中的每一个都必须：

- 18 被试宏平均改善 `>=0.02°`；
- 至少 4/5 outer context 为正；
- subject bootstrap 95% CI 下界 `>0`；
- leave-top 后 `>0`；
- 至少 12/18 被试改善；
- lead、causal ordinary、road-missing 的最坏 outer 被试宏平均回归不超过 `+0.02°`；
- 缺失上下文逐点精确等于 constant g0。

通过也只能作为独立训练侧证据，不能改变 physiology Gate B，也不能与生理增量合并宣称。
