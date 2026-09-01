# Run69 冻结合同：t0+0.4 rolling vehicle 正式 outer OOF

## 1. 授权与 changed estimand

- 合同编号：`RUN69_ROLLING_VEHICLE_OUTER_OOF_V1_20260830`。
- Run69 是 independent result-to-claim 明确授权的唯一下一项实验。
- 当前只实现和合成烟测，不运行五折正式 outer OOF。
- 新任务不是“在 t0 预测完整 1 s 曲线”。它固定为：
  - t0 初始 B_all3 已存在；
  - 等待 0.4 s 后已经观察点 1–8；
  - 使用不晚于 t0+0.4 的车辆信息；
  - 只预测点 9–20，即剩余约 0.6 s。
- 因为 estimand 已改变，Run69 不报告也不使用旧任务的 floor-ratio。

## 2. 唯一允许的信息

Run69 是 vehicle-only：

- 允许：Run67 独立 rolling vehicle cache、严格 nested B_all3、P_full truth 作为训练/评估标签。
- 禁止：physiology、style、behavior context、KD、teacher、post t0+0.4 raw、幅值/road/support 作模型输入或路由。
- 不读取或运行任何 `verify_*.py`。

## 3. 最关键泄漏边界：outer-train B

对每个 outer fold：

1. 读取该 outer 的 Run65 `outer_X_meta_1/2/3.npz`；
2. 只取每个 cache 的 `validation_indices` 与 `validation_base_predictions`；
3. 拼接三块 validation 分片；
4. 必须恰好覆盖当前 outer-train 全部事件一次；
5. 每块 validation 的被试与该块 fit 被试严格不重叠；
6. 当前 outer-test 事件和被试不得出现在任何分片；
7. B_all3 为三条 base curve 的算术均值。

禁止使用普通 Run63 inner-OOF 表中的 train-row 预测，因为这些行的上游基模型可能包含当前 outer-test 被试。代码不会载入该表。

## 4. outer-test B 的处理

Run69 不复用普通 formal Run63 B 曲线。每个 outer 均重新训练冻结的三条基专家：

- fit indices：全部且仅当前 outer-train；
- predict indices：当前 outer-test；
- M2 ExtraTrees、M3 LightGBM、M4 HistGradientBoosting 的结构和超参数来自哈希冻结的 Run63 config/code；
- fit-only imputation 和 Run63 冻结 training weights；
- 重训 seed fold 固定为 90；
- outer-test B_all3 为三条重训预测的算术均值。

代码会输出每折 fit/test 事件数、被试数、零重叠和“未复用 ordinary formal Run63 B”的审计。

## 5. rolling-V

### 47 维输入

严格沿用 Run67：

1. 8 个 observed-prefix error；
2. 7 个 prefix-error diff；
3. 五通道 `[0,0.4]` mean/delta/slope，共 15；
4. initial-tail 第 9–20 点相对 observed point 8，共 12；
5. 五个 validity。

总计 47。所有 raw support 不晚于 t0+0.4。

### 模型

- `Ridge(alpha=100, fit_intercept=True)`；
- fit-only median imputation；
- fit-only standard scaling；
- outer-train subject-equal weights；
- target 为 `truth points9-20 − cross-fitted outer-train B tail`；
- trust 固定 `1.0`，因为冻结 Run67 的 15/15 meta context 都选择 1.0；不再调参。

## 6. pre-only comparator

输入严格为 40 维：

- initial B 全 20 点；
- 五通道 `[-0.4,0]` mean/delta/slope，共 15；
- 五个 validity。

模型同样为 Ridge100。trust 只允许 `{0,0.1,0.25,0.5,1.0}`。

trust 选择只在当前 outer-train 上进行：利用拼接 B 自带的 3 个 subject-disjoint meta assignment，对 pre-only Ridge 做三折 crossfit；用这些 outer-train OOF 行选择 subject-macro 改善最大的 trust，并列取较小 trust。outer-test 不参与。

## 7. initial-tail

`initial_tail` 必须逐点等于当前 outer-test 重训 B_all3 的点 9–20，无额外拟合或收缩。

## 8. 正式 OOF 输出

五折完成后必须得到恰好 2323 行：

- 每个 event_uid 一行且唯一；
- subject、recording、outer_fold 与 P_full 一致；
- 每个事件只由自己的正式 outer-test 模型预测一次；
- 三个模型均为 12 点有限值。

要求写出：

- `oof_predictions_2323.csv`；
- subject-macro 和 pooled MAE；
- 每折指标；
- lead 点 9–12、medium 点 13–15、tail 点 16–20；
- 18 被试配对 bootstrap 2000；
- 每被试与每折改善；
- ordinary/road-missing 保护；
- waiting-cost；
- 单事件延迟；
- coverage/no-duplicate 审计；
- decision、provenance、RESULT_CN。

## 9. waiting-cost

固定等待 `0.4 s`：

- 已观察点数：8；
- 剩余预测点数：12；
- 剩余 horizon：约 0.6 s。

对总体和每个 outer fold、每个模型报告：

- 第9–20点全部12个剩余点的 pooled MAE（`remaining_12pt_mae`）；
- 第9–20点全部12个剩余点的 subject-macro MAE；
- 相对 initial-tail 的改善；
- `改善 / 0.4 s`。

这里的 waiting-cost 指标固定覆盖全部12个尚未发生的点，不是更窄的第16–20点 tail 子段。这张表只量化精度—等待代价，不把 0.4 s 等待伪装成 t0 部署性能。

## 10. 硬门

rolling-V 必须同时超过 initial-tail 和 pre-only：

- 18 被试宏平均改善 `>=0.20°`；
- paired subject bootstrap 95% CI 下界 `>0`；
- 至少 `12/18` 被试改善；
- 至少 `4/5` outer fold 点估计为正。

相对两个参考，以下每个分层的最坏 outer-fold 被试宏平均回归均必须 `<=+0.02°`：

- lead points 9–12；
- causal ordinary；
- road-reference-missing。

ordinary 只由 outer-train stitched B tail 峰值中位数确定；不得使用 outer-test truth 幅值或 30°门。

rolling-V 单事件 transform + Ridge predict + trust addition 的 P95 必须 `<50 ms`。

## 11. 运行纪律

- 合成烟测：`F:/python3.11/python.exe experiment.py --smoke-test`。
- 正式命令：`F:/python3.11/python.exe experiment.py --out_dir=run_1_outer_oof`。
- 正式五折 outer OOF 已在 `run_2_outer_oof/` 完成；`run_1_outer_oof/` 仅为模型拟合前失败记录。
- 输出目录已存在时拒绝覆盖。
- 任何哈希、分折、覆盖、重训、保护或延迟检查失败立即停止，不换阈值或重试搜索。

## 12. 正式结果状态

- `advance=false`；pair、coverage、latency门通过，protection门失败。
- subject-macro MAE：initial-tail `16.8164°`，pre-only `16.7187°`，rolling-V `9.8219°`。
- Fold 3 road-reference-missing 相对 initial/pre-only 分别退化 `+2.5772°/+3.8639°`，超过冻结上限。
- 完整性审计为 `WARN`，数字经独立重算为真；对外必须注明changed estimand、single-seed和proxy-event总体。
