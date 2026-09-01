# Claude 独立复审任务书：365 事件 release-time 驾驶员完整转向曲线模型

## 你的身份与本轮边界

你是本项目的独立、批判性模型审查者。请直接读取本地文件和代码，重建研究问题、数据合同、既有证据与失败经验，然后回答“模型结构应怎样分步骤调整、输入输出如何调整、当前最值得验证的模型是什么”。

本轮只读：

- 不修改任何文件；
- 不运行训练；
- 不发起大规模模型搜索；
- 不把旧路线、提议或完整性检查包装成新实验结论；
- 不因为仓库根目录的 `CLAUDE.md` 存在就默认其 2 s 历史任务是当前任务。它含有历史说明，必须用下面的 Run35–Run40 活证据校正。

请使用中文作答。每一项关键事实都给出本地文件路径，能定位时附行号。明确标注：`已验证事实`、`基于证据的推断`、`待验证建议`。

## 用户真实目的与声明边界

当前正式问题是：在快速转向事件的 release 时刻，只使用 release 及以前可获得的车辆—道路信息，预测 release 后约 1 s 的完整方向盘响应曲线。用户关心的不只是 pooled MAE，还关心方向、幅值、趋势、峰值、尾段、普通/低响应样本是否受伤，以及代表性真实曲线。

不可越过的边界：

- 研究对象只能称为高速快速转向高动态/近失稳代理事件；没有真实 `Fz`、轮离地、`LTR`、`TTR` 真值，不能写成真实侧翻或真实失稳预测。
- 输出是 SILAB/DYNCar logical steering coordinate，不是已经机械标定的手轮角，不能做无依据的固定 `/2.5` 换算。
- 所有 release-time 正式输入必须满足 `X(t) <= release`；未来车辆状态、完整记录离线处理结果、目标派生标签不得进入输入。
- 当前证据是按被试分组的开发性 OOF，不是独立外部确认。

## 当前唯一正式开发合同

先以 Run35 合同原文为准核实，不要直接照抄本任务书：

- `RUN32_365_DEVELOPMENT_SET_V1`；
- 365 个独立事件、17 名被试、75 段 recording；
- 196 个 core（权重 1.0）和 169 个 support（权重 0.55）；83 个 `manual_review` 不进入当前模型；
- 外层 5 折 subject `GroupKFold`；每名被试只进入一个测试折；
- release 前 2 s 的 171 维车辆—道路多尺度摘要是正式主输入；另有 16 路×101 点序列候选，但是否使用必须由现有证据决定；
- 目标为 release 后 `0.05..1.00 s` 的 20 点完整 logical steering 曲线。

不要混入历史 54 条、Anchor-v3 65 条、E08/Rule-v2 285 条、滑窗数或增强副本。它们只能作为历史失败背景，不能替代当前 365 事件合同。

## 必读顺序

### A. 最新交接与冻结起点

1. `<PROJECT_ROOT>\04_project_logs\reports\run35_run39_model_correction_handoff_cn.md`
2. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\protocol_freeze\PROTOCOL_CONTRACT_CN.md`
3. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\FINAL_REPORT_CN.md`
4. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\final_audit\INDEPENDENT_REVIEW_CN.md`
5. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\final_audit\final_validation.json`
6. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\run_corrected_baselines.py`

同时查阅 Run35 的机器裁决与逐事件证据：

- `structured_curve_extratrees/decision.json`
- `style_residual_confirmation/decision.json`
- `physio_matched_ab/decision.json`
- `physio_matched_ab/tables/per_event_matched_ab.csv`
- `multimodal_intersection_audit/decision.json`
- `corrected_baselines/tables/aggregate_metrics.csv`

### B. Run36–Run40：必须吸收的后续证据和失败经验

7. Run36：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run36_causal_temporal_basis_extratrees_20260811\outputs\RESULT_CN.md`
   - 同目录 `decision.json`
   - 重点判断：固定 78 维 release 前时间基为何没有替代 171 摘要/嵌套融合；这对 TCN/Transformer 建议意味着什么。
8. Run37：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run37_identifiability_conditional_variance_audit_20260812\README_CN.md`
   - `...\outputs\RESULT_CN.md`
   - `...\outputs\decision.json`
   - 重点区分：存在显著近邻聚合，不等于已证明模型欠拟合；`inconclusive_small_n` 也不等于完全不可预测。
9. Run38：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run38_physio_residual_correlation_explore_20260812\RESULT_CN.md`
   - 同目录 `decision.json`
10. Run39：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run39_peak_time_label_and_quantile_20260812\RESULT_CN.md`
   - 同目录 `decision_stage0.json`、`decision_stage1.json`、`decision_stage2.json`、`decision_stage3.json`
   - `<PROJECT_ROOT>\04_project_logs\reports\run39_peak_time_execution_plan_cn.md`
11. Run40（独立的 release 后 rolling 合同，不能与 release-time 分数混合）：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run40_conservative_post_anchor_rolling_20260814\RESULT_CN.md`
   - 同目录 `decision.json` 与 `validation/final_validation.json`
   - `<PROJECT_ROOT>\04_project_logs\reports\run40_rolling_update_execution_plan_cn.md`

### C. 研究背景与历史失败，只用于理解，不得覆盖 Run35–Run40

12. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\09_reports\CURRENT_GOAL_FAILURES_AND_DATA_MODEL_GAPS_20260730_CN.md`
13. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\09_reports\PROJECT_SYSTEMATIC_AUDIT_AND_CURRENT_EXECUTION_20260804_CN.md`
14. `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\09_reports\CURRENT_RESEARCH_CARD_20260804_CN.md`

第 12–14 项中的 E08、54/65 条、Anchor-v3、6 折/LOSO 等属于历史阶段。请提炼其中仍然有效的研究目的、声明红线和失败模式，但不要把其样本数、split、指标当成当前合同。

### D. 现有模型结构与数据入口代码

15. Run35 真正基线实现：
   - `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run35_365_protocol_and_structured_curve_20260811\run_corrected_baselines.py`
16. 历史/可复用结构代码：
   - `<PROJECT_ROOT>\02_code\final_code\model\training\v58_modular\modeling.py`
   - `...\v58_modular\losses.py`
   - `...\v58_modular\losses_metrics.py`
   - `...\v58_modular\data.py`
   - `...\v58_modular\config.py`
   - `<PROJECT_ROOT>\02_code\final_code\model\training\conditioned_trajectory_head.py`
   - `<PROJECT_ROOT>\02_code\final_code\model\training\baseline_eval_primary_aux.py`

先回答这些代码是否真的实现当前 365×20 的 Run35 合同。若仍是历史 2 s/其他 split/其他标签任务，只能把模块思想列为可复用，不得拿旧指标作公平对照。

## 已知失败经验：必须逐项回应，不能重复建议

至少覆盖以下路线，并用原文件与数值核实：

1. K=8/K=10 结构化曲线表示：标签重构很好，但 K=8 ExtraTrees 预测改善仅约 0.76%，未过 1% 晋级门；“表示通过”不等于“预测模型通过”。
2. Run36 78 维固定时间基：没有稳定替代 171 维摘要或 `corrected_nested_blend`；不要只因序列模型更复杂就假设会增加信息。
3. 驾驶风格：直接拼接、动态 GMM/Markov、低自由度残差适配均未获得稳定净增益；subject ID embedding 对 17 名不均衡被试尤其危险。
4. 生理：同一 312 事件 matched A/B 中直接 24 维拼接恶化；Run38 也没有找到能稳定解释峰时/峰后回落残差的预注册相关关系。
5. EEG：合法 `causal_online` / `cutoff_safe_teacher` EEG 交集为 0；离线 EEG 只能做机制解释。
6. `post_peak_drop` 不是严格反打/穿零标签；不要把峰后下降称为真实 countersteer/reversal。
7. Run39：坏标签不是峰时误差主因；冻结的 release 前动作相位未过可辨识门；K10 residual interval head 总体过覆盖却只覆盖 50% historical high-under 峰；不重开同一分位区间调参。
8. Run40：release 后新增观测的保守滚动路线虽改善 common，但伤害 ordinary/low-response，最终 no-go；它仍是另一任务，不能伪装成 release-time 提升。
9. 历史 E31–E56：深网、Transformer、多模态、扩散增强、门控、删除弱窗口、合成测试、历史记忆/场景选择等多次出现“局部改善但强响应、普通样本、跨被试或因果边界失败”。只提炼共性教训，不混合历史分数。
10. 自动完整性 `pass`、哈希复现或标签重构通过，不等于科学模型晋级。

## 你必须完成的审查问题

### 1. 先重建现状

- 用一页以内说明研究目标、独立单位、样本结构、输入、输出、split、指标、当前最佳模型、证据边界。
- 列出仓库中哪些“入口/说明”已过时或属于不同合同，防止下一位研究者走错路。
- 给出当前 `corrected_nested_blend` 的组成、输入和输出数据流；指出代码与报告之间任何无法确认的环节。

### 2. 审查输入应该怎样调整

将候选分成四栏：`保留`、`删除/禁用`、`只作分层/解释`、`值得另立合同验证`。

至少讨论：

- 171 维摘要中可能冗余或泄漏风险的特征族；
- 16×101 原始/派生序列是否还有证据支持进入模型；
- release 前短窗口动作状态、道路几何、速度/横摆/侧向加速度/侧倾等应该作为输入还是未来辅助输出；
- 驾驶风格、生理、EEG、subject ID；
- 新输入在 availability、latency、窗口、滤波方向、缺失掩码、训练折内预处理方面的合同；
- Run37 的支持域发现是否支持“置信度/拒绝预测/分域报告”，以及怎样避免把目标邻居信息用到线上。

每个建议说明它针对的是：峰值均值收缩、总体形状、尾段，还是纯粹的不确定性/支持域问题。不要声称 release 前新增模型结构能解决已无证据支持的峰时或峰后回落信息缺失。

### 3. 审查输出应该怎样调整

比较并裁决：

- 直接 20 点完整曲线；
- 固定低秩曲线基系数；
- “保护当前基线 + 低秩残差修正”；
- 峰值/尾段/方向等辅助头；
- 角速度、角加速度作为自由输出或由曲线求导；
- 未来速度、横摆、侧向加速度、侧倾等多任务输出；
- 点预测与不确定性输出；
- release 后 rolling update 与 release-time 输出的分离方式。

输出必须内部一致，避免多头互相矛盾；说明训练目标、重建方式、约束和推理时可用信息。

### 4. 只推荐一个首选模型结构

不要给模型菜单或 sweep。只能给出一个当前最值得做的一次性低自由度候选，并说明为什么它比“整网换 Transformer/TCN/MoE”“再加生理/风格”“继续分位区间”更符合现有证据。

候选必须明确到：

- 输入张量；
- 主干是否冻结；
- 输出参数化；
- 残差/保护门的数学形式；
- 损失函数；
- 防止普通/低响应伤害的机制；
- 外层与内层训练流程；
- 参数自由度上限和唯一允许选择的超参数；
- 与 `corrected_nested_blend` 的公平 OOF 对照。

你可以接受、改造或否决交接文件提出的“保护基线 + 低秩残差峰值锚定头”，但必须给证据理由。若你判断在新增独立数据前不应再训练任何结构，也可以明确给出 `no-run`，但需说明为何这是比一次低自由度验证更合理的选择。

### 5. 给出可执行的分步骤方案和停止规则

按 `Step 0...N` 写最小执行方案。训练前必须冻结：

- 合法输入；
- 目标；
- split；
- matched comparison set；
- 指标；
- 晋级与停止门；
- 随机种子/环境/缓存与内容哈希。

至少同时报告：pooled curve MAE、subject-macro MAE、方向/幅值/峰值、endpoint、tail、趋势、普通/低响应 no-harm、逐被试改善数、被试聚类 CI、代表性真实曲线。明确哪些指标只作诊断，不能当晋级门。

给出一张 `go / no-go` 表：什么结果允许继续，什么结果立即停止，失败后不能如何“微调救活”。

## 最终输出格式

按以下顺序回答：

1. `审查结论摘要`：直接说明保持什么、改什么、首选模型/是否 no-run。
2. `我对项目的准确理解`。
3. `证据与失败经验总表`。
4. `输入调整裁决表`。
5. `输出调整裁决表`。
6. `唯一首选结构`：含必要公式或伪代码。
7. `冻结实验步骤与 go/no-go 门`。
8. `最可能的失败方式与防护`。
9. `仍需用户决定的事项`：只列真正会改变科学合同的选择。
10. `必读证据索引`：列出你实际读取的文件，不要声称读取未打开的文件。

不要只做泛泛综述。要像一个准备签字放行或否决下一次实验的审稿人一样，给出可复核、可执行、对既有失败敏感的结论。
