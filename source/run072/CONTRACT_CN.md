# Run72 受保护 EEG 状态与历史驾驶风格筛查冻结合同

## 1. 证据边界与人口

本轮固定使用 `P_full=2323`、18 名被试、85 条录制。它只产生同一人口的发展性训练侧证据：`same_population_developmental=true`、`external_confirmation=false`、`fresh_strict_subjects=1`。不得删除事件，pooled event 指标只作诊断。

每个 outer context 只读取 Run65 的三个 `outer_k_meta_j.npz`。当前 meta-validation 被试永久不进入该 context 的拟合；outer-test 被试与事件不进入拟合、候选选择、门禁或输出。Run72 不生成 outer-test 预测，`outer_test_opened=false` 必须保持到结束。

`B_all3` 在每个角色内只等于 Run65 `fit_base_predictions` 或 `validation_base_predictions` 三条曲线的算术平均，不读取 Run60 outer OOF 结果替代它。

## 2. 允许输入

- Run71：逐事件独立的 `main46`、`shift46`、`quality11` 与主窗/移位窗 active。46 维是冻结的物理频谱/协方差统计，不是全局切空间；主路径禁止 pyriemann、全局 tangent reference 或 PCA。若 Run71 contract、feature names、46/46/11 形状、哈希、自报 provenance、未来支持或 inactive-NaN 规则任一改变，Run72 在拟合前停止。
- Run64：只读 15 个 `style_*__median` 先前会话中位数特征。`style_available` 只决定 active；原始 `style_prior_session_count` 只决定置乱层 `1/2/3+`，不进入特征。禁止 `recent_*` 和被试 ID。
- Run57/Run63/Run65：172 维因果车辆摘要只用于同容量影子控制；Run65 nested base 只提供训练侧基曲线。
- 真值 20 点曲线只在相应 meta-fit/meta-validation 训练侧角色内用于残差拟合、候选选择和发展性门禁。

风险类别、道路缺失、幅值、被试 ID、event UID、outer fold 均不得进入模型。普通样本也不得再由 `risk_proxy` 或任何标签定义：每个 outer×meta context 只在其 meta-fit 行上计算 `B_all3` 预测曲线的绝对峰值，用“每名 fit 被试总权重相等”的加权中位数冻结阈值；预测绝对峰值不高于阈值即为该 context 的普通样本。这个阈值不读取真值、risk、道路、响应或幅值，冻结后原样应用于该 context 的内部 OOF 行和最终 meta-validation 行。道路参考缺失仍固定为 `road_reference_stratum == road_reference_missing`，只作独立 harm 报告。幅值不参与成员资格、权重、早停、模型选择或门禁。

## 3. 固定模型臂

- `B_all3`：三基曲线算术平均。
- `Q_quality`：11 维质量。
- `C_E`：172 维车辆摘要经 seed `20260830` 固定 Rademacher 矩阵和 reduced QR 到 57 维。
- `C_T`：同一固定方法将 172 维车辆摘要投到 15 维。
- `E_shift`：移位窗 46 维加目标时刻质量 11 维。
- `E_real`：主窗 46 维加目标时刻质量 11 维。
- `T_trait`：15 维先前会话中位数特征。
- `T_permuted`：在每个 fit/validation 角色内部，按 prior-count `1/2/3+` 分层做 donor-subject derangement；同一 target subject/state 固定同一 donor subject，整条 15 维向量一起转移，禁止自捐赠和真值参与。
- `A_additive`：只有 EEG 与 trait 独立门都通过后才允许选择。每个 outer×meta context 只在其 meta-fit cross-fitted E/T 预测上比较三组冻结候选 `(E,T)={(0.75,0.25),(0.5,0.5),(0.25,0.75)}`；先拒绝普通/道路缺失/尾部任一回归超过 `0.02°` 的候选，再最大化相对 B 的 subject-macro 增益。距最佳 `0.01°` 内先选 `0.5/0.5`，再选 subject-macro 平均绝对更新更小者。最终路由固定为：两者 active 时用所选凸组合；仅 EEG active 时逐元素复制 `E_real`；仅 trait active 时逐元素复制 `T_trait`；两者都不 active 时逐元素复制 `B_all3`。A 的 active 是 E/T active 的并集。
- `I_interaction`：只有 `A_additive` 再通过后才实现。EEG 57 维与 trait 15 维分别在当前 fit 角色拟合 2 维 PCA，固定四项为两两乘积；训练标签是 `truth - cross-fitted A_additive`。inactive 精确复制 `A_additive`。

## 4. Ridge 与候选选择

每个 Ridge 明确使用 `active * [1, z]`，`fit_intercept=False`。缺失填补和标准化只在当前 fit-active 行估计。训练样本按被试等权；`alpha={0.1,1,10,100}`，`trust={0,0.1,0.25}`；每个 Run65 meta-fit 内固定再做三折 subject-disjoint cross-fit。三折赋值只用被试、trait active 与 prior-count 层做固定 seed 的确定性排列搜索；每个 fit/validation 角色中，只要某层有目标事件，该层必须至少有两名 donor subject，否则拟合前停止，不能让 `T_permuted` 缩小支持后冒充影子控制。

每个候选先检查三类内部验证 harm：普通样本全曲线、道路参考缺失全曲线、全部事件第16--20点尾部。任一最坏内部 context 回归大于 `0.02°` 即拒绝；剩余候选按 subject-macro 20 点 MAE 增益最大。距最佳 `0.01°` 以内时先选更小 trust，再选更大 alpha。`trust=0` 与 inactive 行必须逐元素精确复制 reference。

`Q_quality/C_E/E_shift/E_real` 使用主窗与移位窗共同 active 支持，且 `E_real` 与 `E_shift` 使用同一目标时刻 `quality11`。所有控制臂独立选择 alpha/trust。

## 5. 支持门

- EEG 全局至少 12 名被试/500 事件；每个 outer-training context 至少 8 名/150 事件；intersection/main 与 intersection/shifted 两个比例均至少 80%。
- trait 全局至少 12 名/500 事件；每个 outer-training context 至少 8 名/150 事件。
- joint 全局至少 10 名/300 事件；每个 outer-training context 至少 7 名/100 事件。

支持不足不删样本、不换定义、不放宽阈值，只能失败并保持精确基线回退。

三类全局/context 支持门必须在任何 Ridge 拟合之前计算。EEG、trait、joint 任一 required support 失败时，所有臂只输出逐元素精确 B 回退，写出 `model_fit_started=false` 的 no-model decision，并停止独立臂、additive 和 interaction。

正式 preflight 还必须逐一检查 15 个最终 meta-validation trait donor 角色：每个 `1/2/3+` 层只要有目标事件，就必须至少有两名 donor subject；支持损失事件必须为 0，自 donor 必须为 0。内部三折角色和最终 meta-validation 角色都要审计。

## 6. 成对效果门

EEG 必须同时满足：`E_real-B >=0.10°`，以及相对 `Q_quality/C_E/E_shift` 各 `>=0.05°`。trait 必须同时满足：`T_trait-B >=0.10°`，相对 `C_T/T_permuted` 各 `>=0.05°`。

每一对还必须：5 个 outer-training context 至少 4 个增益为正；2000 次 subject bootstrap 的 2.5% 下界大于 0；删除最大贡献被试后仍大于 0；18 名被试至少 12 名改善；普通、道路缺失、尾部三类最坏 outer context 回归不超过 `0.02°`。

只有两类独立门都通过，才检查 `A_additive` 相对 B 至少 `0.10°`、相对 E 和 T 各至少 `0.05°`，并使用同一鲁棒性/harm 门。只有 A 通过，才检查 interaction 相对 A 至少 `0.05°`，同样不得放宽其余门。

## 7. 必须产物与停止规则

正式运行必须写出 support、每个 context 的 ordinary threshold/mask 审计、final meta-validation donor 审计、selection、arm、pair、harm、fallback、per-event、trait permutation audit、interaction PCA audit、`decision.json`、`provenance.json` 和中文 `RESULT_CN.md`。additive 的 both/E-only/T-only/neither 四种状态必须分别做精确路由审计；所有 inactive identity、无样本删除、全量支持、nested subject 隔离必须机器检查。

主代理在两轮pre-run审查及四项NO-GO修复闭环后，已授权并完成一次同人口训练侧正式筛查。正式结果为 `independent_gate_failed_no_additive`：EEG与trait独立门均失败，additive/interaction按条件未运行；outer-test始终关闭。该授权不允许第二次搜索、改门或打开lzh。
