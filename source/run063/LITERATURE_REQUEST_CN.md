# 下一步希望补充的文献方向

Run63 已经说明：简单低秩残差在 dense 域几乎没有稳定收益；逐事件软门控有小而稳定的信号，但 dense floor_ratio 改善只有 0.0077，低于预注册的 0.02。因此最需要补的文献，不是新的通用时序大模型，而是以下五类“具体机制”。

## A1：重复事件条件下的个体在线校准/少样本个体化

这是最高优先级。希望回答：同一驾驶员已经完成1至若干次事件后，怎样只用过去事件更新一个很小的校准器，同时保证当前事件不使用未来真值？

优先搜索词：

- personalized time series forecasting few-shot subject-specific online calibration
- driver-specific trajectory prediction few-shot adaptation repeated maneuvers
- predict-then-update calibration fixed backbone longitudinal subjects
- hierarchical Bayesian online adaptation individual random effects forecasting
- meta-learning system identification short time series unseen subject

希望论文具备：按人/设备/患者等实体留出验证；按时间顺序模拟先预测后更新；报告首事件和不同历史数量下的性能；最好有代码。

## A2：安全的 mixture-of-experts / stacking / oracle-gap 学习

Run63 的事后最佳模型空间很大，但软门控只吃到很小一部分。希望寻找“预测每个专家的条件风险/损失”“向固定基线收缩”“失败时精确回退”的方法。

优先搜索词：

- safe mixture of experts regression fallback baseline regret guarantee
- conditional model selection regression predict expert loss
- reliability-aware stacking cross-fitted regression group cross validation
- dynamic ensemble selection continuous regression oracle gap
- gated residual correction frozen backbone safe improvement

希望论文具备：回归而不是纯分类；严格 OOF/cross-fitting；报告相对等权平均的 regret；有 no-harm 或 safe fallback；适用于小样本多输出。

## B1：带协变量的函数型响应与监督低秩回归

希望寻找的不是普通 FPCA 重构，而是“输入 X 直接预测整条函数 Y”的监督式低秩方法，特别是残差曲线。

优先搜索词：

- functional response regression high-dimensional covariates supervised low rank
- functional partial least squares curve response prediction
- reduced-rank regression nonlinear covariates multi-output small sample
- supervised functional principal components regression residual curves
- sparse reduced rank regression multivariate response subject grouped validation

希望论文明确区分：输出表示能否低秩重构，与低秩系数能否从预测时输入中识别。

## B2：多步轨迹的选择性预测、拒答与联合不确定区间

当前模型分歧与事件误差相关约0.485，可能更适合识别“不可靠事件”，而不是改变均值曲线。

优先搜索词：

- selective prediction multi-step time series risk coverage
- conformal trajectory prediction correlated horizons joint coverage
- reject option regression time series forecasting uncertainty
- calibrated ensemble disagreement multi-output regression
- subject-block conformal prediction repeated measures trajectories

希望论文报告整条轨迹联合覆盖、区间宽度、risk-coverage 曲线，以及分组/时间相关条件下的校准边界。

## B3：驾驶员响应的灰箱/混合效应动力学模型

希望寻找能把车辆动力学先验、驾驶员随机效应和数据驱动残差结合起来的方法，而不是另一个黑箱序列编码器。

优先搜索词：

- driver steering response system identification mixed effects
- personalized driver model hierarchical state space steering
- grey-box driver response prediction latent parameter model
- nonlinear mixed effects trajectory prediction human control
- Gaussian process latent force model driver steering behavior
- Hammerstein Wiener NARX driver steering response identification

希望论文有真实驾驶员重复试验、跨驾驶员验证、短时转向响应曲线，并明确哪些变量在预测时可获得。

## 可以暂时不搜的内容

- 泛泛的 LSTM/GRU/TCN/Transformer/S4 性能比较；
- 依赖周车交互网格的普通自动驾驶轨迹预测；
- 只在随机行切分上报告结果的模型；
- 只优化生成多样性、不报告单条预测或联合区间校准的扩散/流模型；
- 使用当前预测事件部分未来真值、却没有明确 warm-start 部署协议的方法；
- 单纯换成 spline/FPCA 名称重做整目标幅值—形状分解。

## 你搜索后最好发给我的材料

每个方向优先给3至5篇，而不是几十篇标题。最好包含：

1. PDF或论文正式链接；
2. GitHub仓库；
3. 论文的训练/测试划分段落；
4. 是否使用测试阶段真值；
5. 数据量、实体数量和是否跨实体验证；
6. 主损失与正式评价指标；
7. 你认为最像我们问题的图或方法页。

如果只能先搜一个方向，请先搜 A1“重复事件后的合法个体校准”；如果能再搜一个，就搜 A2“安全的回归专家门控”。

