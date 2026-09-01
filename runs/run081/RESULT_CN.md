# Research Proposal：Lag–Gain Relation State Network

## Problem Anchor

- **Bottom-line problem：** 在车辆失控/近失稳代理事件的 release 时刻，只使用当时及以前可获得的信息，跨被试预测 release 后 `+0.05…+1.00 s` 的20点完整方向盘响应曲线。
- **Must-solve bottleneck：** 当前 ExtraTrees/LightGBM/HistGBM 都建立在134/172维静态摘要上。它们能表示“过去窗口里有什么”，但弱化了“什么时候发生、正在加速还是衰减、驾驶指令与车辆响应之间相差多久”。新增驾驶员扩大了主体覆盖，但直接合池和直接拼接生理仍没有稳定收益。
- **Non-goals：** 不改成风险分类；不把 release 后观察偷进主任务；不再次尝试 Run62 已失败的幅值—形状因子化；不重新打开Run63低秩残差/软门控或Run64–Run80直接生理/风格拼接；不使用subject ID；不把局部改善写成总体有效。
- **Constraints：** 原18人2323事件、八月275事件、潜在38位驾驶员；共同车辆输入兼容八月缺失`vyaw/vroll`；release前2秒；20点目标；被试不相交；RTX2060 6GB；小模型。
- **Success condition：** LGRS必须稳定超过参数配平Role-TCN，并超过无rate ExtraTrees；subject bootstrap下界>0，至少4/5折正向，四个≥20°幅值档无超过0.01退化，收益不由单个被试支配。

## Technical Gap

树模型和GBM在静态摘要上接近，只说明该摘要+树族趋于饱和。Run62又说明5节点幅值—形状输出和8个手工相位标量都不能解决问题。仍未被干净测试的是：

> 让模型在完整release前序列上，显式估计转向指令到车辆横向响应的多个候选时滞与增益，再把“已观测响应减去期望响应”的关系残差用于未来曲线预测。

普通TCN可以隐式学习这一点，但没有强制它这样做。在小样本跨被试数据中，模型更容易利用幅值、主体或批次差异。LGRS通过固定lag bins和低秩gain限制自由度，使“关系状态”成为可检查的模型对象。

## Method Thesis

- **One sentence：** 在50 Hz release前序列上，用固定物理时滞候选与事件条件化低秩增益估计expected vehicle response，仅将command/context状态与压缩relation residual送入曲线头，从而显式表征驾驶指令—车辆响应错位。
- **Dominant contribution：** Lag–Gain Relation State（LGRS）瓶颈。
- **No supporting trainable contribution：** 首轮无生理、风格、预训练、路由或结构化输出。

## Frozen Inputs and Output

- 时间窗：release前2.0秒。
- 采样：50 Hz，共101点，直接复用Run57 causal cache尺度。
- 共同7通道：`steer_smooth, steer_rate, speed_kmh, ay, roll, curvature, lateral_distance`。
- road missing：道路两通道置0并加入1维mask。
- target：原20点aligned_deg，不改输出形式。
- 同名驾驶员跨旧/八月session必须在同一fold。

### August 101-point alignment contract

八月事件不另写一套序列插值：

1. 从Run78的staged vehicle源读取同一物理字段。
2. 使用Run57/Run76共用的`causal_endpoint_savgol`处理steer/steer_rate。
3. 事件网格固定为`np.linspace(release-2.0, release, 101)`。
4. 每个通道使用同一`causal_hold`，不做双向插值。
5. 与Run57相同，对directional channels乘事件direction。
6. road不可用时curvature/lateral置0并保留missing mask。
7. 从9通道共同定义中删除`yaw_rate/roll_rate`后得到7通道；其余通道顺序不变。

在smoke中，先对Run76公共148事件重建101点序列并逐值比较公共通道；不一致则停止，不训练模型。

## Prefix Representation

每个事件只使用该事件101点前缀做归一化：

\[
\tilde x_c(t)=\frac{x_c(t)-\operatorname{median}(x_c)}{1.4826\operatorname{MAD}(x_c)+\epsilon}
\]

同时保留每通道`median/MAD/release value`形成绝对尺度向量。该向量的缩放参数只在outer-train拟合。

角色划分：

- command `u(t)=[steer_smooth, steer_rate]`
- operating/context `q(t)=[speed, curvature, lateral, road_missing]`
- response `r(t)=[ay, roll]`

方向相关通道保持现有direction alignment。

## Shared Temporal Stem

Plain Role-TCN与LGRS共享：

- 每个角色一个两层causal Conv1d stem；hidden width=16。
- kernel=5；dilation=1和4；SiLU；LayerNorm；dropout=0.10。
- 输出时间长度101，不做stride降采样，避免lag位置含义改变。
- 角色latent：`h_u(t), h_q(t), h_r(t)∈R^16`。

训练臂之间共享同一优化器、head、hidden width和early stopping。通过增加无偏置1×1 projection，使plain control与LGRS总参数差不超过5%。

## Fixed-Lag Conditional Gain Operator

固定lag集合（50 Hz采样点）：

\[
\mathcal L=\{0,1,2,4,6,8,12\}
\]

对应：

\[
\{0,20,40,80,120,160,240\}\text{ ms}
\]

先从operating/context前缀得到事件级8维token，生成头固定为单个affine layer加SiLU：

\[
z_q=\operatorname{SiLU}(W_q[h_q(T),\operatorname{mean}_t h_q(t),s_{abs}]+c_q)\in\mathbb R^8
\]

gain不由自由卷积产生，而由完全写死的rank-2分解生成。全局response factor为：

\[
A=\tanh(A_{raw})\in\mathbb R^{2\times2}
\]

事件条件command×lag factor只用一个线性头：

\[
B(z_q)=0.5\tanh(W_gz_q+c_g)\in\mathbb R^{2\times7\times2}
\]

然后：

\[
g_{j,k,\ell}(z_q)=\sum_{m=1}^{2}A_{j,m}\,B_{k,\ell,m}(z_q)
\]

其中`j∈{ay,roll}`，`k∈{steer,steer_rate}`，`ℓ∈L`。response bias也只用单个线性头：

\[
d(z_q)=W_dz_q+c_d\in\mathbb R^2
\]

实现中禁止把`W_g/W_d`替换为多层MLP。事件最多产生28个gain，但参数通过rank-2共享，避免完全自由的2×2×7动态卷积。

期望响应在原始归一化响应空间计算：

\[
\hat r_j(t)=d_j(z_q)+\sum_{k=1}^{2}\sum_{\ell\in\mathcal L}g_{j,k,\ell}(z_q)\tilde u_k(t-\ell)
\]

越界的`t-ℓ`使用最早prefix值，不使用未来。

关系残差：

\[
e(t)=\tilde r(t)-\hat r(t)
\]

## Relation Bottleneck

LGRS下游禁止直接接收`h_r(t)`、raw response或`\hat r(t)`。它只接收：

1. command latent `h_u`
2. context latent `h_q`
3. residual encoder `h_e=Conv_{causal}(e)`，宽度8
4. gain摘要：每个response的gain质心时滞、总绝对gain、正负gain平衡，共6维
5. absolute scale vector

对`h_u,h_q,h_e`分别取last、全局mean和最近5点mean，拼接后经64维MLP，再用共享线性head输出20点。

因此，若LGRS优于control，不能简单解释为“又看了一遍raw ay/roll”：raw response进入预测的唯一通路是经过低自由度expected-response扣除后的residual。

## Parameter-Matched Controls

1. **ExtraTrees-134D：** 当前强基线。
2. **Plain Raw-TCN：** 7通道一起进入参数配平TCN。
3. **Role-TCN Control：** 与LGRS共享三个role stems、pooling和20点head；把`h_u,h_q,h_r`直接plain fusion，不含lag–gain算子。
4. **LGRS：** 只把response通路替换为fixed-lag operator+residual bottleneck。
5. **LGRS-λ0：** 架构相同，但关系辅助损失权重为0。

主方法证据是LGRS vs Role-TCN Control；ExtraTrees和Plain Raw-TCN回答更宽泛的实用比较。

## Loss and Training Contract

\[
L_{curve}=\frac1{20}\sum_t|\hat y_t-y_t|
\]

\[
L_{rel}=\operatorname{mean}_{j,t}M_{j,t}\,\operatorname{Huber}(\hat r_j(t),\tilde r_j(t))
\]

\[
L=L_{curve}+0.10L_{rel}+0.05\operatorname{Huber}(\Delta\hat y,\Delta y)
\]

- AdamW，learning rate=`3e-4`，weight decay=`1e-3`。
- batch size=64；gradient clip=1.0。
- 最大150 epoch；outer-train内按subject划inner validation；subject-macro MAE patience=15。
- 固定seed=`20260831, 20260832, 20260833`。
- 不做网格；运行后不改lag集合、hidden width或loss权重。
- subject-first采样；旧队列与八月域在batch层面等权。该平衡是协议，不是论文贡献。
- `L_rel`始终在prefix-normalized response空间监督；`M`只屏蔽真实缺失/road无关，不使用未来或标签构造。

## Inference

输入release前101点共同7通道；计算prefix normalization、fixed-lag gains、relation residual和20点曲线。没有release后观测，没有target-subject历史真值，没有生理/style，也没有模型路由。

## Mechanism Diagnostics

1. **lag perturbation：** 评价时把command序列在prefix内循环平移±2/±4点；不重训。若LGRS真正依赖时滞，误差和gain质心应系统变化。
2. **role shuffle（附录级）：** 在同一outer-test内对response通道做被试内事件置换，仅作机制破坏诊断；正式预测仍用真实数据。
3. **gain visualization：** 分被试/幅值档画gain质心和relation residual能量，与曲线继续转向/回舵差异对照。
4. **parameter audit：** 所有神经臂参数量差≤5%。

## One Primary Claim

> 显式固定时滞lag–gain关系瓶颈，比参数配平的普通role fusion更有效地把release前驾驶指令—车辆响应错位编码为跨被试完整转向曲线预测状态。

数据合并方式、生理、风格、自监督均不是本轮claim。

## Minimal Validation

### Core experiment

- 数据：共同7通道，2598事件；同名驾驶员跨批次同fold。
- 主对照：Role-TCN Control vs LGRS，3 seeds。
- 实用对照：ExtraTrees134D、Plain Raw-TCN。
- 主门：LGRS相对Role-TCN subject-macro改善≥0.10°、subject bootstrap下界>0、≥4/5 folds正向、四个≥20°幅值档无>0.01退化。
- 实用门：LGRS相对ExtraTrees改善≥0.20°且同样满足bootstrap/折/幅值保护。

### Necessity ablations

- LGRS vs LGRS-λ0：判断显式response reconstruction是否必要。
- LGRS vs Role-TCN：判断relation bottleneck是否必要。
- lag perturbation：判断模型是否使用时间关系，而非只用幅值。

### Supporting domain table

- original18、August-new20、combined38分别报告subject-macro。
- naive pooled ExtraTrees只作为Run76方向的负对照，不构成第二claim。

## Stop Rule

- 若LGRS不超过参数配平Role-TCN，或CI排除0.10°以上收益：关闭关系状态网络路线。
- 若Role-TCN超过ExtraTrees但LGRS不超过Role-TCN：结论是raw sequence有用，但LGRS不是贡献；不包装为LGRS论文。
- 若两种神经模型均失败：不再换Transformer/Mamba/NCDE；转向道路/事件语义或changed-estimand。
- physiology/style只有在另一个明确课题中讨论，本Run81不打开。

## Compute and Timeline

- 输入101点，模型目标5万–8万参数。
- 单折单seed预计RTX2060小于40分钟；5折×3seed约4–8 GPU小时。
- 首先跑一个outer-train内部smoke和参数配平检查；通过后一次性全量。
