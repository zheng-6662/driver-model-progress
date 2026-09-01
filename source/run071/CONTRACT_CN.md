# Run71 原始 EEG 因果状态缓存冻结合同

## 1. 本轮只做什么

本轮只从 `P_full=2323` 事件清单、专业 `recording_manifest.parquet` 的录制起点/原始 EEG 路径，以及原始 EEG CSV 建立可复用缓存。它不训练模型，不读取目标曲线数值，不按目标或效果挑事件，也不拟合切空间参考、PCA 或任何跨事件变换。

正式人口固定为 Run57-A 的 `pfull_event_manifest.csv` 全部 2323 条。脚本只从这个宽表读取四列：`event_uid`、`subject`、`recording_uid`、遗留锚点列 `primary_release_s`。新产物统一把该时刻称为 `prediction_anchor_s` / 预测起点。

## 2. 严格禁止的输入与动作

- 不读取 `strict_eeg_path`、任何 `.fif`、`offline_research`、`cutoff_safe_teacher`、`causal_online` 数值或未来 QC。
- 不读取 P_full 清单中的目标曲线、峰值、响应分层、风险标签、幅值分层或模型结果。
- 不删除、覆盖已有 Run71 产物。所有写入使用独占创建；同名文件已存在即停止。
- 不生成 `verify_*.py`，不训练模型，不做目标驱动选择。

## 3. 时间和事件锚点

`StorageTime` 是上海本地墙上时间。先按 `Asia/Shanghai` 解释，再转换成 UTC epoch ns。每个事件的预测起点固定为：

`prediction_anchor_ns = recording_manifest.start_time_ns + round(primary_release_s * 1e9)`。

必须报告原始 CSV 第一行时间与 manifest 起点之差，绝对值不超过 100 ms 只作为对齐合理性报告，绝不按事件把时间平移到原始第一行。主窗为 `[anchor-8 s, anchor]`，移位窗为 `[anchor-38 s, anchor-30 s]`，端点包含，二者不重叠。

## 4. 通道、ROI 和缺失处理

固定 32 通道映射与四个 ROI 逐字保存在 `config.json`。原始值非有限或 `abs(value)>=399999` 时无效。每个通道只可使用过去最后一个有效值做最多 20 ms 的前向填充；超过 20 ms 后输入为 0 且有效标志为 0，绝不反向填充。

ROI 在每个时间点只对本 ROI 中“原始有效或过去 20 ms 内因果填充有效”的通道取中位数。至少一个局部通道有效时，该 ROI 的该时间点才有效；否则 ROI 输入为 0、有效标志为 0。全程保留设备原始固定参考，不做共平均重参考。

## 5. 严格前向滤波

每条录制根据所选 EEG 行时间戳的严格正时间差中位数估计 fs，预期约 500 Hz。Nyquist 允许时使用 50 Hz、Q=30 陷波，随后使用 4 阶 1–40 Hz Butterworth SOS；只准 `scipy.signal.sosfilt` 正向运行。

录制开始和每次重置时，`sosfilt_zi` 只乘该段第一个“当前”ROI 样本进行初始化，不借用后续样本。录制内状态连续；时间戳间断 `>0.1 s` 时重置状态，并将断点后 5 s 标成 warmup 无效。录制开头不是人为断点。禁止 `filtfilt`、`sosfiltfilt`、MNE 零相位和重采样。

如果时间不是非降序、StorageTime 无法可靠解析，或 fs 无法支持固定滤波器，整条录制仍保留全部事件，但信号特征不可用并写明原因；不排序、不修时间、不临时换滤波器。

## 6. 活跃判定

主窗和移位窗分别判定。8 s 的预期样本数固定为 `round(8*fs)`；ROI 有效率用有效样本数除以该预期数，因此缺失时间行不会被缩小分母而掩盖。窗口活跃必须同时满足：

1. 至少 3/4 个 ROI 各自有效率 `>=80%`；
2. 窗口终点前最后一个所选 EEG 时间戳的滞后 `<=20 ms`；
3. 窗口中没有任何断点后 warmup 样本。

窗口活跃不删除任何样本。窗口不活跃时该窗全部 46 个信号特征为 NaN，availability/active 标志为 0；质量特征尽可能保留，以说明为何不可用。自然缺失 EEG 的事件也必须保留。

## 7. 固定特征

每个活跃窗口生成 46 个信号特征：

- 24 个频谱特征：4 ROI × theta 4–8 / alpha 8–13 / beta 13–30 Hz × `log(绝对功率+1e-30)` / 相对 1–40 Hz 功率。Welch 段长 2 s、50% 重叠、常数去均值。
- 22 个协方差特征：四 ROI 信号和一阶差分分别用 `LedoitWolf` 估计 4×4 协方差，各自除以 trace 后按固定 ROI 顺序取上三角 10 项，并各保存一个 log trace。

主窗另外保存 11 个固定质量特征。精确名称、顺序和公式全部写入 `config.json`；最终 NPZ 也内嵌 Unicode 名称。主窗和移位窗不拟合切空间参考或 PCA。

## 8. 因果审计与停止边界

允许为了离线建缓存而从录制开头顺序滤到录制末尾，但每条事件提取只能索引该窗口终点及以前的已存前向输出。必须审计所有有限 `max_support - prediction_anchor <= 1e-6 s`；任何未来支持、特征数错误、事件丢失、重复 UID 或 smoke 失败都阻止正式全量写入。

正式全量之前固定 smoke 两例：一个 byx 正常原始 EEG 事件和一个 rjy 自然缺失 EEG 事件。前者必须完成因果提取且支持不越过预测起点；后者必须保留、active=0、46 维信号特征全 NaN。

