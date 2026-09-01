<!-- STIMULUS_CENTERED_MULTIACTION_V2 -->
# 模型状态

## 总体多动作任务

尚未训练正式模型，当前没有可报告的多动作、个体化、生理调制或车辆响应预测性能。Transformer、TCN、GRU、Mamba及其他深度网络本轮均未运行。

只有在刺激语义映射、候选标签人工抽查、样本门和验证协议冻结后，才允许进入模型阶段。

## 历史 release 转向子任务

ExtraTrees-134D仍是旧合并38名驾驶员协议的最强实用基线：subject-macro curve MAE为14.1103度。

Run82中LGRS优于参数配平Role-TCN，但显著落后ExtraTrees。该结论只说明旧单一方向盘均值曲线任务中的神经序列家族没有晋级，不能外推为Transformer不适合新的刺激中心多动作任务。
