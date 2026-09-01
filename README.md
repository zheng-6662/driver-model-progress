<!-- STIMULUS_CENTERED_MULTIACTION_V2 -->
# 极端工况驾驶员模型研究进展

本公开仓库保存可供 GPTPro 审查的项目事实层、历史 Run 卡、聚合结果、公开安全图和必要源代码。当前总体任务已经从“release 后 1 秒方向盘曲线”重置为“极端/高动态刺激后的个体化多动作驾驶员反应”；历史转向任务保留为已完成子任务和基线，不再代表整个课题。

## 当前阅读顺序

1. [GPTPRO_CONTEXT_CN.md](GPTPRO_CONTEXT_CN.md)
2. [PROJECT_BACKGROUND_CN.md](PROJECT_BACKGROUND_CN.md)
3. [CURRENT_STATUS_CN.md](CURRENT_STATUS_CN.md)
4. [AUDIT_INDEX.md](AUDIT_INDEX.md)
5. [多动作任务修正V2审计](audits/multiaction_task_reframe_v2_20260901/README.md)
6. [RUN_INDEX.md](RUN_INDEX.md)
7. [REQUEST_TO_GPTPRO_CN.md](REQUEST_TO_GPTPRO_CN.md)
8. [PUBLISH_WORKFLOW_CN.md](PUBLISH_WORKFLOW_CN.md)

## 当前阶段

状态为 `CONDITIONAL_READY`：已从连续 recording 建立刺激中心候选事件、三通道候选标签、样本/个体化/生理/车辆目标审计，但原始外部触发编号到具体交通动作的权威语义、8月交通触发阈值仍未完全恢复。因此当前可交由 GPTPro 审查数据合同和决策门，尚不能启动正式多动作模型训练。

## 隐私边界

公开仓库不发布原始车辆、生理、EEG、眼动、视频、checkpoint、逐事件预测、一行一驾驶员统计、匿名ID对应表、原始文件清单或本机路径。私有审查包与本仓库物理分离。
