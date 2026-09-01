# Run Index

| Run | Status | One-line conclusion | Card |
|---:|---|---|---|
| 57 | `validated_baseline` | P_full=2323的严格因果基线；ExtraTrees是强基线，但不是全事件无伤模型。 | [Run57](runs/run057/RUN_CARD_CN.md) |
| 58 | `blocked` | 复现锚点口径不一致，按合同停止；道路预览未被正式评价。 | [Run58](runs/run058/RUN_CARD_CN.md) |
| 59 | `diagnostic` | 条件方差/近邻可辨识性审计；地板仅是尺度参照，不是理论不可约噪声。 | [Run59](runs/run059/RUN_CARD_CN.md) |
| 60 | `no_go` | LightGBM/HistGBM在同一静态摘要上没有稳定超过ExtraTrees。 | [Run60](runs/run060/RUN_CARD_CN.md) |
| 61 | `diagnostic_only` | 道路预览残差修正退化；车辆侧已高度隐含弯道条件。 | [Run61](runs/run061/RUN_CARD_CN.md) |
| 62 | `no_go` | 幅值—形状因子化和8个控制相位标量没有形成有效增量。 | [Run62](runs/run062/RUN_CARD_CN.md) |
| 63 | `no_go` | 低秩残差和软门控存在小信号，但没有达到冻结晋级门。 | [Run63](runs/run063/RUN_CARD_CN.md) |
| 64 | `no_go` | 生理、驾驶风格、TCN、FiLM、BIOT等训练侧筛选没有稳定独立增量。 | [Run64](runs/run064/RUN_CARD_CN.md) |
| 65 | `training_side_no_go` | 多模态教师/学生和残差蒸馏训练侧有局部信号，但没有通过进入outer的成对门。 | [Run65](runs/run065/RUN_CARD_CN.md) |
| 66 | `no_go` | 生理—车辆耦合适配器没有通过训练侧门，并伤害普通/尾部层。 | [Run66](runs/run066/RUN_CARD_CN.md) |
| 67 | `no_go` | 异步生理更新Gate A通过、Gate B失败，未打开outer。 | [Run67](runs/run067/RUN_CARD_CN.md) |
| 68 | `no_go` | 生理区间校准没有改善区间宽度、区间分数或风险选择。 | [Run68](runs/run068/RUN_CARD_CN.md) |
| 69 | `changed_estimand_guardrail_fail` | t0+0.4s滚动车辆更新显著改善尾部，但改变估计目标且道路缺失保护门失败。 | [Run69](runs/run069/RUN_CARD_CN.md) |
| 70 | `inventory` | 盘点新鲜多模态外部候选，确认额外车辆/生理/EEG与纵向风格材料。 | [Run70](runs/run070/RUN_CARD_CN.md) |
| 71 | `preprocessing` | 构建2323事件release前因果原始EEG状态缓存；没有训练模型。 | [Run71](runs/run071/RUN_CARD_CN.md) |
| 72 | `no_go` | EEG状态与历史风格在训练侧没有独立增量，outer未打开。 | [Run72](runs/run072/RUN_CARD_CN.md) |
| 73 | `no_go` | 原18人眼动增量未超过车辆基线。 | [Run73](runs/run073/RUN_CARD_CN.md) |
| 74 | `screening` | zyl筛出8条高速度、目标与生理完整事件。 | [Run74](runs/run074/RUN_CARD_CN.md) |
| 75 | `ablation` | 去除vyaw/vroll后整体MAE影响很小，但尾部/个别被试有差异。 | [Run75](runs/run075/RUN_CARD_CN.md) |
| 76 | `no_go` | 直接把早期148条八月事件加入旧训练使原18人MAE退化约0.35度。 | [Run76](runs/run076/RUN_CARD_CN.md) |
| 77 | `no_go` | 八月18人内部车辆+简化生理没有形成增量。 | [Run77](runs/run077/RUN_CARD_CN.md) |
| 78 | `screening` | 八月全量重筛得到275条事件、26位有合格事件被试。 | [Run78](runs/run078/RUN_CARD_CN.md) |
| 79 | `preprocessing` | 完成27位、188个recording的四通道生理正式预处理。 | [Run79](runs/run079/RUN_CARD_CN.md) |
| 80 | `no_go` | 正式清洗生理16维仍未超过车辆或旧生理特征。 | [Run80](runs/run080/RUN_CARD_CN.md) |
| 81 | `method_ready` | 三轮方法审查将LGRS方案细化到READY，但尚无实验正证据。 | [Run81](runs/run081/RUN_CARD_CN.md) |
| 82 | `mechanism_yes_model_no_go` | LGRS稳定超过Role-TCN，但显著落后ExtraTrees；机制增量成立，主模型晋级失败。 | [Run82](runs/run082/RUN_CARD_CN.md) |
