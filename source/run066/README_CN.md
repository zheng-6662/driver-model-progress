# Run66：生理—车辆耦合适配器

这是 Run65 原始生理序列路线达到数学 no-go 后的新训练侧筛选实现。它不再把 30 s 原始生理序列直接送入神经网络，而是只检验四个事先写死、可解释的“生理变化×近期车辆需求”是否能为冻结 B_all3/V 曲线提供低维残差修正。

## 当前完成情况

- 已冻结输入哈希、四个耦合量、三个控制、模型、trust 网格、分折与停止门。
- 已实现 `experiment.py`。
- 正式 Run66 **没有运行**，当前没有可以报告的 Run66 实验数值。
- 只允许合成烟测；烟测不会读取 P_full、Run64 或 nested cache，也不会写结果文件。

详细科学合同见 [CONTRACT_CN.md](CONTRACT_CN.md)，机器可读配置见 [config.json](config.json)。

## 文件

- `config.json`：全部冻结路径、SHA-256、四个特征公式、模型和硬门。
- `CONTRACT_CN.md`：中文协议、时间边界、禁止字段与结果解释边界。
- `experiment.py`：自包含训练侧实现；不从 Run63/64/65 Python 源码动态导入逻辑。

## 两条命令

仅合成烟测：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '<PYTHON_311>\python.exe' '.\experiment.py' --smoke-test
```

以后经人工批准才可运行正式训练侧筛选：

```powershell
& '<PYTHON_311>\python.exe' '.\experiment.py' --out_dir=run_1_training_screen
```

不要在本次实现阶段执行第二条命令。

## 设计概要

1. 每个 outer 只使用它的训练被试；三个 meta context 来自 Run65 已冻结的严格 nested B_all3 缓存。
2. `V_vehicle` 用四个近期车辆需求预测固定 3 维 DCT 残差。
3. 主候选只用四个耦合量修正 cross-fitted V；控制分别是 quality-only、同维 vehicle×vehicle、同 recording 且严格向过去移动的 physiology shadow。
4. 所有 imputer/scaler/Ridge/trust 选择都只在当前 meta-fit；trust 固定为 `0/0.10/0.25`。
5. 生理不可用时保留事件，所有 delta 臂逐点精确回退 V。
6. 结果必须同时给出 B/V/shifted 成对门、18 被试 bootstrap、leave-top、改善被试数，以及 ordinary、road-missing、尾段伤害。

## 预期正式输出

正式运行成功后，新的输出目录会包含：

- `audit/preflight.json`、`audit/context_audit.json`；
- `tables/fit_only_trust_selection.csv`；
- `tables/meta_validation_predictions.csv`；
- `tables/pair_by_outer_context.csv`、`pair_by_subject_and_outer.csv`、`pair_summary_18subjects.csv`；
- `tables/protected_harm_by_outer.csv`、`protected_harm_by_subject.csv`、`protected_harm_summary.csv`；
- `outputs/decision.json`、`outputs/provenance.json`；
- `RESULT_CN.md`、`final_info.json`。

这些仍然只是训练侧发展性证据；`advance_to_outer=true` 也不能写成正式外部泛化改善。
