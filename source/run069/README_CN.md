# Run69：t0+0.4 rolling vehicle 正式 outer OOF

Run69 是 result-to-claim 授权的唯一下一项实验。它把 Run67 训练侧 Stage A 提升到正式五折 outer OOF，但评估任务已经改变：等待 0.4 s、观察点 1–8 后，只预测点 9–20。

正式五折 outer OOF 已在 append-only 的 `run_2_outer_oof/` 完成。`run_1_outer_oof/` 仅保留一次模型拟合前的缓存字符串加载失败记录，没有正式结果。

正式裁决为：`advance=false`。总体和配对改善很强，但 Fold 3 `road_reference_missing` 相对 initial/pre-only 分别退化 `+2.577°/+3.864°`，违反冻结的 `+0.02°` 保护门。完整性审计为 `WARN`：数字真实、无已发现的外层泄漏，但存在 changed-estimand、single-seed 和内置coverage审计部分自证等限定。

## 文件

- `config.json`：冻结输入哈希、changed estimand、泄漏边界、模型、报告和门。
- `experiment.py`：严格 B 拼接、outer-test base 重训、rolling/pre-only Ridge、2323 行 OOF 与完整报告。
- `CONTRACT_CN.md`：中文科学合同。

## 泄漏防线

- outer-train B 只拼接当前 outer 的三个 Run65 nested `validation_base_predictions`。
- 普通 Run63 train-row OOF 预测不读取、不使用。
- outer-test B 默认重训，不复用 ordinary formal B：三条冻结专家只拟合当前 outer-train，然后预测当前 outer-test。
- rolling-V trust 固定 1.0，不在 outer-test 调参。
- pre-only trust 只在 outer-train 三个 subject-disjoint meta 分片上 crossfit 选择。

## 安全烟测

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '<PYTHON_311>\python.exe' '.\experiment.py' --smoke-test
```

烟测只用合成数据，不载入真实 cache、不训练正式基模型、不写文件。

## 已执行的正式命令

```powershell
& '<PYTHON_311>\python.exe' '.\experiment.py' --out_dir=run_2_outer_oof
```

## 预计资源

- 五折 outer-test B 重训是主要开销：每折包含 ExtraTrees、20 个 LightGBM、20 个 HistGradientBoosting；预计约 10–30 分钟，受 CPU 和磁盘影响。
- rolling-V/pre-only 是小型 CPU Ridge，预计额外数分钟。
- GPU/显存需求为零。
- 输出的 2323 行 OOF 表预计为数 MB 至十余 MB。

## 解释边界

Run69 只说明：在等待 0.4 s 并观察前 8 点之后，vehicle-only rolling update 对剩余点 9–20 的正式 subject-disjoint outer OOF 在总体上有大幅收益。它不能写成 t0 时刻完整 1 s 预测性能，也不能使用 physiology/style/context/KD 解释结果。由于道路参考缺失层保护失败，它不是满足全域保护的可推进模型。结果来自单一冻结seed配置和同一proxy-event总体。
