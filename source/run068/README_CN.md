# Run68：rolling-V 尾段生理区间校准

Run68 是当前数据上最后一个生理预测用途。它固定 Run67 `rolling_V` 均值，只允许生理调整第 9–20 点的区间尺度。

## 文件

- `config.json`：冻结输入哈希、模型、80% 同时覆盖、四臂和全部硬门。
- `CONTRACT_CN.md`：中文科学合同。
- `experiment.py`：严格重算 Run67 Stage A context、尺度 OOF、经验校准、评价与裁决。
- `plot.py`：正式训练侧结果存在后生成覆盖—宽度和 risk-coverage 两幅图。
- `notes.txt`：实施记录；本次没有正式结果。

## 本次允许的验证

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '<PYTHON_311>\python.exe' '.\experiment.py' --smoke-test
```

纯合成 smoke 不读取真实数据、不建立输出目录、不运行 5×3 context。

## 以后若经人工批准的唯一训练侧命令

```powershell
& '<PYTHON_311>\python.exe' '.\experiment.py' --run-training-screen --out-dir run_1_training_screen
```

- `--run-training-screen` 是显式保险；缺少它时程序拒绝正式运行。
- 输出目录已存在时拒绝覆盖。
- 程序没有 outer-test 入口。
- 正式运行仍只是同一 P_full 上的 training-side meta-validation，不是独立确认。

## 预期资源

- 复用已存在的 Run67 vehicle/physiology cache；不重新读取原始 2.63/4.62 GiB CSV。
- 重新计算 15 个 Run67 Stage A contexts，再拟合约 180 个小 Ridge 尺度头。
- 预计 CPU 数分钟到十几分钟，内存低于 2 GiB，无 GPU 需求。

