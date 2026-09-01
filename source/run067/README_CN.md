# Run67：t0+0.4 s 异步更新

Run67 把问题拆成两个真正可停止的阶段：先问 0.4 s 新车辆证据能否可靠改善初始尾段；只有答案为“能”，才问同一 0.4 s 内的新生理证据是否还提供独立增量。

当前只完成实现与安全烟测，**没有构建完整缓存、没有启动正式训练侧运行、没有 Run67 数值结果**。

## 文件

- `config.json`：机器可读冻结合同、输入哈希、缓存结构、门与命令。
- `cache_common.py`：P_full 身份、路径、哈希和 append-only 共用逻辑。
- `build_vehicle_cache.py`：从 85 个原始车辆 recording 构造独立 t0+0.4 cache。
- `build_physio_cache.py`：从原始 PhysioLAB 构造 main/quality/pseudo-shifted 三套 15 维 cache；不读 Run64 post cache/mask。
- `experiment.py`：Stage A 全完成并通过后才加载 physiology cache 的训练侧实验。
- `CONTRACT_CN.md`：完整中文科学合同。
- `RUN67B_FOLLOWUP_CN.md`：独立的可选近期行为上下文 trust-modulator 方案；本轮未实现、未运行。

## 安全验证命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& '<PYTHON_311>\python.exe' '.\experiment.py' --smoke-test
& '<PYTHON_311>\python.exe' '.\build_vehicle_cache.py' --smoke-recording
& '<PYTHON_311>\python.exe' '.\build_physio_cache.py' --smoke-recording
```

三条命令都不写 cache/result。后两条各仅读一个 recording。

## 以后经人工批准的完整顺序

```powershell
& '<PYTHON_311>\python.exe' '.\build_vehicle_cache.py'
& '<PYTHON_311>\python.exe' '.\build_physio_cache.py'
& '<PYTHON_311>\python.exe' '.\experiment.py' --out_dir=run_1_training_screen
```

主实验在 Gate A 失败时不会打开 physiology cache。预先构建 physiology cache 只是独立的 raw 特征物化，不等于拟合/评估 Stage B。

## 资源预估

- 车辆 raw：85/85，约 2.63 GiB；完整顺序读取预计数分钟到十几分钟，取决于磁盘。
- 生理 raw：77/85 存在，约 4.62 GiB；CSV 解码与 SHA-256 是主要开销，预计约 15–40 分钟；其余 8 个 recording 保留并回退。
- 模型：全部是小型 Ridge；15 个 meta context、两阶段、少量 trust，预计 CPU 数分钟，显存需求为零。
- 单事件模型延迟在正式运行时逐事件实测；P95 达到或超过 50 ms 即停止。

## 结果边界

Stage A/B 即使全部通过，也仍是同一 P_full 上重复出现的训练侧 meta-validation。不能写成 outer 泛化、独立验证或部署收益。

稳定 prior-session style 已正式停止。为不破坏 Run67 核心的 Stage A→条件 Stage B 边界，六维近期行为上下文调节器被单独记录为 Run67b follow-up，不在 `experiment.py` 中执行，也不参与 physiology Gate B。
