# Run72 使用说明

当前状态：正式同人口训练侧筛查已完成，权威结果在 `run_1_training_screen/`；EEG和trait独立门均失败，additive/interaction未运行，outer-test始终关闭。以下命令保留用于复现输入与测试边界：

1. 正式输入 preflight（只读 Run71/Run65，零模型拟合）：

   `<PYTHON_ENV>\python.exe experiment.py --preflight`

2. 隔离合成 smoke（人工信号，不是科学结果；当前最终证据目录为 `smoke_final4`）：

   `<PYTHON_ENV>\python.exe synthetic_smoke.py --out-dir smoke_final4`

3. 单元测试：

   `<PYTHON_ENV>\python.exe -m pytest -q tests`

正式运行曾由主代理在两轮审查后通过双重锁执行一次；append-only结果目录已存在，不得用新目录改参数重跑或搜索。`config.json` 保留当次授权状态以匹配正式provenance，不表示允许再次执行。

本目录不得读取或执行任何 `verify_*`。上游 Run57/64/65/71 均为只读；所有输出只能写入新的 Run72 子目录，存在即拒绝覆盖。

已保留的 `smoke_final4/` 是当前代码/配置哈希对应的最终合成路径测试。为让小型人工数据穿透 additive/interaction 条件分支，合成脚本只在深拷贝配置中降低支持/增益门，并把 candidate-level 与 pair-level 的人工分层 harm 上限临时设为 `1.0°`；正式 `config.json` 的 `0.02°` 和全部原始门值完全未改。该 formal-flow smoke 的人工事件均为 EEG 与 trait 同时 active，它证明执行顺序、普通mask传播、三候选选择、donor审计、表结构和 interaction 条件路径；`both_active / eeg_only / trait_only / neither` 四状态精确路由由 `tests/test_run72.py` 的独立单元测试覆盖。两者都不能证明真实效果或正式门会通过。
