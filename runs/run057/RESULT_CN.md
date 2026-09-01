# Run57-A V3 严格因果基线结果

## 当前状态

P0 V3复评：`P/P/P/P/P/P/P`，主人口 `P_full=2323`；道路有效2124、道路缺失199。
CPU四臂已完成。GPU门：`通过，Transformer已运行`。

## 主结果

| 模型 | subject-macro曲线MAE° | pooled MAE° | tail5 MAE° | 峰时MAE s | P90峰值低估° |
|---|---:|---:|---:|---:|---:|
| M0_hold | 21.4595 | 22.7174 | 33.5435 | 0.7278 | 88.0408 |
| M0_linear_extrap | 35.7237 | 38.0728 | 73.6880 | 0.2254 | 20.0334 |
| M0_ridge | 14.5872 | 15.1596 | 24.1854 | 0.2057 | 42.1169 |
| M0_extratrees | 13.0214 | 13.6016 | 20.8229 | 0.1952 | 37.4353 |
| M0_transformer | 13.5027 | 13.9651 | 21.7934 | 0.1976 | 34.8998 |

最优点估计模型：`M0_extratrees`，subject-macro曲线MAE `13.0214°`。

## GPU硬门

- `primary_vs_hold`：通过
- `primary_vs_linear`：通过
- `lt30_vs_hold`：通过
- `lt30_vs_linear`：通过
- `30_45_vs_hold`：通过
- `30_45_vs_linear`：通过
- `45_70_vs_hold`：通过
- `45_70_vs_linear`：通过
- `ge70_vs_hold`：通过
- `ge70_vs_linear`：通过

GPU门放行，Transformer只在CPU经典基线过门后运行。

## 人口与解释边界

- 30°幅值不进入人口、权重、早停或GPU门，只用于连续量与分箱报告。
- 道路参考缺失199条全部保留，详见 `stratified_metrics.csv` 的road分层。
- `P_full`比历史参考1801多恢复522条旧事后裁决排除事件；冻结848只作历史子集。
- **新旧 pooled MAE 不可直接比较**。
- V3主人口 release可见幅值/未来完整脉冲峰值比值 median `0.713`、p10 `0.372`。
- 当前仍是在既有2827个离线检测事件宇宙内重建release人口，不等于在线事件发现器已经部署验证。

## 完整性

- 所有输入raw support不晚于release；道路缺失以显式指示进入模型，填补/标准化只在训练数据拟合。
- 五个outer fold都从各自外层训练被试中留3人内验；outer test从未用于超参或早停。
- 幅值箱与道路层、逐被试、保护指标、伤害/被试bootstrap均独立落表。
- `M1_preview`与Run57-B本轮均未授权。
