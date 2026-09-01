# Run72 当前状态与正式结论

Run72 已完成严格因果原始 EEG 状态与先前 session 驾驶特质的同人口训练侧筛查。权威结果目录为 `run_1_training_screen/`。

- 状态：`independent_gate_failed_no_additive`
- `same_population_developmental=true`
- `external_confirmation=false`
- strict fresh subjects=`1`
- `outer_test_opened=false`，outer-test预测行=`0`
- 所有2323事件保留；pooled仅作诊断

正式支持门全部通过：EEG real/shift共同支持2145事件、18名被试；trait 1920事件、16名被试；joint 1765事件、16名被试。

正式效果门均失败：

- EEG相对B、quality、shifted均因15/15 contexts训练侧选择trust=0而精确回退，净增量为0；
- EEG相对同容量车辆控制仅 `+0.00504°`，1/5 context为正，bootstrap下界 `-0.00671°`，leave-top为负，且最坏保护退化 `0.02280° > 0.02°`；
- trait相对B为 `-0.00256°`，相对同容量车辆为 `-0.00333°`，相对permuted trait为 `-0.00256°`，三项均0/5为正；
- trait最坏保护退化 `0.03072° > 0.02°`。

因此 EEG 与 trait 两套独立门都失败。按冻结条件，`A_additive` 与 `I_interaction` 均未运行；这不是它们有数值证据的no-go，而是上游独立门失败后的 `not_run_by_gate`。

完整数值和证据边界见：

- `run_1_training_screen/RESULT_CN.md`
- `run_1_training_screen/decision.json`
- `run_1_training_screen/tables/pair_summary.csv`
- `run_1_training_screen/EXPERIMENT_AUDIT.md`

当前结论只适用于P_full同人口发展性nested筛查，不能写成EEG普遍无用、外部确认或outer-subject正式结果。lzh N=1不因本轮失败自动打开。
