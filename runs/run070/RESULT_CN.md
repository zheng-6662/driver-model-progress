# Run70 新鲜多模态数据盘点

## 结论

- 扫描文件：**84**。
- 聚合 recording key：**43**。
- P_full 外的新鲜已解析被试：**hhx, lc, lzh, zsj**。
- 新鲜 external candidate session：**20**。
- 其中 EEG candidate：**16**。
- longitudinal style 可用被试：**hhx, lc, lzh, zsj**。
- cwh 已在 P_full，所有 cwh session 均排除出新鲜外部被试候选。
- 未知 `2025_09_26_21_32_09` 保持 unresolved；`(hzh)`/`hzh|` 是跨多个明确被试目录重复出现的 companion/entity 标记，不能当作被试证据。
- raw 源文件只读，未复制、未修改、未重命名。

## 被试汇总

| subject   | subject_in_pfull   | fresh_subject   |   total_session_count |   candidate_external_session_count |   candidate_eeg_session_count | longitudinal_style_usable   |
|:----------|:-------------------|:----------------|----------------------:|-----------------------------------:|------------------------------:|:----------------------------|
| cwh       | True               | False           |                     5 |                                  0 |                             0 | False                       |
| hhx       | False              | True            |                     6 |                                  4 |                             0 | True                        |
| lc        | False              | True            |                    12 |                                  6 |                             6 | True                        |
| lzh       | False              | True            |                    13 |                                  5 |                             5 | True                        |
| unknown   | False              | False           |                     1 |                                  0 |                             0 | False                       |
| zsj       | False              | True            |                     6 |                                  5 |                             5 | True                        |

## 新鲜 external candidate sessions

| subject   |   recording_timestamp |   vehicle_union_duration_s |   vehicle_steering_effective_rate_hz |   physio_ecg_effective_rate_hz |   physio_emg_effective_rate_hz |   physio_resp_effective_rate_hz |   vehicle_physio_overlap_s | candidate_eeg_external_session   |
|:----------|----------------------:|---------------------------:|-------------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|---------------------------:|:---------------------------------|
| hhx       |   2025_08_23_19_43_48 |                    357.112 |                               59.854 |                       1000.003 |                       1000.003 |                        1000.003 |                    357.112 | False                            |
| hhx       |   2025_08_23_19_50_08 |                    310.535 |                               59.996 |                       1000.003 |                       1000.003 |                        1000.003 |                    310.535 | False                            |
| hhx       |   2025_08_23_19_55_43 |                    313.852 |                               59.996 |                       1000.003 |                       1000.003 |                        1000.003 |                    313.852 | False                            |
| hhx       |   2025_08_23_20_08_30 |                    332.477 |                               54.680 |                       1000.003 |                       1000.003 |                        1000.003 |                    332.477 | False                            |
| lc        |   2025_08_23_14_18_29 |                    382.324 |                               60.052 |                       1000.003 |                       1000.003 |                        1000.003 |                    382.324 | True                             |
| lc        |   2025_08_23_14_25_11 |                    326.526 |                               54.668 |                       1000.003 |                       1000.003 |                        1000.003 |                    326.526 | True                             |
| lc        |   2025_08_23_14_30_55 |                    328.859 |                               44.365 |                       1000.003 |                       1000.003 |                        1000.003 |                    328.859 | True                             |
| lc        |   2025_08_23_14_36_43 |                    331.473 |                               54.993 |                        999.952 |                        999.952 |                         999.952 |                    331.473 | True                             |
| lc        |   2025_08_23_14_42_36 |                    319.831 |                               59.739 |                       1000.003 |                       1000.003 |                        1000.003 |                    319.831 | True                             |
| lc        |   2025_08_23_15_10_38 |                    120.750 |                               44.944 |                       1000.008 |                       1000.008 |                        1000.008 |                    120.750 | True                             |
| lzh       |   2025_08_23_11_21_56 |                    302.006 |                               59.057 |                       1000.003 |                       1000.003 |                        1000.003 |                    302.006 | True                             |
| lzh       |   2025_08_23_11_30_23 |                    350.934 |                               59.464 |                       1000.003 |                       1000.003 |                        1000.003 |                    350.934 | True                             |
| lzh       |   2025_08_23_11_42_15 |                    321.145 |                               54.130 |                       1000.000 |                       1000.000 |                        1000.000 |                    321.145 | True                             |
| lzh       |   2025_08_23_11_47_52 |                    333.420 |                               53.628 |                       1000.003 |                       1000.003 |                        1000.003 |                    333.420 | True                             |
| lzh       |   2025_08_23_11_53_46 |                    309.635 |                               39.731 |                       1000.003 |                       1000.003 |                        1000.003 |                    309.635 | True                             |
| zsj       |   2025_08_23_16_43_52 |                    483.839 |                               59.935 |                       1000.002 |                       1000.002 |                        1000.002 |                    483.839 | True                             |
| zsj       |   2025_08_23_16_52_17 |                    315.590 |                               60.210 |                       1000.003 |                       1000.003 |                        1000.003 |                    315.590 | True                             |
| zsj       |   2025_08_23_17_06_32 |                    323.317 |                               43.160 |                       1000.003 |                       1000.003 |                        1000.003 |                    323.317 | True                             |
| zsj       |   2025_08_23_17_12_13 |                    309.185 |                               57.085 |                       1000.003 |                       1000.003 |                        1000.003 |                    309.185 | True                             |
| zsj       |   2025_08_23_17_17_38 |                    308.958 |                               60.119 |                       1000.003 |                       1000.003 |                        1000.003 |                    308.958 | True                             |

## 判定阈值

- vehicle duration >=120 s；steering effective rate >=20 Hz；
- ECG/EMG/RESP 各 >=100 Hz；vehicle-physio overlap >=120 s；
- EEG：至少24通道各 >=100 Hz，且 vehicle-physio-EEG overlap >=120 s；
- longitudinal style：至少2个 qualifying sessions，且每个后续 qualifying session 之前已有完成的 qualifying vehicle session。

## 输出

- `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run70_fresh_multimodal_inventory_20260830\corrected_run_2\inventory_files.csv`
- `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run70_fresh_multimodal_inventory_20260830\corrected_run_2\inventory_sessions.csv`
- `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run70_fresh_multimodal_inventory_20260830\corrected_run_2\external_subject_summary.csv`
- `<PROJECT_ROOT>\05_rebuild_from_raw_20260511\03_baselines\run70_fresh_multimodal_inventory_20260830\corrected_run_2\decision.json`
