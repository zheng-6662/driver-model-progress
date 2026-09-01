---
name: vehicle原始道路信号与固定道路标签要区分
description: 被试 vehicle 文件中的原始道路相关信号代表真实行驶过程/真实实验场景数据；需与后续附加的 fixed road labels 区分开。
type: project
originSessionId: 9dc130fe-8050-4e20-80e1-b6023cdbe469
---
被试 vehicle 文件中的原始道路相关信号（例如 `zx1|lanecurvatureXY`、`zx1|lateraldistance`）代表真实行驶过程中的道路数据，反映真实实验场景本身；它们不应与后续附加的固定道路标签（如 `road_type_fixed`、`road_s_ref_m`、`ref_nn_ok`）混为一谈。

**Why:** 用户明确澄清：如果检查的是每位被试车辆数据里的道路数据，那部分就是车辆真实行驶过程中的道路数据，不是后来人为替换的一套外加道路图。

**How to apply:** 后续分析道路数据来源时，先区分“原始 vehicle 内生道路信号”和“后加 fixed/template 标签”。前者默认按真实实验场景理解；真正需要重点核实旧模板/旧场景残留风险的，优先是后者。
