# BOLD 微状态与同条件运动评分

[中文组会报告](reports/Motor_Severity_Association_Report.md) · [固定分析方案](ANALYSIS_PLAN.md) · [公开汇总图](figures/Motor_Severity_Forest.png)

## 复算

在本工作树根目录执行。程序从上一轮行为模块的私有 `Input_Hashes.json` 定位冻结 V6 匹配表和逐扫描数据，从该行为模块读取已核查的历史参照评分，并逐一扫描键核对。`--behavior-private` 可用于指向已有授权的行为工作区。`--v6-private` 可覆盖上轮登记的 V6 路径，文件哈希仍应与原审计一致。若已人工核实原始访视，可通过 `--verified-visits` 与 `--verified-evidence` 提供一份单独输入。

```powershell
$module = 'BOLD_Microstate_Motor_Severity'
python -B -X utf8 "$module/scripts/severity.py"
python -B -X utf8 -m pytest -q -p no:cacheprovider "$module/tests"
```

兼容的包版本见运行记录。绘图需能读取简体中文字体。GEV 使用比例计算，作图转为百分比，出现率和全局切换率由次/秒乘 60 作图，停留时间保留秒。私有目录包含原表哈希、用于验证的匹配输入和逐人散点图。

| 文件 | 内容 |
|---|---|
| [终点清单](results/Endpoint_Registry.csv) | 11 项条件、单位、原影像检验及校正证据 |
| [同条件关联](results/Motor_Severity_Associations.csv) | 11 项原始、置换、阶段及累计校正和留一结果 |
| [累计 15 项关联](results/Cumulative_15_Associations.csv) | 11 项严重度关联与前轮 4 项改善量关联 |
| [FD 敏感性](results/Motor_Severity_FD_Sensitivity.csv) | 同条件 FD 调整的效应、区间和 11 项校正 |
| [既有窄频 ON M4 复算](results/Legacy_ON_M4_Recheck.csv) | 原 Pearson、Fisher 区间和 BH42 校正对照 |
| [复算元数据](results/Run_Metadata.json) | 基线、分析规模、随机种子及软件版本 |
