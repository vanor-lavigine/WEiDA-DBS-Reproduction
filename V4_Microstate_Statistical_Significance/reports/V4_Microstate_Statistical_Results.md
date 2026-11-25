# V4 微状态统计显著性结果

使用 15 名 HC 和 25 对患者的 V3 冻结队列，65 次扫描。AAL78、TR 2.1 s、固定 K=4。A3 的全部 65 次回填标签与版本归档逐帧相符。本轮共同有效窗共 12,574 帧。每个标量对比各运行 10,000 次参与者级置换和 10,000 次 FD 敏感性置换。

## 三个比较及共同方向

- **OFF−HC**。家族内 BH、常规 Holm 和置换 Holm 分别支持 11、7、7 项。同时通过两种 Holm 的项目为 M3 Coverage、M4 Coverage、M4 平均持续时间、M3 Occurrence、M3 GEV、M3 空间离散度、全局切换率。
- **ON−HC**。家族内 BH、常规 Holm 和置换 Holm 分别支持 1、0、0 项。同时通过两种 Holm 的项目为 无。
- **ON−OFF**。家族内 BH、常规 Holm 和置换 Holm 分别支持 10、5、5 项。同时通过两种 Holm 的项目为 M1 Coverage、M4 平均持续时间、M1 Occurrence、M1 GEV、全局切换率。
- **两项 HC 比较同方向**。共 33 项，27 项点估计同方向。同时通过两种各家族 Holm 的有 0 项。

## 四态整体 Coverage

| 对比 | ILR 口径 | 效应距离 | 置换 P | 两项 Holm P | FD 置换 Holm P |
| --- | --- | --- | --- | --- | --- |
| OFF−HC | 对齐四态 | 0.7351 | 0.0004 | 0.0007999 | 0.0007999 |
| ON−HC | 对齐四态 | 0.2374 | 0.6842 | 0.6842 | 0.6574 |
| OFF−HC | 按占比排序 | 0.4337 | 0.0044 | 0.008799 | 0.0232 |
| ON−HC | 按占比排序 | 0.3094 | 0.05239 | 0.05239 | 0.06829 |

对齐四态的 OFF−HC 整体组成通过 Holm，允许在该对比定位状态。ON−HC 的整体组成未达门槛。按占比排序的置换结果属于预设探索性敏感性。该对比的全局探索 Holm P 值见完整表。

## 同时通过常规及置换 Holm 的结果

| 对比 | 指标 | 差值 | 家族 BH P | 家族 Holm P | 置换 Holm P | FD 置换 Holm P | 重训有定义 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ON−OFF | M1 Coverage | +5.93 个百分点 | 0.01059 | 0.01059 | 0.0108 | 0.06239 | 1019/2000 |
| OFF−HC | M3 Coverage | -9.06 个百分点 | 0.006731 | 0.006731 | 0.0104 | 0.05599 | 712/2000 |
| OFF−HC | M4 Coverage | +12.29 个百分点 | 0.01423 | 0.02491 | 0.0399 | 0.08399 | 1450/2000 |
| OFF−HC | M4 平均持续时间 | +4.312 s | 0.0002387 | 0.0002387 | 0.0007999 | 0.0007999 | 1450/2000 |
| ON−OFF | M4 平均持续时间 | -3.069 s | 0.006415 | 0.006415 | 0.008799 | 0.007599 | 1450/2000 |
| ON−OFF | M1 Occurrence | +0.399 次/分 | 0.01332 | 0.01332 | 0.0152 | 0.0112 | 1019/2000 |
| OFF−HC | M3 Occurrence | -0.640 次/分 | 0.007008 | 0.007008 | 0.004 | 0.007999 | 712/2000 |
| ON−OFF | M1 GEV | +1.17 个百分点 | 0.000577 | 0.001058 | 0.0024 | 0.0024 | 1019/2000 |
| OFF−HC | M3 GEV | -2.76 个百分点 | 0.0003339 | 0.0003339 | 0.0033 | 0.005499 | 712/2000 |
| OFF−HC | M3 空间离散度 | +0.0673 | 0.001947 | 0.001947 | 0.0012 | 0.0012 | 712/2000 |
| OFF−HC | 全局切换率 | -1.399 次/分 | 0.001393 | 0.001393 | 0.0021 | 0.0024 | 2000/2000 |
| ON−OFF | 全局切换率 | +0.844 次/分 | 0.01139 | 0.01519 | 0.017 | 0.0024 | 2000/2000 |

99 项标量联合校正后同时通过两套 Holm 的项目共 4 项。OFF−HC M4 平均持续时间、ON−OFF M1 GEV、OFF−HC M3 GEV、OFF−HC M3 空间离散度。所有逐状态条目都在重训中出现无法稳定定位的轮次。表中效应与 P 条件于已经锁定的留出标签。

## BH 通过且常规 Holm 未通过的结果

| 对比 | 指标 | 差值 | 家族 BH P | 家族 Holm P | 置换 Holm P | FD 置换 Holm P | 重训有定义 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OFF−HC | M1 Coverage | -8.06 个百分点 | 0.04341 | 0.09767 | 0.1554 | 0.3078 | 1019/2000 |
| ON−OFF | M3 Coverage | +6.47 个百分点 | 0.03699 | 0.05548 | 0.05999 | 0.1554 | 712/2000 |
| ON−OFF | M4 Coverage | -8.19 个百分点 | 0.03684 | 0.05527 | 0.05999 | 0.1554 | 1450/2000 |
| ON−OFF | M2 平均持续时间 | -2.231 s | 0.04375 | 0.06563 | 0.05519 | 0.0261 | 1658/2000 |
| OFF−HC | M1 GEV | -2.78 个百分点 | 0.04884 | 0.1465 | 0.1503 | 0.2169 | 1019/2000 |
| ON−OFF | M3 GEV | +1.65 个百分点 | 0.044 | 0.11 | 0.07199 | 0.07699 | 712/2000 |
| OFF−HC | M2 空间离散度 | +0.0428 | 0.04687 | 0.1162 | 0.1034 | 0.09349 | 1658/2000 |
| ON−HC | M3 空间离散度 | +0.0402 | 0.04687 | 0.1664 | 0.1512 | 0.1566 | 712/2000 |
| ON−OFF | M3 空间离散度 | -0.0271 | 0.04687 | 0.1664 | 0.1476 | 0.09349 | 712/2000 |
| OFF−HC | M4 空间离散度 | +0.0420 | 0.04687 | 0.1432 | 0.141 | 0.1566 | 1450/2000 |

## 仅 FD 敏感性达到家族 Holm 的补充结果

| 对比 | 指标 | 差值 | 家族 BH P | 家族 Holm P | 置换 Holm P | FD 置换 Holm P | 重训有定义 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ON−OFF | M2 平均持续时间 | -2.231 s | 0.04375 | 0.06563 | 0.05519 | 0.0261 | 1658/2000 |

## 完整检验计数

| 指标 | 对比 | 总计 | 可估计 | Raw | BH | Holm | 置换 Holm | FD 置换 Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coverage | OFF−HC | 4 | 4 | 3 | 3 | 2 | 2 | 0 |
| Coverage | ON−HC | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| Coverage | ON−OFF | 4 | 4 | 3 | 3 | 1 | 1 | 0 |
| 平均持续时间 | OFF−HC | 4 | 4 | 2 | 1 | 1 | 1 | 1 |
| 平均持续时间 | ON−HC | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| 平均持续时间 | ON−OFF | 4 | 4 | 2 | 2 | 1 | 1 | 2 |
| Occurrence | OFF−HC | 4 | 4 | 2 | 1 | 1 | 1 | 1 |
| Occurrence | ON−HC | 4 | 4 | 1 | 0 | 0 | 0 | 0 |
| Occurrence | ON−OFF | 4 | 4 | 1 | 1 | 1 | 1 | 1 |
| GEV | OFF−HC | 4 | 4 | 2 | 2 | 1 | 1 | 1 |
| GEV | ON−HC | 4 | 4 | 1 | 0 | 0 | 0 | 0 |
| GEV | ON−OFF | 4 | 4 | 2 | 2 | 1 | 1 | 1 |
| 空间离散度 | OFF−HC | 4 | 4 | 4 | 3 | 1 | 1 | 1 |
| 空间离散度 | ON−HC | 4 | 4 | 1 | 1 | 0 | 0 | 0 |
| 空间离散度 | ON−OFF | 4 | 4 | 1 | 1 | 0 | 0 | 0 |
| 有向转换 | OFF−HC | 12 | 12 | 5 | 0 | 0 | 0 | 0 |
| 有向转换 | ON−HC | 12 | 12 | 0 | 0 | 0 | 0 | 0 |
| 有向转换 | ON−OFF | 12 | 12 | 3 | 0 | 0 | 0 | 0 |
| 全局切换率 | OFF−HC | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 全局切换率 | ON−HC | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 全局切换率 | ON−OFF | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

BH、Holm 与置换 Holm 的计数采用 Coverage 的整体门槛。各对比的未达显著结果、SD、中位数、IQR 和原单位 95% 区间均在下列数据表。

- [OFF−HC](../results/HC_vs_OFF.csv) · [ON−HC](../results/HC_vs_ON.csv) · [ON−OFF](../results/ON_vs_OFF.csv)
- [Coverage](../results/Coverage_Results.csv) · [Duration](../results/Duration_Results.csv) · [Occurrence](../results/Occurrence_Results.csv) · [完整转换矩阵](../results/Transition_Matrix_Descriptive.csv)
- [非对角转换检验](../results/Transition_Results.csv) · [GEV](../results/GEV_Results.csv) · [Spatial Dispersion](../results/Spatial_Dispersion_Results.csv)
- [FDR](../results/FDR_Results.csv) · [Holm](../results/Holm_Results.csv) · [置换](../results/Permutation_Results.csv) · [探索全局校正](../results/Exploratory_Global_Corrections.csv) · [重训区间](../results/Bootstrap_Results.csv)
- [覆盖率图](../figures/Coverage_Comparison.png) · [段长图](../figures/Duration_Comparison.png) · [出现率图](../figures/Occurrence_Comparison.png) · [转换图](../figures/Transition_Matrix.png) · [GEV 图](../figures/GEV_Comparison.png) · [效应量森林图](../figures/Effect_Size_Forest.png) · [显著性汇总](../figures/Significance_Summary.png)
