# V4 微状态统计结果

使用 105 冻结队列的 40 名参与者、65 次扫描（15 HC、25 对 OFF/ON），AAL78、TR 2.1 s，主分析为 V3 E04 五折留出 K=4。模板、回填标签和每次扫描的 28 项原有状态指标已经逐项核验。空间离散度从同一扫描和相应折模板计算。

## 三个科学问题

1. **OFF 相对 HC。** 家族内两种 Holm 均通过的指标有 4 项。M3 Coverage、M1 Duration、M3 GEV、M3 Microstate-adapted Local Dispersion。
2. **DBS-ON 与 OFF 的配对变化。** 有 3 项达到同一标准。M3 Coverage、M3 GEV、M3 Microstate-adapted Local Dispersion。
3. **向 HC 靠近。** 24/32 项的样本均值距离缩小。预设的单侧趋近置换在 32 项 Holm 后支持 3 项。
   ON−HC 共有 0 项达到两种 Holm 的标准。OFF 有证据而 ON 差异未检出的项数为 4。等效性因缺少预设界值而未判定。

以下家族内检出在 FD 调整置换后未达到 Holm 阈值。M3 Coverage（OFF-HC，校正 P=0.0528）。

GEV 为微状态的拟合解释量。它与 Coverage、离散度共享回填信息，三项的同向变化可能存在数学相关。

微状态主表 96 项中的家族内 BH 和 Holm 分别通过 7、7 项。双 Holm 通过 7 项。96 项一起控制时，原始 Holm 和置换 Holm 同时通过 1 项。该范围下通过的指标为 M3 Coverage（ON-OFF）。六项标签置换不变的整体指标共 18 个对比，其中双 Holm 通过 0 项。

## 家族内显著及未达显著的计数

| 家族 | 对比 | 预设 | 可估计 | Raw | BH | Holm | 置换 Holm | 双 Holm | 总体 Holm | FD 置换 Holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Coverage | OFF-HC | 4 | 4 | 2 | 1 | 1 | 1 | 1 | 0 | 0 |
| Coverage | ON-HC | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Coverage | ON-OFF | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| Duration | OFF-HC | 4 | 4 | 2 | 1 | 1 | 1 | 1 | 0 | 1 |
| Duration | ON-HC | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Duration | ON-OFF | 4 | 4 | 1 | 0 | 0 | 1 | 0 | 0 | 1 |
| Occurrence | OFF-HC | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Occurrence | ON-HC | 4 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Occurrence | ON-OFF | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| GEV | OFF-HC | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| GEV | ON-HC | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| GEV | ON-OFF | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| Spatial Dispersion | OFF-HC | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| Spatial Dispersion | ON-HC | 4 | 4 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Spatial Dispersion | ON-OFF | 4 | 4 | 1 | 1 | 1 | 1 | 1 | 0 | 1 |
| Transition Probability | OFF-HC | 12 | 12 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| Transition Probability | ON-HC | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Transition Probability | ON-OFF | 12 | 12 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |

Raw、BH、Holm、置换 Holm 与 FD 置换 Holm 的各列为 P <0.05 的项数。双 Holm 同时要求 Raw P 和 Permutation P 的家族 Holm <0.05。转移的完整家族含 12 条有向非对角转换 × 3 个对比。

## 家族内两种 Holm 均通过的项目

| 指标 | 对比 | 检验 | 均值差 | 95% CI | g / g_z | Raw P | BH P | Holm P | 置换 P | 置换 Holm P | FD 置换 Holm P | 主指标总体 Holm P |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 Coverage | OFF-HC | Welch t | -0.08245 | [-0.1323, -0.03391] | -1.123 | 0.003854 | 0.02313 | 0.0424 | 0.003 | 0.033 | 0.0528 | 0.343 |
| M3 Coverage | ON-OFF | Wilcoxon signed-rank | +0.05511 | [0.03201, 0.082] | +0.8311 | 0.0004675 | 0.00561 | 0.00561 | 0.0002 | 0.0024 | 0.0036 | 0.04441 |
| M1 Duration | OFF-HC | Welch t | +0.6543 | [0.31, 0.9952] | +1.089 | 0.0007268 | 0.008722 | 0.008722 | 0.0009 | 0.0108 | 0.0156 | 0.06832 |
| M3 GEV | OFF-HC | Mann-Whitney U | -0.03609 | [-0.05421, -0.01812] | -1.402 | 0.0004314 | 0.005177 | 0.005177 | 0.0008 | 0.0096 | 0.0084 | 0.04141 |
| M3 GEV | ON-OFF | paired t | +0.01361 | [0.006282, 0.02204] | +0.6475 | 0.002709 | 0.01625 | 0.0298 | 0.0017 | 0.0187 | 0.0121 | 0.2438 |
| M3 Microstate-adapted Local Dispersion | OFF-HC | Welch t | +0.0673 | [0.03261, 0.09987] | +1.264 | 0.0008302 | 0.008937 | 0.009963 | 0.0004 | 0.0048 | 0.006 | 0.07721 |
| M3 Microstate-adapted Local Dispersion | ON-OFF | paired t | -0.02997 | [-0.04623, -0.01395] | -0.6945 | 0.001489 | 0.008937 | 0.01638 | 0.0015 | 0.0165 | 0.006 | 0.137 |

Coverage、GEV 与转换概率以 0–1 比例计。Duration 以秒计。Occurrence 以每秒段数计。离散度为无量纲的模板距离。标准化效应为独立组 Hedges g 与配对 Hedges g_z。95% CI 条件于固定模板与标签。

## 向 HC 靠近的对比

| 指标 | OFF−HC | ON−HC | 距离缩短 | 95% CI | 单侧置换 P | 32 项 Holm P |
| --- | --- | --- | --- | --- | --- | --- |
| M3 Coverage | -0.08245 | -0.02733 | +0.05511 | [0.008692, 0.07672] | 0.0001 | 0.0032 |
| M3 GEV | -0.03609 | -0.02248 | +0.01361 | [0.006061, 0.02179] | 0.0006 | 0.0186 |
| M3 Microstate-adapted Local Dispersion | +0.0673 | +0.03734 | +0.02997 | [0.01314, 0.04623] | 0.0009 | 0.027 |

距离缩短采用相同 HC 以及相同患者对的组均值。该证据刻画组均值的位置变化。ON−HC 的差异未被检出时，原始差异及区间仍可在主表读出。个体临床改善和统计等效性未在 V4 判定。所有正、负趋近及未达显著项目见 [Towards_HC.csv](../results/Towards_HC.csv)。

## FD 与检验条件

V3 的 CR1 FD 估计及原有校正已原样保留。本次共有 93 个与 V3 有效归档对应的估计、区间和 P 逐项复算相符。另对每个项目以 Freedman–Lane 进行 FD 调整置换。V3 包含 C4（合并患者效应）及双 HC 对比同向的交集检验，完整的 E04/E04_R 归档节选保存在本项目。同一固定队列的特征共享训练参与者，患者访视的噪声与顺序效应可能影响条件推断。

## 完整结果与溯源

- [OFF−HC](../results/HC_vs_OFF.csv) · [ON−HC](../results/HC_vs_ON.csv) · [ON−OFF](../results/ON_vs_OFF.csv) · [全部微状态指标](../results/Microstate_Results.csv)
- [12 条有向转换](../results/Transition_Results.csv) · [BH-FDR](../results/FDR_Results.csv) · [Holm](../results/Holm_Results.csv) · [置换及 FD 调整](../results/Permutation_Results.csv)
- [描述统计](../results/Group_Level_Summary.csv) · [V3 FD 与共同 PD 效应归档](../results/V3_FD_Adjusted_Archive.csv) · [跨方法对照](CROSS_METHOD_COMPARISON.md)
- [输入校验记录](../results/Input_Verification.json) · [统计规程](WEIDA_STATISTICAL_PROTOCOL.md) · [执行与复核](../README.md)
- [组间图](../figures/Group_Comparison.png) · [转换矩阵](../figures/Transition_Matrix.png) · [效应量森林图（含 ON−OFF 配对汇总）](../figures/Effect_Size_Forest.png)
