# Round 5 冻结 A3 整体转移动力学

25 对患者的 12 维 WEiDA 转移向量在 ON/OFF 条件下存在整体变化，**Wald=57.581，P=0.0017998**。整体标准化距离 D=1.518（95% CI [1.426, 2.743]）。平均差向量的欧氏长度为 0.0999（95% CI [0.0769, 0.1519]）。D 是以主收缩协方差衡量的样本距离。两个自助区间均描述效应大小，零假设判定使用置换 P。

Round 4 全部 36 个 WEiDA 路径比较的 BH 和 Holm 通过数仍为 0。整体检验考察联合向量均值，单路径是否通过原家族需分别读表。

## 与 V6 对账

| 定义 | Round 5 | V6 |
|---|---|---|
| 输入向量 | 12 条非对角 P(去向｜当前态) 的 ON−OFF | 相同，25 个完整配对 |
| 自循环 | 包含在每行分母，向量不另列对角元素 | 相同 |
| 协方差 | 样本协方差，0.10 各向同性迹收缩，每次重估 | 相同 |
| 零分布 | 整人同号翻转 12 维，10,000 次，平方 Wald 右尾 | 相同 |
| 种子及区间 | 20261127，9,999 次患者整对 Bootstrap | 7261126，2,000 次 |
| 多重比较 | 一个预设主检验，BH 与 Holm 等于原始 P | 探索性整体家族，2 项 |
| 整体效应 | 欧氏范数及协方差标准化距离 D | 欧氏范数 |

65 个冻结 A3 扫描重数结果与 Round 4 逐扫描表相符。V6 已发布 ON/OFF 的 16/16 个矩阵元素的计数、分母完全一致，概率核验最大误差 1.1102e-16。新种子重算与 V6 整体结论一致，V6 原始 P=0.0017998、Holm P=0.0017998，Round 5 主检验的 BH q=0.0017998、Holm P=0.0017998。V6 分别调整 FD 和 DVARS 的旧 Holm P 为 0.0067993 和 0.0021998，各沿用 V6 原家族。这是同一标签和同一队列上的再分析。

原始协方差秩 12/12，条件数 42.40。主收缩后条件数 16.19。四个固定收缩值的结果如下，敏感性原始 P 用于对照稳定性。

| 迹收缩 | Wald | 原始 P | 条件数 |
|---|---|---|---|
| 0.00 | 92.109 | 0.0060994 | 42.397 |
| 0.05 | 68.698 | 0.0026997 | 23.776 |
| 0.10 | 57.581 | 0.0017998 | 16.186 |
| 0.20 | 46.399 | 0.0014999 | 9.4769 |

## 组成、头动及历史临床证据

ILR 全向量原始 P=0.037096、BH q=0.061827、Holm P=0.14439，完整五项家族 BH 0/5，Holm 0/5。头动敏感性完整八项家族 BH 7/8，Holm 6/8。历史评分的 80 项 Spearman 重核后 BH 通过 0/80。

[配对整体结果](../results/Omnibus_Transition_Results.csv) · [12 维均值向量](../results/Omnibus_Transition_Vector.csv) · [协方差矩阵](../results/Omnibus_Transition_Covariance.csv) · [固定收缩比较](../results/Omnibus_Transition_Sensitivity.csv)

[M4 动力学](Round5_M4_Mechanistic_Report.md) · [组成及稳健性](Round5_Robustness_Report.md) · [历史临床关联](M4_Clinical_Interpretation.md) · [最终解释](Round5_Final_Interpretation.md) · [方案](../ANALYSIS_PLAN.md) · [复现与输入核验](../results/Round5_Run_Manifest.json)

[V6 转移报告](../../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Transition_Dynamics.md) · [Round 4 路径报告](../../round4/transition/reports/A3_Transition_Dynamics_Report.md)
