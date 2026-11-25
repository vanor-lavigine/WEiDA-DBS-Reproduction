# V5 微状态终点优化结果

分析限定在 V4 冻结的 15 名 HC 和 25 名配对患者，65 次扫描、AAL78、TR=2.1 秒。沿用 F1 宽频（上游 0.01–0.08 Hz）、N0、G1、S1 和 A3 λ=0.04，五折模板保持原分配，以第 0 折对齐编号。V4 的 41 项原始哈希和 72 个 Git 文件逐一核对。V5 的覆盖、出现率及转换概率与冻结特征一致。现有标签共计 12,574 个有效帧，分析单位为参与者或患者配对。[输入审计](../manifests/Analysis_Input_Audit.json) · [V4 SHA-256](../manifests/V4_SHA256.csv)

## 六项主要新增检验

| 新终点 | 对比 | Raw P | 参与者置换 P | 平均 FD 调整 P | 六项置换 Holm P | 六项 FD Holm P |
|---|---|---:|---:|---:|---:|---:|
| Coverage Energy | OFF–HC | 0.00040 | 0.00040 | 0.00080 | **0.00240** | **0.00480** |
| Coverage Energy | ON–HC | 0.10529 | 0.10529 | 0.11939 | 0.31587 | 0.35816 |
| 转移矩阵整体 Wald | OFF–HC | 0.00620 | 0.00620 | 0.01910 | **0.03100** | 0.09549 |
| 转移矩阵整体 Wald | ON–HC | 0.32417 | 0.32417 | 0.37546 | 0.43876 | 0.45135 |
| 转移矩阵整体 Wald | ON–OFF | 0.04040 | 0.04040 | 0.05859 | 0.16158 | 0.23438 |
| 健康参考距离缩短 | ON–OFF | 0.21603 | 0.21938 | 0.22568 | 0.43876 | 0.45135 |

多变量主检验的原始 P 来自预定置换分布。距离终点另提供单侧配对 t 的原始 P。原始检验、置换和 FD 分支分别校正。主家族之外还进行全部新增 150 项的联合 Holm。在 150 项联合层级，所有 FD 结果均未过 0.05。[效应及完整家族校正表](../results/All_V5_Statistical_Results.csv)

## Duration 与 Occurrence

所有 M1–M4 的逐人中位停留时间在三个对比均未通过家族 Holm。加入逐人 P75 后，OFF 的 M4 较 HC 长 **5.31 秒**（95% CI 2.88–7.91，置换 Holm 0.00240，FD Holm 0.00720）。剔除连续有效块的首末段后为 **5.22 秒**（2.76–7.68，置换 Holm 0.00360，FD Holm 0.01320）。两种口径的 M4 均通过 150 项未调整置换联合 Holm（0.03000 和 0.04440）。M1–M3 的 OFF–HC，以及四状态的 ON–HC 和 ON–OFF，均未通过停留时间的家族 Holm。V4 原平均停留结果单列冻结。[全部 48 项新停留估计](../results/duration/Duration_Results.csv) · [分布图](../figures/Duration_Distributions.png)

四态 Poisson Pearson χ²/df 为 0.99–1.36，低于固定 NB2 切换阈值 1.5，均保留 Poisson，65 次扫描的每个状态均至少有一个状态段。计数以有效扫描秒数为 offset，标准误按参与者聚类。OFF 相对 HC 的 M3 出现率比为 **0.673**（95% CI 0.555–0.809，计数模型 Holm 0.00300，FD 模型 Holm 0.00722）。相应段率置换 Holm 为 0.00240。FD 模型调整前后的效应、IRR、Bootstrap 区间及四态诊断均保留。ON–OFF 的 M1 段率置换通过家族 Holm（0.03520），计数模型家族 Holm 为 0.05865，FD 模型家族 Holm 为 0.10622。M1 在计数模型的未调整和 FD 两种设定中只达到 BH 层级。ON–HC 的计数组间比较无家族校正支持。[计数模型](../results/occurrence/Occurrence_Count_Models.csv) · [模型诊断](../results/occurrence/Occurrence_Diagnostics.csv) · [独立段率敏感性](../results/occurrence/Occurrence_Rate_Sensitivity.csv)

共同方向采用两个 HC 对比同向且同时存在差异的交集检验。36 个新增共同方向终点无 Holm 或 BH 支持。两个对比的效应与置信区间均保留。[全部共同方向结果](../results/Shared_PD_Direction_Results.csv)

## 质量、重抽及样本量

Mean FD、DVARS 和标准化 DVARS 均覆盖 65 次扫描。未获得统一定义的标准化前 tSNR、足够完整的逐人年龄与性别或可核实的逐扫描用药时间，故 Model 2 未估计。原始侧车标识连接到 24 对 DICOM 访视，全部同日先 ON 后 OFF，间隔中位 109.98 分钟（四分位 97.04–142.52）。余下 1 对的顺序未核实。配对无法分离固定顺序、间隔与刺激条件。[协变量表](../results/covariates/Covariate_Availability.csv) · [扫描顺序](../results/covariates/Scan_Order_and_Intervals.json)

主流程的 114 项可重抽效应及 12 项 FD 计数模型均得到 2,000/2,000 次有效 Bootstrap。所有结果以固定五折模板为条件。冻结 V4 在重新训练时四态皆可匹配的成功率为 549/2,000，适用于另一个包含模板不确定性的分析层级。[重抽审计](../results/bootstrap/All_Methods_Bootstrap_Audit.csv) · [V4 稳定性](../../V4_Microstate_Statistical_Significance/reports/Microstate_Stability_Report.md)

保守规划采用双侧 α=0.05/6，考虑 15% 失访。若目标标准化 d=0.35，80% 功效需实际招募 236 名 HC 和 236 名配对 PD（708 次计划扫描），90% 功效需各 298 名（894 次）。d=0.40 时分别需各 182 名和各 229 名。单独做配对、80% 功效在 d=0.35 或 0.40 下需招募 122 或 95 名患者。以本样本距离效应 dz≈0.160 作规划敏感性，80% 功效需要约 563 名配对患者。减半效应时约需 2,236 名配对患者。该敏感性估算有很大的先导效应不确定性。功效计算以标准化均值差模型为近似。[样本量全表](../results/power/Sample_Size_Planning.csv) · [固定 PD 25 人的 HC 增量](../results/power/Fixed_PD25_HC_Increase.csv)

发表论文报告 16 名 HC，现有原始分组表只登记 15 名。另有一份 HC 命名扫描且已能提取候选数据，受试者唯一身份、符合纳入标准和对应论文名单尚未证实，因此保留原 15 名 HC。本轮无符合核查条件的扩展队列。[第 16 名 HC 审计](../results/covariates/Sixteenth_HC_Audit.json) · [原论文](https://doi.org/10.1038/s41531-026-01354-3)
