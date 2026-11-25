# 模拟校准与微状态稳定性

## 历史 V3 数值复核

读取原 V3 总表 33,898 行及三个对比和同向表。逐表与总表中的对应切片相符。复算 E04 的 4 个预定主检验及 E04_R 的 144 个 CR1 对比。V3 E04 的 OFF−HC 组成原始 P=0.0429，四项 Holm P=0.1716。V3 的四项主检验无 Holm 显著结果。V3 各方案的家族 BH、Holm 及跨方案 Holm 有完整记录。
[历史主检验复算](../results/V3_Primary_Reproduction.csv) · [原校正计数](../results/V3_Archive_Threshold_Counts.csv) · [V3 统计归档](../../V3_BOLD_Microstate_Method_Optimization/results/All_Methods_Comparison.csv)

V3 E02 和 E04 的五折空间模板一致，已复核 40 个对应。最低 |r|=0.907，最低双向余量=0.059。E04_R 的历史全流程重训只有 96/2,000 次同时满足四态识别。E09、F3 等分支也有状态匹配不全。历史固定模板区间和只在可匹配时定义的重训效应需要按各自口径解释。实际何项是假阳性无法由组间 P 或匹配失败单独判定。

[V3 重训归档](../results/V3_Bootstrap_Archive.csv) · [V3 原统计方法](../../V3_BOLD_Microstate_Method_Optimization/reports/statistics_methods.md) · [V3 模拟偏差报告](../../V3_BOLD_Microstate_Method_Optimization/reports/Method_Bias_Report.md)

## 峰间回填与正则化的零假设核查

重新计算 V3 逐 TR 与峰间回填的模拟切换拒绝率。分别在各自 500 个队列上，S0 为 6.6% 和 6.0%，S5 为 42.6% 和 8.6%。另逐队列读取 V4 Reliability 的 1200 个归档，独立复建六项核心 Holm 家族。S5 改变观测噪声和 FD，潜在状态序列已知一致，其拒绝率衡量既定噪声机制下的敏感性。

| 方法 | S5 高噪声家族拒绝数 | 最差 S5 Wilson 上界 | 互斥训练标签 ARI | 互斥训练四态全匹配 |
| --- | --- | --- | --- | --- |
| A3 | 21/200 | 15.52% | 0.287 | 33.3% |
| A0 | 21/200 | 15.52% | 0.244 | 0.0% |
| A1 | 22/200 | 16.09% | 0.321 | 33.3% |
| A2 | 23/200 | 16.66% | 0.204 | 0.0% |

高噪声上界须不超过 15%。四种方法均未通过。A3 与 A0 同为最低上界，A3 的互斥训练标签 ARI 和四态全匹配率较高。这是使用另一固定方案的依据。实际清洁条件下的准确性、较强噪声下的错误拒绝率和真实队列推断的可靠性分别报告。

传统 8 项 HC 家族的重算直接使用模拟归档中的逐指标 FD 调整 CR1 P，可检查本次家族边界在历史模拟中的表现。这套重算没有生成新合成队列。当前 Welch/秩检验和 10,000 次置换的联合规则缺少另外的全流程模拟校准。

| 情景 | 方案 | 指标家族 | 队列数 | 8项全可估计 | Holm 至少一项拒绝 |
| --- | --- | --- | --- | --- | --- |
| S0 | A0 | coverage | 200 | 200 | 2.0% |
| S0 | A0 | duration | 200 | 200 | 5.0% |
| S0 | A0 | occurrence | 200 | 200 | 3.5% |
| S0 | A1 | coverage | 200 | 200 | 5.0% |
| S0 | A1 | duration | 200 | 200 | 7.5% |
| S0 | A1 | occurrence | 200 | 200 | 3.5% |
| S0 | A3 | coverage | 200 | 200 | 8.0% |
| S0 | A3 | duration | 200 | 200 | 4.5% |
| S0 | A3 | occurrence | 200 | 200 | 7.0% |
| S5_high | A0 | coverage | 200 | 200 | 6.0% |
| S5_high | A0 | duration | 200 | 200 | 4.5% |
| S5_high | A0 | occurrence | 200 | 200 | 7.0% |
| S5_high | A1 | coverage | 200 | 200 | 8.5% |
| S5_high | A1 | duration | 200 | 200 | 2.5% |
| S5_high | A1 | occurrence | 200 | 200 | 9.0% |
| S5_high | A3 | coverage | 200 | 200 | 5.5% |
| S5_high | A3 | duration | 200 | 200 | 6.5% |
| S5_high | A3 | occurrence | 200 | 200 | 6.5% |

[逐项重算](../results/Reliability_Traditional_Family_Recount.csv) · [核心校准复核](../results/Reliability_Core_Calibration_Recount.csv) · [V3 峰间核查](../results/V3_Peak_Null_Recount.csv)

## A3 的模板与完整重训

观察五折的 40 个状态对应全部通过。最小 |r|=0.892，最小双向余量=0.262。使用互斥训练集检验时，15 项划分中仅 5 项四态同时通过。较高的五折相似度与独立训练重现能力有各自的含义。

成功运行 2,000/2,000 次全流程重训。各状态匹配次数与旧 A3 同种子审计一致。

| 状态 | 有对应的重训次数 | 比例 |
| --- | --- | --- |
| M1 | 1019 | 50.95% |
| M2 | 1658 | 82.90% |
| M3 | 712 | 35.60% |
| M4 | 1450 | 72.50% |
| 四态同时可匹配 | 549 | 27.45% |

本轮只有全局指标或排序组成坐标达到所有重训均有定义。具体状态及有向转换的完整重训区间均留空，部分匹配区间标记为 matched-only。下表的区间为可在全部重训中计算且不跨零的逐项描述性区间。

| 对比 | 指标 | 有定义 | 原单位差值 | 重训 95% CI |
| --- | --- | --- | --- | --- |
| OFF−HC | switching_rate | 2000 | -0.02331 | [-0.03379, -0.007809] |
| OFF−HC | sorted_ilr_3 | 2000 | -0.3947 | [-0.525, -0.09339] |
| ON−OFF | switching_rate | 2000 | 0.01407 | [0.002774, 0.0213] |

[完整重训结果](../results/Bootstrap_Results.csv) · [模板对应](../results/State_Matching.csv) · [四张参考图](../figures/State_Templates.png) · [历史复核日志](../results/Historical_Audit_Verification.json)
