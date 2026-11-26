# V9 结果与解释

**同一批 25 名患者中，ON 相对 OFF 的 M1 GFP² 权重增量主要来自占用项。M1 与 M4 连续表达变化均未获得与运动总分改善相关的校正后证据。**

1. **原始评分对账。** V8 重算纠正了工作簿的运动徐缓与第二表总分公式。原 WEiDA4-1 参照的 50 条总分及 25 名患者的改善分逐条一致。V8 日期不限分析为 Holm3 0/3、BH9 0/9、Holm12 0/12。既有震颤 × M4 持续时间的 P = 0.07649、BH9 q = 0.56820，按预设双侧置换判定。 [V8 归档与评分对账](V8_Archive_and_Score_Reconciliation.md)
2. **M1 权重拆分。** 占用贡献 +0.928 个 GEV 百分点（95% CI [0.395, 1.481]），相对幅度贡献 −0.073（[−0.346, 0.179]）。主分析新两项中只有占用项通过 Holm2（P = 0.00680）和 Holm8（P = 0.02480）。占用项的 Holm8 在 ΔDVARS 调整后通过（P = 0.04340），在 ΔFD 调整后未通过（P = 0.08889）。 [M1 分解与敏感性](M1_Occupancy_Amplitude_Decomposition.md)
3. **连续模板表达。** 全窗表达的平均配对变化为 M1 +0.01403、M4 −0.00952。65 次扫描均重算出与冻结统计一致的四态硬标签 GEV，各折保留原模板编号与有效窗。 [连续表达方法与描述结果](Continuous_Template_Expression.md)
4. **临床关联。** 主模型中每增加 0.01 表达，拟合的改善量变化为 M1 +0.825 分、M4 +0.539 分。两项区间均跨零，Holm2 0/2，累计 Holm14 0/14；两套头动敏感性及 22 人日期核实子集的结论一致。逐人留一时 M1 的系数曾接近零，M4 的点估计方向在各次留一中一致。 [临床模型与留一诊断](Motor_Association_Followup.md)

这些发现适用于 AAL78、K=4、TR=2.1 秒、0.01–0.08 Hz、N0/G1/S1、λ=0.04 的五折冻结模板和本队列。24 个有独立时钟的配对均先 ON 后 OFF，扫描先后与条件的效应在本队列难以分辨。25 人临床分析保留 1 对日期证据不足和 2 对日期冲突，22 人日期核实分析提供固定敏感性核对。占用项支持本队列的影像条件差异；中介机制和临床改善仍需要独立证据。 [V6 扫描顺序审计](../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/ON_OFF_Data_Quality_Report.md)

在拥有锁定输入的工作区，从仓库根目录运行以下命令。运行记录含输入基线提交、版本、验证误差及重抽设置；具体输入哈希在私有索引 `private/manifests/Input_Hashes.json`。

```powershell
python -B -X utf8 BOLD_Microstate_Expression_Followup/scripts/verify_v8_archive.py
python -B -X utf8 BOLD_Microstate_Expression_Followup/scripts/run_v9.py
python -B -X utf8 -m pytest -q -p no:cacheprovider BOLD_Microstate_Expression_Followup/tests
```

V7、V8、V9 的 21 项测试全部通过。复跑 V9 后，七份统计 CSV 逐字节一致。两组累计校正分别覆盖登记的 V7+V9 影像检验和 V8+V9 临床检验。

[冻结方案](../ANALYSIS_PLAN.md) · [假设和统计配置](../configs/protocol.json) · [运行记录](../results/Run_Manifest.json) · [V7/V9 联合校正](../results/V7_V9_Joint_Correction.csv) · [V8/V9 联合校正](../results/V8_V9_Clinical_Joint_Correction.csv)
