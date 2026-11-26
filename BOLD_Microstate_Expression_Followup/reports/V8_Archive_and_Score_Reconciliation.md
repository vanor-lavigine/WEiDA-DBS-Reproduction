# V8 归档与原始运动评分对账

**重算纠正了原工作簿的公式错误。旧 WEiDA4-1 主分析的 50 条总分、25 人的 OFF−ON 改善分和 13 个可比领域或小计均与重算值一致。** V8 日期不限分析的三项总分和九项领域关联均未通过相应的多重比较校正。

## 队列与完整统计

原评分主表有 27 名 STN 患者及 54 条完整评分。其中 25 人的两侧评分按原采集名册、研究编号、刺激条件和冻结扫描键连接到 V7 影像。其余 2 人缺少符合冻结标准的完整影像对。27 人的 STN 靶点与身份均有原始证据，其中 1 人的人口学表姓名异写通过唯一研究编号及原采集姓名核对。 [V8 队列流转](../../BOLD_Microstate_STN_Original_Scores/results/STN_Cohort_Flow.csv)

| V8 分析集合 | 患者数 | 纳入依据 | Holm3 | BH9 | Holm12 | 全精度结果 |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 原日期筛选 | 23 | 22 对两侧日期证据齐全、1 对独立证据不足 | 0/3 | 0/9 | 0/12 | [三项总分](../../BOLD_Microstate_STN_Original_Scores/results/Primary_Associations.csv) · [九项领域](../../BOLD_Microstate_STN_Original_Scores/results/Exploratory_Domain_Associations.csv) |
| 日期不限 | 25 | 22 对核实、1 对证据不足、2 对日期冲突 | 0/3 | 0/9 | 0/12 | [三项总分](../../BOLD_Microstate_STN_Original_Scores/results/date_unrestricted/Primary_Associations.csv) · [九项领域](../../BOLD_Microstate_STN_Original_Scores/results/date_unrestricted/Exploratory_Domain_Associations.csv) |
| 双侧日期核实 | 22 | 两侧评分日期均与独立 DICOM 采集日期一致 | 0/3 | 0/9 | 0/12 | [敏感性全表中的 Verified_Visit](../../BOLD_Microstate_STN_Original_Scores/results/date_unrestricted/Motion_and_Visit_Sensitivity.csv) |

每轮归档含 12 条主口径模型记录和 48 条敏感性模型记录。22 人子集的全部 12 行在两轮中完全一致。25 人分析放宽了日期纳入标准，原有日期证据等级逐对保留。原表标注 UPDRS-III，量表版本仍未独立核实。 [23 与 25 人逐项对照](../../BOLD_Microstate_STN_Original_Scores/results/date_unrestricted/Comparison_23_vs_25.csv) · [25 人 V8 报告](../../BOLD_Microstate_STN_Original_Scores/reports/STN_Date_Unrestricted_Associations.md)

## 33 项计分

| 核对项 | 重算与对账结果 |
| --- | --- |
| 运动总分 | 相加主评分表 G:X 与 AB:AP 的 33 个 0–4 分临床条目。两份工作簿重叠的 54 条 STN 评分逐条、逐项一致。 |
| 强直 | 主表 I:M 共 5 项。 |
| 运动徐缓 | 主表 N:X 共 11 项。原 Y 列公式使用 J:X，额外纳入 J:M 四肢强直，导致 49/54 条高 1–16 分。 |
| 震颤 | 主表 AG:AP 共 10 项，含姿势性 2 项、动作性 2 项和静止性 6 项。 |
| 第二表 AT 总分 | 20/54 条公式包含 AI 预测列。14 条实际高 4–15 分，差值等于同一行五项 AI 预测之和。重算以两份表中对齐的 33 个临床条目计分。 |
| 旧参照逐扫描对账 | 依患者、扫描和条件对账。总分改变 0/50 条，13 个可比领域或小计每项均有 0/50 条改变。 |
| 旧参照逐患者对账 | OFF−ON 总分改善改变 0/25 人。原日期筛选的 23 人中改变 0/23 人。 |

[完整列映射与公式对账](../../BOLD_Microstate_STN_Original_Scores/reports/STN_Original_Score_Audit.md) · [对账统计](../../BOLD_Microstate_STN_Original_Scores/results/Score_Reconciliation_Summary.csv) · [旧参照模型对照](../../BOLD_Microstate_STN_Original_Scores/results/Legacy_Score_Replacement_Associations.csv)

## 既有 E8 复算

**25 人的震颤改善与 M4 平均持续时间变化估计为正向，预设双侧置换及多重校正均未通过。** 模型结局是 OFF−ON 震颤评分（分），影像项为 ON−OFF 的 M4 平均持续时间（秒），调整 OFF 震颤评分和 ON−OFF 平均 FD。在协变量相同的条件下，M4 平均持续时间变化每增加 1 秒，拟合的震颤改善量增加 0.14409 分。

| 有效配对 | β（分/秒） | 百分位 Bootstrap 95% CI | 双侧置换 P | BH9 q | Holm12 P |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | +0.14409025 | [0.01093595, 0.30334723] | 0.07649235 | 0.56820318 | 0.91790821 |

区间和置换使用同一批 25 名患者、同一单位和同一完整模型。百分位 Bootstrap 对完整患者行联合重抽，按 2,000 次重拟合所得系数的 2.5% 和 97.5% 分位数取区间。Freedman–Lane 以仅含 OFF 震颤评分和 ΔFD 的约简模型为零假设参照，置换其患者残差并重拟合完整模型，按 10,000 次置换中与观测值至少同样极端的 |HC3 t| 计算双侧 P。两套抽样均无失败。区间与 P 值来自不同的构造，无需在零值判定上相互对应。E8 的判定采用原预设双侧置换及校正。 [全精度复算记录](../results/V8_Archive_Verification.json)

## 归档与复算

从报告基线 `dc9c732dbbf7d3cd9005bbd908d75f5ffb1ce987` 建立 V9 后，导入 V8 完整本地归档 `6b2978c97fc38bc05f3ce577b60e66941c173768`。导入提交为 `2b44ada9fc1439d33a61d6a8438fc76c248561d8`，34 个跟踪文件的 Git 子树与原归档相同。核对了两份原分析锁中的 17 个输入项 SHA-256，重读原表并复做身份、条件及临床对账，重建的 23 人和 25 人临床输入逐字节吻合。评分对账、队列流转和评分分布汇总也逐字节吻合。输入指纹存于忽略提交的私有复核记录。

在可访问原锁定输入的工作区，从 V9 仓库根目录运行以下命令。复算使用原模型代码、同一重抽规则和种子 `7261126`。

```powershell
python -B -X utf8 BOLD_Microstate_Expression_Followup/scripts/verify_v8_archive.py
```

[V8 锁定方案](../../BOLD_Microstate_STN_Original_Scores/ANALYSIS_PLAN.md) · [模型代码](../../BOLD_Microstate_STN_Original_Scores/scripts/regression.py) · [原运行记录](../../BOLD_Microstate_STN_Original_Scores/results/Run_Verification.json) · [日期不限运行记录](../../BOLD_Microstate_STN_Original_Scores/results/date_unrestricted/Verification.json)
