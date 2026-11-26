# V10 M1 占用贡献的头动与可靠性复核

固定删帧后，25 名患者的 M1 占用贡献仍为正，平均 **+0.793 个 GEV 百分点**，约为原窗估计的 **85.5%**。两项敏感性校正后 P = 0.03980，累计十项校正后 P = 0.15918。原窗发现保留正向证据，删帧结果提供了同一队列内的敏感性支持。

## 原窗的 FD 调整影响

三套模型均使用同一批 25 人。未调整效应为配对平均，调整效应为 ΔFD 或 ΔDVARS 为零时的截距。括号内为 95% 患者 Bootstrap 百分位区间，标准误和效应均为 GEV 百分点。

| 原窗模型 | 效应 [95% CI] | 标准误 | Bootstrap 标准误 | 双侧置换 P | Holm2 P | Holm8 P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 未调整 | +0.928 [0.395, 1.481] | 0.283（配对均值） | 0.278 | 0.00340 | 0.00680 | 0.02480 |
| 调整 ΔFD | +0.785 [0.219, 1.290] | 0.293（HC3） | 0.272 | 0.01270 | 0.02540 | 0.08889 |
| 调整 ΔDVARS | +0.943 [0.353, 1.554] | 0.315（HC3） | 0.299 | 0.00620 | 0.01240 | 0.04340 |

调整 FD 后，点估计下降 **0.142 个百分点（15.3%）**。常规标准误从 0.283 到 0.293，使用同样患者重抽衡量的标准误从 0.278 到 0.272，区间宽度由 1.086 到 1.071。精度总体相近。原始 P 上升，Holm8 的家族仍是八项、占用项在两模型中原始 P 均排第 2，FD 模型仍通过 Holm2。未调整模型用均值符号翻转，调整模型用残差符号翻转和 HC3 截距统计量，P 的变动同时反映估计和推断构造。结果显示 FD 调整后效应有所减弱，现有量级和精度无法据此将原窗差异归为头动伪影。每行均有 10,000 次有效置换、2,000 次有效 Bootstrap。 [全精度复算](../results/V9_Occupancy_Motion_Replay.csv)

## 逐帧质量规则与有效暴露

65 次扫描的 fMRIPrep 21.0.2 原始逐帧 FD 与冻结 confounds 副本逐项相符，扫描键、刺激条件和平均 FD 也相符。Power FD 按相邻 TR 的平移绝对变化与按 50 mm 半径换算的旋转绝对变化求和，单位为 mm。FD 的首 TR 天然缺少前一时点，原档案该位置为 NA。三条 209 帧的 ROI 序列缺的是首 TR，均从原 210 TR 时钟的第 2 个 TR 对齐。原先 V9 对短序列位置的文字和边界检查已纠正，原逐帧测量本身一直使用冻结标签的有效窗。 [fMRIPrep 21.0.2 代码](https://github.com/nipreps/fmriprep/blob/21.0.2/fmriprep/workflows/bold/confounds.py) · [Nipype FD 接口](https://nipype.readthedocs.io/en/1.4.1/api/generated/nipype.algorithms.confounds.html#framewisedisplacement) · [运行记录](../results/Run_Manifest.json)

按原时钟排除 **FD > 0.50 mm** 的 TR 及前 1、后 2 个 TR，再与原 A3 共同窗取交集。正式口径使用同一折 G1 模板、S1、λ=0.04，以及删帧前 F1/N0 原始有效窗的平均 GFP²，在每个连续块独立重新解码。原窗 S1 标签逐帧复现 65/65。删帧后 282 个连续块单独复算一致，65 次扫描的留帧数、四态 Coverage、GEV 与旧 V4 质量窗口结果一致，GEV 最大绝对误差为 $5.55\times10^{-17}$。 [冻结流程](../ANALYSIS_PLAN.md) · [运行记录](../results/Run_Manifest.json)

| 条件 | 扫描数 | 剔除帧数 / 原共同窗帧数 | 单次剔除帧数中位数 | 保留秒数中位数 [范围] | 连续块总数（单次中位数） |
| --- | ---: | ---: | ---: | ---: | ---: |
| HC | 15 | 418 / 2,912 | 10 | 380.1 [165.9, 424.2] | 54（3） |
| OFF | 25 | 518 / 4,852 | 4 | 396.9 [98.7, 422.1] | 94（2） |
| ON | 25 | 698 / 4,810 | 12 | 382.2 [115.5, 424.2] | 134（2） |

三种口径均有 25 个可分解的完整患者配对。ON 的平均保留时长比 OFF 短 18.65 秒，删帧前平均相差 −3.53 秒。质量掩膜口径和重新解码口径保留相同 TR；重新解码改变了患者 26 次扫描中的 100 个保留帧标签。扫描曝光范围、各条件 C、W、R、F、GEV 均值以及配对可用性在汇总表中逐口径报告。 [扫描曝光与读数](../results/Scan_Window_Summary.csv) · [扫描测量状态](../results/Scan_Measurement_Status.csv) · [配对可用性](../results/Pair_Availability.csv)

## 删帧后贡献与累计校正

M1 占用比例为 $C$，M1 占保留窗 GFP² 的权重为 $W$，相对幅度为 $R=W/C$，$F$ 为 M1 帧内 GFP² 加权平方相关。三种口径各用自己的有效帧计算 $W=CR$、$K_C+K_R=C_W$、$K_C+K_R+C_F=\Delta\mathrm{GEV}$。75 个口径与患者配对均通过分解恒等式。

| 口径 | 占用贡献 $K_C$ [95% CI] | 相对幅度贡献 $K_R$ [95% CI] |
| --- | ---: | ---: |
| 原窗 | +0.928 [0.395, 1.481] | −0.073 [−0.346, 0.179] |
| 删帧后沿用冻结标签，描述性 | +0.851 [0.192, 1.475] | −0.138 [−0.383, 0.102] |
| 删帧后 S1 重新解码，正式敏感性 | +0.793 [0.143, 1.394] | −0.102 [−0.355, 0.140] |

以相同的 25 个配对作比较，删帧本身使 $K_C$ 下降 0.077 个百分点，重新解码额外下降 0.057。两步合计 −0.134 [−0.411, 0.101] 个百分点，表示窗口及解码改变下的配对估计差。 [同患者三口径和分步差值](../results/Window_Comparison.csv)

| 正式敏感性端点 | n | 双侧置换 P | Holm2 P | 累计 Holm10 P | 配对 $d_z$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| $K_C$ | 25 | 0.01990 | 0.03980 | 0.15918 | +0.497 |
| $K_R$ | 25 | 0.41826 | 0.41826 | 1.00000 | −0.165 |

各有 10,000 次有效患者符号翻转、2,000 次有效患者 Bootstrap。累计 Holm10 包含原 V7/V9 未调整八项和两项预定删帧敏感性，沿用完整原始置换 P。十项中原 M1 权重与原占用贡献仍通过校正（均 P = 0.03100），两项新增删帧条目未通过。 [正式敏感性](../results/Censored_Contribution_Tests.csv) · [累计十项完整结果](../results/V7_V9_V10_Holm10.csv)

## 患者影响与模板可靠性

原窗 $K_C$ 与 ΔCoverage 的 Spearman ρ = +0.938，二者共享计算中的占用差。$K_C$ 与 ΔFD 的 ρ = +0.205，ΔCoverage 与 ΔFD 的 ρ = +0.147。删帧导致的 $K_C$ 变化与 ON−OFF 保留秒数差的变化呈描述性负相关（ρ = −0.341，留一范围 −0.440 至 −0.254）。这表明保留时长差值得纳入解释。逐一留出单个患者时，原窗 $K_C$ 为 +0.812 至 +1.046，FD 调整截距为 +0.636 至 +0.905，重新解码后为 +0.652 至 +0.938，均保持正向。完整样本用于推断，留一只诊断影响。 [分布与头动关联](../results/Patient_Diagnostic_Distributions.csv) · [关联及留一](../results/Patient_Diagnostic_Associations.csv) · [模型留一](../results/Leave_One_Out_Summary.csv)

两套 V4 正式重训存档逐轮一致。M1 有 **1,019/2,000 次（50.95%）** 成功匹配。仅在这些轮次中，ON−OFF Coverage 和 GEV 的平均差为 +5.965 和 +1.200 个百分点，95% 分位区间分别为 [1.849, 10.017]、[0.441, 1.977]。另 981 次的相关状态结果保留缺失。旧档案缺每轮逐患者逐条件的 $C,W,F$，无法恢复 $K_C$ 的全流程重训区间。 [重训档案复核与扩展所需字段](Template_Refit_Audit.md)

现有证据支持以下表述。**在本队列和冻结的 A3 模板下，ON 相比 OFF 的 M1 占用对 GEV 增量的贡献为正；FD 调整和固定删帧后点估计仍为正，严格累计校正下的敏感性支持有限。** 24 个可核实顺序的配对均先 ON 后 OFF，现有设计只能解释为条件差异。M1 在完整重训中的识别率进一步限制了模板无关的推断。 [扫描顺序来源](../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/ON_OFF_Data_Quality_Report.md)

## 复算与发布

从仓库根目录使用已核验的源档案运行下列命令。V7、V8、V9 的 21 项软件验证测试通过，V10 新增 6 项软件验证测试通过，11 份 V10 统计 CSV 复跑逐字节一致。公开哈希记录既有 V8/V9 配置与统计表的指纹，以及不含路径的私有输入集合 SHA-256。 [输入指纹](../results/Input_Fingerprints.json) · [V8 评分对账汇总](../../BOLD_Microstate_STN_Original_Scores/results/Score_Reconciliation_Summary.csv) · [V9 总报告](../../BOLD_Microstate_Expression_Followup/reports/Final_Interpretation.md)

~~~powershell
python -B -X utf8 BOLD_Microstate_M1_Robustness/scripts/run_v10.py
python -B -X utf8 BOLD_Microstate_M1_Robustness/scripts/audit_refits.py
python -B -X utf8 -m pytest -q -p no:cacheprovider BOLD_Microstate_M1_Robustness/tests BOLD_Microstate_Expression_Followup/tests BOLD_Microstate_STN_Original_Scores/tests BOLD_Microstate_Mechanistic_Decomposition/tests
~~~
