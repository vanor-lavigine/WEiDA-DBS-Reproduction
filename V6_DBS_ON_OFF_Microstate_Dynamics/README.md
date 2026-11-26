# V6 DBS ON–OFF Microstate Dynamics

本研究使用 25 名有 ON/OFF 配对扫描的 PD 患者和 15 名健康对照。患者是推断单位。冻结的 A3、AAL78、四态、TR=2.1 s、F1 宽频 0.01–0.08 Hz、N0、G1、S1、λ=0.04 全部沿用。V4 和 V5 两个提交的文件逐一做过 SHA-256 校验。

| 入口 | 内容 |
|---|---|
| [最终科学解释](reports/V6_Final_Scientific_Interpretation.md) | 十个研究问题和证据层级 |
| [配对结果](reports/V6_ON_OFF_Microstate_Results.md) · [异质性](reports/V6_Paired_Response_Heterogeneity.md) | 两个主要终点、次要结果、个体变化 |
| [转移](reports/V6_Transition_Dynamics.md) · [健康参考](reports/V6_Normalization_Analysis.md) · [UPDRS](reports/V6_UPDRS_Association.md) | 整体与路径、距离、临床访视资格 |
| [数据质量](reports/ON_OFF_Data_Quality_Report.md) · [冻结引用](V4_V5_FROZEN_REFERENCE.json) · [哈希表](manifests/V4_V5_SHA256.csv) | 25 对核验、扫描时序及输入谱系 |
| [固定方案](ANALYSIS_PLAN.md) · [配置](configs/protocol.json) · [核验](logs/Run_Verification.json) | 统计细节、可重现性及质量检查 |
| [41 项标量](results/ON_OFF_Feature_Summary.csv) · [整体检验](results/ON_OFF_Omnibus_Results.csv) · [重抽审计](results/Paired_Bootstrap_Audit.csv) | 效应、置信区间、原始 P、BH、Holm 和敏感性 |

公开图为真实逐人数据产生的组级汇总。授权工作树运行后生成逐人图和配对源表。`private/manifests/ON_OFF_Pair_Manifest.csv` 保留来源扫描 IDs、质量及访视信息。`private/results/ON_OFF_Subject_Features.csv`、`ON_OFF_Paired_Differences.csv`、`Paired_Bootstrap_Draws.csv` 保留逐人或逐重抽的完整输出。本地八张逐人图在 `private/figures/`。所有图以 600 DPI PNG 和原生矢量 SVG/PDF 导出。Git 忽略 `private/`。

标量统计以 ON−OFF 为方向。`raw_p` 由预定分布规则选取，两个原始检验也并列保留。`permutation_p` 是配对符号翻转的 P。`fd_p` 与 `dvars_p` 每个患者仍只提供一条差值。`order_subset_p` 对已核实顺序的 24 对执行相同配对检验。`fd_effect` 是 ΔFD=0 的截距。转换路径的 `*_gate_open` 指示对应整体家族是否通过。`*_evidence_class` 标记校正层级。健康参考距离的正变化另定义为 OFF 距离减 ON 距离。

## 复算

从 V6 工作树根目录运行。现有历史布局下，工作树与 `WEiDA-DBS-V4`、`WEiDA-DBS-V3`、原始 105 冻结清单和访问审计同一父目录。完整复算须有权读取这些本地档案。环境依赖 Python、NumPy、pandas、SciPy、statsmodels、scikit-learn、Matplotlib、Pillow、Numba、openpyxl 和 pytest。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$v6 = 'V6_DBS_ON_OFF_Microstate_Dynamics'
python "$v6/scripts/prepare.py"
python "$v6/scripts/analyze.py"
python "$v6/scripts/make_figures.py"
python -m pytest -q -p no:cacheprovider "$v6/tests"
python "$v6/scripts/verify.py"
```

UPDRS 的 `BLOCKED` 来源于评分与扫描访视连接缺失。如果有审核过的配对原始记录，可把记录放在被忽略的 `private/inputs/Verified_Visit_UPDRS.csv`，字段为 `participant_id`、`on_scan_id`、`off_scan_id`、`updrs_iii_on`、`updrs_iii_off` 和 `visit_evidence`。最后一列登记 `direct_visit_key` 或 `confirmed_assessment_date`，用于描述已核实的连接依据。输入记录留在本地。
