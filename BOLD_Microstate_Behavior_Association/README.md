# 冻结 BOLD 微状态与运动评分的配对关联

[组会报告](reports/BOLD_Microstate_Behavior_Association_Report.md) · [评分与访视核查](reports/Behavior_Data_Audit.md) · [预先固定方案](ANALYSIS_PLAN.md)

## 在授权工作区复算

在新工作树根目录运行。V6 的 `WEIDA_V6_DATA_ROOT` 指向各历史工作树与评分证据目录的共同父目录。新工作树与原 `WEiDA-DBS-V6` 并列时即可使用默认值。也可用 `--v6-private` 指向原 V6 实验的 `private` 目录。其他输入位置可用 `--help` 查看。

```powershell
$experiment = 'BOLD_Microstate_Behavior_Association'
python -B -X utf8 "$experiment/scripts/run.py" --audit-only
python -B -X utf8 "$experiment/scripts/run.py"
python -B -X utf8 -m pytest -q -p no:cacheprovider "$experiment/tests"
```

程序只读四份冻结 V6 表，并核查评分来源和协议。输入 SHA-256 与逐人证据保存在本模块 `private/manifests` 和 `private/evidence`。计算产物分别写入 `private/verified` 与 `private/archived_unverified`，逐人散点图在 `private/figures`。Python 使用 NumPy、pandas、SciPy、statsmodels、openpyxl、Matplotlib 和 pytest。绘图环境须提供简体中文字体，如 Microsoft YaHei 或 Noto Sans CJK SC。软件版本见 [运行记录](results/Run_Metadata.json)。

### 已审定访视输入

`private/inputs/Verified_Visit_UPDRS.csv` 记录 `participant_id`、`on_scan_id`、`off_scan_id`、`updrs_iii_on`、`updrs_iii_off`、`visit_evidence`。证据类型为 `direct_visit_key` 或 `confirmed_assessment_date`。仅在人工已经核对原件、条件及访视时填写。配套私有 `Verified_Visit_Evidence.csv` 每侧一行，包含 `participant_id`、`condition`、`scan_id`、`source_path`、`record_locator`、`visit_basis`、`assessment_date`。位置字段可以记录表名与行号。日期证据检查原记录评估日期是否与对应扫描日期吻合。用药和量表版本证据一并登记在来源索引或人工审计资料。

### 历史导出输入

默认读取已逐行核对的历史参照版。如果能够在授权工作区访问原 WEiDA 项目根，可运行 `--archive-project <路径>`。该根应包含各版本的 `result` 表（P、LT、两条件的经验矩阵）及用于原有队列顺序的 `data` 文件名。程序调用原历史 P/LT 行指纹匹配，在相同 50 个扫描键上核对每个可读版本，然后计算来源与两侧总分一致子集的敏感性。

## 核查产物

| 文件 | 内容 |
|---|---|
| [Verified_Visit_Associations.csv](results/Verified_Visit_Associations.csv) | 核实访视的四项关联或状态 |
| [Archived_Score_Associations.csv](results/Archived_Score_Associations.csv) | 三版历史评分各四项，含不可估计原因 |
| [Score_Source_Sensitivity.csv](results/Score_Source_Sensitivity.csv) | 各版效应及可核实一致子集的四项结果 |
| [四项效应图](figures/Association_Effect_Summary.png) | 组级 Spearman ρ、95% CI、Holm P |

`raw_p` 为常规双侧 Spearman P，`permutation_p` 为患者间置换 P。`fd_p` 为 FD 秩回归的 Freedman–Lane 置换 P。三个模型均以独立的四项家族校正。置换次数列只计非观测排列。Bootstrap 常数记录可能重叠，失败数以不可估计的重抽为准。
