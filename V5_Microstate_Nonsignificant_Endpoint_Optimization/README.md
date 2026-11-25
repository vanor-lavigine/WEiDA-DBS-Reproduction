# V5 Microstate Nonsignificant Endpoint Optimization

在 15 名 HC 和 25 名 STN-DBS 患者的冻结 BOLD 队列中，复用 V4 的 A3 五折模板与标签，检验四态占用、整体转换结构和到健康参考的配对变化。V5 是同队列探索性再分析。

| 入口 | 内容 |
|---|---|
| [最终科学解释](reports/V5_Final_Scientific_Interpretation.md) | 疾病相关证据、刺激推断和后续验证 |
| [主报告](reports/V5_Microstate_Optimization_Report.md) | 六项主检验、停留、计数、质量和功效 |
| [Coverage](reports/V5_Coverage_Analysis_Report.md) · [Transition](reports/V5_Transition_Matrix_Report.md) · [健康参考](reports/V5_Restoration_Analysis_Report.md) · [噪声校准](reports/V5_Noise_Calibration_Report.md) | 指标定义、效应、图与敏感性分析 |
| [预定方案](ANALYSIS_PLAN.md) · [参数](configs/protocol.json) | 分析单位、统计量、校正、重抽与模拟门槛 |
| [V4 冻结清单](V4_FROZEN_SIGNIFICANCE_REGISTRY.csv) · [哈希](manifests/V4_SHA256.csv) | 原提交和全部历史证据层级 |
| [V5 全表](results/All_V5_Statistical_Results.csv) · [六项主检验](results/Primary_Six_Results.csv) | 每项检验的效应、区间、P、FD、BH、Holm 和解释门槛 |
| [统计与产物核验](logs/Run_Verification.json) | 置换、自助、模拟、数据完整性和软件版本 |

CSV 的独立组效应以列中第一个条件减第二个条件。`restoration` 特例为 OFF 距离减 ON 距离。`raw_p` 和 `permutation_p` 分开保留，具体检验见 `*_p_basis`。`fd_p` 对计数模型表示调整 FD 的聚类 GLM P，FD 段率置换另列。`*_holm_p` 与 `*_fdr_p` 是指定家族值，`joint_*` 跨新终点。`*_gate_open` 决定单态或路径是否进入相应推断。`*_supported` 已将闸门与校正一起考虑。状态未见时停留 NA。置信区间为所列效应的固定模板自助区间。

## 运行

从仓库根目录运行下列 PowerShell 命令。Python 依赖 NumPy、SciPy、pandas、statsmodels、Numba、openpyxl、Matplotlib 和 Pillow。完整复算需授权的本地档案，默认布局为 V5 工作树与 `WEiDA-DBS-V4`、`WEiDA-DBS-V3`、原冻结 105 ROI 与质控档案位于同一父目录。V4 固定标签需在本地可读。全部患者级派生文件和模拟断点写入 `private/`，Git 忽略此目录。发布结果包含匿名汇总及纯合成队列。

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$v5 = 'V5_Microstate_Nonsignificant_Endpoint_Optimization'
python "$v5/scripts/v5_data.py"
python "$v5/scripts/noise_calibration.py" --workers 10
python "$v5/scripts/run_analysis.py"
python "$v5/scripts/audit_and_power.py"
python "$v5/scripts/make_figures.py"
python "$v5/scripts/verify.py"
```

V4 基准 SHA 为 `2d66fb12b5883aa3a81f30c5cf9bd749d6ffaaad`，对应标签 `v4-microstate-frozen-2026-11-25`。
