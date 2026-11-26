# BOLD 微状态变化的机制分解 V7

本模块复用冻结 A3 标签和五折模板。25 名患者提供 ON/OFF 配对，15 名健康对照用于核验完整的 65 次扫描及描述取样。主分析在六项配对分解检验内做 Holm 校正。

[最终报告](reports/V7_Final_Mechanistic_Interpretation.md) · [锁定方案](ANALYSIS_PLAN.md) · [配置](configs/protocol.json) · [主结果](results/Mechanistic_Primary_Statistics.csv) · [核验记录](logs/Run_Verification.json)

## 复算

在 V7 工作树根目录运行。读取上轮定向分析的私有输入索引以定位 V6 冻结表，用 V4 冻结收据定位档案与 ROI 时序。可通过 `--workspace` 指定与收据相符的根目录。帧标签、患者信息、私有配对图和实际本机命令保存在忽略目录 `private/`。

```powershell
$v7 = 'BOLD_Microstate_Mechanistic_Decomposition'
$inputIndex = 'PATH_TO_TARGETED_INPUT_HASHES.json'
$aalVolume = 'PATH_TO_AAL1ctx_MNI.nii.gz'
$aalLabels = 'PATH_TO_AAL1ctx_MNI.csv'
$yeoVolume = 'PATH_TO_YEO7_TIGHT_MNI152.nii.gz'
python -B -X utf8 "$v7/scripts/run_v7.py" --audit-only --prior-input-index $inputIndex
python -B -X utf8 "$v7/scripts/run_v7.py" --prior-input-index $inputIndex
python -B -X utf8 "$v7/scripts/network.py" --aal-map $aalVolume --aal-labels $aalLabels --yeo-map $yeoVolume
python -B -X utf8 "$v7/scripts/render_v7.py"
python -B -X utf8 -m pytest -q -p no:cacheprovider "$v7/tests"
```

影像分析环境见运行记录。体素图谱注释还使用 nibabel，输入版本和图谱指纹见[图谱核验](logs/Network_Atlas_Verification.json)。图件使用 600 DPI PNG 及矢量 PDF。公开的图 2 为配对患者的组级均值和区间，私有图 2 保留 25 条真实连接线。
