# A3 微状态方法、结果与流程冻结

## 冻结的分析

同一 V3 冻结队列包含 15 名 HC、25 名配对患者及 65 次扫描。使用 AAL78、TR 2.1 s、固定 K=4、F1/N0、A3 正则化回填与 λ=0.04。状态编号以 G1 折 0 的四张模板为锚，组间效应使用五折留出特征。预定对比为 OFF−HC、ON−HC 与 ON−OFF。标量和 ILR 检验使用参与者级置换，逐项保留校正结果。已有 2,000 轮完整重训结果及状态匹配情况与本次提交一起归档。[锁定参数](../configs/protocol.json) · [统计方法](../reports/Statistical_Methods.md)

OFF−HC 的四态 Coverage 组成置换 Holm P=0.0008，FD 调整后仍为 0.0008。固定特征下，OFF 的 M4 平均持续时间比 HC 长约 4.31 秒。全局切换率的重训区间支持 OFF 相对 HC 降低和 ON 相对 OFF 增加。四态同时可匹配的重训轮次为 549/2,000。使用本提交的结果时保留家族校正、噪声校准和重训可识别性的限制。[结果报告](../reports/V4_Microstate_Statistical_Results.md) · [重训与噪声报告](../reports/Microstate_Stability_Report.md)

## 检查公开快照

从实验目录运行以下命令，用 Python 标准库核对冻结配置、统计表、报告、PNG、流程代码和算法依赖快照。文本 SHA-256 统一以 LF 换行计算，PNG 按原始字节计算。`results/Final_Verification.json` 保留原工作区运行时的验证记录。

```powershell
python scripts/verify_freeze.py
```

依赖快照位于 `workflow/source/`，路径与工作区中的源文件布局对应。装载器和私有输入的内容指纹记录在清单中。逐参与者材料由原授权工作区提供。

## 在授权工作区复算

复算使用原目录拓扑。`WEiDA-DBS-V4` 中包含本实验、原 V4 Reliability 实验、本地 V3 实现及前序统计脚本。并列的 `WEiDA-DBS-V3` 提供冻结统计归档。原 ROI 和混杂副本、V4 回填归档及模拟诊断从授权工作区读取。先核对这套已锁定的依赖与档案，再运行分析。

```powershell
conda env create -f workflow/environment.yml
conda activate microstate-significance-v4
$research_workspace = (Resolve-Path ../..).Path
python scripts/verify_freeze.py --workspace $research_workspace
python scripts/audit_history.py
python scripts/micro_inference.py
python scripts/full_bootstrap.py --workers 10
python scripts/micro_figures.py
python scripts/write_reports.py
python scripts/verify.py
```

`$research_workspace` 指向同时含 `WEiDA-DBS-V3` 和 `WEiDA-DBS-V4` 的数据工作区。重训批次可在私有缓存中续跑。模拟复核读取历史归档的逐队列结果。本次快照记录在冻结后证据和方法不变时可核验的结果。
