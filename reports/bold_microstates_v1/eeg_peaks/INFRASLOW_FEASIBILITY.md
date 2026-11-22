# 同一低频 BOLD 信号上比较亚稳态与微状态的可行性

2026 年 11 月 22 日。在已核验的真实 AAL78 衍生时序上检查两条分析所需的输入。105 为 25 对 STN 患者和 15 名健康对照，SyN9 为 26 对和 15 名健康对照，其中 24 对患者重合。SyN9 上游 XCP-D 带通为 0.01–0.08 Hz。105 使用归档 BIDS-fMRIpost 波形，其上游缓存的完整滤波记录仍缺失。[输入与配对依据](../legacy_105/README.md) · [SyN9 处理记录](../../2026-11-21-v100-syn9-stn-hc.md)

两频段的状态共现和 ΔFD 调整关联已在两套真实数据上完成。[四组并列结果](../same_band/README.md)

## 两个候选频段的实测支持

0.04–0.07 Hz 取自 WEiDA 原文。0.01–0.027 Hz 取自静息态 fMRI 文献的 slow-5 分带。这两段在本次可行性检查前选定。两套数据均以 TR=2.1 秒计算。表中周期数为频率乘以 210 帧的名义采样窗 441 秒，105 的三次较短扫描各有 209 帧。[WEiDA 原文方法](https://pmc.ncbi.nlm.nih.gov/articles/PMC13280393/) · [Zuo 等的 slow-5 分带](https://pmc.ncbi.nlm.nih.gov/articles/PMC2856476/)

| 频段 | 每次约含周期数 | 105 每次扫描的空间振幅峰，中位数［范围］ | SyN9 每次扫描的空间振幅峰，中位数［范围］ |
|---|---:|---:|---:|
| 0.04–0.07 Hz | 17.6–30.9 | 48［45–52］ | 48［43–51］ |
| 0.01–0.027 Hz | 4.4–11.9 | 18［15–20］ | 17［14–19］ |

两频段均对每个 ROI 线性去趋势、去均值并作二阶 Butterworth 双向带通。同一滤波结果一路计算 78 区 Hilbert 相位，另一路复用 ROI 时间标准化、空间标准差局部极大值和忽略极性的 AAHC。105 的 65 次和 SyN9 的 67 次扫描在两个频段均得到有限的相位和四张扫描级单位空间图。两种独立的峰定位实现给出了相同帧号。四张图由预设 K=4 决定，模板能否稳定复现、两路状态能否对应及 ON/OFF 效应仍待检验。空间振幅峰是聚类可用的地形样本，峰数也受相邻帧自相关影响。[WEiDA 滤波和局部亚稳态实现](https://github.com/chenfei-ye/WEIDA-DBS/blob/main/MSanalysis.m) · [选峰方法](PLAN.md)

若目标为 0.01 Hz 以下，SyN9 当前衍生物在这一范围受到上游高通衰减。441 秒在 0.005 Hz 仅约 2.2 个周期，0.001 Hz 约 0.44 个周期。研究这一区间需要核实高通前时序及其漂移控制，若要稳健估计低端的相位和状态切换，还需更长记录。

## 比较的对象

在同一扫描的同频信号上，BOLD 微状态用幅度空间图定义，量化持续时间、出现率和覆盖率。WEiDA 用脑区间相位关系的加权特征向量定义状态。论文中的局部亚稳态度量该状态内特征向量相对于群体质心的离散程度。载波频带、状态驻留时长以及该离散程度是三个不同的量。[WEiDA 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13280393/) · [Hancock 等对亚稳态及其观测指标的讨论](https://doi.org/10.1038/s41583-024-00883-1)

- 建议将 0.04–0.07 Hz 作为主要比较，将 slow-5 单列为探索性比较。各频段都用同一滤波输入分别计算幅度和相位，核对 ROI、原始帧号、滤波边缘，并在两路均有效的帧上比较。slow-5 的相位状态和局部亚稳态需要在该频段重算。
- 要回答两种状态是否同时出现，可为每次扫描建立四乘四的共现表，计算不依赖状态编号的调整互信息。扫描内循环移位一侧标签，并保持有效时间段边界，可以给出考虑时间自相关的参照。组级推断以患者为单位。
- 要回答局部亚稳态变化是否伴随驻留变化，可事先指定论文关注的 WEiDA 状态 2，固定其空间匹配，在每套处理内检验患者的局部亚稳态 ON−OFF 差与四个 BOLD 幅度状态持续时间差的关联，加入同一患者的 ΔFD。只检验持续时间时四项一起作 Holm 校正。若同时纳入出现率和覆盖率，则十二项一起校正。两套处理的患者部分重合，适合评估处理流程敏感性。
- 若目标是验证真实 EEG 微状态与 BOLD 的时间联系，需要同步 EEG–fMRI。可以提取每个 EEG 微状态的匹配时间序列，经血流动力学响应卷积后再与 BOLD 对齐。[Britz 等同步 EEG–fMRI 研究](https://cibm.ch/wp-content/uploads/1-s2.0-S105381191000220X-main.pdf)

## 实际复算命令

在仓库根目录从 PowerShell 运行。环境为 Python 3.12.7、NumPy 1.26.4、SciPy 1.13.1、pandas 2.2.2。命令仅输出扫描级检查的汇总计数。

~~~powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:BLIS_NUM_THREADS='1'
$env:PYTHONUTF8='1'
@'
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, detrend, filtfilt, find_peaks, hilbert
sys.path.insert(0, str(Path("experiments/bold_microstates_v1").resolve()))
from compare_legacy_105 import REPORTS, read_inputs
from peak_states import K, prepare_scan

tr = 2.1
columns = pd.read_csv(REPORTS / "templates.tsv", sep="\t", index_col=0).columns.tolist()
cohorts = read_inputs(Path(sys.argv[1]), columns)
print("Python", sys.version.split()[0], "NumPy", np.__version__, "SciPy", scipy.__version__, "pandas", pd.__version__)
print("source,band_Hz,scans,scans_with_4_maps_and_finite_phases,peaks_min,peaks_median,peaks_max")
for source, scans in zip(("105", "SyN9"), cohorts):
    for low, high in ((0.04, 0.07), (0.01, 0.027)):
        b, a = butter(2, [low, high], btype="bandpass", fs=1/tr)
        padlen = 3 * (max(len(a), len(b)) - 1)
        counts = []
        successful = 0
        for scan in scans.values():
            residual = detrend(scan.values, axis=0)
            residual -= residual.mean(axis=0)
            band = filtfilt(b, a, residual, axis=0, padlen=padlen)
            analytic = hilbert(band, axis=0)
            phases_ok = np.isfinite(np.angle(analytic)).all() and (np.abs(analytic) > 0).all()
            z = (band - band.mean(axis=0)) / band.std(axis=0, ddof=1)
            peaks = find_peaks(z.std(axis=1))[0]
            counts.append(len(peaks))
            if len(peaks) >= K:
                prepared = prepare_scan(band, scan.mean_fd)
                np.testing.assert_array_equal(peaks, prepared.peaks)
                norms = np.linalg.norm(prepared.individual, axis=1)
                if phases_ok and norms.shape == (K,) and np.allclose(norms, 1):
                    successful += 1
        print(f"{source},{low:g}-{high:g},{len(scans)},{successful},{min(counts)},{np.median(counts):g},{max(counts)}")
'@ | python -B - 'E:\NIT\神经调控数据'
~~~
