# WEiDA-DBS Reproduction

## V100 预处理交接

截至 2026 年 11 月 16 日，V100 上这批修复 BIDS 副本的 **126 个静息态 BOLD run** 已完成 fMRIPrep 和 XCP-D。122 个 run 走 CIFTI，4 个 run 走 NIfTI。每个 run 均保留 210 个时间点，TR 为 2.1 秒。计数单位为 run，独立受试人数及 ON/OFF 配对仍需依据原始队列证据确认。

| 阶段 | 本批已核验结果 |
| --- | --- |
| BIDS 输入 | 126 个 BOLD run。迁移清单记录 510 个已核验文件。独立 BIDS Validator 退出码为 0，记录 0 个错误、1404 个警告。 |
| FreeSurfer 8.2.0 | 122 个 CIFTI run 复用已核验的 FreeSurfer 表面，逐条复核源重建的构建版本均为 8.2.0。fMRIPrep 批次内重跑 `recon-all` 为 0。 |
| fMRIPrep 25.2.7 | 126 个 run 通过退出码、成功日志和时间轴核验。126 个 run 均有 T1w 空间和 MNI152NLin2009cAsym 2 mm 体积 BOLD。122 个 run 另有 fsLR 91k CIFTI。 |
| XCP-D 26.1.0 | 122 个 CIFTI run 和 4 个 NIfTI run 通过输出核验，最终跳过 0 个。实际峰值并发 34 个任务。 |
| ROI 与静态 FC | 630 份 ROI 时间序列 TSV、630 份 Pearson 矩阵 TSV，另有 610 对原生 CIFTI `.ptseries.nii` / `.pconn.nii`。 |
| 影像和来源 QC | 126 个 run 的输出人工影像 QC 标记为待复核。上游历史记录有 11 个 run 已通过当时的技术准入，其余保留原有来源、身份或解剖复核状态。 |

### 输入来源

处理所用输入可追溯到内部迁移的修复 BIDS 副本。该工作目录的名称带有 `ds005237`，本批文件实测为 210 帧、TR 2.1 秒，公开 TCP `ds005237` 的采集说明为 TR 0.8 秒，且队列定义不同。当前记录以内部修复批次为来源，原始扫描到研究队列的映射和与论文作者数据的一致性仍待核验。[OpenNeuro 数据集维护者说明](https://github.com/OpenNeuroDatasets/ds005237) · [WEiDA 论文数据可用性声明](https://doi.org/10.1038/s41531-026-01354-3)

### 实际处理设置

**fMRIPrep 25.2.7** 对 `rest` 做 participant 级处理，指定 `--output-spaces T1w MNI152NLin2009cAsym:res-2`。122 个 run 提供 FreeSurfer `subjects_dir`，使用 `--fs-no-resume --cifti-output 91k`。4 个 run 使用 `--fs-no-reconall`，保留体积分支。126 个 run 均有 slice-timing 执行记录和 210 行 confounds。本批无可用 fieldmap，未执行 SDC。独立 BIDS Validator 审计与运行记录一起保留，fMRIPrep 启动使用 `--skip-bids-validation`。输入只读挂载，控制器结束时复核了输入完整性。fMRIPrep 于北京时间 2026 年 11 月 14 日 12:47 完成，峰值并发 20 个任务。[fMRIPrep 25.2.7 输出文档](https://fmriprep.org/en/25.2.7/outputs.html)

| XCP-D 26.1.0 实际参数 | 设置 |
| --- | --- |
| 输入与工作模式 | `--input-type fmriprep --mode linc --task-id rest`，按 run 使用 `--file-format cifti` 或 `nifti` |
| 去噪 | 36P，即运动参数、全局信号、白质信号和脑脊液信号及其导数和平方项。启用 despike，头半径按 `--head-radius auto` 估计。 |
| 时间域 | `--dummy-scans 0 --fd-thresh 0 --combine-runs n`，关闭基于 FD 的删帧，保留 210 帧。带通为 0.01–0.08 Hz，阶数 2。 |
| 空间域 | `--smoothing 0 --min-coverage 0.2`。CIFTI 输入为 fsLR 91k（91,282 个 grayordinates），NIfTI 输入为 MNI152NLin2009cAsym 2 mm。 |
| 图谱及 QC | `4S156Parcels`、`4S256Parcels`、`4S456Parcels`、`4S1056Parcels`、`HCP`。启用 run-wise Pearson 矩阵及 LINC QC。 |
| 队列 | 每 run 独立输出，单任务报错记为 `SKIPPED_ERROR` 后队列继续。两例试跑首次因公共 TemplateFlow 资源缺失退出，补齐后均通过。首次日志与状态保留。 |

XCP-D 于北京时间 2026 年 11 月 15 日 14:01 完成。以上是这批实际运行参数，参考教程的 LINC 配置，ROI 分析的附加平滑设为 0 mm。[rs-fMRI 预处理教程](https://chenfei-ye.github.io/zh/post/202406_rs_fmri_run/) · [XCP-D 26.1.0 输出文档](https://xcp-d.readthedocs.io/en/26.1.0/outputs.html)

### 图谱维度和缺失值

每个 run 的 TSV 时间序列为 `210 × N`，对应静态 Pearson 矩阵为 `N × N`。已逐对核对时间序列列名与矩阵轴，五套图谱各自的 ROI 标签和顺序在两种格式的所有输出中一致。每份输出还通过了时间轴、矩阵对称性、取值范围及抽样 Pearson 重算核验。

| 图谱 | ROI 数 `N` | CIFTI 无有效 FC 的 ROI 总数（122 个 run） | NIfTI 无有效 FC 的 ROI 总数（4 个 run） | NIfTI 单个 run 最多 |
| --- | ---: | ---: | ---: | ---: |
| 4S156Parcels | 156 | 0 | 12 | 5 |
| 4S256Parcels | 256 | 0 | 21 | 8 |
| 4S456Parcels | 456 | 0 | 52 | 22 |
| 4S1056Parcels | 1056 | 0 | 177 | 70 |
| HCP | 19 | 0 | 0 | 0 |

无有效 FC 的 ROI 按矩阵对角线缺失计数，保留软件输出的 `n/a`。四个 NIfTI run 在 4S 图谱中均有此情况。固定图谱和处理格式后再比较批次内的实验结果，ROI 覆盖条件与缺失值处理应随实验一起记录。

## 在 V100 上验证输入

将 `XCPD_ROOT` 设为你在 V100 上有权限读取的 XCP-D 批次根目录。`results.tsv` 每行对应一个 run 和一套图谱，通过 `format`、`atlas` 和 `label` 筛选。`roi_time_series`、`fc_matrix`、`qc_report` 指向对应的原生导数和 QC 报告。`n_frames`、`n_regions` 和 `n_invalid_regions` 用于检查输入形状及缺失。`final-verification.json` 保存汇总，单 run 的状态及命令记录在批次目录内。

下面的只读检查已在 V100 各抽取一个 CIFTI / NIfTI 的 4S156 结果，两例均通过。检查使用 Python 3.8+ 标准库，验证头部、形状、矩阵对角线缺失数及一对可用 ROI 的 Pearson 相关系数。

```bash
export XCPD_ROOT='REPLACE_WITH_XCPD_BATCH_DIRECTORY'
python3 - <<'PY'
import csv
import math
import os
from pathlib import Path
from statistics import fmean


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle, delimiter="\t"))


def number(token):
    return math.nan if token.lower() in {"", "n/a", "na"} else float(token)


root = Path(os.environ["XCPD_ROOT"])
with (root / "results.tsv").open(newline="", encoding="utf-8") as handle:
    index = list(csv.DictReader(handle, delimiter="\t"))

for fmt in ("cifti", "nifti"):
    row = next(item for item in index if item["format"] == fmt
               and item["atlas"] == "4S156Parcels"
               and item["label"] == "COMPUTATIONAL_CHECKS_PASS")
    ts_header, *timepoints = read_tsv(row["roi_time_series"])
    fc_header, *matrix_rows = read_tsv(row["fc_matrix"])
    roi_names = fc_header[1:]
    if ts_header[0].lower() == "time" and ts_header[1:] == roi_names:
        ts_header = ts_header[1:]
        timepoints = [frame[1:] for frame in timepoints]
    assert ts_header == roi_names == [line[0] for line in matrix_rows]
    assert len(timepoints) == int(row["n_frames"]) == 210
    assert len(matrix_rows) == len(roi_names) == int(row["n_regions"]) == 156
    assert all(len(frame) == 156 for frame in timepoints)
    assert all(len(line) == 157 for line in matrix_rows)
    missing = sum(not math.isfinite(number(line[i + 1]))
                  for i, line in enumerate(matrix_rows))
    assert missing == int(row["n_invalid_regions"])

    usable = []
    for i, column in enumerate(zip(*timepoints)):
        values = list(map(number, column))
        if all(map(math.isfinite, values)) and max(values) > min(values):
            mean = fmean(values)
            usable.append((i, [value - mean for value in values]))
        if len(usable) == 2:
            break
    assert len(usable) == 2
    (i, a), (j, b) = usable
    scale = math.sqrt(math.fsum(v * v for v in a) * math.fsum(v * v for v in b))
    pearson = math.fsum(u * v for u, v in zip(a, b)) / scale
    observed = number(matrix_rows[i][j + 1])
    assert math.isclose(pearson, observed, abs_tol=2e-5, rel_tol=1e-5)
    print(f"{fmt} 4S156Parcels 210x156 / 156x156 PASS")
PY
```

## WEiDA 验证实验

### 先验证数据接口和动态计算

1. 用上面的只读脚本确认两种格式的输入可读。用一份有限值齐全的 CIFTI `4S156Parcels` 时间序列做首个运行测试，保留矩阵列顺序。TSV 是时间点 × ROI，作者 MATLAB 计算核心接收 ROI × 时间点。
2. 将 ROI 时间序列送入去趋势、带通、Hilbert 相位、逐帧相位一致性及加权特征向量计算。逐帧相位一致性是 `cos(φ_i(t) − φ_j(t))`，矩阵应对称且对角线为 1。按作者代码采用 `TR=2.1` 秒和 0.04–0.07 Hz 窄带时，实验记录需写明输入已经过 XCP-D 的 0.01–0.08 Hz 带通。静态 Pearson 矩阵可用于核对输入，动态计算沿时间轴使用 ROI 时间序列。
3. 一份完整 run 应给出 210 个逐帧特征，每个特征长度与所用图谱的有效 ROI 数一致。再用多份同图谱、同格式的 run 测试 `K=4` 聚类及状态时间线、占比、驻留时间和转移概率。占比按 run 检查合计约为 1，驻留时间按 TR 转为秒，转移概率按 run 边界计算。比较不同重复运行的状态时，先按质心匹配状态，记录随机种子、聚类设置和缺失值规则。[作者 MATLAB 实现](https://github.com/chenfei-ye/WEIDA-DBS/blob/main/StateSpace_modelling.m)

### 对齐论文主实验

作者主分析使用 78 个 AAL 皮层 ROI，论文报告 fMRIPrep 21.0.2、fieldmap-less SDC、FreeSurfer 6.0 的 9 自由度 BOLD-to-T1w 配准，以及 BIDS-fMRIpost 和 36 个混杂回归量。作者 `StateSpace_modelling.m` 从 AAL78 CSV 去掉首列索引后转置，示例文件尾缀为 `AAL78_MNI_basic36_sm6.csv`。该入口按 16 个 HC、25 个 STN-DBS OFF 和 25 个 STN-DBS ON run 拟合状态空间，使用 0.04–0.07 Hz 带通、Hilbert 相位和 K-means（`K=4`、平方欧氏距离、300 次初始化、最大 500 次迭代）。入口的条件分配依赖文件排序，接入当前数据前应以有来源证据的 HC / STN-OFF / STN-ON 清单核对每一条记录及配对。[论文方法](https://doi.org/10.1038/s41531-026-01354-3) · [作者数据入口和聚类实现](https://github.com/chenfei-ye/WEIDA-DBS/blob/main/StateSpace_modelling.m)

本次已交接的成果是 fMRIPrep / XCP-D 计算产物。用当前 4S 时间序列可做 WEiDA 方法的技术验证。论文主图的对齐仍需确认队列来源和人工 QC，落实作者的体积预处理及 AAL78 ROI 映射，再按作者输入契约生成时间序列。状态占比、驻留时间、转移概率和论文组间结果均待实验验证。作者将 tTIS 的预测入口单列在 `WEiDA_pred.m`。[作者代码与输入说明](https://github.com/chenfei-ye/WEIDA-DBS)
