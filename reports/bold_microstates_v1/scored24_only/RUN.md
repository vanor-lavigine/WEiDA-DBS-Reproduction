# 真实数据计算与核验

2026 年 11 月 24 日。本轮基于实验分支 `experiment/bold-microstates-v1` 的 `a1dbdde478ce37291be6f307858a0c1fed598c82`。原实验协议位于 `2c52ba865bba99ff182f0e68bd9cc1b4ac7b9537`，本次独立建模的入组条件、两种方法和检验家族另有记录。[本轮方案](PLAN.md) · [原实验协议](../../../experiments/bold_microstates_v1/PROTOCOL.md)

| 阶段 | 实际运行与检查 |
|---|---|
| 纳入 | 先在 105 读取归档输入的访视标识，把 50 条访视同 V100 的成对患者清单求交集。这一步只读取名单。得到的 24 对均有可核对的原始 BOLD、T1w 和 ROI 输出溯源记录。 |
| 建模 | V100 以只读方式挂载经核准的输入索引和 SyN9 导数，另建私有结果目录。逐一核对预期文件摘要、78 个脑区名称、顺序、TR、矩阵大小、无重复数值内容、全部帧可用，以及两份身份清单中的 OFF/ON。拟合清单只允许扫描键、患者键、条件和 ROI 文件校验字段。 |
| 环境 | V100 在原有 XCP-D 镜像中运行 Python 3.12.12、NumPy 2.2.6、SciPy 1.15.2、pandas 2.2.3，容器禁网、数值库单线程。105 上的统计环境为 Python 3.10.20、NumPy 1.26.4、SciPy 1.14.1、pandas 2.3.3，数值库单线程。本机出图使用 Python 3.12.7、Matplotlib 3.9.2。 |
| 评分行核验 | 读取归档 `scl_P_stn.csv`、`scl_LT_stn.csv`。用原有影像指纹将每个评分行唯一对应到一次具体访视，再用已核验的访视连接键连接到本轮指标。主评分 `WEiDA4-1_atlas1_group0` 与 `WEiDA3-42_atlas1_group2` 的全部 50 个总分一致，每访视均由 12 个分项复算。备用总分来自 `WEiDA4-3_atlas1_group2`。 |
| 独立复核 | 从 96 条逐帧序列按游程重新计数。384 条扫描与状态组合的帧数、段数和三项指标全部复现，最大绝对差为 0。用独立宽表对照均值和 SciPy Spearman 结果，120 条报告效应最大绝对差为 1.12×10⁻¹⁶ 以下。核对了 8 张模板和 21 组 Holm 校正。 |

## 实跑命令与参数

V100 容器在本次私有工作目录中执行下列命令。`INPUT_ROOT` 代指现场核验过的输入索引所在目录。逐人清单和原始机器路径保存在受限环境。

```bash
/app/.pixi/envs/xcp-d/bin/python -B scored_cohort_fit.py --roster fit_roster.tsv --input-root "$INPUT_ROOT" --output fit
```

拟合生成 `fit/templates.tsv`、`fit/metrics.tsv`、`fit/sequences.tsv`、`fit/fit_summary.json`。组模板和拟合汇总可在本报告查看。训练只读取被选中 48 次扫描的 ROI 时序及其溯源索引。AAHC 的四张候选图先在扫描内估计，再以全部 48 套候选图作为不同初始化进行组级对齐。24 人的两个条件共同学习一套组模板，同一方法的全部 48 次扫描均使用这套模板。[拟合代码](../../../experiments/bold_microstates_v1/scored_cohort_fit.py) · [峰值与 AAHC 实现](../../../experiments/bold_microstates_v1/peak_states.py)

105 实际通过内存管道使用 `"$SCORE_PYTHON" -B -` 启动统计，在原机调用 `scored_cohort_stats.aggregate(payload, project, private_tmp)`，临时配对表写入内存文件系统。`SCORE_PYTHON` 指向表中记录版本的解释器。仓库脚本提供同一函数的命令行入口。输入包仅含本次访视连接键、随机盐和影像指标，临床分值由函数在评分机上读取。

```bash
python -B scored_cohort_stats.py --project "$CLINICAL_PROJECT" --private-tmp /dev/shm < "$PRIVATE_PAYLOAD"
```

三种评分分析均调用相同的 `stats.analyze`。固定 `k=4`、`clinical_scale="UPDRS3"`、`n_permutations=10000`、`n_bootstrap=2000`、`seed=20261121`。条件效应用配对差均值的双侧符号翻转检验。临床相关的双侧置换在患者之间打乱评分秩次，bootstrap 每次整人重采样。对同一问题的不同状态和指标作 Holm 校正，并另算两种方法合并的同家族 Holm 值。备用评分与总分一致子集只重复两个临床家族。[评分与统计代码](../../../experiments/bold_microstates_v1/scored_cohort_stats.py)

以下两个公开命令可在仓库根目录运行，输入只需已发布的组级结果。

```bash
python experiments/bold_microstates_v1/verify_scored_cohort.py reports/bold_microstates_v1/scored24_only
python experiments/bold_microstates_v1/plot_scored_cohort.py reports/bold_microstates_v1/scored24_only/stats.csv reports/bold_microstates_v1/scored24_only/clinical_effects.png
```

本轮另外运行了仓库的 `test_core.py`、`test_stats.py`、`test_peak_states.py`、`test_clinical_archive.py` 和 `test_pipeline.py`，共 33 项通过。这些是软件测试。报告中涉及患者、扫描、影像指标与评分的统计结果全部来自真实队列。[数值核验记录](verification.json)
