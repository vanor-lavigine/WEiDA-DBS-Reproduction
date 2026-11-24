# 运行与核验记录

本轮于 2026 年 11 月 24 日在实验分支 `experiment/bold-microstates-v1` 基础提交 `399a1bc17b5f1b6111fd8f185983ea4b7cb64e9f` 上执行。输入为已冻结的真实 V100 SyN9 导数、24 人 AAHC 组模板及 105 评分归档。下列命令以变量代替数据所在私有路径。[计算前约定](PLAN.md) · [临床结果](README.md)

| 环节 | 输入与实际核验 |
|---|---|
| V100 回填 | 核对 24 个 OFF/ON 对、原有时序文件摘要、78 个脑区的名称及模板顺序、210 帧与 TR=2.1 秒。确认每次扫描的平均 FD 有效。使用原有 XCP-D 环境，容器禁网，输入挂载只读，数值库单线程。 |
| 状态序列 | 逐帧方法的 10,080 帧与上次冻结标签完全一致。峰间方法使用 2,185 个空间标准差峰，最终保留 9,375 帧（93.01%）。OFF 4,680 帧、741 个完整段，ON 4,695 帧、734 个完整段。两种方法的各状态帧数、段数均独立重新计数。 |
| 105 评分 | 在评分所在机器读取既有主版归档和与其总分一致的副本，核对两份总分以及 12 个分项之和。用既有访视键连接 48 条扫描记录，逐人计算相对改善率。三组临床比较均有 24 个患者独立观测。 |
| 统计核验 | 105 两次实算全部 84 行的 r、原始 p 及逐项置信区间完全相同。首次用 SciPy 的 BH 实现，复算用仓库内等价的 NumPy 实现，q 的最大绝对差为 1.12×10⁻¹⁶ 以下。逐帧方法还复现了上轮四个状态的 OFF 持续时间 Spearman ρ，最大绝对差为 5.56×10⁻¹⁷ 以下。 |
| 软件环境 | V100 使用 Python 3.12.12、NumPy 2.2.6、SciPy 1.15.2、pandas 2.2.3。105 使用 Python 3.10.20、NumPy 1.26.4、SciPy 1.14.1、pandas 2.3.3。本机出图使用 Python 3.12.7、Matplotlib 3.9.2。 |

## 命令和私有输入

以下是 V100 容器实际执行的 Python 命令。`INPUT_INDEX` 为已核验的输入索引，`SCORED_FIT` 含上一轮 24 人的拟合清单、组模板及逐帧序列。`backfit` 是本次独立的输出目录。原有代码在 `PYTHONPATH` 中。在这一步，评分没有进入状态标签的计算。[状态回填代码](../../../../experiments/bold_microstates_v1/li_clinical_backfit.py)

```bash
/app/.pixi/envs/xcp-d/bin/python -B li_clinical_backfit.py --index "$INPUT_INDEX" --training "$SCORED_FIT" --output backfit
```

105 实际以 `env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$SCORE_PYTHON" -B -` 在内存中调用 `li_clinical_stats.aggregate(payload, project)`。`payload` 包含既有的访视连接键、随机盐和本轮回填指标，分值在 105 原位读取。以下脚本入口执行相同函数。[临床统计代码](../../../../experiments/bold_microstates_v1/li_clinical_stats.py)

```bash
"$SCORE_PYTHON" -B li_clinical_stats.py --project "$CLINICAL_PROJECT" < "$PRIVATE_PAYLOAD"
```

每种方法有 4 个状态 × 3 个指标，加全部状态段的平均持续时间和总出现率，共 14 项。每个指标对应 3 个临床问题。两套回填各自对 42 个双侧 Pearson 检验作 BH 校正，阈值 q<0.01。`ci_low`、`ci_high` 是根据 Fisher z 计算的逐项 95% 区间，以本轮固定模板为条件。逐患者置换使用 10,000 次、种子 `20261124`，偏相关调整当次 OFF 的平均帧间位移（FD），自由度 21。其余敏感性分析见原约定。[方案](PLAN.md)

## 汇总、作图与测试

以下命令在仓库根目录执行。`CLINICAL_FIRST` 与 `CLINICAL_REPEAT` 分别代表两次真实临床运行的组级输出，两次计算使用同一批输入。仅重绘图片时，已提交的 `stats.csv` 即可使用。

```bash
ANALYSIS=experiments/bold_microstates_v1
PREVIOUS=reports/bold_microstates_v1/scored24_only
REPORT="$PREVIOUS/eeg_clinical_alignment"
python -B "$ANALYSIS/export_li_clinical.py" "$BACKFIT_QC" "$CLINICAL_FIRST" "$CLINICAL_REPEAT" "$PREVIOUS/stats.csv" "$REPORT"
python -B "$ANALYSIS/plot_li_clinical.py" "$REPORT/stats.csv" "$REPORT/clinical_alignment.png"
python -m pytest -q "$ANALYSIS/tests"
```

本机实验测试套件 47 项通过，其中新增的两个合成数据用例检查访视配对、个人改善率、缺失状态和多重检验校正。图和 84 行统计表中的 BOLD 结果来自真实患者。EEG 对照使用原文图 4 公布的组级系数。[公开汇总代码](../../../../experiments/bold_microstates_v1/export_li_clinical.py) · [绘图代码](../../../../experiments/bold_microstates_v1/plot_li_clinical.py) · [数值核验记录](verification.json) · [Li 等，图 4](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.697909/full#F4)
