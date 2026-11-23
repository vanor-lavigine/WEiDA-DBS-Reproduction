# 真实数据、运行环境与复核

本轮从实验分支 `experiment/bold-microstates-v1` 的 `c885288b75bd8760c733a761919c21642934df95` 开始。首次指定提交 `2c52ba865bba99ff182f0e68bd9cc1b4ac7b9537`、本轮分支及 V100 当初用于统计的 `stats.py` 内容相同。冻结模板与仓库中的 AAL78 状态权重逐字节核对（换行统一后）。真实数据本轮独立执行两次，第二次增加效应量复算和作图用分组均值。[冻结拟合的原始命令](../RUN.md) · [本轮计算代码](../../../experiments/bold_microstates_v1/clinical_archive_rerun.py)

## 输入与评分匹配

| 阶段 | 实际输入与检查 |
|---|---|
| V100 | 从真实有符号 K=4 拟合的 `analysis_k4/metrics.tsv`、`identity_used.tsv` 和 `input/metadata.verified.tsv` 读取 268 条扫描×状态指标。检查训练参数、14,070 个有效帧、67 条扫描身份、原始打包清单和平均 FD。每条扫描的去噪 ROI 序列校验值与打包清单对应，原始 BOLD 和 T1w 校验记录与 105 来源对应。 |
| 105 | 在评分存放机器读取 `scl_P_stn.csv`、`scl_LT_stn.csv`。每行的状态占比和停留时间与组 2、组 3 的经验矩阵联合匹配，严格要求唯一对应、条件相符、50 条全部命中，绝对误差容限为 `1e-8`。核对有编号的成像输入、冻结的患者映射及三份存档的分项。 |
| 安全连接 | 两机均验证既有 SSH 主机密钥。传递扫描级成像指标和一次性 HMAC 连接键。逐人评分在 105 原机参与计算。原统计函数生成的逐人配对表放在 RAM 临时目录，退出后清理。仓库接收经过字段筛选的效应量、区间、样本数和分组均值。 |
| 统计 | 家族 A/B/C 分别固定为 12/12/4 项。主评分与备用评分各用 24 人，版本完全一致子集用 17 人。全程保留最初按 67 次扫描训练的模板，B/C 每个端点均有对应方案的全部患者。每项区间单独计算。 |
| 环境 | 临床统计在 105 使用 Python 3.10.20、NumPy 1.26.4、pandas 2.3.3、SciPy 1.14.1，数值库各用单线程。切换率、同帧共现复核与作图使用 Windows Python 3.12.7、NumPy 1.26.4、pandas 2.2.2、SciPy 1.13.1、Matplotlib 3.9.2。 |

## 实跑命令

下列路径变量指向本地私有适配目录、实验脚本目录、数据目录和新输出目录。启动器从 V100 读取冻结指标，在内存中核对来源后调用 105 的统计入口。两个环境均使用原有凭据。

```powershell
python -B -X utf8 -u "$private/run_federated.py"
python -B -X utf8 -u "$private/run_federated.py" --output aggregate_verified.json
python -B -X utf8 "$private/export_report.py"
python -B -X utf8 "$exp/matched24_switch_rate.py" --data-root . --output "$private/switch_recheck"
python -B -X utf8 "$exp/matched24_cells.py" --data-root . --out "$private/cooccurrence_recheck"
python -B -X utf8 "$exp/plot_clinical_archive.py" "$report"
python -B -X utf8 -m pytest -q -p no:cacheprovider "$exp/tests"
```

临床统计在原机调用 `clinical_archive_rerun.aggregate(payload, project, private_tmp)`，其内部调用原实验 `stats.analyze`，参数为 `k=4, clinical_scale="UPDRS3", n_permutations=10000, n_bootstrap=2000, seed=20261121`。每台机器各自从受限目录读取原输入。仓库脚本也支持从标准输入接收同格式的私有数据包。

```bash
python -B clinical_archive_rerun.py --project "$ARCHIVE_PROJECT" --private-tmp /dev/shm < "$PRIVATE_IMAGING_PACKET"
```

私有包的 `verified` 记录冻结模型的真实数据标记、模式、极性、K、成对人数和帧数。`metrics_tsv` 为原扫描级指标 TSV 文本。`visits` 每条含 `run_id, person, cohort, condition, fd, scan_key`，其中 `person` 为保持原排序的匿名编号，`scan_key` 按原扫描标签小写和此次随机盐进行 HMAC-SHA256 计算。`salt` 为该盐的十六进制表示。程序要求同一人的两个条件均保有对应的评分。

## 验证

- 对 60 条汇总逐项独立重算了患者数和效应量，最大绝对差约为 `1.11e-16`。七组家族 Holm 值通过独立排序计算。两次真实统计给出完全相同的效应量、区间及 p 值。[数值核验证据](verification.json)
- 26 对的 12 个刺激效应、区间和原始及校正 p 与冻结报告逐项相等。24 人的影像切换率关联汇总、32 个状态共现检验及其配套 S2 表也与历史 CSV 逐字节相同。[原配对结果](../stats.csv) · [原共现结果](../matched24_primary/cooccurrence_cells/README.md)
- 45 项代码测试通过，包括本轮新增的两个合成排列样例，分别验证乱序评分行可恢复、指纹重复时拒绝配对。报告中的统计来自真实队列。[合成测试代码](../../../experiments/bold_microstates_v1/tests/test_clinical_archive.py)
