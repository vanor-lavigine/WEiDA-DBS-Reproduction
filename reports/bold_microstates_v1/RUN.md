# V100 运行记录

实验分支 `experiment/bold-microstates-v1`。拟合和统计程序来自起始提交 `2c52ba865bba99ff182f0e68bd9cc1b4ac7b9537`，在 V100 用 `git get-tar-commit-id` 核对了传输的 Git 归档。新增的代码将既有 SyN9 清单适配为实验输入，建立只读容器环境，并复核和导出组层结果。状态拟合与统计程序保持起始提交的实现。

## 软件与运行位置

| 项目 | 实跑值 |
|---|---|
| 主机 | V100，Ubuntu 22.04.5 LTS，Linux 6.8.0-138-generic，x86_64 |
| Docker Engine | 28.1.1 |
| 本地镜像 | `pennlinc/xcp_d:26.1.0` |
| 镜像 ID | `sha256:92a4a0ac1d2be762484ce11c0de2183718c517d8d91b59ef942ea6095e309183` |
| Python | 3.12.12，conda-forge，`/app/.pixi/envs/xcp-d/bin/python` |
| NumPy / SciPy | 2.2.6 / 1.15.2 |
| pandas / Matplotlib | 2.2.3 / 3.10.8 |
| 线程 | CPU，OpenBLAS、MKL、OpenMP、BLIS 各限制为 1 |
| 主流程时段 | 北京时间 16:08:10–16:10:23，退出码 0 |
| 有符号 K=5–7 | 北京时间 16:11:42–16:15:37，退出码 0 |
| unsigned K=4 | 北京时间 16:11:42–16:14:08，退出码 0 |
| 测试 | 26 项通过，12.207 秒，手算或合成输入 |

V100 的 `cohort.tsv` 和 `syn9-stn-hc-oib_7oau/plan.json` 通过运行审计编号、扫描条件和 BIDS 标签连接。纳入来自同一批次的 `finished.json`，与 `scan-disposition.tsv`、`pair-disposition.tsv` 和原 WEiDA MATLAB 打包记录一致。打包的 `复现输入清单.csv`、`复现输入核验.json`、`AAL78脑区顺序.json` 在本轮私有输入目录中分别保存为 `packing-manifest.csv`、`packing-verification.json`、`aal78-order.json`。输入适配逐扫描核对 `xcpd-command.json` 的科学参数和 `verification.json` 的提取证据，读取各扫描目录的 `roi-timeseries.tsv`，另外从已留存的 fMRIPrep confounds TSV 汇总 FD 和 DVARS。 [数据批次报告](../2026-11-21-v100-syn9-stn-hc.md) · [输入验证](input_summary.json)

## 主流程命令

将起始提交的 `experiments/bold_microstates_v1` 归档解包至独立运行目录的 `repo/` 后，运行以下命令。变量展开到实际批次和本轮目录。`v100_python.sh` 使用本地镜像 ID 调用 Python，挂载 SyN9 来源为只读，输出到本轮目录。

```bash
RUN=/data1/student/ruidong/bold_microstates_v1_20261121_Jp2fwhug
SYN9_ROOT=/data1/student/ruidong/weida_stn_own_20261116_j_srnj4i
BATCH="$SYN9_ROOT/syn9-stn-hc-oib_7oau"
IMAGE=sha256:92a4a0ac1d2be762484ce11c0de2183718c517d8d91b59ef942ea6095e309183
cd "$RUN/repo"
E="$RUN/repo/experiments/bold_microstates_v1"
server_python() {
  bash "$E/v100_python.sh" "$RUN" "$SYN9_ROOT" "$IMAGE" "$@"
}

server_python -m unittest discover -s experiments/bold_microstates_v1/tests -v
server_python "$E/adapt_syn9.py" \
  --batch "$BATCH" --cohort "$SYN9_ROOT/cohort.tsv" \
  --pack-manifest "$RUN/packing-manifest.csv" \
  --pack-verification "$RUN/packing-verification.json" \
  --roi-order "$RUN/aal78-order.json" --out "$RUN/input"

server_python "$E/run.py" fit --runs "$RUN/input/runs.tsv" --out "$RUN/pilot_k4" \
  --mode pilot --k 4 --data-kind real --polarity signed \
  --n-init 100 --max-iter 300 --seed 20261121 --stability-repeats 2
server_python "$E/run.py" fit --runs "$RUN/input/runs.tsv" \
  --metadata "$RUN/input/metadata.verified.tsv" --out "$RUN/analysis_k4" \
  --mode analysis --k 4 --data-kind real --polarity signed \
  --n-init 100 --max-iter 300 --seed 20261121 --stability-repeats 2
server_python "$E/run.py" stats --fit-dir "$RUN/analysis_k4" \
  --metadata "$RUN/input/metadata.verified.tsv" --out "$RUN/inference_k4" \
  --permutations 10000 --bootstrap 2000 --seed 20261121
```

临床版本和分数在验证清单中为空，故此次统计命令没有指定量表。若后续获得配对评分，保持模板和身份映射，使用补齐的私有验证清单及实际量表名称在新目录运行同一统计入口。

## 描述性敏感性和汇总生成

主 K=4 的 268 条扫描×状态指标通过覆盖率恒等式校验后，开始如下两组拟合作业。有符号 K=5–7 顺序执行，unsigned K=4 单独执行。各作业都使用相同的扫描、ROI 空间、seed、初始化数与迭代上限，患者等权。K=5–7 和 unsigned 仅运行拟合。

```bash
for k in 5 6 7; do
  server_python "$E/run.py" fit --runs "$RUN/input/runs.tsv" \
    --metadata "$RUN/input/metadata.verified.tsv" --out "$RUN/analysis_k${k}" \
    --mode analysis --k "$k" --data-kind real --polarity signed \
    --n-init 100 --max-iter 300 --seed 20261121 --stability-repeats 2
done
server_python "$E/run.py" fit --runs "$RUN/input/runs.tsv" \
  --metadata "$RUN/input/metadata.verified.tsv" --out "$RUN/unsigned_k4" \
  --mode analysis --k 4 --data-kind real --polarity unsigned \
  --n-init 100 --max-iter 300 --seed 20261121 --stability-repeats 2
server_python "$E/summarize_v100.py" --run-dir "$RUN" --out "$RUN/share"
```

汇总脚本在 V100 从原 BOLD 矩阵和共同模板重算 14,070 个标签，再用逐帧标签重算每个状态的三个指标和两种转移计数。配对效应从已验证的患者映射按端点重算，Holm 用各家族的全体预设位置复核。仓库保存组层汇总、图表及模板权重。单次扫描的时序、标签、标准化参数、患者配对表和运行日志保留在私有运行目录。 [数值核验](numerical_checks.json) · [汇总统计](stats.csv)
