# 运行与复核记录

计算在 Windows 工作站完成。输入为已有的 105 与 V100 SyN9 归档。读取既有身份表、ON/OFF 对照、AAL78 轴、文件校验和、头动记录和同帧 MATLAB 输入包。V100 的批次完成文件另以只读 SSH 执行 `sha256sum` 复核，实测 SHA-256 为 `d96eabb442092b7db8e849267a4f1af4a24598ed1396838cdba23f4d66e94ea0`，与归档收据一致。原始影像预处理已在该批次完成。

以下为本轮使用的入口及参数。各折只读同一输入，可以按下列分片并行，也可以顺序执行。MATLAB 的每个折单独把随机种子重置为 0。编号 0 为全样本，1–24 为各删除一人的内部折。进入公开结果时，每折换成基于私有摘要排序的匿名编号。模板脚本和结果导出脚本检查同名产物以防覆盖。复跑已提交版本时，使用空的中间目录，并将 `$reports` 设为新的输出位置，例如 `local_results/matched24_recheck/report`。

```powershell
Set-Location 'E:\NIT\微状态\WEiDA-DBS-Reproduction'
$dataRoot = 'E:\NIT\神经调控数据'
$matched = Join-Path $dataRoot 'V100脑深部刺激WEiDA复现_20261118\代码与辅助\同人群精确度对照'
$cache = 'local_results/matched24_primary'
$reports = 'reports/bold_microstates_v1/matched24_primary'
$exp = 'experiments/bold_microstates_v1'
$matlab = 'G:\matlab\bin\matlab.exe'
$env:PYTHONIOENCODING = 'utf-8'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
New-Item -ItemType Directory -Path $cache
$cacheAbs = (Resolve-Path -LiteralPath $cache).Path

foreach ($offset in 0..3) {
    & $matlab -singleCompThread -sd (Get-Location).Path -batch "addpath(fullfile(pwd,'experiments','bold_microstates_v1')); matched24_phase('$matched','$cacheAbs',${offset}:4:24)"
}
foreach ($offset in 0..2) {
    $foldSet = @(0..24 | Where-Object { ($_ % 3) -eq $offset })
    python -B "$exp/matched24_data.py" --data-root $dataRoot --cache $cache --folds @foldSet
}

# 0 同时完成全样本检验及固定模板的全部 24 个删除折。正数折使用重拟合模板。
python -B "$exp/matched24_analysis.py" --data-root $dataRoot --cache $cache --folds 0
$groups = @(@(1,2,3),@(4,5,6,7),@(8,9,11),@(10,15),@(12,13,14),@(16,17,19),@(18,23),@(20,21,22,24))
foreach ($foldSet in $groups) {
    python -B "$exp/matched24_analysis.py" --data-root $dataRoot --cache $cache --folds @foldSet
}

# 追溯既有定义。作者 105 全样本归档缺少逐帧特征，单独重建后核对全部 13,647 个旧标签。
& $matlab -singleCompThread -sd (Get-Location).Path -batch "addpath(fullfile(pwd,'experiments','bold_microstates_v1')); matched24_legacy_features('$dataRoot','$cacheAbs')"
python -B "$exp/matched24_bridge.py" --data-root $dataRoot --cache $cache

python -B -m pytest -q -p no:cacheprovider "$exp/tests"
python -B "$exp/verify_matched24.py" --data-root $dataRoot --cache $cache
python -B "$exp/matched24_report.py" --cache $cache --destination $reports
```

主频段两套已对齐的 `inputs.mat` 与 `author_matched_k4.mat`、作者特征脚本、参考图的摘要及 Python 包版本写入 [环境与输入核验记录](results/audit.json)。每折的 MATLAB 版本、拟合帧数和评分窗口见 [质控表](results/fold_quality.csv)。原始试验有 38 项构造数据测试，本轮用共同帧、共同完整对以及同步重抽差值新增 3 项。真实结果另外用独立的插值、段计数、组内/组间方差分解和秩回归实现复核。[真实数据复算](results/verification.json)
