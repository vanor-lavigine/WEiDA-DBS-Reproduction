# V4 WEiDA 统计框架与 BOLD 微状态

在 WEiDA-DBS-Reproduction 的本地 `WEiDA-DBS-V4` 工作树中运行。本项目只读取 V3 主方法的冻结数据及其归档输出，并读取同一批扫描的本地 WEiDA 归档。分析对象为 15 名 HC 和 25 名配对患者，共 65 次 AAL78 扫描。

- [统计规程及指标映射](reports/WEIDA_STATISTICAL_PROTOCOL.md)
- [微状态分析结果](reports/V4_STATISTICAL_RESULTS.md)
- [同扫描 WEiDA 对照](reports/CROSS_METHOD_COMPARISON.md)
- [PD 与 HC 的补充比较](reports/PD_DISEASE_ASSOCIATION.md)
- [输入及代码运行版本校验](results/Input_Verification.json)

## 重跑与检查

在本目录运行以下命令。沿用当前 Python 环境的 numpy、pandas、SciPy、statsmodels、matplotlib 与 Pillow。脚本使用既有的 V3 加载和 CR1 代码，需要当前目录结构下的 V3 私有归档和原有 WEiDA 归档。

```powershell
python scripts/run_v4.py
python scripts/verify_v4.py
python scripts/pd_disease.py
```

`results/` 是逐对比和逐状态的统计表。`coverage`、`gev` 和转换值均按 0–1 比例写入 CSV，Occurrence 单位为段/秒。`private/` 保存核对参与者映射、逐扫描特征与归档指纹所需的本地文件。该目录和含患者级配对连线的 PNG 由本项目的 `.gitignore` 管理。

PD 与 HC 的补充分析使用现有 K=4 特征。`pooled_*` 每位患者的 ON 和 OFF 等权平均后与 HC 比较，要求该指标在两次访视均有定义。`both_*` 使用 OFF−HC 和 ON−HC 的同向交集，取同向时两个双侧 P 的较大者。加权模型 `c4_cr1_*` 同时利用两次扫描，并按参与者控制依赖与 FD。补充分析在看到 V4 主结果后开展，每种方法的全部主指标分别进入患者平均和同向交集的校正家族。
