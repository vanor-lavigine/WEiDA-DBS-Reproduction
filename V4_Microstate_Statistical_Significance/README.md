# V4 Microstate Statistical Significance

同一 V3 冻结队列的微状态显著性再分析。分析选择、模板、实际结果与重训可识别性分别见下列入口。

- [微状态统计结果](reports/V4_Microstate_Statistical_Results.md)
- [统计方法及校正家族](reports/Statistical_Methods.md)
- [噪声校准与模板稳定性](reports/Microstate_Stability_Report.md)
- [帕金森相关性及论文表述](reports/Scientific_Interpretation.md)
- [冻结范围、复算流程和公开校验](workflow/FROZEN_WORKFLOW.md)
- [结果与源码 SHA-256 清单](workflow/Freeze_Manifest.json)

利用工作树中的 V3 实施代码及已验证的 V4 Reliability 本地缓存，从本目录执行。

```powershell
python scripts/audit_history.py
python scripts/micro_inference.py
python scripts/full_bootstrap.py --workers 10
python scripts/micro_figures.py
python scripts/write_reports.py
python scripts/verify.py
```

冻结方案位于 `configs/protocol.json`。逐人特征、逐轮重抽和计算缓存均写入本地 `private/`。旧实验从原目录只读。`results/` 与 `figures/` 保存汇总产物。
