# V8 STN 原始评分复核与运动关联

本模块复用 V7 冻结影像。原始评分、患者映射、日期、输入指纹及逐人图保存在私有工作区。

[评分核查](reports/STN_Original_Score_Audit.md) · [关联报告](reports/STN_Microstate_Motor_Associations.md) · [锁定方案](ANALYSIS_PLAN.md) · [运行核验](results/Run_Verification.json)

在 V8 工作树根目录运行以下命令。默认用工作区根中的三个原始工作簿和已核查的上轮私有输入索引。程序也可通过参数读取相同来源在授权工作区中的副本。

```powershell
$v8 = 'BOLD_Microstate_STN_Original_Scores'
python -B -X utf8 "$v8/scripts/run_v8.py" --audit-only
python -B -X utf8 "$v8/scripts/run_v8.py"
python -B -X utf8 -m pytest -q -p no:cacheprovider "$v8/tests"
```

统计运行先复算旧四项 Spearman 及头动秩模型，再执行已锁定的三项总分、九项分域和四套敏感性模型。每份输出的推断单位为患者。

## 按身份与 DBS 条件配对的 25 人复算

在相同工作树中运行 `python -B -X utf8 BOLD_Microstate_STN_Original_Scores/scripts/date_unrestricted.py`。这项分析纳入两对日期冲突的评分，逐条保留其访视标记。结果和逐项样本对照存于 `results/date_unrestricted/`。[25 人复算报告](reports/STN_Date_Unrestricted_Associations.md)
