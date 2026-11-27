"""Render Chinese reports from the public Round 5 summaries."""

import pandas as pd


METRICS = {
    "duration_seconds": "持续时间（秒）", "coverage": "占用率（%）",
    "occurrence_per_minute": "出现率（段/有效分钟）", "P44": "M4→M4",
    "exit_rate": "M4 总离开概率", "pooled_entry": "非 M4 来源汇总进入概率",
    "P41": "M4→M1", "P42": "M4→M2", "P43": "M4→M3",
    "P41_exit": "退出后去 M1", "P42_exit": "退出后去 M2", "P43_exit": "退出后去 M3",
    "P14": "M1→M4", "P24": "M2→M4", "P34": "M3→M4",
    "enter_count": "块内进入次数", "exit_count": "块内退出次数", "episode_count": "段数",
    "coverage_duration_component": "驻留时间分量（百分点）",
    "coverage_occurrence_component": "出现率分量（百分点）",
}
PATH_SYMBOLS = {"omnibus_12": "12 维向量", "P41": "M4→M1", "P34": "M3→M4", "P44": "M4→M4"}


def number(value, digits=5):
    return "NA" if pd.isna(value) else f"{value:.{digits}g}"


def interval(estimate, low, high, scale=1):
    if any(pd.isna(v) for v in (estimate, low, high)):
        return "NA"
    return f"{estimate*scale:+.3f} [{low*scale:+.3f}, {high*scale:+.3f}]"


def ci(row, field="mean_diff", scale=1):
    return interval(row[field], row.ci95_low, row.ci95_high, scale)


def list_tests(table, q="fdr_q", h="holm_p"):
    n = len(table)
    return f"BH {int(table[q].lt(.05).sum())}/{n}，Holm {int(table[h].lt(.05).sum())}/{n}"


def md_table(head, rows):
    return ("| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" +
            "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows))


def omnibus_report(tables, archive, audit):
    p0 = tables["Omnibus_Transition_Results"].iloc[0]
    sensitivity = tables["Omnibus_Transition_Sensitivity"]
    validation = audit["v6_archive_transition_parity"]
    ilr = tables["Transition_ILR_Omnibus"]
    whole = ilr.loc[ilr.scope.eq("all_rows")].iloc[0]
    motion = tables["Transition_Motion_Sensitivity"]
    clinical = tables["M4_Clinical_Associations"]
    clinical_pass = int(clinical.fdr_q.lt(.05).sum())
    paths = [
        ("输入向量", "12 条非对角 P(去向｜当前态) 的 ON−OFF", "相同，25 个完整配对"),
        ("自循环", "包含在每行分母，向量不另列对角元素", "相同"),
        ("协方差", "样本协方差，0.10 各向同性迹收缩，每次重估", "相同"),
        ("零分布", "整人同号翻转 12 维，10,000 次，平方 Wald 右尾", "相同"),
        ("种子及区间", "20261127，9,999 次患者整对 Bootstrap", "7261126，2,000 次"),
        ("多重比较", "一个预设主检验，BH 与 Holm 等于原始 P", f"探索性整体家族，{int(archive.family_size)} 项"),
        ("整体效应", "欧氏范数及协方差标准化距离 D", "欧氏范数"),
    ]
    stability = md_table(("迹收缩", "Wald", "原始 P", "条件数"), [
        (f"{r.ridge:.2f}", f"{r.wald_statistic:.3f}", number(r.raw_p), number(r.test_condition_number))
        for r in sensitivity.itertuples()
    ])
    return f"""# Round 5 冻结 A3 整体转移动力学

25 对患者的 12 维 WEiDA 转移向量在 ON/OFF 条件下存在整体变化，**Wald={p0.wald_statistic:.3f}，P={number(p0.raw_p)}**。整体标准化距离 D={p0.standardized_effect:.3f}（95% CI [{p0.standardized_ci95_low:.3f}, {p0.standardized_ci95_high:.3f}]）。平均差向量的欧氏长度为 {p0.vector_norm:.4f}（95% CI [{p0.vector_norm_ci95_low:.4f}, {p0.vector_norm_ci95_high:.4f}]）。D 是以主收缩协方差衡量的样本距离。两个自助区间均描述效应大小，零假设判定使用置换 P。

Round 4 全部 36 个 WEiDA 路径比较的 BH 和 Holm 通过数仍为 0。整体检验考察联合向量均值，单路径是否通过原家族需分别读表。

## 与 V6 对账

{md_table(('定义', 'Round 5', 'V6'), paths)}

65 个冻结 A3 扫描重数结果与 Round 4 逐扫描表相符。V6 已发布 ON/OFF 的 {validation['edges_verified']}/16 个矩阵元素的计数、分母完全一致，概率核验最大误差 {number(validation['maximum_absolute_probability_difference'])}。新种子重算与 V6 整体结论一致，V6 原始 P={number(archive.raw_p)}、Holm P={number(archive.holm_p)}，Round 5 主检验的 BH q={number(p0.fdr_q)}、Holm P={number(p0.holm_p)}。V6 分别调整 FD 和 DVARS 的旧 Holm P 为 {number(archive.fd_holm_p)} 和 {number(archive.dvars_holm_p)}，各沿用 V6 原家族。这是同一标签和同一队列上的再分析。

原始协方差秩 {p0.raw_covariance_rank}/12，条件数 {p0.raw_condition_number:.2f}。主收缩后条件数 {p0.test_condition_number:.2f}。四个固定收缩值的结果如下，敏感性原始 P 用于对照稳定性。

{stability}

## 组成、头动及历史临床证据

ILR 全向量原始 P={number(whole.raw_p)}、BH q={number(whole.fdr_q)}、Holm P={number(whole.holm_p)}，完整五项家族 {list_tests(ilr)}。头动敏感性完整八项家族 {list_tests(motion.loc[motion.correction_slot.eq(True)], 'bh_q')}。历史评分的 80 项 Spearman 重核后 BH 通过 {clinical_pass}/80。

[配对整体结果](../results/Omnibus_Transition_Results.csv) · [12 维均值向量](../results/Omnibus_Transition_Vector.csv) · [协方差矩阵](../results/Omnibus_Transition_Covariance.csv) · [固定收缩比较](../results/Omnibus_Transition_Sensitivity.csv)

[M4 动力学](Round5_M4_Mechanistic_Report.md) · [组成及稳健性](Round5_Robustness_Report.md) · [历史临床关联](M4_Clinical_Interpretation.md) · [最终解释](Round5_Final_Interpretation.md) · [方案](../ANALYSIS_PLAN.md) · [复现与输入核验](../results/Round5_Run_Manifest.json)

[V6 转移报告](../../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Transition_Dynamics.md) · [Round 4 路径报告](../../round4/transition/reports/A3_Transition_Dynamics_Report.md)
"""


def m4_report(tables):
    effects = tables["M4_Dynamics_Paired_Effects"].set_index("metric")
    groups = tables["M4_Dynamics_Decomposition"].pivot(index="metric", columns="condition", values="mean")
    names = ["duration_seconds", "coverage", "occurrence_per_minute", "P44", "exit_rate",
             "P41", "P42", "P43", "P41_exit", "P42_exit", "P43_exit", "P14", "P24", "P34", "pooled_entry",
             "enter_count", "exit_count", "episode_count", "coverage_duration_component",
             "coverage_occurrence_component"]
    in_prob = {"coverage", "exit_rate", "pooled_entry", "coverage_duration_component", "coverage_occurrence_component"}
    rows = []
    for name in names:
        effect = effects.loc[name]
        multiplier = 100 if (name in in_prob or name.startswith("P")) else 1
        mean = [number(groups.loc[name, cond]*multiplier) if name in groups.index else "NA" for cond in ("HC", "OFF", "ON")]
        rows.append((METRICS[name], *mean, ci(effect, scale=multiplier), number(effect.hedges_g),
                     number(effect.round4_fdr_q), number(effect.round4_holm_p)))
    maintenance = effects.loc["P44"]
    destinations = groups.loc[["P41_exit", "P42_exit", "P43_exit"]]
    main_exit = "，".join(f"{condition} 为 M{destinations[condition].idxmax()[2]}"
                         f"（{100*destinations[condition].max():.2f}%）" for condition in ("HC", "OFF", "ON"))
    duration = effects.loc["coverage_duration_component"].mean_diff
    occurrences = effects.loc["coverage_occurrence_component"].mean_diff
    intervals = md_table(("指标", "HC", "OFF", "ON", "ON−OFF [95% CI]", "配对 g_z", "原 BH q", "原 Holm P"), rows)
    return f"""# Round 5 M4 维持、进入与退出

同一有效窗内，M4 自循环平均由 {100*groups.loc['P44', 'OFF']:.2f}% 降至 {100*groups.loc['P44', 'ON']:.2f}%。ON−OFF 为 {ci(maintenance, scale=100)} 个百分点，25 对中 {int(maintenance.duration_self_loop_both_lower)} 对的自循环与驻留时间同时降低。自循环和总离开概率是同一个二分构成，差值及配对区间互为反数。

M4 的内部段计数满足占用率 = 持续时间 × 每秒出现率。逐人对称分解的平均结果显示，驻留缩短对应 {100*duration:+.2f} 个占用百分点，出现率变化对应 {100*occurrences:+.2f} 个百分点，两项之和与平均占用率差一致。M4 仍有相近的块内进入和退出次数。驻留缩短是占用减少的主要算术分量。离开 M4 后最高的组均去向，{main_exit}。进入和退出方向的构成变化保留为机制线索。

{intervals}

概率组均值为百分比，概率配对差为百分点。步长为一个 TR。P41 使用全部 M4 来源邻对，P41_exit 只以离开 M4 的邻对为分母。P14/P24/P34 各使用本来源态的全部有效邻对。家族校正直接来自 Round 4，两个分母口径各有原 36 项槽位。自循环、出现率、计数及分解使用描述性区间。

![M4 四项动力学及占用分解](../figures/M4_Dynamics_Decomposition.png)

![入口和退出条件去向](../figures/M4_Entry_Exit_Comparison.png)

[M4 组级分解](../results/M4_Dynamics_Decomposition.csv) · [M4 配对效应](../results/M4_Dynamics_Paired_Effects.csv) · [三对比的进入与退出](../results/M4_Entry_Exit_Analysis.csv)

逐人配对表和配对轨迹图留在 Round 5 的 Git 忽略目录。
"""


def robustness_reports(tables, audit):
    tests = tables["Transition_ILR_Omnibus"]
    checks = tables["Transition_ILR_Sensitivity"]
    motion = tables["Transition_Motion_Sensitivity"]
    order = tables["Transition_Order_Sensitivity"]
    models = motion.loc[motion.correction_slot.eq(True)]
    report = md_table(("分析", "协变量", "终点", "n", "效应 [95% CI]", "P", "BH q", "Holm P"), [
        ({"adjusted_intercept": "调整截距", "covariate_association": "头动关联"}[r.analysis],
         r.covariate, PATH_SYMBOLS[r.endpoint], r.n, interval(r.effect, r.ci95_low, r.ci95_high),
         number(r.raw_p), number(r.bh_q), number(r.holm_p)) for r in models.itertuples()
    ])
    support = order.loc[order.endpoint.eq("omnibus_12") & order.status.eq("ok")]
    group_report = md_table(("配对范围", "有效对数", "先 ON", "顺序未知", "P41", "P44", "P34"), [
        (label, int(group.n.iloc[0]), int(group.on_first.iloc[0]), int(group.order_unverified.iloc[0]),
         *[ci(group.set_index("endpoint").loc[edge], "effect", 100) for edge in ("P41", "P44", "P34")])
        for label, group in order.loc[order.analysis.isin(support.analysis)].groupby("analysis", sort=False)
    ])
    quality = audit["quality"]
    exposure = quality.get("counts", {})
    quality_observed = "固定窗保留的有效邻对为 " + "、".join(
        f"{condition} {exposure[condition]['adjacency_pairs']:,} 个" for condition in ("OFF", "ON")
    ) + "。" if exposure else ""
    quality_rows = motion.loc[motion.analysis.eq(quality.get("mode", "fd_censor_frozen_labels"))]
    quality_table = md_table(("固定质量窗终点", "有效对数", "效应 [95% CI]", "状态"), [
        (PATH_SYMBOLS[r.endpoint], int(r.n),
         interval(r.effect, r.ci95_low, r.ci95_high, 1 if r.endpoint == "omnibus_12" else 100), r.status)
        for r in quality_rows.itertuples()
    ])
    motions = {r.covariate: r for r in models.loc[models.analysis.eq("adjusted_intercept") &
                                                 models.endpoint.eq("omnibus_12")].itertuples()}
    time = order.loc[order.analysis.eq("scan_interval")].iloc[0]
    motion_report = f"""# 转移概率的头动、顺序及日期稳健性

两项头动输入沿用冻结的平均 FD 和标准化 DVARS（均为 ON−OFF）。两个调整向量与未调整向量的夹角余弦分别为 FD {motions['FD'].cosine_to_unadjusted:.4f}、DVARS {motions['std_DVARS'].cosine_to_unadjusted:.4f}。Δ头动为零均落在观测范围，固定模型条件数为 {motions['FD'].design_condition_number:.2f} 和 {motions['std_DVARS'].design_condition_number:.2f}。25 对均有头动参数。

{report}

整体效应用概率均值向量长度。头动关联斜率长度以每 1 个样本头动标准差计，单路径截距效应为 0–1 概率差。调整为条件差值对单项 Δ头动的回归，在 Δ头动=0 估计截距。截距检验采用零截距缩减模型，斜率检验采用仅截距缩减模型，均以患者级向量残差整体翻转。条件推断依赖缩减模型和残差符号对称性。向量检验沿用 HC3 与 0.10 迹收缩，区间整对重抽。八项合并校正，{list_tests(models, 'bh_q')}。P34 调整值及 12 个方向系数见完整表。

固定质量窗采用既有 FD>0.50 mm 标记及前一、后二帧规则。只在原冻结回填标签上应用固定掩膜，原窗口内的断点隔开相邻计数。{quality_observed}P41 与 P44 的调整检验属于上述头动家族，原始路径仍沿用 Round 4 校正结果。

{quality_table}

路径差以百分点显示。可核实扫描顺序的 {audit['on_first']} 对均先 ON 后 OFF，另 {audit['order_unknown']} 对未知。OFF−ON 时间间隔中位数为 {60*time.effect:.2f} 分钟，范围 {60*time.interval_min:.2f}–{60*time.interval_max:.2f} 分钟。数据缺少反向顺序样本，顺序与刺激条件的因果作用无法分离。

{group_report}

旧日期核实子集有 {audit['legacy_date_verified']} 对。较严格的 Round 4 原件复核支持 {audit['round4_both_same_day']} 对两端均同日，完整同次访视仍为 0 对。日期和顺序子集只作方向核对，均值差及区间以百分点表示。

[全部头动模型及质量窗](../results/Transition_Motion_Sensitivity.csv) · [扫描顺序和日期子集](../results/Transition_Order_Sensitivity.csv) · [既有访视审计](../../round4/audit/Clinical_Timing_Audit_Report.md)
"""
    ilr_rows = md_table(("来源范围", "维度", "n", "原始 P", "BH q", "Holm P"), [
        (r.scope, r.dimensions, r.n_pairs, number(r.raw_p), number(r.fdr_q), number(r.holm_p))
        for r in tests.itertuples()
    ])
    directional = checks.loc[(checks.from_state.eq("M4") & checks.to_state.isin(("M1", "M4"))) |
                             (checks.from_state.eq("M3") & checks.to_state.eq("M4"))]
    scale = md_table(("路径", "原尺度", "Jeffreys 后", "ILR 中心回变换"), [
        (f"{r.from_state}→{r.to_state}", f"{100*r.original_difference:+.2f}", f"{100*r.smoothed_difference:+.2f}",
         f"{100*r.aitchison_center_difference:+.2f}") for r in directional.itertuples()
    ])
    stability = f"""# Round 5 组成型及条件稳健性

向每个已观察来源的四个转移计数单元各加 0.5 再闭合。患者 ON 和 OFF 的原始矩阵分别有 {int(checks.zero_counts_ON.sum())} 和 {int(checks.zero_counts_OFF.sum())} 个零计数单元。四个来源行使用固定 M1–M4 Helmert 基，ILR1 比较 M1/M2，ILR2 比较前两列的几何均值和 M3，ILR3 比较前三列的几何均值和 M4。

{ilr_rows}

全向量加四个来源行共五个槽，{list_tests(tests)}。原始概率尺度的整体证据没有在 ILR 家族校正后复现。下表三个关注路径在平滑及 ILR 中心回变换后保持原尺度变化方向，数值为 ON−OFF 的百分点。ILR 中心回变换是组的 Aitchison 中心差，估计对象与算术平均差有所不同。

{scale}

[完整 ILR 结果](../results/Transition_ILR_Omnibus.csv) · [12 个坐标](../results/Transition_ILR_Results.csv) · [零计数及三个尺度](../results/Transition_ILR_Sensitivity.csv) · [头动、顺序和日期核对](Transition_Robustness_Report.md)
"""
    return stability, motion_report


def clinical_report(tables):
    data = tables["M4_Clinical_Associations"]
    joint = tables["M4_Joint_Response_Analysis"].iloc[0]
    baseline = tables["M4_Baseline_Adjusted_Models"]
    test_slots = baseline.loc[baseline.hypothesis.eq(True)]
    names = {("M4", "M1", "total"), ("M4", "M4", "total"), ("M4", "M4", "bradykinesia")}
    spot = data.loc[[tuple(x) in names for x in data[["from_state", "to_state", "clinical_endpoint"]].to_numpy()]]
    candidates = md_table(("既往候选路径", "终点", "n", "ρ [95% CI]", "原始 P", "BH80 q"), [
        (f"{r.from_state}→{r.to_state}", {"total": "运动总分", "bradykinesia": "运动徐缓"}[r.clinical_endpoint],
         r.n_pairs, interval(r.rho, r.ci95_low, r.ci95_high), number(r.raw_p), number(r.fdr_q))
        for r in spot.itertuples()
    ])
    report = md_table(("模型", "项", "系数或 ΔR² [95% CI]", "P", "BH q", "Holm P"), [
        (r.model, r.term, interval(r.estimate, r.ci95_low, r.ci95_high), number(r.raw_p), number(r.fdr_q),
         number(r.holm_p)) for r in test_slots.itertuples()
    ])
    return f"""# M4 动态响应与历史运动评分

Round 4 全部 {len(data)} 个预定义 Spearman 已按原种子、原患者键、双侧扫描键和原检验方向复算。原始 P、BH80 及参与者配对 Bootstrap 区间均核对相符。当前 BH80 通过 {int(data.fdr_q.lt(.05).sum())}/80。

{candidates}

25 对患者的联合模型固定使用 ΔP44/ΔP41/ΔP34，R²={joint.r_squared:.3f}，调整 R²={joint.adjusted_r_squared:.3f}，成组重抽的样本内 95% 区间 [{joint.ci95_low:.3f}, {joint.ci95_high:.3f}]。整人重排绝对改善的 P={number(joint.raw_p)}，该单项探索检验的 q 和 Holm 与 P 相同。最大 VIF={joint.maximum_vif:.3f}，标准化设计条件数 {joint.standardized_condition_number:.3f}。样本内 R² 区间描述拟合大小，推断以重排检验为准。预测量协方差与相关见诊断表。

基线调整采用 ON 总分对 OFF 总分及预设路径变化回归。完整七项探索家族 {list_tests(test_slots)}。路径系数单位为每 1 个百分点转移变化对应的量表点数，联合块用 ΔR² 描述效应、用 HC3 Wald F 检验。

{report}

绝对改善定义为 OFF−ON，保留改善、不变与恶化。HC 采用任务规定的零分作描述参照，仅患者进入关联。评分与扫描没有完整双条件同期原件链接（0/25），上述关联属于历史评分探索，联合拟合为样本内结果。

[80 项复核](../results/M4_Clinical_Associations.csv) · [联合置换](../results/M4_Joint_Response_Analysis.csv) · [相关、协方差与 VIF](../results/M4_Joint_Response_Predictor_Diagnostics.csv) · [基线调整](../results/M4_Baseline_Adjusted_Models.csv) · [临床参照](../results/M4_Clinical_Reference_Summary.csv)
"""


def reports(tables, legacy_table, acquisition_audit, frozen_audit):
    legacy = legacy_table.loc[legacy_table.endpoint.eq("transition_omnibus")].iloc[0]
    robustness, motion = robustness_reports(tables, acquisition_audit)
    p0 = tables["Omnibus_Transition_Results"].iloc[0]
    p1 = tables["M4_Dynamics_Paired_Effects"].set_index("metric")
    ilr = tables["Transition_ILR_Omnibus"]
    joint = tables["M4_Joint_Response_Analysis"].iloc[0]
    ending = f"""# Round 5 最终解释

**A｜群体转移组织。** 25 对患者的原始概率尺度 12 维整体检验 P={number(p0.raw_p)}，与 V6 使用的同一冻结队列结论一致。固定收缩值的敏感性结果见主表。ILR 家族 {list_tests(ilr)}。可核实的 24 对均先 ON 后 OFF，条件效应与固定扫描顺序的作用无法分离。

**B｜M4 动态。** ON−OFF 自循环差 {ci(p1.loc['P44'], scale=100)} 个百分点。M4→M1 为 {ci(p1.loc['P41'], scale=100)}，M3→M4 为 {ci(p1.loc['P34'], scale=100)} 个百分点，沿用 Round 4 家族的校正证据。驻留与占用减少相互对应，M4 总离开概率与自循环互补。主要的退出去向和入口变化保留为描述性线索。

**C｜个体临床关联。** 既有 Spearman 无 BH80 通过项。三项 M4 变化联合关联的样本内 R²={joint.r_squared:.3f}，整人重排 P={number(joint.raw_p)}。该结果来自历史评分，完整同期配对为 0/25。群体条件差异与个体运动改善关联各有独立统计问题。

[整体转移](Round5_Transition_Optimization_Report.md) · [M4 分解](Round5_M4_Mechanistic_Report.md) · [稳健性](Round5_Robustness_Report.md) · [临床关联](M4_Clinical_Interpretation.md)
"""
    return {
        "Round5_Transition_Optimization_Report.md": omnibus_report(tables, legacy, frozen_audit),
        "Round5_M4_Mechanistic_Report.md": m4_report(tables),
        "Round5_Robustness_Report.md": robustness,
        "Round5_Final_Interpretation.md": ending,
        "Transition_Robustness_Report.md": motion,
        "M4_Clinical_Interpretation.md": clinical_report(tables),
    }
