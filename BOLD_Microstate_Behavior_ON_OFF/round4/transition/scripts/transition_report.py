"""Render the report from the same aggregate tables used by the plots."""

import math


ENDPOINTS_CN = {
    "total": "运动总分", "rigidity": "强直", "bradykinesia": "运动徐缓",
    "tremor": "震颤", "gait_balance": "三项步态与平衡",
}
CONTRASTS = ("OFF-HC", "ON-OFF", "ON-HC")
CONDITIONS = ("HC", "OFF", "ON")


def number(x, digits=4):
    return f"{x:.{digits}g}" if math.isfinite(x) else "NA"


def percent(x):
    return f"{100*x:.2f}%" if math.isfinite(x) else "NA"


def delta_ci(row):
    if not math.isfinite(row.mean_diff):
        return "NA"
    ci = f"[{100*row.ci95_low:+.2f}, {100*row.ci95_high:+.2f}]" if math.isfinite(row.ci95_low) else "[NA]"
    return f"{100*row.mean_diff:+.2f} {ci}"


def counts(frame):
    return int(frame.passes_fdr.sum()), int(frame.passes_holm.sum())


def discovery_list(frame, field):
    selected = frame.loc[frame[field].eq(True)]
    if selected.empty:
        return "无"
    return "、".join(f"{row.from_state}→{row.to_state}（{100*row.mean_diff:+.2f} 个百分点）"
                    for row in selected.itertuples(index=False))


def main_tables(primary):
    sections = []
    text_header = ("| 转移 | n | 第一组均值 | 第二组均值 | 均值差［95% CI］（百分点） | 原始 P | BH q | Holm P |\n"
                   "|---|---:|---:|---:|---:|---:|---:|---:|")
    for contrast in CONTRASTS:
        part = primary.loc[primary.contrast.eq(contrast)]
        rows = []
        for row in part.itertuples(index=False):
            n = f"{int(row.n_pairs)} 对" if row.paired else f"{row.n_a}/{row.n_b}"
            rows.append(f"| {row.from_state}→{row.to_state} | {n} | {percent(row.mean_a)} | {percent(row.mean_b)} | "
                        f"{delta_ci(row)} | {number(row.raw_p)} | {number(row.fdr_q)} | {number(row.holm_p)} |")
        sections.append("### " + contrast.replace("-", "−") + "\n\n" + text_header + "\n" + "\n".join(rows))
    return "\n\n".join(sections)


def m4_transition_descriptions(group):
    rows = ["| 分母口径 | 路径 | HC | OFF | ON |", "|---|---|---:|---:|---:|"]
    for method in ("WEiDA", "ExitConditional"):
        t = group.loc[group.method.eq(method) & (group.from_state.eq("M4") | group.to_state.eq("M4"))]
        wide = t.pivot(index=["from_state", "to_state"], columns="condition", values="mean")
        for (source, target), entries in wide.iterrows():
            cells = " | ".join(percent(entries[c]) for c in CONDITIONS)
            rows.append(f"| {method} | {source}→{target} | {cells} |")
    return "\n".join(rows)


def clinical_summary(clinical):
    tested = int(clinical.raw_p.notna().sum())
    passes = clinical.loc[clinical.passes_fdr]
    if not len(passes):
        return (f"80 个预设关联槽位中 {tested} 项可估计，BH80 通过 0 项，最小 q 为 "
                f"{number(clinical.fdr_q.min())}。所检验的转移概率变化与五项运动改善量缺少校正后的关联证据。")
    rows = []
    for row in passes.itertuples(index=False):
        rows.append(f"- {row.from_state}→{row.to_state} 与{ENDPOINTS_CN[row.clinical_endpoint]}，"
                    f"n={row.n_pairs} 对，ρ={number(row.rho)}，95% CI "
                    f"[{number(row.ci95_low)}, {number(row.ci95_high)}]，BH q={number(row.fdr_q)}。")
    return f"80 个预设关联槽位中 {tested} 项可估计，BH80 通过 {len(passes)} 项。\n\n" + "\n".join(rows)


def report_text(group, comparisons, clinical, m4, reference, quality, parity):
    primary = comparisons.loc[comparisons.method.eq("WEiDA") & comparisons.scope.eq("prespecified")]
    supplementary = comparisons.loc[comparisons.method.eq("ExitConditional")]
    summary = []
    for contrast in CONTRASTS:
        p = primary.loc[primary.contrast.eq(contrast)]
        b, h = counts(p)
        summary.append(f"| {contrast.replace('-', '−')} | {p.raw_p.notna().sum()}/12 | {b} | "
                       f"{discovery_list(p, 'passes_fdr')} | {h} | {discovery_list(p, 'passes_holm')} |")
    primary_summary = "\n".join(summary)
    b_main, h_main = counts(primary)
    b_exit, h_exit = counts(supplementary)
    diagonals = comparisons.loc[comparisons.method.eq("WEiDA") & comparisons.from_state.eq("M4") &
                                comparisons.to_state.eq("M4")]
    self_change = diagonals.loc[diagonals.contrast.eq("ON-OFF")].iloc[0]
    self_means = group.loc[group.method.eq("WEiDA") & group.from_state.eq("M4") & group.to_state.eq("M4")]
    self_means = self_means.set_index("condition")["mean"]
    exits = group.loc[group.method.eq("ExitConditional") & group.from_state.eq("M4")]
    peak = exits.loc[exits.groupby("condition")["mean"].idxmax()].set_index("condition")
    peak_text = "，".join(f"{c} 最多转向 {peak.loc[c, 'to_state']}（{percent(peak.loc[c, 'mean'])}）" for c in CONDITIONS)
    exit_tests = supplementary.loc[supplementary.contrast.eq("ON-OFF") & supplementary.from_state.eq("M4")]
    edge_text = "、".join(f"{row.from_state}→{row.to_state} {100*row.mean_diff:+.2f} 个百分点"
                          f"（BH q={number(row.fdr_q)}，Holm P={number(row.holm_p)}）"
                          for row in exit_tests.itertuples(index=False))
    ent_tests = primary.loc[primary.contrast.eq("ON-OFF") & primary.to_state.eq("M4")]
    ent_text = "、".join(f"{row.from_state}→{row.to_state} {100*row.mean_diff:+.2f} 个百分点"
                         f"（BH q={number(row.fdr_q)}，Holm P={number(row.holm_p)}）"
                         for row in ent_tests.itertuples(index=False))
    context = m4.loc[(m4.scope.eq("group")) & m4.method.eq("M4_context")].pivot(
        index="metric", columns="condition", values="mean")
    context_pair = m4.loc[(m4.scope.eq("contrast")) & m4.method.eq("M4_context")].set_index("metric")
    m4_rows = []
    for field, label, scale in (
        ("m4_entry_given_other_origin", "从非 M4 来源进入 M4（%）", 100),
        ("m4_exit_given_m4_origin", "从 M4 来源离开（%）", 100),
        ("duration_seconds_M4", "持续时间（秒）", 1),
        ("coverage_M4", "占用率（%）", 100),
        ("occurrence_per_minute_M4", "出现率（段/有效分钟）", 1),
        ("enters_M4", "块内进入次数", 1),
        ("exits_M4", "块内退出次数", 1),
    ):
        row = context.loc[field]
        average = " | ".join(f"{row[c]*scale:.2f}" for c in CONDITIONS)
        pair = context_pair.loc[field]
        m4_rows.append(f"| {label} | {average} | {scale*pair.mean_diff:+.2f} "
                       f"[{scale*pair.ci95_low:+.2f}, {scale*pair.ci95_high:+.2f}] |")
    timed_table = "\n".join(m4_rows)
    hc_edges = reference.loc[reference.method.eq("WEiDA") & reference.from_state.ne(reference.to_state)]
    closer = int(hc_edges.gap_reduction.gt(1e-12).sum())
    further = int(hc_edges.gap_reduction.lt(-1e-12).sum())
    hc_self = reference.loc[reference.method.eq("WEiDA") & reference.from_state.eq("M4") &
                            reference.to_state.eq("M4")].iloc[0]
    hc_self_direction = "降至" if hc_self.gap_reduction > 0 else "变为"
    nominal_hc = primary.loc[primary.contrast.eq("ON-HC")].sort_values("raw_p").iloc[0]
    scans, trs, blocks, adjacent = (int(quality[x].sum()) for x in (
        "n_scans", "valid_trs", "blocks", "valid_adjacent_pairs",
    ))
    return f"""# A3 BOLD 微状态完整四态转移概率

源数据核验通过 **{scans}/65 次扫描**（HC 15、OFF 25、ON 25），以及 25 个精确 ON/OFF 配对。WEiDA 主家族 36 项中，BH 通过 **{b_main} 项**，Holm 通过 **{h_main} 项**。退出条件概率单列为补充家族。

| 对比 | 可估计 | BH36 通过 | 方向与均值差 | Holm36 通过 | 方向与均值差 |
|---|---:|---:|---|---:|---|
{primary_summary}

主矩阵组均值为逐扫描概率等权平均，包含对角自循环。差矩阵使用第一组减第二组，ON−OFF 使用完整配对。图中的 n 对应各单元有定义的样本。

| HC | OFF | ON |
|:---:|:---:|:---:|
| ![HC 转移矩阵](../figures/Transition_WEiDA_HC.png) | ![OFF 转移矩阵](../figures/Transition_WEiDA_OFF.png) | ![ON 转移矩阵](../figures/Transition_WEiDA_ON.png) |

| OFF−HC | ON−OFF | ON−HC |
|:---:|:---:|:---:|
| ![OFF 减 HC](../figures/Transition_WEiDA_OFF_minus_HC.png) | ![ON 减 OFF](../figures/Transition_WEiDA_ON_minus_OFF.png) | ![ON 减 HC](../figures/Transition_WEiDA_ON_minus_HC.png) |

## M4 的进入、驻留和退出

M4→M4 的平均概率为 HC **{percent(self_means['HC'])}**、OFF **{percent(self_means['OFF'])}**、ON **{percent(self_means['ON'])}**。ON−OFF 的均值差和逐项自助区间为 **{delta_ci(self_change)} 个百分点**（{int(self_change.n_pairs)} 对）。自循环属于描述性比较。同一有效窗内，ON 的 M4 持续时间及占用率也低于 OFF，出现率的差值较小。

![M4 自循环配对汇总](../figures/M4_SelfLoop_Paired_Summary.png)

离开 M4 后，{peak_text}。使用退出条件分母的三条 ON−OFF 去向差为 {edge_text}。

按全部合格来源邻对计算，三条进入 M4 的 ON−OFF 差异为 {ent_text}。每个入口保留各自来源状态的分母。将非 M4 来源邻对合并所得的补充进入概率汇总如下。

| 指标 | HC | OFF | ON | ON−OFF［95% CI］ |
|---|---:|---:|---:|---:|
{timed_table}

概率和占用率行的差值用百分点。Occurrence 以连续段数除以有效分钟，块内进入次数排除块首即处于 M4 的段。每次扫描的 M4 自循环次数等于 M4 帧数减段数，其分母等于 M4 帧数减以 M4 结束的块数。持续时间等于 M4 总秒数除以段数。自循环与持续时间的方向一致性属于这一数据结构的描述，神经活动强度和临床恢复需其他证据。

{m4_transition_descriptions(group)}

![离开 M4 后的去向及变化](../figures/M4_Exit_Directions.png)

## 健康参照和既往结果

使用相同 HC 样本和各边完整患者对，WEiDA 的 12 条非对角路径中，ON 组均值到 HC 的绝对距离在 {closer} 条缩小、{further} 条扩大。M4 自循环距离从 {100*hc_self.off_absolute_gap:.2f} {hc_self_direction} {100*hc_self.on_absolute_gap:.2f} 个百分点。ON−HC 中原始 P 最小的 {nominal_hc.from_state}→{nominal_hc.to_state}，均值差及区间为 {delta_ci(nominal_hc)} 个百分点（P={number(nominal_hc.raw_p)}）。距离变化属于样本描述，等效性需要预先界定差异容许范围。

同一原窗的持续时间和占用结果见上表。此前，M4 持续时间的 ON−OFF 配对变化通过 BH4，第三轮的 20 项 Coverage 与相对运动改善关联无 BH 通过项。V6 的 12 维整体变化和路径证据使用不同的置换和校正家族，本轮重数与 V6 已发布的 ON/OFF 矩阵完成 16/16 元素数值对账。V4 E04 的转移结果对应另一套方法。 [持续时间报告](../../../reports/Behavior_ON_OFF_Report.md) · [Coverage 报告](../../../round3/reports/ON_OFF_Coverage_Clinical_Rate_Report.md) · [V6 转移报告](../../../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Transition_Dynamics.md) · [V4 E04 结果](../../../../V4_WEiDA_Statistical_Framework_Microstate/reports/V4_STATISTICAL_RESULTS.md)

## 转移变化与历史运动评分

{clinical_summary(clinical)}

检验使用同一患者的转移变化 ON−OFF 和绝对改善 OFF−ON。总分由 33 项人工条目重算，正式量表版本待核实。三项步态与平衡为探索性合计。HC 采用任务给定的零分设置，影像参照只使用 HC 转移概率。原始访视审计在 25 人中确认完整同期配对 0 人，故患者关联属于历史评分探索。原时间戳可核实的 24 对均先 ON 后 OFF，余下 1 对顺序未知。 [完整临床 BH80](../results/Transition_Clinical_Associations.csv) · [同期访视审计](../../audit/Clinical_Timing_Audit_Report.md)

## 所有主要概率检验

下列 36 项共同参与 BH 和 Holm。括号内 95% 区间是均值差区间。秩检验的统计量和效应量、Hedges g／配对 g_z 及各自区间保存在完整比较表。

![36 条组间比较、效应区间和校正](../figures/Transition_WEiDA_Forest.png)

{main_tables(primary)}

退出条件概率的 36 项补充检验中，BH 通过 {b_exit} 项，Holm 通过 {h_exit} 项。退出条件分母的非对角行独立归一化。 [完整组比较](../results/Transition_Group_Comparisons.csv) · [BH 与 Holm 校正](../results/Transition_Statistical_Corrections.csv) · [退出条件概率](../results/Transition_Probability_ExitConditional.csv)

## 复现和质量核验

严格按冻结 A3 留出回填标签、V6 原共同有效块与已发布状态编号计算。各扫描有效 TR、块数和邻对之和分别为 **{trs:,}、{blocks}、{adjacent:,}**。校验了逐扫描 4×4 计数、状态频数、连续段边界、进入与退出、概率行和及与原 Coverage、Duration、Occurrence 的对应。V6 已发表计数与分母完全吻合，概率均值最大核验误差 {parity['maximum_absolute_probability_difference']:.3g}。

三组使用 V4 Shapiro 门槛下的双侧 Welch、Mann–Whitney U、配对 t 或 Wilcoxon。区间为 9,999 次参与者级百分位 Bootstrap，配对整对重抽。结论条件于固定模板、标签和本队列。

[冻结方案](../ANALYSIS_PLAN.md) · [完整检验清单](../configs/test_registry.csv) · [运行清单与命令](../results/Transition_Run_Manifest.json) · [扫描核验汇总](../results/Transition_QC_Summary.csv) · [WEiDA 组概率](../results/Transition_Probability_WEiDA.csv) · [M4 汇总](../results/M4_Transition_Summary.csv) · [HC 参照](../results/Transition_HC_Reference.csv)

逐扫描计数和概率、逐人 M4 配对图及源文件指纹保存在工作树 Git 忽略的 `private/round4/transition/`。
"""
