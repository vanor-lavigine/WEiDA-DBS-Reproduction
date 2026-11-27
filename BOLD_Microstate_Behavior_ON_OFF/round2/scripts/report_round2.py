"""Build the release report solely from validated summaries and archived context."""

import math

CLINICAL = {
    "total": "运动总分", "rigidity": "强直", "bradykinesia": "运动徐缓",
    "tremor": "震颤", "gait_balance": "步态与平衡", "remaining_four": "其余四项",
}
PATTERNS = {
    "both_shorter": "M2、M4 均缩短", "only_m2_shorter": "只有 M2 缩短",
    "only_m4_shorter": "只有 M4 缩短", "both_longer": "M2、M4 均延长", "zero_in_either": "任一态为零差",
}


def number(value, digits=2):
    if not math.isfinite(value):
        return "NA"
    formatted = f"{value:.{digits}f}"
    return formatted.replace("-", "−") if formatted[0] == "-" else formatted


def stats_text(row, prefix, rate=False):
    values = [getattr(row, f"{prefix}_{stat}") for stat in ("mean", "median", "q25", "q75")]
    if rate:
        values = [100*x for x in values]
    mean, median, q1, q3 = (number(x) for x in values)
    return f"{mean} / {median} [{q1}, {q3}]"


def write_report(path, profile, composition, patterns, states, reference, brain_audit, issues):
    score = profile.set_index("endpoint")
    total, tremor = score.loc["total"], score.loc["tremor"]
    combo = patterns.set_index("pattern")["count"]
    n = int(total["n"])
    lines = [
        "# STN DBS ON/OFF 第二轮报告",
        "",
        "沿用原冻结 A3 共同窗及 25 名患者配对。这一轮描述改善的构成和个体变化，并估计 M4 持续时间与"
        "同一冻结队列 15 名 HC 均值的距离变化。原临床 BH5、持续时间 BH4 和 20 项关联均保留原校正范围。"
        "本轮只对平均 M4 配对距离变化给出嵌套 Bootstrap 区间，未追加 P 值。",
        "",
        "## 一、临床改善的构成与一致性",
        "",
        f"**总分 {int(total.n_improved)}/{n} 人下降。** OFF 和 ON 平均分别为 "
        f"{number(total.off_mean)} 和 {number(total.on_mean)} 分，平均改善 {number(total.improvement_mean)} 分。"
        f"逐人改善率的平均值为 {100*total.rate_mean:.1f}%。下降为分值方向，临床重要改善阈值未设定。"
        "总分沿用 33 项人工评分重算口径，正式量表版本待核实。",
        "",
        "下表分值列依次为均值、中位数和四分位范围。改善率列使用百分比，分母为该项 OFF 分数大于零的人数。",
        "",
        "| 指标 | OFF（分） | ON（分） | 改善量 OFF−ON（分） | 改善 / 不变 / 变差 | 有效改善率 n | 改善率（%） | 原 BH5 q |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in profile.itertuples():
        counts = " / ".join(f"{getattr(row, f'n_{key}')}（{getattr(row, f'{key}_fraction'):.0%}）"
                            for key in ("improved", "unchanged", "worse"))
        lines.append(f"| {CLINICAL[row.endpoint]} | {stats_text(row, 'off')} | {stats_text(row, 'on')} | "
                     f"{stats_text(row, 'improvement')} | {counts} | {row.rate_n} | {stats_text(row, 'rate', True)} | "
                     f"{row.legacy_bh5_q:.4g} |")
    lines += [
        "",
        f"震颤在全部 {n} 人中为 {int(tremor.n_improved)} 人改善、{int(tremor.n_unchanged)} 人不变、"
        f"{int(tremor.n_worse)} 人变差。OFF 震颤大于零的 {int(tremor.off_positive_n)} 人中有 "
        f"{int(tremor.off_positive_n_improved)} 人改善（{tremor.off_positive_improved_fraction:.1%}）。"
        f"另 {int(tremor.off_zero_n)} 人 OFF 震颤已经为零，该条目没有进一步下降的空间。"
        "这些人的原始分数与绝对变化均保留，改善率记为 NA。",
        "",
        "![五项临床评分变化方向](../figures/Clinical_Response_Proportions.png)",
        "",
        "[临床连续量、比例与分母](../results/Clinical_Response_Profile.csv) · [原临床 BH5](../../results/Clinical_Paired_Results.csv)",
    ]
    if composition is None:
        lines += ["", f"五项分解待补充。{issues['clinical_composition']}。"]
    else:
        ranked = composition.loc[composition.component.ne("total")].sort_values("improvement_mean", ascending=False)
        first, second = tuple(ranked.itertuples())[:2]
        lines += [
            "",
            f"**总分改善点数以{CLINICAL[first.component]}和{CLINICAL[second.component]}两部分较多。** "
            f"平均分别改善 {number(first.improvement_mean)} 和 {number(second.improvement_mean)} 分，"
            f"合计占平均总分改善的 {first.fraction_of_total_mean+second.fraction_of_total_mean:.1%}。"
            f"根据字典重算了两个条件的 {2*n} 次评分。强直、运动徐缓、震颤、步态与平衡和其余四项逐人相加均等于总分。"
            "其余四项是言语、面部表情、起立和姿势。",
            "",
            "| 总分部分 | 条目数 | OFF 均值（分） | ON 均值（分） | 平均改善（分） | 占平均总分改善 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in composition.itertuples():
            lines.append(f"| {CLINICAL[row.component]} | {row.item_count} | {number(row.off_mean)} | "
                         f"{number(row.on_mean)} | {number(row.improvement_mean)} | "
                         f"{row.fraction_of_total_mean:.1%} |")
        lines += [
            "",
            "分值比例为每部分的平均改善除以平均总分改善。各域条目数不同，比例只描述这套量表的改善点数构成。",
            "",
            "![平均运动总分改善的构成](../figures/Clinical_Improvement_Composition.png)",
            "",
            "[总分分解](../results/Clinical_Improvement_Composition.csv) · [原人工条目字典](../../results/Score_Item_Dictionary.csv)",
        ]
    lines += [
        "",
        "## 二、脑动态的个体组合与健康参照",
        "",
        f"**{int(combo['both_shorter'])}/{n} 人的 M2 和 M4 都缩短，"
        f"{int(combo['only_m2_shorter']+combo['only_m4_shorter'])}/{n} 人只缩短其中一态。** "
        "两态的组均值缩短由部分相同患者共同支持，患者内仍有多种方向组合。"
        "方向取自实际 ON−OFF，1e−12 秒容差内单列为零差。",
        "",
        "| 持续时间方向组合 | 人数 / 总配对数 | 比例 |",
        "|---|---:|---:|",
    ]
    for row in patterns.itertuples():
        lines.append(f"| {PATTERNS[row.pattern]} | {row.count}/{row.n_pairs} | {row.proportion:.0%} |")
    lines += [
        "",
        "| 状态 | 缩短 / 不变 / 延长 | 平均变化（秒） | 中位变化与四分位范围（秒） |",
        "|---|---|---:|---:|",
    ]
    for row in states.itertuples():
        lines.append(f"| {row.state} | {row.shorter} / {row.unchanged} / {row.longer} | {number(row.mean_s)} | "
                     f"{number(row.median_s)} [{number(row.q1_s)}, {number(row.q3_s)}] |")
    lines += [
        "",
        "第一轮 M2 和 M4 的持续时间 BH4 q 分别为 0.04375 和 0.006415。第二轮方向计数是配对描述。",
        "",
        "![M2 和 M4 的变化组合与幅度](../figures/M2_M4_Response_Patterns.png)",
        "",
        "[方向组合](../results/M2_M4_Response_Patterns.csv) · [持续时间幅度](../results/M2_M4_Duration_Change_Summary.csv)"
        " · [原持续时间 BH4](../../results/Duration_Paired_Results.csv)",
    ]
    if reference is None:
        lines += ["", f"M4 健康参照距离待补充。{issues['m4_hc_reference']}。"]
    else:
        (r,) = reference.itertuples(index=False)
        result = ("区间全部在零以上，支持当前固定模型下 ON 时平均个体距离较小。"
                  if r.ci95_low_s > 0 else
                  "区间全部在零以下，支持当前固定模型下 ON 时平均个体距离较大。"
                  if r.ci95_high_s < 0 else "区间包含零，平均个体距离的变化方向仍有不确定性。")
        lines += [
            "",
            "### M4 距离本队列 HC 均值",
            "",
            f"**{r.n_closer} 人更接近，{r.n_farther} 人更远离，{r.n_unchanged} 人不变。** "
            f"个体到参照均值的平均距离从 {number(r.off_distance_mean_s)} 秒降至 {number(r.on_distance_mean_s)} 秒，"
            f"平均配对距离改善为 {number(r.distance_change_mean_s)} 秒"
            f"（95% 百分位区间 {number(r.ci95_low_s)} 至 {number(r.ci95_high_s)} 秒）。{result}",
            "",
            f"HC 参照是 {r.n_hc} 人原共用有效窗 M4 持续时间的算术均值 {number(r.hc_center_s)} 秒。"
            f"HC 中位数 {number(r.hc_median_s)} 秒，四分位范围 "
            f"{number(r.hc_q1_s)} 至 {number(r.hc_q3_s)} 秒。此范围是样本描述。",
            "",
            "| 距离统计 | 均值（秒） | 中位数（秒） | 四分位范围（秒） |",
            "|---|---:|---:|---:|",
        ]
        for label, prefix in (("OFF 到参照的绝对距离", "off_distance"), ("ON 到参照的绝对距离", "on_distance"),
                              ("个体距离变化（OFF 距离减 ON 距离）", "distance_change")):
            lines.append(f"| {label} | {number(getattr(r, prefix+'_mean_s'))} | "
                         f"{number(getattr(r, prefix+'_median_s'))} | "
                         f"{number(getattr(r, prefix+'_q1_s'))} 至 {number(getattr(r, prefix+'_q3_s'))} |")
        lines += [
            "",
            f"组均值本身到参照的距离为 OFF {number(r.off_mean_duration_distance_to_hc_s)} 秒、"
            f"ON {number(r.on_mean_duration_distance_to_hc_s)} 秒。主要效应使用上表的个体绝对距离配对差，"
            "先对每人计算绝对距离，再取平均。"
            f"M4 缩短的 {brain_audit['n_m4_shorter']} 人中，"
            f"{brain_audit['n_closer_and_m4_shorter']} 人同时更接近 HC 均值。",
            f"{r.bootstrap_draws:,} 次重抽每次同时抽取 {r.n_hc} 名 HC 和 {r.n_pairs} 个完整患者对。"
            "每次重估一个 HC 均值，供该轮全部 ON 与 OFF 使用。",
            "",
            "![M4 健康参照分布、患者距离与区间](../figures/M4_HC_Reference.png)",
            "",
            "[健康参照距离汇总](../results/M4_HC_Reference_Summary.csv)",
        ]
    lines += [
        "",
        "### 既有时间分解与空间注释",
        "",
        "V7 原窗结果显示，M4 的平均持续时间由 OFF 11.98 秒降到 ON 8.92 秒，"
        "占用比例由 37.02% 降到 28.82%，出现率均值从 1.848 增到 1.886 段/有效分钟。"
        "每次扫描的平均持续时间等于 M4 总占用时间除以 M4 段数。"
        "这组点估计呈现总占用减少、连续驻留缩短、出现率均值变化较小。"
        "V7 占用分量原 Holm6 P=0.1150，保留为描述性分解。"
        " [V7 时间分解](../../../BOLD_Microstate_Mechanistic_Decomposition/reports/V7_M4_Duration_Decomposition.md)",
        "",
        "V7 折 0 的 AAL78–Yeo7 注释以落入 Yeo7 的平方模板权重为分母。"
        "M2 主要份额包括 Default 32.6%、Ventral Attention 14.7%，M4 包括 Default 31.3%、Somatomotor 18.1%。"
        "Yeo7 映射部分分别覆盖 M2 和 M4 整体平方模板权重的 39.9% 和 39.8%。"
        "这些占比描述已映射部分的空间分布，整个状态保留 M2/M4 编号。激活和抑制需要另外的测量证据。"
        " [V7 空间注释](../../../BOLD_Microstate_Mechanistic_Decomposition/reports/V7_Functional_Network_Annotation.md)",
        "",
        "V6 四态 Coverage 的 Aitchison 距离平均缩短 0.124，嵌套 95% CI −0.191 至 0.412，"
        "健康参照家族 Holm P=0.21938，原结论仍为未获得校正支持。该指标衡量四态占用组成，"
        "本轮衡量单态持续时间到 HC 均值的绝对距离。"
        " [V6 四态占用距离](../../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Normalization_Analysis.md) · "
        "[V6 配对异质性](../../../V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Paired_Response_Heterogeneity.md)",
        "",
        "结论适用于同一冻结模型与本队列参照。全脑动态恢复正常、与 HC 等效及因果疗效仍未建立。"
        "既有顺序审计中，24 对可核实患者均先 ON 后 OFF，另 1 对顺序未知。"
        "共享图使用汇总值，逐人配对轨迹在本地私有目录。",
        "",
        "[第二轮方案](../ANALYSIS_PLAN.md) · [输入指纹与核验](../results/Round2_Run_Manifest.json)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
