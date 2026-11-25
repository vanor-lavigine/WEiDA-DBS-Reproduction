"""Build complete comparison tables and Chinese reports from the frozen tests."""

import numpy as np
import pandas as pd


DISPLAY = {
    "coverage_entropy": "占用熵",
    "global_pooled_duration": "合并段长",
    "switching_rate": "切换率",
    "global_occurrence": "总段出现率",
    "total_gev": "总 GEV",
    "global_dispersion": "全扫描平均离散度",
}

COMPARISON_COLUMNS = ("n_a", "n_b", "test", "effect_size", "ci95_low", "ci95_high", "standardized_effect",
                      "standardized_ci95_low", "standardized_ci95_high", "raw_p", "fdr_p", "holm_p",
                      "permutation_p", "permutation_holm_p", "fd_permutation_holm_p", "family_holm_and_permutation_holm")


def comparison_table(ms, weida, mapping, source_ok):
    peers = weida.set_index(["metric", "contrast"])
    rows = []
    for a in ms.itertuples():
        if not source_ok:
            status, key = "WEiDA scan-level evidence unavailable", None
        elif a.family == "Spatial Dispersion" or a.metric == "global_dispersion":
            status, key = "conceptual analogue with different geometry", None
        elif a.family == "GEV" or a.metric == "total_gev":
            status, key = "Microstate readout without WEiDA counterpart", None
        elif a.scope == "overall":
            status, key = "comparable overall metric", a.metric
        elif mapping:
            chunks = a.metric.split("_")
            key = "_".join([chunks[0], *(str(mapping[int(i)]) for i in chunks[1:])])
            status = "qualified state correspondence"
        else:
            status, key = "state correspondence failed", None
        row = dict(family=a.family, contrast=a.contrast, microstate_metric=a.metric, microstate_state=a.state,
                   weida_metric=key or "", status=status, raw_unit=a.unit)
        row.update({f"microstate_{col}": getattr(a, col, np.nan) for col in COMPARISON_COLUMNS})
        if key is not None:
            b = peers.loc[(key, a.contrast)]
            row.update(weida_state=b.state, **{f"weida_{col}": b.get(col, np.nan) for col in COMPARISON_COLUMNS})
            finite = np.isfinite(a.effect_size) and np.isfinite(b.effect_size)
            if finite:
                row.update(direction_agreement=np.sign(a.effect_size) == np.sign(b.effect_size),
                           standardized_effect_difference=a.standardized_effect - b.standardized_effect)
        rows.append(row)
    return pd.DataFrame(rows)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def value(number, signed=False):
    if not np.isfinite(number):
        return "—"
    return f"{number:+.4g}" if signed else f"{number:.4g}"


def label(row):
    if row.metric in DISPLAY:
        return DISPLAY[row.metric]
    family = "Microstate-adapted Local Dispersion" if row.family == "Spatial Dispersion" else row.family
    return f"{row.state} {family}"


def difference_rows(data):
    return [
        [label(r), r.contrast, r.test, value(r.effect_size, True), f"[{value(r.ci95_low)}, {value(r.ci95_high)}]",
         value(r.standardized_effect, True), value(r.raw_p), value(r.fdr_p), value(r.holm_p), value(r.permutation_p),
         value(r.permutation_holm_p), value(r.fd_permutation_holm_p), value(getattr(r, "all_primary_holm_p", np.nan))]
        for r in data.itertuples()
    ]


DETAIL_HEADER = ["指标", "对比", "检验", "均值差", "95% CI", "g / g_z", "Raw P", "BH P", "Holm P",
                 "置换 P", "置换 Holm P", "FD 置换 Holm P", "主指标总体 Holm P"]


def evidence_counts(data, alpha):
    counts = []
    for (family, contrast), d in data.groupby(["family", "contrast"], sort=False):
        counts.append([family, contrast, len(d), d.status.eq("ok").sum(), d.raw_p.lt(alpha).sum(),
                       d.fdr_p.lt(alpha).sum(), d.holm_p.lt(alpha).sum(), d.permutation_holm_p.lt(alpha).sum(),
                       d.family_holm_and_permutation_holm.sum(), d.all_primary_holm_p.lt(alpha).sum(),
                       d.fd_permutation_holm_p.lt(alpha).sum()])
    return table(["家族", "对比", "预设", "可估计", "Raw", "BH", "Holm", "置换 Holm", "双 Holm", "总体 Holm", "FD 置换 Holm"], counts)


def report_micro(out, ms, close, audit, alpha):
    core = ms[ms.scope.eq("primary")]
    support = core[core.family_holm_and_permutation_holm]
    n = support.contrast.value_counts()
    joint96 = core.all_primary_holm_p.lt(alpha) & core.all_primary_permutation_holm_p.lt(alpha)
    global_support = ms.scope.eq("overall") & ms.family_holm_and_permutation_holm
    detected = close[close.scope.eq("primary") & close.toward_holm_p.lt(alpha)]
    descriptive = close[close.scope.eq("primary") & close.closer_in_sample.eq(True)]
    baseline_fdr = support[~support.fd_permutation_holm_p.lt(alpha)]
    novelty = support[support.family.eq("GEV")]
    entire_family_labels = "、".join(f"{label(r)}（{r.contrast}）" for r in core[joint96].itertuples())
    once = {}
    for contrast in core.contrast.unique():
        names = [label(r) for r in support[support.contrast.eq(contrast)].itertuples()]
        once[contrast] = '、'.join(names) if names else '无'

    main_points = [
        f"1. **OFF 相对 HC。** 家族内两种 Holm 均通过的指标有 {n.get('OFF-HC', 0)} 项。{once['OFF-HC']}。",
        f"2. **DBS-ON 与 OFF 的配对变化。** 有 {n.get('ON-OFF', 0)} 项达到同一标准。{once['ON-OFF']}。",
        f"3. **向 HC 靠近。** {len(descriptive)}/32 项的样本均值距离缩小。预设的单侧趋近置换在 32 项 Holm 后支持 {len(detected)} 项。",
        f"   ON−HC 共有 {n.get('ON-HC', 0)} 项达到两种 Holm 的标准。OFF 有证据而 ON 差异未检出的项数为 "
        f"{close.loc[close.scope.eq('primary'), 'off_detected_on_not_detected'].sum()}。等效性因缺少预设界值而未判定。",
    ]
    if len(baseline_fdr):
        detail = '、'.join(f"{label(r)}（{r.contrast}，校正 P={value(r.fd_permutation_holm_p)}）"
                          for r in baseline_fdr.itertuples())
        main_points.append(f"\n以下家族内检出在 FD 调整置换后未达到 Holm 阈值。{detail}。")
    if len(novelty):
        main_points.append("\nGEV 为微状态的拟合解释量。它与 Coverage、离散度共享回填信息，三项的同向变化可能存在数学相关。")

    q3 = [[label(r), value(r.off_minus_hc, True), value(r.on_minus_hc, True), value(r.toward_hc_gain, True),
           f"[{value(r.ci95_low)}, {value(r.ci95_high)}]", value(r.toward_p), value(r.toward_holm_p)]
          for r in detected.itertuples()]
    overall_note = f"该范围下通过的指标为 {entire_family_labels}。" if entire_family_labels else ""
    text = [
        "# V4 微状态统计结果", "",
        "使用 105 冻结队列的 40 名参与者、65 次扫描（15 HC、25 对 OFF/ON），AAL78、TR 2.1 s，主分析为 V3 E04 五折留出 K=4。"
        "模板、回填标签和每次扫描的 28 项原有状态指标已经逐项核验。空间离散度从同一扫描和相应折模板计算。", "",
        "## 三个科学问题", "", *main_points, "",
        f"微状态主表 {len(core)} 项中的家族内 BH 和 Holm 分别通过 {core.fdr_p.lt(alpha).sum()}、{core.holm_p.lt(alpha).sum()} 项。"
        f"双 Holm 通过 {len(support)} 项。96 项一起控制时，原始 Holm 和置换 Holm 同时通过 {joint96.sum()} 项。{overall_note}"
        f"六项标签置换不变的整体指标共 {ms.scope.eq('overall').sum()} 个对比，其中双 Holm 通过 {global_support.sum()} 项。", "",
        "## 家族内显著及未达显著的计数", "", evidence_counts(core, alpha), "",
        "Raw、BH、Holm、置换 Holm 与 FD 置换 Holm 的各列为 P <0.05 的项数。双 Holm 同时要求 Raw P 和 Permutation P 的家族 Holm <0.05。"
        "转移的完整家族含 12 条有向非对角转换 × 3 个对比。", "",
        "## 家族内两种 Holm 均通过的项目", "",
        table(DETAIL_HEADER, difference_rows(support)) if len(support) else "无。", "",
        "Coverage、GEV 与转换概率以 0–1 比例计。Duration 以秒计。Occurrence 以每秒段数计。离散度为无量纲的模板距离。"
        "标准化效应为独立组 Hedges g 与配对 Hedges g_z。95% CI 条件于固定模板与标签。", "",
        "## 向 HC 靠近的对比", "",
        table(["指标", "OFF−HC", "ON−HC", "距离缩短", "95% CI", "单侧置换 P", "32 项 Holm P"], q3) if q3 else "无通过 32 项 Holm 的指标。", "",
        "距离缩短采用相同 HC 以及相同患者对的组均值。该证据刻画组均值的位置变化。ON−HC 的差异未被检出时，原始差异及区间仍可在主表读出。"
        "个体临床改善和统计等效性未在 V4 判定。所有正、负趋近及未达显著项目见 [Towards_HC.csv](../results/Towards_HC.csv)。", "",
        "## FD 与检验条件", "",
        f"V3 的 CR1 FD 估计及原有校正已原样保留。本次共有 {audit['v3_fd_reproduced_rows']} 个与 V3 有效归档对应的估计、区间和 P 逐项复算相符。"
        "另对每个项目以 Freedman–Lane 进行 FD 调整置换。V3 包含 C4（合并患者效应）及双 HC 对比同向的交集检验，完整的 E04/E04_R 归档节选保存在本项目。"
        "同一固定队列的特征共享训练参与者，患者访视的噪声与顺序效应可能影响条件推断。", "",
        "## 完整结果与溯源", "",
        "- [OFF−HC](../results/HC_vs_OFF.csv) · [ON−HC](../results/HC_vs_ON.csv) · [ON−OFF](../results/ON_vs_OFF.csv) · [全部微状态指标](../results/Microstate_Results.csv)",
        "- [12 条有向转换](../results/Transition_Results.csv) · [BH-FDR](../results/FDR_Results.csv) · [Holm](../results/Holm_Results.csv) · [置换及 FD 调整](../results/Permutation_Results.csv)",
        "- [描述统计](../results/Group_Level_Summary.csv) · [V3 FD 与共同 PD 效应归档](../results/V3_FD_Adjusted_Archive.csv) · [跨方法对照](CROSS_METHOD_COMPARISON.md)",
        "- [输入校验记录](../results/Input_Verification.json) · [统计规程](WEIDA_STATISTICAL_PROTOCOL.md) · [执行与复核](../README.md)",
        "- [组间图](../figures/Group_Comparison.png) · [转换矩阵](../figures/Transition_Matrix.png) · [效应量森林图（含 ON−OFF 配对汇总）](../figures/Effect_Size_Forest.png)", "",
    ]
    (out / "reports" / "V4_STATISTICAL_RESULTS.md").write_text("\n".join(text), encoding="utf-8")


def report_comparison(out, ms, weida, cross, match, mapping, audit):
    ms_found = ms[ms.scope.eq("primary") & ms.family_holm_and_permutation_holm]
    w_found = weida[weida.scope.eq("primary") & weida.family_holm_and_permutation_holm]
    overall = cross[cross.status.eq("comparable overall metric")]
    comparable = cross[cross.status.isin(["comparable overall metric", "qualified state correspondence"])]
    if comparable.empty:
        shared = comparable
    else:
        shared = comparable[comparable.microstate_family_holm_and_permutation_holm.eq(True) &
                            comparable.weida_family_holm_and_permutation_holm.eq(True) & comparable.direction_agreement.eq(True)]
    by_contrast = []
    for g, d in overall.groupby("contrast", sort=False):
        by_contrast.append([g, len(d), d.direction_agreement.eq(True).sum(), d.direction_agreement.eq(False).sum(),
                            d.microstate_family_holm_and_permutation_holm.eq(True).sum(),
                            d.weida_family_holm_and_permutation_holm.eq(True).sum()])
    detail = [[DISPLAY[r.microstate_metric], r.contrast, value(r.microstate_standardized_effect, True),
               value(r.weida_standardized_effect, True), value(r.microstate_fdr_p), value(r.weida_fdr_p),
               value(r.microstate_holm_p), value(r.weida_holm_p), value(r.microstate_permutation_holm_p),
               value(r.weida_permutation_holm_p)] for r in overall.itertuples()]
    qual = match[match.hungarian_selected] if not match.empty else match
    if audit["weida_available"]:
        wire = [
            "从本地作者代码 R2025b、K=4、种子 0 档案读取 65 次逐扫描标签及特征。核对扫描编号、条件、配对键、ROI SHA、有效帧和 MAT 阵列。"
            "两种方法读取的 ROI 阵列逐元素相同。WEiDA 时钟覆盖 13,647 个有效帧。三次扫描各缺一帧。",
            "逐扫描重新计算占用率、作者边界删失段长及 4×4 下一 TR 转换，并与归档逐项核对。原特征与原质心复核通过后，"
            "按原 VAR 公式重算局部 metastability。参见 [原 WEiDA 方法](WEIDA_STATISTICAL_PROTOCOL.md) 和 [输入校验](../results/Input_Verification.json)。",
        ]
    else:
        wire = [f"WEiDA 逐人档案未通过核验，无法直接比较。原因见 [Input_Verification.json](../results/Input_Verification.json)。"]
    if not qual.empty:
        spatial = (f"五个留出折各有四个 Hungarian 候选。{qual.pair_pass.sum()}/{len(qual)} 同时达到 |r|≥0.60 和双向余量≥0.05，"
                   f"候选 |r| 范围 {value(qual.abs_spatial_r.min())}–{value(qual.abs_spatial_r.max())}。"
                   f"跨折排列一致性{'通过' if qual.same_order_in_all_folds.all() else '未通过'}。")
        spatial += "完整四态对应通过。" if mapping else "完整四态对应未建立，跨方法同类比较限定在无状态编号的整体指标。"
    else:
        spatial = "WEiDA 状态映射无法评估。"
    status = ("局部离散程度的疾病差异在两种表征中都有体现。" if
              (ms_found.family.eq("Spatial Dispersion") & ms_found.contrast.eq("OFF-HC")).any() and
              (w_found.family.eq("Local Metastability (WEiDA VAR)") & w_found.contrast.eq("OFF-HC")).any() else
              "两种表征的局部离散效应分别见逐状态结果。")
    micro_specific = ms_found[ms_found.family.eq("GEV")]
    coverage_pattern = ms_found[ms_found.family.eq("Coverage")]
    onoff_w = w_found[w_found.contrast.eq("ON-OFF")]
    found_duration = ms_found[ms_found.family.eq("Duration")]
    w_duration = weida.loc[weida.family.isin(["Duration", "WEiDA native Dwell Time"]), "family_holm_and_permutation_holm"].sum()
    direction_disagrees = overall[overall.direction_agreement.eq(False)] if len(overall) else overall
    reversals = '、'.join(f"{DISPLAY[r.microstate_metric]}（{r.contrast}）" for r in direction_disagrees.itertuples())
    novel = '、'.join(f"{label(r)}（{r.contrast}，均值差 {value(r.effect_size, True)}）" for r in micro_specific.itertuples())
    occupancy = '、'.join(f"{label(r)}（{r.contrast}，均值差 {value(r.effect_size, True)}）" for r in coverage_pattern.itertuples())
    content = [
        "# WEiDA 与 BOLD 微状态同扫描对照", "",
        "## 队列与可比性", "", *wire, "",
        "原论文使用 16 名 HC 与 25 对患者。本地比较严格使用 V3 的 15 名 HC 与 25 对患者，论文的逐人身份映射未得到核验。"
        "WEiDA 原复跑使用 0.04–0.07 Hz 的相位特征和全队列模板，V3 E04 使用已有 0.01–0.08 Hz 缓存幅值和五折模板。"
        "同扫描控制了输入队列和 ROI 顺序，分析带宽和模板训练设计仍有区别。跨方法差异不能单独归因于状态表征。", "",
        spatial, "",
        "本地 WEiDA 编号 W1–W4 和 V3 微状态 M1–M4 各自保留。局部 VAR 使用 WEiDA 特征图及原始方差尺度。"
        "微状态适配离散度使用中心化单位幅值图及极性不敏感的平均距离。二者作为各方法内部的局部离散证据阅读。"
        "原 WEiDA 段长删去边界段，原转换分母包含自循环。用于同定义对照的 WEiDA 段长和转换从归档标签重新计算，原口径另有完整结果。", "",
        "## 研究结论", "",
        "| 原论文发现 | V4 在冻结队列中得到的证据 |", "| --- | --- |",
        "| V-VA/SM 的占用率 OFF 低于 HC，ON 相对 OFF 增加。LMB-SM 在 OFF 的占用率更高。 | "
        + (f"微状态检测到占用率变化。{occupancy}。" if occupancy else "微状态占用率达到双 Holm 的对比数为 0。")
        + "论文状态身份在微状态中无法验证。 |",
        "| V-VA/SM 的局部 VAR 在 OFF 较高，ON 相对 OFF 降低。 | " + status
        + f"本地 WEiDA 有 {len(onoff_w)} 个 ON−OFF 主指标同时通过两种 Holm。 |",
        "| 论文报告通往 V-VA/SM 的若干转换变化，组比较使用未经校正的 P。 | "
        f"本地将 12 条转换和三个对比全部纳入校正。微状态和 WEiDA 的双 Holm 转换结果分别为 "
        f"{ms_found.family.eq('Transition Probability').sum()} 项和 {w_found.family.eq('Transition Probability').sum()} 项。 |", "",
        "占用率及局部离散的结果与原论文部分现象方向相近。四态对应未达到预设标准，无法确认微状态复现了论文所指的具体 PMS 中的疾病异常与刺激变化。"
        "同扫描 WEiDA 与微状态在整体指标的直接比较见下表。论文的疾病、刺激和转换结果来自 Fig. 3–4。"
        "[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC13280393/)", "",
        (f"微状态特有的已检出读出为 {novel}。GEV 反映幅值图对模板的拟合和占用，结果与同状态的 Coverage 及离散度存在定义关联。"
         if novel else "微状态特有的 GEV 项没有达到双 Holm。"), "",
        (f"微状态段长通过双 Holm 的对比为 {'、'.join(f'{label(r)}（{r.contrast}，均值差 {value(r.effect_size, True)} s）' for r in found_duration.itertuples())}。"
         f"本地 WEiDA 的同定义状态段长与边界删失段长合计 {w_duration} 项通过双 Holm。" if audit["weida_available"] else ""), "",
        "对照表的方向一致指两种方法的均值差同号。方向不一致或仅一侧检出属于描述性对照。两种方法差异大小的显著性未另作交互检验。", "",
        "## 定义可对齐的效应", "",
        f"在 {len(comparable)} 个可对齐的指标 × 对比中，{len(shared)} 项同时达到双 Holm 且效应同向。切换率、总段出现率和合并段长由共同的段定义推导，存在机械相关。", "",
        (f"点估计方向相反的整体对比为 {reversals}。" if reversals else ""), "",
        table(["对比", "整体指标数", "同向", "反向", "微状态双 Holm", "WEiDA 双 Holm"], by_contrast) if by_contrast else "没有可直接比较的整体指标。", "",
        table(["整体指标", "对比", "微状态 g", "WEiDA g", "微状态 BH", "WEiDA BH", "微状态 Holm", "WEiDA Holm",
               "微状态置换 Holm", "WEiDA 置换 Holm"], detail) if detail else "", "",
        "## 本地 WEiDA 通过两种 Holm 的主指标", "",
        table(DETAIL_HEADER, difference_rows(w_found)) if len(w_found) else "无。", "",
        "全部效应、区间及未达显著结果见 [WEiDA_Results.csv](../results/WEiDA_Results.csv) 与 [Microstate_Results.csv](../results/Microstate_Results.csv)。"
        "[Cross_Method_Comparison.csv](../results/Cross_Method_Comparison.csv) 保留实际可比性标记。"
        "[State_Matching.csv](../results/State_Matching.csv) 记录逐折全矩阵，包含有符号相关、绝对相关、行列余量与门槛结果。"
        "[跨方法效应图](../figures/Cross_Method_Concordance.png)", "",
    ]
    (out / "reports" / "CROSS_METHOD_COMPARISON.md").write_text("\n".join(content), encoding="utf-8")
