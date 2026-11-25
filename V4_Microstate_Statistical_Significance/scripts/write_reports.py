"""Write the scientific and stability narratives from the validated result tables."""

import json

import numpy as np
import pandas as pd

from micro_inputs import HERE


OUT = HERE / "results"
REPORT = HERE / "reports"
ALPHA = .05
FAMILY_NAMES = {"Coverage": "Coverage", "Duration": "平均持续时间", "Occurrence": "Occurrence",
                "GEV": "GEV", "Spatial Dispersion": "空间离散度", "Transition Probability": "有向转换",
                "Switching Rate": "全局切换率"}
CONTRAST = {"OFF-HC": "OFF−HC", "ON-HC": "ON−HC", "ON-OFF": "ON−OFF"}


def csv(name):
    return pd.read_csv(OUT / f"{name}.csv")


def info(name):
    return json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))


def md(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(str(x) for x in row) + " |" for row in rows)])


def number(x):
    return f"{x:.4g}" if np.isfinite(x) else "—"


def mark(row):
    state = row.state + " " if isinstance(row.state, str) else ""
    return state + FAMILY_NAMES.get(row.family, row.family)


def effect(row):
    if row.family in ("Coverage", "GEV", "Transition Probability"):
        return f"{100*row.effect_size:+.2f} 个百分点"
    if row.family in ("Occurrence", "Switching Rate"):
        return f"{60*row.effect_size:+.3f} 次/分"
    if row.family == "Duration":
        return f"{row.effect_size:+.3f} s"
    return f"{row.effect_size:+.4f}"


def count_rows(tests):
    counts = []
    for (family, code), d in tests.groupby(["family", "contrast"], sort=False):
        counts.append([FAMILY_NAMES[family], CONTRAST[code], len(d), d.status.eq("ok").sum(), d.raw_p.lt(ALPHA).sum(),
                       d.interpret_fdr.sum(), d.interpret_holm.sum(), d.permutation_support.sum(),
                       d.fd_permutation_holm_p.lt(ALPHA).sum()])
    return counts


def details(selected):
    return [[CONTRAST[r.contrast], mark(r), effect(r), number(r.fdr_p), number(r.holm_p),
             number(r.permutation_holm_p), number(r.fd_permutation_holm_p), f"{r.defined:.0f}/2000"] for r in selected.itertuples()]


def primary_report(tests, comp, shared, boot, audit):
    u = tests.merge(boot.loc[boot.scheme.eq("full"), ["contrast", "metric", "defined", "full_interval_excludes_zero"]],
                    on=["contrast", "metric"], validate="one_to_one")
    main = u[u.joint_holm_support]
    global_ok = u[u.all_scalar_holm_p.lt(ALPHA) & u.all_scalar_permutation_holm_p.lt(ALPHA)]
    bh_only = u[u.interpret_fdr & ~u.interpret_holm]
    fdx = u[u.fd_permutation_holm_p.lt(ALPHA) & ~u.joint_holm_support]
    layout = ["对比", "指标", "差值", "家族 BH P", "家族 Holm P", "置换 Holm P", "FD 置换 Holm P", "重训有定义"]
    ilr = [[CONTRAST[r.contrast], "对齐四态" if r.metric == "ilr" else "按占比排序", number(r.effect_size), number(r.permutation_p),
            number(r.permutation_holm_p), number(r.fd_permutation_holm_p)] for r in comp.itertuples()]
    support = [f"{CONTRAST[i.contrast]} {mark(i)}" for i in global_ok.itertuples()]
    conclusions = []
    for code in CONTRAST:
        f = u.loc[u.contrast.eq(code)]
        some = f[f.joint_holm_support]
        listing = "、".join(mark(r) for r in some.itertuples()) or "无"
        conclusions.append(f"- **{CONTRAST[code]}**。家族内 BH、常规 Holm 和置换 Holm 分别支持 "
                           f"{f.interpret_fdr.sum()}、{f.interpret_holm.sum()}、{f.permutation_support.sum()} 项。"
                           f"同时通过两种 Holm 的项目为 {listing}。")
    shared_n = int((shared.holm_p.lt(ALPHA) & shared.permutation_holm_p.lt(ALPHA)).sum())
    both_name = "、".join(support) if support else "无"
    lines = ["# V4 微状态统计显著性结果", "",
             f"使用 15 名 HC 和 25 对患者的 V3 冻结队列，65 次扫描。AAL78、TR 2.1 s、固定 K=4。"
             f"A3 的全部 65 次回填标签与版本归档逐帧相符。本轮共同有效窗共 {audit['frames_total']:,} 帧。"
             "每个标量对比各运行 10,000 次参与者级置换和 10,000 次 FD 敏感性置换。", "",
             "## 三个比较及共同方向", "", *conclusions,
             f"- **两项 HC 比较同方向**。共 {len(shared)} 项，{shared.estimated_same_direction.sum()} 项点估计同方向。"
             f"同时通过两种各家族 Holm 的有 {shared_n} 项。", "",
             "## 四态整体 Coverage", "",
             md(["对比", "ILR 口径", "效应距离", "置换 P", "两项 Holm P", "FD 置换 Holm P"], ilr), "",
             "对齐四态的 OFF−HC 整体组成通过 Holm，允许在该对比定位状态。ON−HC 的整体组成未达门槛。"
             "按占比排序的置换结果属于预设探索性敏感性。该对比的全局探索 Holm P 值见完整表。", "",
             "## 同时通过常规及置换 Holm 的结果", "", md(layout, details(main)), "",
             f"99 项标量联合校正后同时通过两套 Holm 的项目共 {len(global_ok)} 项。{both_name}。"
             "所有逐状态条目都在重训中出现无法稳定定位的轮次。表中效应与 P 条件于已经锁定的留出标签。", "",
             "## BH 通过且常规 Holm 未通过的结果", "", md(layout, details(bh_only)), "",
             "## 仅 FD 敏感性达到家族 Holm 的补充结果", "", md(layout, details(fdx)), "",
             "## 完整检验计数", "",
             md(["指标", "对比", "总计", "可估计", "Raw", "BH", "Holm", "置换 Holm", "FD 置换 Holm"], count_rows(u)), "",
             "BH、Holm 与置换 Holm 的计数采用 Coverage 的整体门槛。各对比的未达显著结果、SD、中位数、IQR 和原单位 95% 区间均在下列数据表。", "",
             "- [OFF−HC](../results/HC_vs_OFF.csv) · [ON−HC](../results/HC_vs_ON.csv) · [ON−OFF](../results/ON_vs_OFF.csv)",
             "- [Coverage](../results/Coverage_Results.csv) · [Duration](../results/Duration_Results.csv) · [Occurrence](../results/Occurrence_Results.csv) · [完整转换矩阵](../results/Transition_Matrix_Descriptive.csv)",
             "- [非对角转换检验](../results/Transition_Results.csv) · [GEV](../results/GEV_Results.csv) · [Spatial Dispersion](../results/Spatial_Dispersion_Results.csv)",
             "- [FDR](../results/FDR_Results.csv) · [Holm](../results/Holm_Results.csv) · [置换](../results/Permutation_Results.csv) · [探索全局校正](../results/Exploratory_Global_Corrections.csv) · [重训区间](../results/Bootstrap_Results.csv)",
             "- [覆盖率图](../figures/Coverage_Comparison.png) · [段长图](../figures/Duration_Comparison.png) · [出现率图](../figures/Occurrence_Comparison.png) · [转换图](../figures/Transition_Matrix.png) · [GEV 图](../figures/GEV_Comparison.png) · [效应量森林图](../figures/Effect_Size_Forest.png) · [显著性汇总](../figures/Significance_Summary.png)", ""]
    (REPORT / "V4_Microstate_Statistical_Results.md").write_text("\n".join(lines), encoding="utf-8")


def stability_report(boot, check, history):
    match = csv("State_Matching")
    rank = csv("Fallback_Method_Selection")
    core = csv("Reliability_Core_Calibration_Recount")
    analysis = csv("Reliability_Traditional_Family_Recount")
    v3 = csv("V3_Peak_Null_Recount")
    low = v3.loc[v3.scenario.eq("S0")].set_index("method").rate
    high = v3.loc[v3.scenario.eq("S5")].set_index("method").rate
    core_s5 = core[core.scenario.eq("S5_high")].set_index("method")
    count = {f"M{i+1}": n for i, n in enumerate(check["matches_by_state"])}
    qualified = boot.loc[boot.scheme.eq("full") & boot.full_interval_excludes_zero]
    rows = []
    for r in rank.itertuples(index=False):
        method = r[0]
        row = core_s5.loc[method]
        rows.append([method, f"{row.core_any_reject}/{row.replicates}", f"{100*r.worst_s5_wilson_upper:.2f}%",
                     f"{r.ari_mean:.3f}", f"{r.all_axis_qualified_fraction*100:.1f}%"])
    trad = analysis.loc[analysis.scenario.isin(["S0", "S5_high"]) & analysis.method.isin(["A0", "A1", "A3"])]
    trad_rows = [[r.scenario, r.method, r.family, r.n, r.all_8_estimable, f"{r.holm_any_rate*100:.1f}%"] for r in trad.itertuples()]
    bootstrap_states = [[key, n, f"{100*n/check['draws']:.2f}%"] for key, n in count.items()]
    bootstrap_states.append(["四态同时可匹配", check["full_composition_matches"],
                             f"{100*check['full_composition_matches']/check['draws']:.2f}%"])
    portions = [[CONTRAST[r.contrast], r.metric, r.defined, number(r.mean_difference),
                 f"[{number(r.ci95_low)}, {number(r.ci95_high)}]"] for r in qualified.itertuples()]
    lines = ["# 模拟校准与微状态稳定性", "", "## 历史 V3 数值复核", "",
             f"读取原 V3 总表 {history['all_v3_rows_read']:,} 行及三个对比和同向表。逐表与总表中的对应切片相符。"
             f"复算 E04 的 {history['original_primary_tests_reproduced']} 个预定主检验及 E04_R 的 {history['e04_r_cr1_reproduced']} 个 CR1 对比。"
             "V3 E04 的 OFF−HC 组成原始 P=0.0429，四项 Holm P=0.1716。V3 的四项主检验无 Holm 显著结果。"
             "V3 各方案的家族 BH、Holm 及跨方案 Holm 有完整记录。",
             "[历史主检验复算](../results/V3_Primary_Reproduction.csv) · [原校正计数](../results/V3_Archive_Threshold_Counts.csv) · [V3 统计归档](../../V3_BOLD_Microstate_Method_Optimization/results/All_Methods_Comparison.csv)", "",
             f"V3 E02 和 E04 的五折空间模板一致，已复核 40 个对应。最低 |r|={history['e04_min_abs_r']:.3f}，"
             f"最低双向余量={history['e04_min_margin']:.3f}。E04_R 的历史全流程重训只有 96/2,000 次同时满足四态识别。"
             "E09、F3 等分支也有状态匹配不全。历史固定模板区间和只在可匹配时定义的重训效应需要按各自口径解释。"
             "实际何项是假阳性无法由组间 P 或匹配失败单独判定。", "",
             "[V3 重训归档](../results/V3_Bootstrap_Archive.csv) · [V3 原统计方法](../../V3_BOLD_Microstate_Method_Optimization/reports/statistics_methods.md) · [V3 模拟偏差报告](../../V3_BOLD_Microstate_Method_Optimization/reports/Method_Bias_Report.md)", "",
             "## 峰间回填与正则化的零假设核查", "",
             f"重新计算 V3 逐 TR 与峰间回填的模拟切换拒绝率。分别在各自 500 个队列上，S0 为 {100*low.F1_TR_common:.1f}% 和 {100*low.F1_peak_common:.1f}%，"
             f"S5 为 {100*high.F1_TR_common:.1f}% 和 {100*high.F1_peak_common:.1f}%。"
             f"另逐队列读取 V4 Reliability 的 {history['formal_synthetic_cohorts']} 个归档，独立复建六项核心 Holm 家族。"
             "S5 改变观测噪声和 FD，潜在状态序列已知一致，其拒绝率衡量既定噪声机制下的敏感性。", "",
             md(["方法", "S5 高噪声家族拒绝数", "最差 S5 Wilson 上界", "互斥训练标签 ARI", "互斥训练四态全匹配"], rows), "",
             "高噪声上界须不超过 15%。四种方法均未通过。A3 与 A0 同为最低上界，A3 的互斥训练标签 ARI 和四态全匹配率较高。"
             "这是使用另一固定方案的依据。实际清洁条件下的准确性、较强噪声下的错误拒绝率和真实队列推断的可靠性分别报告。", "",
             "传统 8 项 HC 家族的重算直接使用模拟归档中的逐指标 FD 调整 CR1 P，可检查本次家族边界在历史模拟中的表现。"
             "这套重算没有生成新合成队列。当前 Welch/秩检验和 10,000 次置换的联合规则缺少另外的全流程模拟校准。", "",
             md(["情景", "方案", "指标家族", "队列数", "8项全可估计", "Holm 至少一项拒绝"], trad_rows), "",
             "[逐项重算](../results/Reliability_Traditional_Family_Recount.csv) · [核心校准复核](../results/Reliability_Core_Calibration_Recount.csv) · [V3 峰间核查](../results/V3_Peak_Null_Recount.csv)", "",
             "## A3 的模板与完整重训", "",
             f"观察五折的 40 个状态对应全部通过。最小 |r|={match.abs_correlation.min():.3f}，最小双向余量={match.bidirectional_margin.min():.3f}。"
             "使用互斥训练集检验时，15 项划分中仅 5 项四态同时通过。较高的五折相似度与独立训练重现能力有各自的含义。", "",
             f"成功运行 {check['fully_refitted']:,}/{check['draws']:,} 次全流程重训。各状态匹配次数与旧 A3 同种子审计一致。", "",
             md(["状态", "有对应的重训次数", "比例"], bootstrap_states), "",
             "本轮只有全局指标或排序组成坐标达到所有重训均有定义。具体状态及有向转换的完整重训区间均留空，部分匹配区间标记为 matched-only。"
             "下表的区间为可在全部重训中计算且不跨零的逐项描述性区间。", "",
             md(["对比", "指标", "有定义", "原单位差值", "重训 95% CI"], portions), "",
             "[完整重训结果](../results/Bootstrap_Results.csv) · [模板对应](../results/State_Matching.csv) · [四张参考图](../figures/State_Templates.png) · [历史复核日志](../results/Historical_Audit_Verification.json)", ""]
    (REPORT / "Microstate_Stability_Report.md").write_text("\n".join(lines), encoding="utf-8")


def interpretation(tests, comp, shared, boot):
    q = tests.set_index(["metric", "contrast"])
    ilr = comp.loc[comp.metric.eq("ilr")].set_index("contrast")
    switch = [(c, q.loc[("switching_rate", c)], boot.loc[boot.scheme.eq("full") & boot.metric.eq("switching_rate") &
                                                         boot.contrast.eq(c)].iloc[0]) for c in CONTRAST]
    rows = [[CONTRAST[c], f"{60*r.effect_size:+.3f}", number(r.permutation_holm_p), number(r.fd_permutation_holm_p),
             f"[{60*b.ci95_low:+.3f}, {60*b.ci95_high:+.3f}]"] for c, r, b in switch]
    shared_ok = int((shared.holm_p.lt(ALPHA) & shared.permutation_holm_p.lt(ALPHA)).sum())
    switch_off = q.loc[("switching_rate", "OFF-HC")]
    switch_on = q.loc[("switching_rate", "ON-OFF")]
    m4 = q.loc[("duration_4", "OFF-HC")]
    ng = q.loc[("gev_3", "OFF-HC")]
    dist = q.loc[("dispersion_3", "OFF-HC")]
    lines = ["# 疾病相关性与可写入论文的结论", "",
             "## 帕金森相关的结果", "",
             f"在固定 A3 模板的探索性比较中，OFF−HC 的四态 Coverage 整体组成存在差异，置换 Holm P={number(ilr.loc['OFF-HC'].permutation_holm_p)}。"
             f"M3 占有率和出现率较低，M4 占有率和平均段长较高。M4 的平均段长差 {m4.effect_size:+.3f} s，在 99 项标量联合 Holm 下仍有固定特征差异证据。"
             "这些条目描述已接受 DBS 的患者关机时与 HC 的区别。", "",
             f"OFF 与 ON 均有经校正的同方向差异的指标数为 {shared_ok}。ON−HC 的整体组成和单指标 Holm 检验没有检出。"
             "健康组对照和患者组内部开关机回答不同问题。诊断在两次患者访视之间相同，配对结果刻画该读出随扫描条件变化的程度。"
             "顺序固定的患者访视设计和噪声校准缺口影响因果解释。", "",
             "## 重训后仍可解释的整体变化", "",
             md(["对比", "切换率差（次/分）", "置换 Holm P", "FD 置换 Holm P", "2,000 轮重训 95% CI（次/分）"], rows), "",
             "探索性全局切换率是可在全部重训中计算的直接动态证据。OFF−HC 方向为减少，ON−OFF 方向为增加。"
             f"两处变化的探索全局置换 Holm P 分别为 {number(switch_off.global_permutation_holm_p)} 和 {number(switch_on.global_permutation_holm_p)}，"
             f"联合探索家族与模型后分别为 {number(switch_off.global_model_union_permutation_holm_p)} 和 {number(switch_on.global_model_union_permutation_holm_p)}。"
             "V3 E04 的两项 HC 切换率对比也未达到原主家族 Holm。当前结果的重训稳定性适用于 A3 及其时间窗。",
             "[探索全局校正](../results/Exploratory_Global_Corrections.csv) · [重训结果](../results/Bootstrap_Results.csv)", "",
             "## 可以写入论文的描述", "",
             "> 在已观察的 V3 冻结队列中，固定四态微状态的探索性再分析检出 OFF 与健康对照的 Coverage 组成差异。"
             "在不依赖具体状态编号的汇总中，OFF 相对健康对照的切换更少，ON 相对 OFF 的切换更多。"
             "两个切换率效应的全流程参与者重训区间均未跨零。高噪声校准与逐状态重训可识别性限制了机制推断。", "",
             "## 仍属探索性的发现", "",
             f"M3 GEV 差为 {100*ng.effect_size:+.2f} 个百分点，适配空间离散度差为 {dist.effect_size:+.4f}。"
             "它们在固定 A3 下有组差异，离散度刻画状态内平均模板距离，GEV 同时受状态占用和拟合程度影响。"
             "局部空间离散度和临床严重程度、发生机制之间的关系在本实验没有直接估计。",
             "单状态结果的完整重训匹配不足。它们适合作为带有条件限定的结果或后续假设。"
             "论文中的 PMS 命名与 M1–M4 的对应没有在本轮建立。ON−HC 的未检出保留为未建立等效性。", ""]
    (REPORT / "Scientific_Interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    tests, comp, shared, boot = [csv(f) for f in ("All_Scalar_Results", "Coverage_ILR_Results", "Shared_PD_Direction_Results", "Bootstrap_Results")]
    primary_report(tests, comp, shared, boot, info("Analysis_Verification"))
    stability_report(boot, info("Bootstrap_Verification"), info("Historical_Audit_Verification"))
    interpretation(tests, comp, shared, boot)
    print("Three result, stability, and interpretation reports written.")


if __name__ == "__main__":
    main()
