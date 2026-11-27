"""Render the Chinese release report from the two fixed tables."""

import numpy as np

from round3_io import CFG

LABELS = {"total":"运动总分", "rigidity":"强直", "bradykinesia":"运动徐缓", "tremor":"震颤",
          "gait_balance":"三项步态与平衡"}
EVIDENCE = {"fdr_supported":"通过 BH20", "nominal_only":"仅名义 P<0.05", "no_evidence":"证据不足",
            "unestimable":"不可估计"}


def num(x):
    return f"{x:.6g}" if np.isfinite(x) else "NA"


def consistency(a, b):
    if not (np.isfinite(a) and np.isfinite(b)):
        return "不可比较"
    if a == 0 or b == 0:
        return "零相关" if a == b else "一项为零"
    return "同向" if np.sign(a) == np.sign(b) else "反向"


def detail(rate, absolute):
    return (f"{rate.state} × {LABELS[rate.clinical_endpoint]}（ρ={num(rate.rho)}，"
            f"原始 P={num(rate.raw_p)}，q={num(rate.fdr_q)}，n={rate.n}。"
            f"同样本绝对减分 ρ={num(absolute.rho)}，{consistency(rate.rho, absolute.rho)}）")


def write_report(path, main, secondary, coverage, eligibility):
    a_rows = list(main.itertuples(index=False))
    b_rows = list(secondary.itertuples(index=False))
    paired_rows = list(zip(a_rows, b_rows, strict=True))
    main_pass = [(a, b) for a, b in paired_rows if a.passes_fdr]
    main_nominal = [(a, b) for a, b in paired_rows if a.evidence == "nominal_only"]
    comparisons = [consistency(a.rho, b.rho) for a, b in paired_rows]
    n_pair = int(coverage.n_pairs.iloc[0])
    tremor = eligibility.set_index("clinical_endpoint").loc["tremor"]
    framing = (
        f"**主分析 {len(main_pass)}/20 项通过 BH-FDR。** "
        + ("、".join(detail(a, b) for a, b in main_pass) + "。"
           if main_pass else "四态与五项改善率均未获得校正后的关联证据。")
    )
    lines = [
        "# DBS ON–OFF 状态占用率变化与运动改善率的关联", "",
        framing, "",
        "正 ρ 表示占用增加越多、改善率越高。负 ρ 表示占用减少越多、改善率越高。"
        f"同样本绝对减分对照中，{comparisons.count('同向')} 格同向、{comparisons.count('反向')} 格反向、"
        f"{comparisons.count('零相关')+comparisons.count('一项为零')} 格涉及零相关、"
        f"{comparisons.count('不可比较')} 格不可比较。对照另行 BH20，通过 {int(secondary.passes_fdr.sum())}/20 项。",
        "",
    ]
    if main_nominal:
        lines += ["仅达到名义 P<.05 的条目为 " + "、".join(detail(a, b) for a, b in main_nominal)
                  + "。这些条目未通过主分析 BH20。", ""]
    elif not main_pass:
        lines += [f"主分析原始 P<.05 也为 0/20。最小原始 P={num(main.raw_p.min())}，"
                  f"最小 q={num(main.fdr_q.min())}。", ""]
    lines += [
        f"冻结队列共 {n_pair} 名 STN 患者。震颤改善率纳入 {int(tremor.n_valid_rate)} 人，"
        f"{int(tremor.n_off_zero)} 人 OFF 评分为零。其绝对减分在 {int(tremor.n_valid_absolute)} 人中可计算，"
        f"同样本对照仍限制在与改善率相同的 {int(tremor.n_valid_rate)} 人。HC 不进入相关。",
        "",
        "## 主分析完整结果", "",
        "| 状态 | 运动终点 | n | ρ | 原始 P | BH20 q | 证据 | 同样本绝对减分 ρ | 方向比较 |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for a, b in paired_rows:
        if (a.state, a.clinical_endpoint, a.n) != (b.state, b.clinical_endpoint, b.n):
            raise ValueError("Main and secondary result slots differ")
        lines.append(f"| {a.state} | {LABELS[a.clinical_endpoint]} | {a.n} | {num(a.rho)} | {num(a.raw_p)} | "
                     f"{num(a.fdr_q)} | {EVIDENCE[a.evidence]} | {num(b.rho)} | {consistency(a.rho, b.rho)} |")
    lines += [
        "", "![覆盖率变化与运动改善率主分析](../figures/Coverage_Clinical_Rate_Heatmap.png)", "",
        "[主分析 CSV](../results/Coverage_Clinical_Rate_Associations.csv) · "
        "[绝对减分对照 CSV](../results/Coverage_Clinical_Absolute_Associations.csv) · "
        "[临床改善率可用人数](../results/Clinical_Rate_Eligibility.csv)",
        "", "## Coverage 描述与方法", "",
        "| 状态 | OFF 均值（%） | ON 均值（%） | ON−OFF 均值（百分点） | 增加 / 不变 / 减少人数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        lines.append(f"| {row.state} | {100*row.off_mean:.2f} | {100*row.on_mean:.2f} | "
                     f"{100*row.delta_mean:+.2f} | {row.n_increased} / {row.n_unchanged} / {row.n_decreased} |")
    lines += [
        "",
        "Coverage 取同一有效窗内四态各自的有效时间比例。四态合计为 1，每对患者四态差值和为 0，"
        "因此状态变化反映相对时间分配，网络激活或抑制需要另外的测量证据。M1–M4 保留 A3 原编号。",
        "",
        "沿用 AAL78、K=4、TR=2.1 秒、0.01–0.08 Hz、N0/G1/S1、λ=0.04、原共同窗及固定五折。"
        "每名患者以患者键和两侧扫描键匹配。占用差为 ON−OFF，绝对减分为 OFF−ON，改善率为绝对减分除以 OFF。"
        "分母为零的该项改善率保留 NA，其他终点以及零改善和恶化个体照常保留。",
        "",
        "沿用 WEiDA 的双侧 Spearman＋BH-FDR 关联框架。本轮预设四态 × 五项构成主 BH20，"
        "绝对减分对照用同样本并各自校正 20 项。采用标准并列秩。样本不足 3 或变量恒定时保留 NA，"
        "校正运算中该槽按 P=1 计入。总分沿用 33 项人工条目重算，正式量表版本待核实。"
        "步态、步态冻结及姿势平稳度三项组成步态与平衡探索性合计。既有临床比较和持续时间关联沿用其原校正范围。",
        "",
        "这是同一队列下的探索性延伸。组均值占用变化和临床改善各自描述条件差异，"
        "相关必须由患者内两项变化共同判定。既有审计中 24 对可核实患者均先 ON 后 OFF，另 1 对顺序未知。",
        "",
        "[四态 Coverage 描述](../results/Coverage_Change_Summary.csv) · "
        "[第三轮方案](../ANALYSIS_PLAN.md) · [固定检验清单](../configs/test_registry.csv) · "
        "[运行清单](../results/Round3_Run_Manifest.json) · "
        "[原临床与持续时间报告](../../reports/Behavior_ON_OFF_Report.md) · "
        "[第二轮报告](../../round2/reports/ON_OFF_Round2_Report.md) · "
        "[WEiDA 既有统计规程](../../../V4_WEiDA_Statistical_Framework_Microstate/reports/WEIDA_STATISTICAL_PROTOCOL.md)",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
