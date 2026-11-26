"""Aggregate release figures and complete private plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavior_protocol import BEHAVIORS, PRIVATE, ROOT, STATES

NAMES = dict(total="Motor total", rigidity="Rigidity", bradykinesia="Bradykinesia",
             tremor="Tremor", gait_balance="Gait and balance")
CN = dict(total="运动总分", rigidity="强直", bradykinesia="运动徐缓", tremor="震颤", gait_balance="步态与平衡")
NAVY, TEAL, GRAY = "#182e45", "#087d79", "#687789"


def num(x, digits=4):
    return f"{x:.{digits}g}" if np.isfinite(x) else "NA"


def style(ax, axis="y"):
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color="#e6edf3", linewidth=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b7c5d3")


def save(fig, filename, private):
    out = (PRIVATE if private else ROOT) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / filename, dpi=145, facecolor="white")
    plt.close(fig)


def plot_clinical(frame, scores, private):
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7.1), layout="constrained")
    fig.suptitle("Motor scores in paired DBS visits", x=.02, ha="left", fontsize=17, fontweight="bold", color=NAVY)
    for ax, r in zip(axs.ravel()[:5], scores.itertuples(), strict=True):
        off, on = (frame[f"{r.endpoint}_{condition}"].to_numpy(float) for condition in ("off", "on"))
        keep = np.isfinite(off) & np.isfinite(on)
        off, on = off[keep], on[keep]
        if private:
            for a, b in zip(off, on, strict=True):
                ax.plot([0, 1], [a, b], "o-", lw=.75, ms=3, color=TEAL, alpha=.3)
        elif len(off) >= 5:
            for i, group in enumerate((off, on)):
                q1, q3 = np.percentile(group, [25, 75])
                ax.vlines(i, q1, q3, color="#aed4d0", lw=8)
        ax.plot([0, 1], [r.off_mean, r.on_mean], "o-", color=NAVY, lw=2)
        ax.set(xticks=[0, 1], xticklabels=["OFF", "ON"], xlim=(-.37, 1.37), ylabel="Score points")
        ax.set_title(f"{NAMES[r.endpoint]} · n={r.n}\nBH5 q={num(r.fdr_q, 3)}",
                     loc="left", color=NAVY, fontweight="semibold")
        style(ax)
    ax = axs.ravel()[5]
    y = np.arange(len(BEHAVIORS))
    ax.hlines(y, scores.ci95_low, scores.ci95_high, lw=1.7, color=TEAL)
    ax.plot(scores.mean_change, y, "o", color=NAVY)
    ax.axvline(0, color=GRAY, lw=.85)
    ax.set(yticks=y, yticklabels=[NAMES[m] for m in BEHAVIORS], xlabel="OFF − ON, points")
    ax.invert_yaxis()
    ax.set_title("Mean paired improvement and 95% CI", loc="left", color=NAVY, fontsize=10)
    style(ax, "x")
    save(fig, "Clinical_Paired_Changes_Individual.png" if private else "Clinical_Paired_Changes.png", private)


def plot_association(frame, results, private):
    fig, axs = plt.subplots(4, 5, figsize=(17.2, 12.1), layout="constrained")
    fig.suptitle("BOLD duration changes and motor improvement", x=.02, ha="left",
                 fontsize=18, fontweight="bold", color=NAVY)
    for i, state in enumerate(STATES):
        for j, behavior in enumerate(BEHAVIORS):
            ax = axs[i, j]
            r = results.loc[results.state.eq(state) & results.behavior.eq(behavior)].iloc[0]
            x = frame[f"duration_{i+1}_change"].to_numpy(float)
            y = frame[f"{behavior}_improvement"].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            observed = np.column_stack((x[keep], y[keep]))
            if private:
                ax.scatter(observed[:, 0], observed[:, 1], color=TEAL, alpha=.68, s=16, linewidths=0)
            elif len(observed) >= 5:
                order = np.lexsort((observed[:, 1], observed[:, 0]))
                centroids = [g.mean(axis=0) for g in np.array_split(observed[order], min(5, len(observed)//5))]
                ax.scatter(*np.asarray(centroids).T, s=26, color=TEAL)
            ax.axhline(0, lw=.6, color=GRAY)
            ax.axvline(0, lw=.6, color=GRAY)
            ax.set_title(f"{state} · {NAMES[behavior]}\nρ={num(r.rho, 3)}   BH20 q={num(r.fdr_q, 3)}   n={r.n}",
                         loc="left", color=NAVY, fontsize=9)
            if i == 3:
                ax.set_xlabel("Duration ON − OFF (s)")
            if j == 0:
                ax.set_ylabel("Score OFF − ON (points)")
            style(ax)
    save(fig, "Duration_Motor_Associations_Individual.png" if private else "Duration_Motor_Associations.png", private)


def association_summary(frame):
    return "、".join(f"{r.state} × {CN[r.behavior]}（ρ={num(r.rho)}，q={num(r.fdr_q)}）"
                    for r in frame.loc[frame.passes_fdr].itertuples()) or "无"


def audit_report(clinical_audit, imaging_audit, flow):
    import json

    lines = [
        "# 数据与人工评分条目核查", "",
        f"确认 STN 临床配对 {flow['clinical_reference']} 对，冻结影像配对 {flow['imaging']} 对，"
        f"患者与两扫描键精确连接 {flow['main']} 对，日期核实 {flow['date_verified']} 对，"
        f"固定质量窗匹配 {flow['quality']} 对。",
        "", "第一评分文件的实际纳入患者工作表按双层表头确认 33 个人工条目。"
        "Q:R、AA:AB、AE 为预测列，AI 为全身自发动作人工评分。AT 只用于对账。"
        "第二评分文件复核同一身份和访视的人工条目及旧派生结果。人口学工作簿与既有身份链确认 STN 靶点。",
        "", "两份评分表的 54 次 STN 访视逐项一致。旧 AT 在 7 人的 14 次访视中多计 4–15 分，"
        "差额全部来自排除的预测列。第二评分文件的旧运动徐缓小计混入肢体强直，49 次访视与重算值不同。"
        "主样本中 6 人震颤 OFF 为零，仅该改善率留空。",
        "", "原表运动检查总分，33 项重算，正式版本待核实。人工条目范围 0–4，分值增大对应运动损害加重。"
        "步态与平衡仅为步态、步态冻结、姿势平稳度之和。分析未按工作表名称或这三项划分患者亚型。",
        "", "[评分方向参考](https://mds.movementdisorders.org/UserFiles/New%20UPDRS%207%203%2008%20final.pdf) · "
        "[条目字典](../results/Score_Item_Dictionary.csv) · "
        "[既有评分核查](../../BOLD_Microstate_STN_Original_Scores/reports/STN_Original_Score_Audit.md)",
        "", "## 临床来源核查汇总", "", "\x60\x60\x60json",
        json.dumps(clinical_audit, ensure_ascii=False, indent=2), "\x60\x60\x60",
        "", "## 影像源与质量核查汇总", "", "\x60\x60\x60json",
        json.dumps(imaging_audit, ensure_ascii=False, indent=2), "\x60\x60\x60",
    ]
    (ROOT / "reports/Data_and_Score_Audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def report(scores, duration, main, date, quality, flow):
    clinical_names = "、".join(CN[key] for key in scores.loc[scores.passes_fdr & scores.mean_change.gt(0), "endpoint"]) or "无"
    durations = "、".join(f"{r.endpoint}（{num(r.mean_change)} 秒）"
                         for r in duration.loc[duration.passes_fdr].itertuples()) or "无"
    significant = main.loc[main.passes_fdr]
    dims = "、".join(dict.fromkeys(CN[x] for x in significant.behavior))
    lines = [
        "# STN 患者 DBS ON–OFF 运动行为变化及 BOLD 微状态关联", "",
        f"主分析 {flow['main']} 对患者。运动改善量为 OFF−ON，持续时间变化为 ON−OFF。", "",
        f"1. **ON 时改善的评分**　{clinical_names}通过本轮 BH5。",
        f"2. **持续时间变化**　{durations}通过本轮 BH4。括号内为平均 ON−OFF。",
        f"3. **行为与持续时间的关联**　主分析 BH20 通过项 {len(significant)}/20。"
        + (f"{association_summary(main)}。" if len(significant) else
           f"最小 q={num(main.fdr_q.min())}。"),
        f"4. **对应的症状维度**　{('通过校正的改善指标为' + dims + '。') if dims else '总分和各分项均未获得校正后的关联证据。'}",
        f"5. **固定敏感性**　日期核实子集（{flow['date_verified']} 对）的 BH20 通过项为{association_summary(date)}。"
        f"头动质量窗（{flow['quality']} 对）通过项为{association_summary(quality)}。主分析以原冻结共同窗为准。",
        "", "[数据与条目核查](Data_and_Score_Audit.md) · [冻结方案](../ANALYSIS_PLAN.md) · [运行清单](../results/Run_Manifest.json)",
        "", "## 临床与持续时间配对结果", "",
        "| 指标 | n | OFF 均值 | ON 均值 | 平均变化 [95% CI] | 检验 | 原始 P | BH q |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    labels = dict(paired_t="配对 t", Wilcoxon_signed_rank="Wilcoxon 符号秩", all_zero="全零差", NA="NA")
    for table in (scores, duration):
        for r in table.itertuples():
            name = CN.get(r.endpoint, r.endpoint + " 持续时间（秒）")
            lines.append(f"| {name} | {r.n} | {num(r.off_mean)} | {num(r.on_mean)} | "
                         f"{num(r.mean_change)} [{num(r.ci95_low)}, {num(r.ci95_high)}] | "
                         f"{labels[r.test]} | {num(r.raw_p)} | {num(r.fdr_q)} |")
    lines += [
        "", "总分采用原表运动检查总分，33 项重算，正式版本待核实。高分对应更重的损害。"
        "改善率只作补充，OFF=0 时记为 NA。均值差区间为 9,999 次患者整对 Bootstrap 百分位。"
        "显著性依据各检验家族的 q 值。图中连线连接同一批患者的条件均值，浅色条为条件内四分位距。",
        "", "![临床配对图](../figures/Clinical_Paired_Changes.png)", "",
        "[临床配对结果](../results/Clinical_Paired_Results.csv) · [持续时间结果](../results/Duration_Paired_Results.csv) · "
        "[临床全样本描述](../results/Clinical_Reference_Descriptives.csv)",
        "", "## 20 项预定 Spearman 与固定敏感性", "",
        "ρ<0 表示持续时间缩短越多，改善越大。ρ>0 表示持续时间增加越多，改善越大。"
        "四态乘五项全部进入同一个 BH20 家族。日期和头动敏感性各用同一清单独立校正。",
        "", "| 状态 | 改善指标 | n | ρ | 原始 P | BH20 q | 日期 ρ / q | 质量 ρ / q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r, day, fd in zip(main.itertuples(), date.itertuples(), quality.itertuples(), strict=True):
        if (r.state, r.behavior) != (day.state, day.behavior) or (r.state, r.behavior) != (fd.state, fd.behavior):
            raise ValueError("Sensitivity rows do not match the main slots")
        lines.append(f"| {r.state} | {CN[r.behavior]} | {r.n} | {num(r.rho)} | {num(r.raw_p)} | "
                     f"{num(r.fdr_q)} | {num(day.rho)} / {num(day.fdr_q)} | {num(fd.rho)} / {num(fd.fdr_q)} |")
    lines += [
        "", "共享散点按时间差排序，每点平均至少五人。标注的 ρ 和 q 从患者级数据计算。"
        "私有图保留完整散点和临床逐人配对轨迹。",
        "", "![关联散点图](../figures/Duration_Motor_Associations.png)", "",
        "[主关联全表](../results/Duration_Motor_Associations.csv) · [固定敏感性全表](../results/Association_Sensitivities.csv)",
        "", "## 文献和既往检验", "",
        "本地《脑电微状态》是 Michel 与 Bréchet 的综述。Guo 等先评估 EEG 微状态的测量可靠性，"
        "再报告 D 持续时间与 UPDRS 正相关。本轮以此设计全四态的配对探索。"
        "冻结 BOLD A3 状态的持续时间以秒计。EEG D 和 BOLD M4 的对应关系未建立。"
        " [综述](https://pubmed.ncbi.nlm.nih.gov/42103554/) · [Guo 等](https://pubmed.ncbi.nlm.nih.gov/38531478/)",
        "", "V4 已检验同一队列的四态持续时间，原始检验的 BH4 支持 M2 和 M4 缩短，q 分别为 .04375 和 .006415。"
        "另一项历史分析已用 Spearman 检验 M4 持续时间与总分变化，ρ=−.279，原始 P=.1774，置换 Holm4=.5612、BH4=.3742。"
        "V8 日期不限制回归曾检验含 OFF 基线及 ΔFD 的 M4 持续时间与运动改善。"
        "总分 P=.6942、Holm3=1，震颤 P=.07649、BH9=.56820、Holm12=.9179。"
        "V9 的 M1/M4 连续表达与总分改善调整检验 Holm2 为 .51245 和 .44036，联合 Holm14 均为 1。"
        "本轮在同队列复核方法并扩展症状维度，原历史家族校正仍按各报告保留。"
        " [V4 完整表](../../V4_Microstate_Statistical_Significance/results/All_Scalar_Results.csv) · "
        "[早期同队列行为分析](../../BOLD_Microstate_Behavior_Association/reports/BOLD_Microstate_Behavior_Association_Report.md) · "
        "[V8 日期不限制分析](../../BOLD_Microstate_STN_Original_Scores/reports/STN_Date_Unrestricted_Associations.md) · "
        "[V9 解释](../../BOLD_Microstate_Expression_Followup/reports/Final_Interpretation.md)",
        "", "24 对可核实的扫描均先 ON 后 OFF，另 1 对顺序未知。"
        "评分记录两种 DBS 条件均为药物 OFF。主样本有 2 对日期记录冲突、1 对日期证据不足，仍保留已确认的身份和 DBS 条件。"
        "本轮结果解释限于 ON/OFF 条件相关变化和指标间关联。",
    ]
    (ROOT / "reports/Behavior_ON_OFF_Report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
