"""Draw the clinical plots from shareable aggregate tables only."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


COLORS = ["#21657a", "#b65b36", "#417452", "#67518e"]
METRICS = {"coverage":"占比", "occurrence_per_min":"出现率", "duration_sec":"持续时间"}


def forest(ax, rows, heading):
    rows = rows.reset_index(drop=True)
    labels = []
    for i, row in enumerate(rows.itertuples()):
        ax.plot([row.ci_low,row.ci_high],[i,i], color=COLORS[row.state-1], lw=2.4, solid_capstyle="round")
        ax.scatter(row.estimate, i, color=COLORS[row.state-1], s=38, zorder=3)
        labels.append(f"M{row.state}  {METRICS[row.metric]}")
    ax.axvline(0, color="#7a8389", lw=1, ls="--")
    ax.set(yticks=range(len(rows)), yticklabels=labels, ylim=(len(rows)-.45,-.55),
           xlim=(-1,1), xticks=[-1,-.5,0,.5,1], xlabel="Spearman ρ")
    ax.set_title(heading, loc="left", fontsize=13, pad=13)
    ax.grid(axis="x", color="#e5e7e8", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)


def draw(report):
    stats = pd.read_csv(report/"stats.csv")
    main = stats[stats.scenario.eq("archive_main")]
    severity = main[main.family.eq("clinical_off_duration")]
    change = main[main.family.eq("clinical_change")]
    names = {font.name for font in font_manager.fontManager.ttflist}
    choices = [name for name in ("Microsoft YaHei","Noto Sans CJK SC","Source Han Sans SC","SimHei") if name in names]
    if not choices:
        raise ValueError("A Chinese font is required to render the report figures")
    with plt.rc_context({"font.family":[choices[0],"DejaVu Sans"], "axes.spines.top":False,
                         "axes.spines.right":False,"figure.facecolor":"white","axes.unicode_minus":False}):
        fig, axes = plt.subplots(1,2,figsize=(10.6,5.4),layout="constrained",gridspec_kw={"wspace":.1})
        forest(axes[0], severity, "关机时持续时间与运动总分")
        forest(axes[1], change, "脑状态变化与运动改善分数")
        fig.suptitle("24 名 STN 患者的临床关联及 95% 区间",fontsize=16)
        fig.savefig(report/"clinical_correlations.png",dpi=185)
        plt.close(fig)
        bins = pd.read_csv(report/"severity_bins.csv")
        if len(bins)!=12 or bins.n.min()<5:
            raise ValueError("Invalid plot disclosure groups")
        fig, axes = plt.subplots(2,2,figsize=(7.6,5.8),sharex=True,sharey=True,layout="constrained")
        for state,ax in enumerate(axes.flat,1):
            s=bins[bins.state.eq(state)].sort_values("mean_duration_sec")
            ax.scatter(s.mean_duration_sec,s.mean_archived_score,color=COLORS[state-1],s=60)
            ax.set_title(f"M{state}",loc="left",fontweight="bold",color=COLORS[state-1])
            ax.grid(alpha=.18)
            ax.set_axisbelow(True)
            if state in (1,3):
                ax.set_ylabel("组均 UPDRS3（分）")
            if state in (3,4):
                ax.set_xlabel("组均 OFF 持续时间（秒）")
        fig.suptitle("各状态按 OFF 持续时间三等分，每点 8 人",fontsize=15)
        fig.savefig(report/"off_duration_group_means.png",dpi=185)
        plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    draw(parser.parse_args().report)
