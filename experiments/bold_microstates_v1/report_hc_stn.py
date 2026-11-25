"""Export the 105 healthy-control contrasts and plot their participant-level intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


METRICS = {"duration_s": ("平均持续时间", "秒"), "occurrence_s": ("出现率", "次/分"),
           "coverage": ("时间占比", "百分点")}
COLORS = {"OFF": "#096971", "ON": "#a44b20"}


def verify(r, table, previous):
    assert r["data_kind"] == "real" and r["counts"] == {"HC":15, "OFF":25, "ON":25}
    assert r["hc_matched_sha256"] == 15 and r["axis"] == previous["axis"]
    original = previous["bands"]["weida_band"]
    primary = table[table.analysis == "group_templates"].set_index(["condition", "state", "metric"])
    for condition in ("OFF", "ON"):
        np.testing.assert_allclose(r["maps"][condition], original["maps"][condition], rtol=0, atol=1e-10)
        for row in original["paired"]:
            scale = {"duration_s":1, "occurrence_s":60, "coverage":100}[row["metric"]]
            obs = primary.loc[(condition, row["state"], row["metric"])]
            np.testing.assert_allclose(obs.mean_patient, row[condition.lower()]*scale, atol=1e-10)
    for _, family in table.groupby("analysis"):
        assert len(family)==28 and not family.duplicated(["condition","state","metric"]).any()
        summary = family[["mean_patient", "sd_patient", "n_patient",
                          "mean_hc", "sd_hc", "n_hc"]].to_numpy().T
        std = stats.ttest_ind_from_stats(*summary)
        np.testing.assert_allclose(std.pvalue, family.p_raw, atol=1e-12)
        welch = stats.ttest_ind_from_stats(*summary, equal_var=False)
        np.testing.assert_allclose(welch.pvalue, family.p_welch, atol=1e-12)
        by_p = family.sort_values("p_raw")
        expected = np.minimum(1, np.maximum.accumulate(by_p.p_raw.to_numpy()*np.arange(28,0,-1)))
        np.testing.assert_allclose(by_p.p_holm28, expected)
        np.testing.assert_allclose(family.delta, family.mean_patient-family.mean_hc, atol=1e-12)
        np.testing.assert_allclose((family.ci_low+family.ci_high)/2, family.delta, atol=1e-12)


def axes_style(ax):
    ax.axvline(0, color="#465764", lw=1, ls="--")
    ax.spines[["right","left","top"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", labelsize=9)


def interval(ax, row, y, mark=False):
    color = COLORS[row.condition]
    ax.errorbar(row.delta, y, xerr=[[row.delta-row.ci_low], [row.ci_high-row.delta]],
                fmt="o" if row.condition=="OFF" else "s", color=color, ms=6, capsize=3, lw=1.7)
    if mark:
        ax.annotate("*", (row.delta,y), xytext=(0,5 if row.condition=="OFF" else -13), textcoords="offset points",
                    ha="center", color=color, size=13)


def overall_plot(t, out):
    fig, axes = plt.subplots(1,2, figsize=(10.4,3.8))
    fig.subplots_adjust(left=.17,right=.97,top=.77,bottom=.26,wspace=.32)
    fig.suptitle("患者相对健康组的总体差异随模板选择改变", fontsize=16)
    for ax, metric in zip(axes, ["duration_s","occurrence_s"]):
        axes_style(ax)
        for i,method in enumerate(["group_templates","common_template"]):
            for cond,offset in [("OFF",.10),("ON",-.10)]:
                row = t[(t.analysis==method)&(t.state=="all")&(t.metric==metric)&(t.condition==cond)].iloc[0]
                interval(ax,row,1-i+offset)
        label,unit=METRICS[metric]
        ax.set(title=label if metric=="duration_s" else "总出现率", xlabel=f"患者 − 健康组（{unit}）",
               yticks=[1,0], yticklabels=["各组独立模板","三组共同模板"] if ax is axes[0] else [],
               ylim=(-.45,1.48))
    legend(fig, y=.058)
    fig.savefig(out/"overall_switching.png",dpi=165)
    plt.close(fig)


def legend(fig, y):
    handles=[Line2D([],[],marker=m,color=COLORS[c],lw=1.7,label=label)
             for c,m,label in [("OFF","o","关机 − 健康"),("ON","s","开机 − 健康")]]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.50,y),ncol=2,frameon=False,fontsize=9)


def states_plot(t, out):
    fig, axes = plt.subplots(1,3,figsize=(11.6,4.7))
    fig.subplots_adjust(left=.065,right=.98,top=.83,bottom=.27,wspace=.34)
    fig.suptitle("四个 BOLD 微状态的患者与健康组差异",fontsize=16)
    for ax, (metric,(title,unit)) in zip(axes,METRICS.items()):
        axes_style(ax)
        for i,state in enumerate(["M1","M2","M3","M4"]):
            for condition,offset in [("OFF",.11),("ON",-.11)]:
                row=t[(t.analysis=="group_templates")&(t.state==state)&(t.metric==metric)&(t.condition==condition)].iloc[0]
                interval(ax,row,3-i+offset,mark=row.p_raw<.05)
        ax.set(title=title,xlabel=f"患者 − 健康组（{unit}）",yticks=[3,2,1,0],
               yticklabels=["M1","M2","M3","M4"],ylim=(-.5,3.55))
    legend(fig,y=.075)
    fig.text(.5,.038,"点为组均值差，横线为逐项 95% 置信区间。* 表示原始 p < 0.05。",ha="center",fontsize=9)
    fig.savefig(out/"state_differences.png",dpi=165)
    plt.close(fig)


def export(path):
    out=path.parent
    r=json.loads(path.read_text(encoding="utf8"))
    previous=json.loads((out.parent/"eeg_replacement_105/results.json").read_text(encoding="utf8"))
    t=pd.DataFrame(r["comparisons"])
    verify(r,t,previous)
    for name,part in t.groupby("analysis",sort=False):
        part.drop(columns="analysis").to_csv(out/(name+".csv"),index=False)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if f in fonts),None)
    if not font:
        raise RuntimeError("Chinese font required to render figures")
    plt.rcParams["font.family"]=[font]
    overall_plot(t,out)
    states_plot(t,out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path)
    export(parser.parse_args().results)
