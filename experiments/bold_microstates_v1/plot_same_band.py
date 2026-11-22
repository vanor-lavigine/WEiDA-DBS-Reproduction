"""Plot patient-weighted co-occurrence excesses and paired change associations."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

BANDS = [("primary","0.04–0.07 Hz"),("slow5","slow-5")]
METRICS = [("duration_sec","持续时间"),("occurrence_per_min","出现率"),("coverage","覆盖率")]


def plot(report):
    for family in ("Microsoft YaHei","Noto Sans CJK SC","SimHei"):
        if family in {font.name for font in font_manager.fontManager.ttflist}:
            plt.rcParams["font.family"] = family
            break
    results = report/"results"
    cells = pd.read_csv(results/"state_pairs.csv")
    fig, axes = plt.subplots(2,2,figsize=(8.1,7.7),layout="constrained")
    colors = LinearSegmentedColormap.from_list("joint",["#24617a","#f9f8f4","#ae4824"])
    for row,(band,title) in enumerate(BANDS):
        for col,source in enumerate(("105","SyN9")):
            ax = axes[row,col]
            data = cells[(cells.source==source)&(cells.band==band)].pivot(index="micro",columns="weida",values="excess_pp")
            im = ax.imshow(data,cmap=colors,vmin=-4,vmax=4)
            for (i,j),value in np.ndenumerate(data.to_numpy()):
                ax.text(j,i,f"{value:+.2f}",ha="center",va="center",fontsize=10,
                        color="white" if abs(value)>2.5 else "#182c36")
            ax.set(xticks=range(4),xticklabels=["S1","S2","S3","S4"],yticks=range(4),
                   yticklabels=["M1","M2","M3","M4"],title=f"{source}  ·  {title}")
            ax.tick_params(length=0)
            if row==1:
                ax.set_xlabel("WEiDA 相位状态")
            if col==0:
                ax.set_ylabel("BOLD 幅度状态")
            for spine in ax.spines.values():
                spine.set_visible(False)
    bar = fig.colorbar(im,ax=axes.ravel().tolist(),orientation="horizontal",shrink=.68,aspect=36,pad=.05)
    bar.set_label("实际共现时间占比 − 循环移位期望（百分点）")
    fig.savefig(report/"cooccurrence.png",dpi=175,bbox_inches="tight")
    plt.close(fig)

    associations = pd.read_csv(results/"associations.csv").set_index(["source","band","state","metric"])
    fig, axes = plt.subplots(1,4,figsize=(12,5.8),sharey=True,layout="constrained")
    order = [(state,metric) for state in range(1,5) for metric,_ in METRICS]
    labels = [f"M{state}  {label}" for state in range(1,5) for _,label in METRICS]
    for ax,(source,(band,title)) in zip(axes,[(s,b) for b in BANDS for s in ("105","SyN9")]):
        ax.axvline(0,color="#82939b",lw=1)
        for i,(state,metric) in enumerate(order):
            r = associations.loc[source,band,state,metric]
            color = "#af461d" if r.p_holm<.05 else "#34627a"
            ax.plot([r.ci_low,r.ci_high],[i,i],color=color,lw=1.8)
            ax.plot(r.partial_r,i,marker="D" if r.p_holm<.05 else "o",ms=5,color=color)
        for y in (2.5,5.5,8.5):
            ax.axhline(y,color="#d8dfe0",lw=.8)
        ax.set(title=f"{source}  ·  {title}",xlim=(-1,1),xticks=[-1,-.5,0,.5,1],ylim=(11.7,-.7))
        ax.set_yticks(np.arange(12),labels)
        ax.tick_params(length=0,pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.supxlabel("控制 ΔFD 的偏 Spearman 相关（线段为逐项 95% 区间，菱形表示 Holm p < 0.05）",fontsize=10)
    fig.savefig(report/"associations.png",dpi=165,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report",type=Path,required=True)
    plot(cli.parse_args().report)
