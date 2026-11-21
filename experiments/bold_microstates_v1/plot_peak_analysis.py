"""Draw the method comparisons and spatial correspondence from group summaries."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def plot(root):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei","Noto Sans CJK SC","SimHei","DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    stats = pd.read_csv(root / "results/paired_stats.csv")
    midpoint = pd.read_csv(root / "midpoint/paired_stats.csv").assign(variant="midpoint_spatial")
    stats = pd.concat([stats,midpoint],ignore_index=True)
    variants = ["old_signed","peak_interp","peak_direct","common24_peak","midpoint_spatial"]
    labels = ["原有符号法","选峰＋峰间插值","同模板逐帧回填","共同24对与共用模板","中点按空间图归类"]
    sources = [("105", "#006c78", "D", -.12),("SyN9", "#ac5428", "o", .12)]
    fig, axs = plt.subplots(2,2,figsize=(10.2,6.4),sharex="col",sharey=True)
    fig.subplots_adjust(left=.205,right=.97,top=.81,bottom=.10,hspace=.30,wspace=.17)
    fig.suptitle("头动相同时，M2 与 M4 的 ON−OFF 差值",y=.985,fontsize=16)
    legend = [Line2D([],[],color=c,marker=m,label=src) for src,c,m,_ in sources]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.52,.932),ncol=2,frameon=False)
    for r,state in enumerate((2,4)):
        for c,(metric,unit,factor) in enumerate([("coverage","覆盖率（百分点）",100),("duration_sec","持续时间（秒）",1)]):
            ax=axs[r,c]
            ax.axhspan(.65,1.35,color="#edf4f6",zorder=0)
            ax.axvline(0,ls="--",lw=.9,color="#666",zorder=1)
            for src,color,mark,offset in sources:
                v=stats.loc[(stats.source==src)&(stats.state==state)&(stats.metric==metric)].set_index("variant").loc[variants]
                y=np.arange(5)+offset
                ax.hlines(y,v.adjusted_ci_low*factor,v.adjusted_ci_high*factor,color=color,lw=1.6)
                ax.scatter(v.adjusted_delta*factor,y,color=color,marker=mark,s=24,zorder=3)
            ax.set(title=f"M{state}  {unit}",yticks=range(5),yticklabels=labels,ylim=(4.55,-.55))
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(axis="y",length=0,pad=7)
            if r==1: ax.set_xlabel("ON−OFF")
    fig.savefig(root/"paired_effects.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)

    matches = pd.read_csv(root/"results/spatial_matches.csv")
    fig,axs=plt.subplots(1,3,figsize=(10.8,3.95))
    fig.subplots_adjust(left=.072,right=.905,bottom=.17,top=.78,wspace=.26)
    fig.suptitle("原有符号状态与新无符号模板的空间对应",y=.99,fontsize=16)
    datasets=[("105_old_to_peak","105 独立拟合"),("SyN9_old_to_peak","SyN9 独立拟合"),
              ("SyN9_old_to_shared_peak","共用模板（以原 SyN9 为参照）")]
    for ax,(pair,title) in zip(axs,datasets):
        a=matches.loc[matches.comparison==pair].pivot(index="reference",columns="target",values="abs_r").values
        im=ax.imshow(a,vmin=0,vmax=1,cmap="viridis")
        for i,j in np.ndindex(a.shape):
            ax.text(j,i,f"{a[i,j]:.2f}",ha="center",va="center",color="white" if a[i,j]<.5 else "#111",fontsize=11)
            if i==j:
                ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor="#ffcb65",lw=2))
        ax.set(xticks=range(4),xticklabels=["M1","M2","M3","M4"],yticks=range(4),
               yticklabels=["M1","M2","M3","M4"],xlabel="新无符号模板",title=title)
        ax.tick_params(length=0)
    axs[0].set_ylabel("原有符号状态")
    cax=fig.add_axes([.92,.215,.014,.51])
    fig.colorbar(im,cax=cax,label="绝对空间相关 |r|")
    fig.savefig(root/"spatial_correspondence.png",dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--report-dir",type=Path,required=True)
    plot(cli.parse_args().report_dir)
