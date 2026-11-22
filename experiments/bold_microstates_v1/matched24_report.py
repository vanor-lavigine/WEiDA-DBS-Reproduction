"""Export only aggregate tables and plot prior, matched, and deletion estimates."""

import argparse
import json
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from compare_legacy_105 import REPORTS
from run import new_output
from same_band_analysis import SOURCES


def plot(out,current,loo):
    for name in ("Microsoft YaHei","Noto Sans CJK SC","SimHei"):
        if name in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [name,"DejaVu Sans"]
            break
    else:
        raise RuntimeError("Install a Chinese font to render the report figure")
    fig,axs = plt.subplots(3,1,figsize=(9.3,7.6),height_ratios=(.85,1,1),layout="constrained")
    prior = pd.read_csv(REPORTS/"same_band/results/associations.csv").query("band=='primary' and state==1 and metric=='duration_sec'").set_index("source")
    colors = {"105":"#007B83","SyN9":"#B34F1F"}
    ax=axs[0]
    for si,src in enumerate(SOURCES):
        for j,df in enumerate((prior,current)):
            row = df.loc[src]
            ax.errorbar(row.partial_r,j+(si-.5)*.24,
                        xerr=[[row.partial_r-row.ci_low],[row.ci_high-row.partial_r]],
                        fmt=("o","s")[si],color=colors[src],ms=6,lw=1.5,capsize=3,label=src if j==0 else None)
    ax.set_yticks([0,1],["上轮各自拟合","本轮共享模板"])
    ax.set_ylim(1.55,-.55)
    ax.set_xlim(-1,1)
    ax.axvline(0,color="#525A60",lw=.8)
    ax.set_xlabel("控制 ΔFD 的偏 Spearman ρ（线段为逐项 95% 区间）")
    ax.legend(loc="upper right",ncol=2,frameon=False)
    ax.set_title("ΔS2 局部亚稳态与 ΔM1 平均持续时间",loc="left",fontweight="bold",fontsize=13)
    for ax,mode,title in zip(axs[1:],("fixed","refit"),("固定共享模板，每次仅从统计中剔除一人","每次先剔除一人，再拟合两种共享模板并重算关联")):
        part = loo[loo["mode"]==mode].sort_values("fold")
        pos = np.arange(len(part))+1
        for i,(src,col) in enumerate(zip(SOURCES,("r_105","r_syn9"))):
            ax.plot(pos+(i-.5)*.17,part[col],linestyle="none",marker=("o","s")[i],ms=4.8,
                    color=colors[src],label=src)
            ax.axhline(current.loc[src,"partial_r"],color=colors[src],ls="--",lw=1,alpha=.8)
        ax.axhline(0,color="#525A60",lw=.7)
        ax.set_ylim(-.85,.65)
        ax.set_xlim(.3,24.7)
        ax.set_ylabel("偏 Spearman ρ")
        ax.set_title(title,loc="left",fontweight="bold",fontsize=11)
        ax.set_xticks([1,4,8,12,16,20,24],["J01","J04","J08","J12","J16","J20","J24"])
        ax.grid(axis="y",alpha=.15)
    axs[2].set_xlabel("每点为剔除一位患者后的组相关，虚线为全样本估计")
    for ax in axs:
        ax.spines[["right","top"]].set_visible(False)
    fig.savefig(out/"m1_influence.png",dpi=175,bbox_inches="tight",facecolor="white")
    plt.close(fig)


def run(cache,destination):
    full = pd.read_csv(cache/"associations.csv")
    current = full.query("state==1 and metric=='duration_sec'").set_index("source")
    names = [f"J{k:02d}" for k in range(1,25)]
    fixed = pd.read_csv(cache/"fixed.csv")
    refit = pd.concat([pd.read_csv(cache/f"refit_{name}.csv") for name in names])
    families = pd.concat([fixed,refit]).sort_values(["mode","fold","source","state","metric"])
    assert families.groupby(["mode","fold","source"]).size().eq(12).all() and len(families)==1152
    assert not families.duplicated(["mode","fold","source","state","metric"]).any()
    focus = families.query("state==1 and metric=='duration_sec'")
    loo = focus[focus.source=="105"][["fold","mode","n_pairs","partial_r","p_holm"]].merge(
        focus[focus.source=="SyN9"][["fold","mode","n_pairs","partial_r","p_holm"]],
        on=["fold","mode","n_pairs"],validate="one_to_one",suffixes=("_105","_syn9"))
    loo = loo.rename(columns={"partial_r_105":"r_105","partial_r_syn9":"r_syn9"})
    loo["gap_syn9_minus_105"] = loo.r_syn9-loo.r_105
    assert len(loo)==48
    focal = pd.read_csv(cache/"focal_gap.csv").iloc[0]
    summary = []
    for mode,group in loo.groupby("mode"):
        for key in ("r_105","r_syn9","gap_syn9_minus_105"):
            delta = abs(group[key]-focal[key])
            q = dict(mode=mode,quantity=key,min=group[key].min(),max=group[key].max(),
                     median=group[key].median(),negative=int((group[key]<0).sum()),positive=int((group[key]>0).sum()),
                     largest_absolute_change=delta.max(),largest_change_fold=group.loc[delta.idxmax(),"fold"])
            if key!="gap_syn9_minus_105":
                p = group["p_holm_"+key[2:]]
                q.update(holm_below_05=int((p<.05).sum()),minimum_holm=p.min(),maximum_holm=p.max())
            summary.append(q)
    quality,matches = [],[]
    for name in ["all",*names]:
        qc = json.loads((cache/f"qc_{name}.json").read_text(encoding="utf8"))
        matches.extend(dict(fold=name,**r) for r in qc.pop("matches"))
        quality.append(qc)
    out = new_output(destination/"results")
    for f in ("audit.json","verification.json","associations.csv","focal_gap.csv","cooccurrence.csv",
              "shared_templates.tsv","descriptive_bridge.csv","previous_definition_loo.csv","template_stability.csv"):
        copyfile(cache/f,out/f)
    for name,frame in (("leave_one_out",loo),("influence_summary",pd.DataFrame(summary)),("leave_one_out_families",families),
                       ("fold_quality",pd.DataFrame(quality)),("template_matching",pd.DataFrame(matches))):
        frame.to_csv(out/(name+".csv"),index=False,na_rep="NA")
    plot(destination,current,loo)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    run(args.cache,args.destination)
