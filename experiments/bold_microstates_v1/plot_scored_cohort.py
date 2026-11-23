"""Render the scored-cohort clinical figure using group-level results."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(csv_file, target):
    tests=pd.read_csv(csv_file)
    main=tests[tests.scenario.eq("archive_main")]
    metrics=["duration_sec","occurrence_per_min","coverage"]
    fig,(severity,change)=plt.subplots(1,2,figsize=(11.5,6.1),gridspec_kw={"width_ratios":[1.38,1]})
    fig.subplots_adjust(left=.12,right=.95,top=.83,bottom=.21,wspace=.29)
    labels=[]
    image=np.empty((8,3))
    for i,(method,prefix,color) in enumerate((("AAHC","A","#087867"),("signed","F","#475e9e"))):
        for j in range(4):
            y=4*i+j
            subset=main[main.method.eq(method)&main.state.eq(j+1)]
            row=subset[subset.family.eq("clinical_off_duration")].iloc[0]
            severity.plot([row.ci_low,row.ci_high],[y,y],color=color,linewidth=1.8)
            severity.plot(row.estimate,y,"o",ms=7,color=color)
            severity.text(.99,y,f"{row.estimate:+.2f}",ha="right",va="center",fontsize=9,color=color)
            changes=subset[subset.family.eq("clinical_change")].set_index("metric").loc[metrics]
            image[y]=changes.estimate
            labels.append(prefix+str(j+1))
    severity.set(yticks=np.arange(8),yticklabels=labels,ylim=(7.65,-.65),xlim=(-1.02,1.02),
                 xticks=[-1,-.5,0,.5,1],xlabel="Spearman correlation")
    severity.axvline(0,ls="--",lw=1,color="#69747c")
    severity.grid(axis="x",alpha=.18)
    severity.spines[["top","right","left"]].set_visible(False)
    severity.set_title("OFF duration and OFF motor severity",fontsize=13,loc="left",pad=13)
    heat=change.imshow(image,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
    for (y,x),rho in np.ndenumerate(image):
        change.text(x,y,f"{rho:+.2f}",ha="center",va="center",fontsize=10,color="#101820" if abs(rho)<.65 else "white")
    change.set(yticks=np.arange(8),yticklabels=labels,xticks=range(3),xticklabels=["Duration","Episodes/min","Occupancy"])
    change.tick_params(axis="both",length=0)
    change.spines[:].set_visible(False)
    change.set_title("Change with DBS and motor improvement",fontsize=13,loc="left",pad=13)
    for ax in (severity,change):
        ax.axhline(3.5,color="#78868c",lw=.8)
    barax=fig.add_axes([.61,.088,.30,.02])
    fig.colorbar(heat,cax=barax,orientation="horizontal",ticks=[-1,-.5,0,.5,1],label="Spearman correlation")
    fig.text(.12,.948,"BOLD states and UPDRS-III in 24 paired STN patients",fontsize=18,weight="bold")
    fig.text(.12,.890,"A1–A4  peak-based AAHC     •     F1–F4  signed all-frame clustering",fontsize=10.5)
    fig.text(.12,.10,"Dots and lines show estimates and pointwise 95% intervals.\n"
                      "Imaging change = ON − OFF.  Motor improvement = OFF − ON.",fontsize=9,color="#39444a",va="top")
    fig.savefig(target,dpi=180,facecolor="white")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    plot(args.csv,args.output)
