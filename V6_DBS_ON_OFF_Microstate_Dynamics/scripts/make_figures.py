"""Render scientific figures from real paired scans and their public summaries."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from analyze import HETEROGENEITY_KEYS, PRIORITY_CONTEXT, paired_inputs
from paired import draws, interval
from v6_common import HERE, PRIVATE, RESULT, read_json


OFF_COLOR, ON_COLOR, HC_COLOR = "#b34936", "#16816d", "#305b89"
INK, MUTED, GRID = "#1b3046", "#4e6175", "#dce3e9"
STATES = ["M1","M2","M3","M4"]
NAMES = {"gev_1":"M1 GEV", "duration_4":"M4 mean dwell", "occurrence_1":"M1 episodes",
         "switching_rate":"Global switching", **{f"coverage_{i}":f"M{i} coverage" for i in range(1,5)}}

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "text.color":INK, "axes.labelcolor":INK,
                     "axes.titlecolor":INK, "axes.spines.top":False, "axes.spines.right":False,
                     "axes.spines.left":False, "axes.spines.bottom":False, "xtick.color":MUTED,"ytick.color":MUTED,
                     "axes.axisbelow":True, "svg.fonttype":"none", "pdf.fonttype":42})


def save(fig,directory,number,name,title):
    directory.mkdir(parents=True,exist_ok=True)
    fig.suptitle(title,fontsize=15,fontweight="bold")
    for extension in ("png","svg","pdf"):
        path = directory / f"Figure_{number:02d}_{name}.{extension}"
        fig.savefig(path,dpi=600,facecolor="white")
        if extension=="svg":
            body = path.read_text(encoding="utf-8")
            path.write_text("\n".join(line.rstrip() for line in body.splitlines())+"\n",encoding="utf-8",newline="\n")
    plt.close(fig)


def scale(key):
    if key=="duration_4":
        return 1.,"seconds","s"
    if key in ("occurrence_1","switching_rate"):
        return 60.,"events per valid minute","/min"
    return 100.,"percent","pp"


def paired_panel(ax,a,b,label,unit,private=True,subtitle=True):
    ax.set(xticks=[0,1],xticklabels=["OFF","ON"],xlim=(-.24,1.24),ylabel=unit)
    if subtitle:
        ax.set_title(label,fontsize=10,fontweight="bold")
    ax.grid(axis="y",color=GRID,lw=.7)
    if private:
        tint = np.where(b>=a,ON_COLOR,OFF_COLOR)
        for left,right,c in zip(a,b,tint):
            ax.plot([0,1],[left,right],color=c,alpha=.30,lw=.8,zorder=1)
        ax.scatter(np.zeros(len(a)),a,s=13,color=OFF_COLOR,alpha=.66,zorder=2)
        ax.scatter(np.ones(len(b)),b,s=13,color=ON_COLOR,alpha=.66,zorder=2)
    else:
        bootstrap = draws(len(a))
        for j,values,c in zip((0,1),(a,b),(OFF_COLOR,ON_COLOR)):
            lo,hi,_ = interval(values[bootstrap].mean(axis=1))
            ax.vlines(j,lo,hi,color=c,lw=2,zorder=3)
    ax.plot([0,1],[a.mean(),b.mean()],color=INK,linewidth=1.6,zorder=3)
    ax.scatter([0,1],[a.mean(),b.mean()],s=38,color=[OFF_COLOR,ON_COLOR],edgecolors="white",linewidths=.7,zorder=4)


def figure_overview(directory,on,off,private):
    fig,axes = plt.subplots(2,4,figsize=(11.2,5.7),layout="constrained")
    for ax,key in zip(axes.flat,PRIORITY_CONTEXT):
        factor,unit,_ = scale(key)
        paired_panel(ax,off[key].to_numpy()*factor,on[key].to_numpy()*factor,NAMES[key],unit,private)
    fig.supxlabel("Each thin line links the same patient across conditions." if private else
                   "Group means and 95% paired-bootstrap intervals. Each condition contains the same 25 patients.",fontsize=9)
    save(fig,directory,1,"Paired_Microstate_Overview","DBS ON and OFF microstate measures")


def delta_forest(ax,row,factor,unit):
    estimate,low,high = factor*np.array([row.mean_difference,row.ci95_low,row.ci95_high])
    ax.axvline(0,color=GRID,ls="--",lw=1.5)
    ax.hlines(0,low,high,lw=2.5,color=INK)
    ax.scatter(estimate,0,color=ON_COLOR if estimate>0 else OFF_COLOR,s=55,zorder=3)
    width = max(abs(high-low),abs(estimate))*.28
    ax.set(xlim=(min(0,low)-width,max(0,high)+width),ylim=(-.72,.85),yticks=[],xlabel=f"ON − OFF ({unit})")
    ax.set_title("Mean paired difference and 95% interval",fontsize=10,fontweight="bold")
    ax.text(.5,.77,f"{estimate:+.2f}   [{low:+.2f}, {high:+.2f}]",transform=ax.transAxes,ha="center",fontweight="bold")
    ax.text(.5,.21,f"Cohen d$_z$ = {row.dz:+.2f}\nHolm P = {row.permutation_holm_p:.4f}   •   FD Holm P = {row.fd_holm_p:.4f}",
            transform=ax.transAxes,ha="center",linespacing=1.52)


def priority(directory,on,off,tests,private):
    for k,key,name in [(2,"gev_1","M1_GEV"),(3,"duration_4","M4_Duration")]:
        factor,unit,short = scale(key)
        a,b = off[key].to_numpy()*factor,on[key].to_numpy()*factor
        fig,(lx,rx) = plt.subplots(1,2,figsize=(8.6,4.1),layout="constrained",width_ratios=[1,1.25])
        row = tests.loc[key]
        if private:
            values = np.r_[a,b]
            margin = .07*np.ptp(values)
            bounds = [values.min()-margin,values.max()+margin]
            lx.plot(bounds,bounds,ls="--",color=GRID,lw=1)
            lx.scatter(a,b,s=25,alpha=.8,c=np.where(b>=a,ON_COLOR,OFF_COLOR),edgecolors="white",linewidth=.5)
            lx.set(xlim=bounds,ylim=bounds,aspect="equal",xlabel=f"OFF ({unit})",ylabel=f"ON ({unit})")
            lx.set_title("Each point is one patient pair",fontsize=10,fontweight="bold")
        else:
            paired_panel(lx,a,b,"Paired group means",unit,False)
        delta_forest(rx,row,factor,short)
        fig.supxlabel(f"{int(row.n_positive)} increased, {int(row.n_negative)} decreased, {int(row.n_tied)} unchanged across 25 pairs",fontsize=9)
        save(fig,directory,k,name,NAMES[key]+" after switching DBS condition")


def event_rates(directory,on,off,tests,private):
    fig,axes = plt.subplots(1,2,figsize=(8.5,4.1),layout="constrained")
    for ax,key in zip(axes,["occurrence_1","switching_rate"]):
        a,b = off[key].to_numpy()*60,on[key].to_numpy()*60
        row = tests.loc[key]
        paired_panel(ax,a,b,NAMES[key],"Events per valid minute",private)
        ax.set_xlabel(f"Mean Δ {row.mean_difference*60:+.2f}/min   •   FD Holm P {row.fd_holm_p:.4f}",fontsize=9)
    save(fig,directory,4,"Occurrence_and_Switching","Episodes and switches per minute")


def matrices(directory):
    table = pd.read_csv(RESULT / "Transition_Matrices.csv")
    omni = pd.read_csv(RESULT / "ON_OFF_Omnibus_Results.csv").set_index("endpoint").loc["transition_omnibus"]
    keys = ("off_mean_probability","on_mean_probability","mean_difference")
    matrices = [100*table.pivot(index="source",columns="destination",values=x).loc[STATES,STATES].to_numpy() for x in keys]
    peak = max(matrices[0].max(),matrices[1].max())
    difference_peak = np.max(np.abs(matrices[2]))
    fig,axes = plt.subplots(1,3,figsize=(10.5,4.5),layout="constrained")
    for ax,label,m,i in zip(axes,["OFF","ON","ON − OFF"],matrices,range(3)):
        signed = i==2
        total = difference_peak if signed else peak
        ax.imshow(m,cmap="RdBu_r" if signed else "Blues",vmin=-total if signed else 0,vmax=total)
        for (a,b),x in np.ndenumerate(m):
            ax.text(b,a,f"{x:+.1f}" if signed else f"{x:.1f}",va="center",ha="center",color="white" if abs(x)>.67*total else INK,fontsize=9)
        ax.set(title=label,xlabel="Destination",xticks=range(4),xticklabels=STATES,yticks=range(4),yticklabels=STATES)
    axes[0].set_ylabel("Source")
    fig.supxlabel(f"Conditional percentages and percentage-point differences. 12D Wald Holm P {omni.permutation_holm_p:.4f}, FD {omni.fd_holm_p:.4f}.",fontsize=9)
    save(fig,directory,5,"Transition_Dynamics","Transition matrices with self-transitions")


def heterogeneity(directory,delta,private):
    keys = [*HETEROGENEITY_KEYS]
    labels = [NAMES[key] for key in keys]
    if private:
        profile = pd.read_csv(PRIVATE / "results/Paired_Response_Patterns.csv").set_index("participant_id")
        profile = profile.sort_values(["exploratory_ward_cluster","group_direction_cosine"],ascending=[True,False])
        d = delta.loc[profile.index,keys]
        z = d / d.std(ddof=1)
        fig,(ax,bx) = plt.subplots(1,2,figsize=(8.8,7.7),layout="constrained",width_ratios=[2.3,1.])
        heat = ax.imshow(z.to_numpy(),cmap="RdBu_r",norm=TwoSlopeNorm(vmin=-2.5,vcenter=0,vmax=2.5),aspect="auto")
        ax.set(xticks=range(4),xticklabels=labels,yticks=range(len(z)),yticklabels=z.index)
        ax.tick_params(axis="x",rotation=19)
        ax.tick_params(axis="y",labelsize=7)
        fig.colorbar(heat,ax=ax,shrink=.62,label="Paired change / SD of change")
        y = np.arange(len(z))
        color = np.where(profile.exploratory_ward_cluster.eq(1),HC_COLOR,ON_COLOR)
        bx.axvline(0,color=GRID,ls="--",lw=1)
        bx.scatter(profile.group_direction_cosine,y,c=color,s=15)
        bx.set(ylim=(len(z)-.5,-.5),xlim=(-1.08,1.08),yticks=[],xlabel="Cosine to other patients' mean direction")
    else:
        rows = pd.read_csv(RESULT / "Paired_Change_Directions.csv").set_index("endpoint").loc[keys]
        fig,(ax,bx) = plt.subplots(1,2,figsize=(9.5,4.3),layout="constrained",width_ratios=[1.28,1])
        left = np.zeros(len(keys))
        for name,c,label in [("n_negative",OFF_COLOR,"Decreased"),("n_tied",GRID,"Tied"),("n_positive",ON_COLOR,"Increased")]:
            height = rows[name].to_numpy()
            ax.barh(labels,height,left=left,color=c,label=label,height=.56)
            for y,n in enumerate(height):
                if n:
                    ax.text(left[y]+n/2,y,str(n),ha="center",va="center",color="white" if label!="Tied" else INK,fontsize=8)
            left += height
        ax.invert_yaxis()
        ax.set(xlim=(0,25),xlabel="Paired patients",title="Directions of the observed changes")
        ax.legend(loc="lower center",bbox_to_anchor=(.5,-.32),ncol=3,frameon=False,fontsize=8)
        mat = delta[keys].corr(method="spearman").to_numpy()
        bx.imshow(mat,cmap="RdBu_r",vmin=-1,vmax=1)
        for (i,j),v in np.ndenumerate(mat):
            bx.text(j,i,f"{v:+.2f}",ha="center",va="center",color="white" if abs(v)>.68 else INK,fontsize=8)
        short = ["M1 GEV","M4 dwell","M1 rate","Switching"]
        bx.set(xticks=range(4),xticklabels=short,yticks=range(4),yticklabels=short,title="Spearman co-change")
        bx.tick_params(axis="x",rotation=28)
    save(fig,directory,6,"Paired_Response_Heterogeneity","Individual responses vary across measures")


def clinical(directory,private):
    audit = read_json(RESULT / "UPDRS_Input_Audit.json")
    if audit["status"]=="BLOCKED":
        fig,ax = plt.subplots(figsize=(8.2,3.1),layout="constrained")
        ax.axis("off")
        ax.text(.5,.64,"UPDRS association not executed",transform=ax.transAxes,ha="center",fontsize=18,fontweight="bold")
        ax.text(.5,.37,"Historical score exports lack a verified assessment-to-scan visit link.",transform=ax.transAxes,ha="center",color=MUTED)
    else:
        results = pd.read_csv(RESULT / "UPDRS_Association_Results.csv").set_index("endpoint")
        fig,axes = plt.subplots(2,2,figsize=(8.3,6),layout="constrained")
        if private:
            d = pd.read_csv(PRIVATE / "results/UPDRS_Association_Pairs.csv")
        for ax,key in zip(axes.flat,HETEROGENEITY_KEYS):
            row = results.loc[key]
            if private:
                ax.scatter(d[key]*scale(key)[0],d.updrs_off_minus_on,color=HC_COLOR,s=20)
                ax.set(xlabel=f"ON − OFF ({scale(key)[2]})",ylabel="UPDRS OFF − ON")
            else:
                ax.hlines(0,row.ci95_low,row.ci95_high,color=HC_COLOR,lw=2)
                ax.scatter(row.rho,0,color=HC_COLOR)
                ax.set(xlim=(-1,1),yticks=[],xlabel="Spearman ρ and paired-bootstrap interval")
            ax.set_title(f"{NAMES[key]}   ρ = {row.rho:.2f}")
    save(fig,directory,7,"UPDRS_Association","Microstate changes and motor scores")


def health(directory,private):
    r = pd.read_csv(RESULT / "Restoration_Results.csv").iloc[0]
    fig,(ax,bx) = plt.subplots(1,2,figsize=(9.2,4.2),layout="constrained",width_ratios=[1,1.25])
    if private:
        d = pd.read_csv(PRIVATE / "results/HC_Reference_Distances.csv")
        paired_panel(ax,d.Distance_OFF_to_HC.to_numpy(),d.Distance_ON_to_HC.to_numpy(),"Distance to HC center","Aitchison distance",True)
    else:
        ax.bar(["OFF","ON"],[r.mean_off,r.mean_on],color=[OFF_COLOR,ON_COLOR],width=.58)
        ax.vlines([0,1],[r.mean_off_ci95_low,r.mean_on_ci95_low],[r.mean_off_ci95_high,r.mean_on_ci95_high],color=INK,lw=1.7)
        ax.set(title="Distance to HC center",ylabel="Mean Aitchison distance",ylim=(0,max(r.mean_off_ci95_high,r.mean_on_ci95_high)*1.14))
    bx.axvline(0,color=GRID,ls="--",lw=1.4)
    bx.hlines(0,r.ci95_low,r.ci95_high,color=ON_COLOR,lw=2.4)
    bx.scatter(r.effect_size,0,color=ON_COLOR,s=51,zorder=3)
    bx.set(title="Mean paired shortening and 95% interval",yticks=[],ylim=(-.7,.78),xlabel="OFF − ON distance (positive = closer during ON)")
    bx.text(.5,.18,f"Holm P {r.permutation_holm_p:.3f}   •   {int(r.n_closer)}/25 closer",transform=bx.transAxes,ha="center",fontsize=9)
    save(fig,directory,8,"HC_Reference_Distance","DBS condition and healthy-reference distance")


def main():
    _,on,off,delta,_ = paired_inputs()
    tests = pd.read_csv(RESULT / "ON_OFF_Feature_Summary.csv").set_index("endpoint")
    for private,directory in [(False,HERE / "figures"),(True,PRIVATE / "figures")]:
        figure_overview(directory,on,off,private)
        priority(directory,on,off,tests,private)
        event_rates(directory,on,off,tests,private)
        matrices(directory)
        heterogeneity(directory,delta,private)
        clinical(directory,private)
        health(directory,private)
    print("Exported eight public and eight restricted figures in 600 DPI PNG, SVG and PDF.")


if __name__ == "__main__":
    main()
