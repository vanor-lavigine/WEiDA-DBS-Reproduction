"""Render public figures using only aggregate analysis outputs."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from v5_data import CFG, HERE, RESULT


OUT = HERE / "figures"
STATES = ["M1", "M2", "M3", "M4"]
GROUPS = ["HC", "OFF", "ON"]
GROUP_COLOR = {"HC":"#345681", "OFF":"#bc4a36", "ON":"#218374"}
STATE_COLOR = ["#245480", "#477c76", "#bf6b43", "#d9b15f"]
CONTRASTS = CFG["contrasts"]


plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                     "axes.spines.right":False, "axes.labelcolor":"#253443", "text.color":"#253443",
                     "axes.titlecolor":"#172d46", "xtick.color":"#465563", "ytick.color":"#465563"})


def read(name):
    return pd.read_csv(RESULT / f"{name}.csv")


def save(fig, name, title):
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.savefig(OUT / f"{name}.png", dpi=180, facecolor="white")
    plt.close(fig)


def p_label(p):
    return f"{p:.4f}"


def coverage():
    data = read("coverage/Coverage_Composition").pivot(index="condition", columns="state", values="mean").loc[GROUPS, STATES]
    fig, ax = plt.subplots(figsize=(8,3.8), layout="constrained")
    left = np.zeros(3)
    for state, color in zip(STATES, STATE_COLOR):
        values = data[state].to_numpy()*100
        ax.barh(GROUPS, values, left=left, height=.64, color=color, label=state)
        for i, (l, v) in enumerate(zip(left, values)):
            ax.text(l+v/2, i, f"{v:.1f}%", va="center", ha="center", color="white" if state!="M4" else "#26333d", fontweight="bold")
        left += values
    ax.invert_yaxis()
    ax.set(xlim=(0,100), xlabel="Mean percentage of valid BOLD frames")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5,-.17))
    save(fig,"Coverage_Composition","Four-state coverage on frozen V4 templates")


def matrix_plot(mats, title, result, caption, difference=False):
    fig, axes = plt.subplots(1,3, figsize=(11.8,4.4), layout="constrained")
    vmax = max(np.abs(x).max() for x in mats.values())
    for ax, (label, m) in zip(axes, mats.items()):
        matrix = ax.imshow(m, cmap="RdBu_r" if difference else "Blues", vmin=-vmax if difference else 0, vmax=vmax)
        for (i,j), val in np.ndenumerate(m):
            ax.text(j,i,f"{val:+.1f}" if difference else f"{val:.1f}",ha="center",va="center",
                    color="white" if abs(val)>.65*vmax else "#122338", fontsize=9)
        ax.set(title=label, xlabel="Destination", xticks=range(4), xticklabels=STATES, yticks=range(4), yticklabels=STATES)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Source")
    fig.colorbar(matrix, ax=axes, shrink=.68, label="Percentage-point change" if difference else "Mean conditional probability (%)")
    fig.supxlabel(caption, fontsize=9)
    save(fig,result,title)


def transitions():
    raw = read("transition/Transition_Matrices")
    groups = {x:100*raw.loc[raw.condition.eq(x)].pivot(index="source",columns="destination",values="mean_probability").loc[STATES,STATES].to_numpy() for x in GROUPS}
    p = read("transition/Transition_Omnibus").set_index("contrast")
    terms = "   |   ".join(f"{c}  {p_label(p.loc[c, 'permutation_holm_p'])} / {p_label(p.loc[c, 'fd_holm_p'])}" for c in CONTRASTS)
    matrix_plot(groups,"Full transition matrices, including self-transitions","Transition_Matrix_Omnibus",f"Six-test Holm P (permutation / FD):  {terms}")
    changes = {c:groups[c.split("-")[0]]-groups[c.split("-")[1]] for c in CONTRASTS}
    matrix_plot(changes,"Full-matrix mean differences","Transition_Differences","Cell values are descriptive. See the localization table for gated inference.",True)


def duration():
    data = read("duration/Duration_Summaries")
    names = ["Median dwell, window", "P75 dwell, window", "Median dwell, edge", "P75 dwell, edge"]
    fig, axes = plt.subplots(2,2, figsize=(10,7), layout="constrained", sharex=True, sharey=True)
    for ax, name in zip(axes.flat,names):
        block = data.loc[data.definition.eq(name)].set_index(["condition","state"])
        for i, group in enumerate(GROUPS):
            x = np.arange(4)+(i-1)*.22
            z = block.loc[group].loc[STATES]
            color = GROUP_COLOR[group]
            ax.vlines(x,z.q1,z.q3,color=color, lw=6, alpha=.55)
            ax.scatter(x,z["median"],color=color,s=22, label=group, zorder=2)
        ax.set(title=name, xticks=np.arange(4), xticklabels=STATES)
        ax.grid(axis="y", alpha=.18)
    axes[0,0].legend(frameon=False, ncol=3, fontsize=8)
    fig.supylabel("Participant-level dwell statistic (s)")
    fig.supxlabel("Dots show group medians. Bars span the first to third quartile.", fontsize=9)
    save(fig,"Duration_Distributions","Within-state dwell distributions")


def occurrence():
    data = read("occurrence/Occurrence_Summaries").set_index(["condition","state"])
    fig, ax = plt.subplots(figsize=(8,4.8),layout="constrained")
    for i, g in enumerate(GROUPS):
        z = data.loc[g].loc[STATES]
        ax.errorbar(np.arange(4)+(i-1)*.21,z.mean_rate_per_min, yerr=z.sd_rate_per_min/np.sqrt(z.n),
                    color=GROUP_COLOR[g],fmt="o", capsize=4, ms=6, label=g)
    ax.legend(frameon=False, ncol=3)
    ax.set(xticks=np.arange(4), xticklabels=STATES, ylabel="Mean episodes per valid minute ± SE")
    ax.grid(axis="y", alpha=.23)
    fig.supxlabel("Inference uses exposure-adjusted participant-clustered count models.",fontsize=9)
    save(fig,"Occurrence_Group_Comparison","Microstate episode rates")


def distances():
    z = read("restoration/Restoration_Results").iloc[0]
    fig, (ax, bx) = plt.subplots(1,2,figsize=(9.7,4.2),layout="constrained",width_ratios=[1,1.25])
    ax.bar(["OFF","ON"],[z.mean_off,z.mean_on],color=[GROUP_COLOR["OFF"],GROUP_COLOR["ON"]],width=.6)
    for i, v in enumerate([z.mean_off,z.mean_on]):
        ax.text(i,v+.025,f"{v:.3f}",ha="center",fontweight="bold")
    ax.set(ylabel="Mean Aitchison distance", ylim=(0,1.35*max(z.mean_off,z.mean_on)),title="Distance to HC center")
    bx.axvline(0, color="#748494", linestyle="--", lw=1)
    bx.hlines(0,z.ci95_low,z.ci95_high,color=GROUP_COLOR["ON"],lw=3)
    bx.scatter(z.effect_size,0,s=58,color=GROUP_COLOR["ON"],zorder=3)
    bx.text(.5,.16,f"Holm P = {p_label(z.permutation_holm_p)}   |   FD = {p_label(z.fd_holm_p)}",transform=bx.transAxes,ha="center")
    bx.set(xlabel="OFF − ON distance (positive = ON is closer)", title="Mean paired change and bootstrap 95% CI",yticks=[],ylim=(-.8,.8))
    save(fig,"HC_Reference_Distance","Distance to the healthy coverage composition")


def forest():
    r = read("duration/Duration_Results")
    fig, axes = plt.subplots(2,3,figsize=(12,6.6),layout="constrained",sharey=True,sharex=True)
    for i, name in enumerate(["Median dwell, window","P75 dwell, window"]):
        for j, c in enumerate(CONTRASTS):
            ax = axes[i,j]
            z = r.loc[r.family.eq(name) & r.contrast.eq(c)].set_index("endpoint").loc[STATES]
            ax.axvline(0, color="#748494", linestyle="--", lw=1)
            for y, (state, row) in enumerate(z.iterrows()):
                color = STATE_COLOR[y]
                ax.hlines(y,row.ci95_low,row.ci95_high,color=color,lw=2)
                ax.scatter(row.effect_size,y,color=color,s=36,zorder=2)
            ax.set(title=f"{name.split(',')[0]} · {c}", yticks=np.arange(4), yticklabels=STATES,ylim=(3.6,-.6))
            ax.grid(axis="x", alpha=.2)
    fig.supxlabel("Mean difference in participant-level dwell statistic (s)")
    save(fig,"Bootstrap_Forest","Fixed-template bootstrap, 2,000 resamples")


def noise():
    data = read("simulation/Calibration_by_Scenario")
    fig, axes = plt.subplots(1,2,figsize=(10.1,4.6),layout="constrained")
    colors = {"S0":"#576673", "S5_moderate":"#218374", "S5_high":"#bc4a36"}
    for name, tint in colors.items():
        q = data.loc[data.scenario.eq(name)].sort_values("penalty")
        axes[0].plot(q.penalty,100*q.wilson_upper,marker="o",color=tint,label=name.replace("_"," "))
    axes[0].axhline(15, color="#bc4a36", ls="--", lw=1, alpha=.6)
    axes[0].axhline(10, color="#576673", ls="--", lw=1, alpha=.6)
    axes[0].set(title="False-positive calibration",ylabel="Wilson upper 95% bound (%)", xlabel="Switch penalty λ")
    axes[0].legend(frameon=False, fontsize=8)
    s = data.loc[data.scenario.eq("S0")].sort_values("penalty")
    axes[1].plot(s.penalty,s.ari,marker="o",color="#345681",label="Adjusted Rand index")
    axes[1].plot(s.penalty,s.short_recall,marker="s",color="#218374",label="Short-episode recall")
    axes[1].set(title="Sequence recovery in S0", xlabel="Switch penalty λ", ylim=(0,1))
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.axvline(CFG["baseline"]["lambda"],color="#8695a3",lw=1,ls=":")
        ax.grid(axis="y", alpha=.15)
    save(fig,"Noise_Calibration","No switch penalty met all simulation gates")


def comparison():
    old = pd.read_csv(HERE / "V4_FROZEN_SIGNIFICANCE_REGISTRY.csv").query("metric == 'ilr'").set_index("contrast")
    new = read("coverage/Coverage_Energy").set_index("contrast")
    fig, axes = plt.subplots(1,2,figsize=(9.8,3.6),layout="constrained",sharex=True)
    for ax, p0, p1, label in zip(axes,["permutation_holm_p","fd_permutation_holm_p"],
                                ["permutation_holm_p","fd_holm_p"],["Unadjusted permutation", "FD-adjusted permutation"]):
        for i, c in enumerate(CONTRASTS[:2]):
            for s,(z,p,marker,color) in enumerate([(old,p0,"o","#345681"),(new,p1,"s","#218374")]):
                ax.scatter(z.loc[c,p],i+(.13 if s else -.13),marker=marker,color=color,s=50,
                           label=["V4 ILR–Wald","V5 Energy"][s] if i==0 else None)
        ax.axvline(.05, color="#bc4a36", linestyle="--", lw=1)
        ax.set(title=label, xscale="log",xlim=(.0004,1.1),xlabel="Holm P in each version's primary family",
               yticks=[0,1],yticklabels=CONTRASTS[:2],ylim=(1.55,-.5))
    axes[0].legend(frameon=False,fontsize=8,loc="center left")
    save(fig,"V4_V5_Comparison","Coverage evidence in the same frozen cohort")


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    for generate in (coverage, transitions, duration, occurrence, distances, forest, noise, comparison):
        generate()
