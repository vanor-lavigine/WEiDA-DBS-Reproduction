"""Scientific figures for the complete Microstate-only analysis."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats

from micro_inputs import HERE, STATE


GROUPS = ("HC", "OFF", "ON")
PALETTE = {"HC": "#16888A", "OFF": "#B85B38", "ON": "#4B5EA9"}
CONTRASTS = ("OFF-HC", "ON-HC", "ON-OFF")
FAMILY_COLORS = {"Coverage": "#405E9C", "Duration": "#16888A", "Occurrence": "#A96031", "GEV": "#8661A7",
                 "Spatial Dispersion": "#A83D65", "Transition Probability": "#586377"}
DESCRIPTIONS = [("coverage", "Coverage", "% of labeled frames", 100, "Coverage_Comparison"),
                ("duration", "Mean duration", "Seconds", 1, "Duration_Comparison"),
                ("occurrence", "Occurrence rate", "Episodes per minute", 60, "Occurrence_Comparison"),
                ("gev", "Global explained variance by state", "% of total GFP-weighted variation", 100, "GEV_Comparison")]


def theme():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#94A2B3", "text.color": "#243148",
                         "axes.labelcolor": "#243148", "xtick.color": "#46536A", "ytick.color": "#46536A",
                         "axes.titleweight": "bold", "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(HERE / "figures" / f"{name}.png", dpi=176, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def group_metric(df, key, title, unit, scale, file):
    fig, ax = plt.subplots(figsize=(8.3, 4.9), layout="constrained")
    for g, condition in enumerate(GROUPS):
        values = [scale * df.loc[df.condition.eq(condition), f"{key}_{s}"].dropna().to_numpy() for s in STATE]
        means = [a.mean() if len(a) else np.nan for a in values]
        radius = [stats.t.ppf(.975, len(a)-1) * stats.sem(a) if len(a)>1 else np.nan for a in values]
        ax.errorbar(np.arange(1, 5) + (g-1)*.19, means, yerr=radius, color=PALETTE[condition], marker="o",
                    linestyle="none", capsize=3, linewidth=1.35, markersize=6, zorder=3)
    ax.set(title=title, xticks=list(STATE), xticklabels=[f"M{i}" for i in STATE], ylabel=unit, xlim=(.55, 4.45))
    ax.grid(axis="y", color="#E1E6EE", linewidth=.85)
    handles = [Line2D([], [], marker="o", linestyle="none", color=PALETTE[g], label=f"{g} (n={15 if g == 'HC' else 25})")
               for g in GROUPS]
    ax.legend(handles=handles, ncol=3, loc="best", frameon=False)
    fig.supxlabel("A3 held-out estimates  ·  Condition means and pointwise 95% t intervals", fontsize=9.5)
    save(fig, file)


def transitions(df):
    matrices = np.full((4, 4, 4), np.nan)
    patients = df.loc[df.condition.isin(("OFF", "ON"))]
    for i, j in np.ndindex(4, 4):
        key = f"transition_{i+1}_{j+1}"
        for k, cond in enumerate(GROUPS):
            matrices[k, i, j] = df.loc[df.condition.eq(cond), key].mean()
        p = patients.pivot(index="subject", columns="condition", values=key).dropna()
        matrices[3, i, j] = (p.ON - p.OFF).mean()
    fig, axes = plt.subplots(1, 4, figsize=(14.8, 4.3), layout="constrained")
    fig.suptitle("Adjacent-TR transition probabilities", fontsize=17, fontweight="bold")
    mid = max(.01, np.nanmax(np.abs(matrices[-1]), initial=.01))
    maximum = max(.01, np.nanmax(matrices[:3], initial=.01))
    for k, (ax, matrix) in enumerate(zip(axes, matrices)):
        paired = k == 3
        cmap = plt.get_cmap("RdBu_r" if paired else "YlGnBu").copy()
        cmap.set_bad("#E6EAF1")
        plot = ax.imshow(matrix, cmap=cmap, vmin=-mid if paired else 0, vmax=mid if paired else maximum)
        for i, j in zip(*np.nonzero(np.isfinite(matrix))):
            p = matrix[i, j]
            high = abs(p) > .7*mid if paired else p > .65*maximum
            ax.text(j, i, f"{p:+.3f}" if paired else f"{p:.3f}", ha="center", va="center", fontsize=8.5,
                    color="white" if high else "#243148")
        ax.set(title="ON − OFF" if paired else GROUPS[k], xlabel="Destination", xticks=range(4), yticks=range(4),
               xticklabels=[f"M{i}" for i in STATE], yticklabels=[f"M{i}" for i in STATE])
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        if not paired:
            native = plot
    axes[0].set_ylabel("Source")
    fig.colorbar(native, ax=axes[:3].tolist(), label="Probability", shrink=.67)
    fig.colorbar(plot, ax=axes[3], label="Paired difference", shrink=.67)
    fig.supxlabel("Diagonal cells show self-transitions. The twelve directed off-diagonal edges enter inference.", fontsize=9.5)
    save(fig, "Transition_Matrix")


def forest(tests):
    primary = tests.loc[tests.family.ne("Switching Rate")]
    order = primary.loc[primary.contrast.eq("OFF-HC"), "metric"].tolist()
    lohi = primary[["standardized_ci95_low", "standardized_ci95_high"]].to_numpy()
    scale = max(1, np.nanmax(np.abs(lohi), initial=1))*1.12
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 12.4), sharey=True, layout="constrained")
    fig.suptitle("State and transition effect sizes", fontsize=17, fontweight="bold")
    for ax, c in zip(axes, CONTRASTS):
        ax.axvline(0, color="#8490A4", linewidth=1.0, linestyle="--", zorder=1)
        rows = primary.loc[primary.contrast.eq(c)].set_index("metric").loc[order]
        for j, row in enumerate(rows.itertuples()):
            if j % 2 == 0:
                ax.axhspan(j-.5, j+.5, linewidth=0, color="#F0F3F8", zorder=0)
            if not np.isfinite(row.standardized_effect):
                continue
            color = FAMILY_COLORS[row.family]
            ax.hlines(j, row.standardized_ci95_low, row.standardized_ci95_high, color=color, linewidth=1.25, zorder=2)
            ax.scatter(row.standardized_effect, j, edgecolor=color, facecolor=color if row.joint_holm_support else "white",
                       s=31, linewidth=1.05, zorder=3)
        ax.set(title=c.replace("-", " − "), xlim=(-scale, scale), ylim=(len(order)-.5, -.5),
               xlabel="Hedges g_z" if c == "ON-OFF" else "Hedges g")
        ax.tick_params(axis="y", length=0)
    names = {"Transition Probability": "Transition", "Spatial Dispersion": "Dispersion"}
    labels = [f"{names.get(row.family, row.family)}  {row.state.replace('->', '→')}"
              for row in primary.loc[primary.contrast.eq("OFF-HC")].itertuples()]
    axes[0].set_yticks(range(len(order)), labels, fontsize=9)
    fig.supxlabel("Fixed-feature 95% participant bootstrap intervals  ·  Filled points pass both family Holm tests and the Coverage gate", fontsize=9.8)
    save(fig, "Effect_Size_Forest")


def significance(tests, composition):
    fields = [("raw_p", "Raw P"), ("fdr_p", "Family BH"), ("holm_p", "Family Holm"), ("permutation_holm_p", "Permutation Holm"),
              ("fd_permutation_holm_p", "FD permutation Holm")]
    families = ["Coverage ILR", "Coverage", "Duration", "Occurrence", "Transition Probability", "GEV", "Spatial Dispersion", "Switching Rate"]
    all_rows = pd.concat([tests, composition], ignore_index=True)
    fig, axes = plt.subplots(1, len(fields), figsize=(15.5, 5.05), sharey=True, layout="constrained")
    fig.suptitle("Significance across contrasts and correction rules", fontsize=17, fontweight="bold")
    for ax, (col, title) in zip(axes, fields):
        colors = np.full((len(families), len(CONTRASTS)), np.nan)
        labels = {}
        for i, f in enumerate(families):
            for j, c in enumerate(CONTRASTS):
                rows = all_rows.loc[all_rows.family.eq(f) & all_rows.contrast.eq(c)]
                n = len(rows)
                hits = rows[col].lt(.05).sum()
                gated = n and f == "Coverage" and c != "ON-OFF" and not rows.passes_coverage_gate.any() and col != "raw_p"
                if n:
                    colors[i, j] = hits/n
                labels[i, j] = (f"{hits}/{n}{'†' if gated and hits else ''}" if n else "—")
        cmap = plt.get_cmap("Blues").copy()
        cmap.set_bad("#EDF0F4")
        ax.imshow(colors, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for (i, j), label in labels.items():
            ax.text(j, i, label, ha="center", va="center", color="white" if colors[i, j] > .6 else "#243148", fontsize=9)
        ax.set(title=title, xticks=range(3), xticklabels=["OFF−HC", "ON−HC", "ON−OFF"])
        ax.tick_params(axis="both", length=0, labelsize=8.6)
        ax.spines[:].set_visible(False)
    axes[0].set_yticks(range(len(families)), families)
    fig.supxlabel("Counts below 0.05 / planned rows  ·  † Coverage localization withheld by the whole-composition gate", fontsize=9.7)
    save(fig, "Significance_Summary")


def templates():
    templates = pd.read_csv(HERE / "results/Reference_Templates.csv")
    matrix = templates.iloc[:,4:]
    mx = abs(matrix.to_numpy()).max()
    fig, axes = plt.subplots(2, 1, figsize=(16, 6.6), layout="constrained")
    fig.suptitle("Four fixed AAL78 reference templates", fontsize=17, fontweight="bold")
    for ax, side in zip(axes, ("_L", "_R")):
        hemisphere = matrix.loc[:, matrix.columns.str.endswith(side)]
        image = ax.imshow(hemisphere.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-mx, vmax=mx)
        ax.set(title="Left" if side == "_L" else "Right", yticks=range(4), yticklabels=[f"M{i}" for i in STATE],
               xticks=range(hemisphere.shape[1]), xticklabels=[i[:-2].replace("_", " ") for i in hemisphere.columns])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, label="Template coefficient", shrink=.72)
    fig.supxlabel("Fold 0 defines the state numbering. The backfit is insensitive to an overall change of sign.", fontsize=9.7)
    save(fig, "State_Templates")


def main():
    theme()
    (HERE / "figures").mkdir(exist_ok=True)
    data = pd.read_csv(HERE / "private/Subject_Level_Microstate_Features.csv")
    results = pd.read_csv(HERE / "results/All_Scalar_Results.csv")
    ilr = pd.read_csv(HERE / "results/Coverage_ILR_Results.csv")
    for description in DESCRIPTIONS:
        group_metric(data, *description)
    transitions(data)
    forest(results)
    significance(results, ilr)
    templates()


if __name__ == "__main__":
    main()
