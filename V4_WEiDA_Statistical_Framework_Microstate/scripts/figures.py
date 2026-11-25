"""Render the five scientific figures from the complete result tables."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats


GROUPS = ("HC", "OFF", "ON")
COLORS = {"HC": "#18878D", "OFF": "#B45A35", "ON": "#485DB0"}
METRICS = [("coverage", "Coverage", "% of frames", 100), ("duration", "Mean duration", "Seconds", 1),
           ("occurrence", "Occurrence", "Episodes / min", 60), ("gev", "GEV", "% of global variance", 100),
           ("dispersion", "Local dispersion", "Unit-vector distance", 1)]
FAMILY_COLORS = {"Coverage": "#3E68A8", "Duration": "#22857D", "Occurrence": "#996134", "GEV": "#865DA0",
                 "Spatial Dispersion": "#AA4661", "Transition Probability": "#546074"}
CONTRASTS = ("OFF-HC", "ON-HC", "ON-OFF")
TITLES = {"OFF-HC": "OFF − HC", "ON-HC": "ON − HC", "ON-OFF": "ON − OFF"}
SHARED = {"coverage_entropy": "Occupancy entropy", "global_pooled_duration": "Pooled duration",
          "switching_rate": "Switching rate", "global_occurrence": "Total occurrence"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": "#263247", "axes.titleweight": "bold",
                         "axes.edgecolor": "#96A1AE", "text.color": "#1A263A", "xtick.color": "#344156",
                         "ytick.color": "#344156", "figure.facecolor": "white", "savefig.facecolor": "white"})


def save(fig, destination):
    fig.savefig(destination, dpi=175, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def group_comparison(data, destination):
    fig, axes = plt.subplots(2, 3, figsize=(12.9, 7.1), layout="constrained")
    fig.suptitle("BOLD microstate features by condition", fontsize=18, fontweight="bold")
    for ax, (key, title, unit, scale) in zip(axes.flat, METRICS):
        for k, group in enumerate(GROUPS):
            samples = [data.loc[data.condition.eq(group), f"{key}_{i}"].dropna().to_numpy() * scale
                       for i in range(1, 5)]
            means = [s.mean() if len(s) else np.nan for s in samples]
            ci = [stats.t.ppf(.975, len(s) - 1) * stats.sem(s) if len(s) > 1 else np.nan for s in samples]
            ax.errorbar(np.arange(1, 5) + (k - 1) * .17, means, yerr=ci, marker="o", markersize=5,
                        color=COLORS[group], linewidth=1.3, capsize=3, linestyle="none", zorder=3)
        ax.set(title=title, ylabel=unit, xticks=range(1, 5), xticklabels=[f"M{i}" for i in range(1, 5)], xlim=(.55, 4.45))
        ax.grid(axis="y", color="#DCE2E9", linewidth=.65)
    legend = axes[-1, -1]
    legend.axis("off")
    handles = [Line2D([], [], color=COLORS[g], marker="o", linestyle="none", label=f"{g}  ·  n={15 if g == 'HC' else 25}")
               for g in GROUPS]
    legend.legend(handles=handles, loc="center", frameon=False, title="Mean and 95% CI", labelspacing=1, fontsize=12)
    save(fig, destination)


def paired_figure(data, destination):
    fig, axes = plt.subplots(5, 4, figsize=(11.5, 11.5), sharex=True, sharey="row", layout="constrained")
    fig.suptitle("Within-patient changes from DBS OFF to ON", fontsize=18, fontweight="bold")
    pd = data.loc[data.condition.isin(["ON", "OFF"])]
    for row, (key, title, unit, scale) in enumerate(METRICS):
        for col, ax in enumerate(axes[row]):
            piv = pd.pivot(index="subject", columns="condition", values=f"{key}_{col + 1}").dropna()
            pairs = piv[["OFF", "ON"]].to_numpy().T * scale
            ax.grid(axis="y", color="#E2E6EC", linewidth=.6)
            if pairs.size:
                ax.plot([0, 1], pairs, color="#728396", alpha=.26, linewidth=.8)
                means = pairs.mean(axis=1)
                ax.plot([0, 1], means, color="#2E384A", linewidth=2.0, zorder=3)
                ax.scatter([0, 1], means, color=[COLORS["OFF"], COLORS["ON"]], edgecolor="white", linewidth=.9, zorder=4)
            ax.set(xticks=[0, 1], xticklabels=["OFF", "ON"], xlim=(-.2, 1.2))
            if col == 0:
                ax.set_ylabel(f"{title}\n{unit}", fontsize=9.5)
            if row == 0:
                ax.set_title(f"M{col + 1}")
    fig.supxlabel("Each pale line links the two scans from one patient. Bold segments join the condition means.", fontsize=10)
    save(fig, destination)


def transition_figure(data, inference, destination):
    use = data.loc[data.condition.isin(["OFF", "ON"])]
    matrices = {g: np.full((4, 4), np.nan) for g in (*GROUPS, "ON − OFF")}
    support = inference.set_index(["metric", "contrast"]).family_holm_and_permutation_holm
    positive = np.zeros((4, 4), bool)
    for i in range(4):
        for j in range(4):
            if i == j:
                continue
            col = f"transition_{i + 1}_{j + 1}"
            for g in GROUPS:
                matrices[g][i, j] = data.loc[data.condition.eq(g), col].mean()
            piv = use.pivot(index="subject", columns="condition", values=col).dropna()
            matrices["ON − OFF"][i, j] = (piv.ON - piv.OFF).mean()
            positive[i, j] = support.loc[(col, "ON-OFF")]
    hi = max(.01, *(float(np.nan_to_num(matrices[g], nan=0).max()) for g in GROUPS))
    change = max(.01, float(np.nan_to_num(np.abs(matrices["ON − OFF"]), nan=0).max()))
    fig, axes = plt.subplots(1, 4, figsize=(14.6, 4.1), layout="constrained")
    fig.suptitle("Where transitions go when a microstate ends", fontsize=18, fontweight="bold")
    palettes = [plt.get_cmap("YlGnBu").copy(), plt.get_cmap("RdBu_r").copy()]
    for palette in palettes:
        palette.set_bad("#ECEFF3")
    for ax, (g, matrix) in zip(axes, matrices.items()):
        diff = g == "ON − OFF"
        im = ax.imshow(matrix, cmap=palettes[int(diff)], vmin=-change if diff else 0, vmax=change if diff else hi)
        for i, j in zip(*np.nonzero(np.isfinite(matrix))):
            val = matrix[i, j]
            color = "white" if (abs(val) / change > .70 if diff else val / hi > .64) else "#1A263A"
            string = f"{val:+.2f}" if diff else f"{val:.2f}"
            if diff and positive[i, j]:
                string += "*"
            ax.text(j, i, string, ha="center", va="center", fontsize=9, color=color)
        ax.set(title=g, xlabel="Destination", xticks=range(4), yticks=range(4),
               xticklabels=[f"M{i}" for i in range(1, 5)], yticklabels=[f"M{i}" for i in range(1, 5)])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if diff:
            fig.colorbar(im, ax=ax, shrink=.67, label="Paired difference")
        else:
            base = im
    axes[0].set_ylabel("Source")
    fig.colorbar(base, ax=axes[:3].tolist(), shrink=.67, label="Conditional probability")
    if positive.any():
        fig.supxlabel("* Both family Holm tests pass.", fontsize=9.5)
    save(fig, destination)


def forest(data, destination):
    core = data.loc[data.scope.eq("primary")]
    names = core.loc[core.contrast.eq(CONTRASTS[0])]
    boundaries = core[["standardized_ci95_low", "standardized_ci95_high"]].to_numpy()
    limits = max(1.05, np.nanmax(np.abs(boundaries), initial=1.05)) * 1.07
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 12.5), sharey=True, layout="constrained")
    fig.suptitle("Microstate effect sizes across all prespecified contrasts", fontsize=18, fontweight="bold")
    abbreviations = {"Transition Probability": "Transition", "Spatial Dispersion": "Local dispersion"}
    for ax, contrast in zip(axes, CONTRASTS):
        ax.axvline(0, color="#738097", linestyle="--", linewidth=1)
        d = core.loc[core.contrast.eq(contrast)].set_index("metric").loc[names.metric]
        for y, row in enumerate(d.itertuples()):
            if y % 2 == 0:
                ax.axhspan(y - .5, y + .5, color="#F1F4F8", linewidth=0, zorder=0)
            if not np.isfinite(row.standardized_effect):
                continue
            col = FAMILY_COLORS[row.family]
            ax.hlines(y, row.standardized_ci95_low, row.standardized_ci95_high, color=col, linewidth=1.25)
            ax.scatter(row.standardized_effect, y, s=32, marker="o", edgecolor=col, linewidth=1.1,
                       facecolor=col if row.family_holm_and_permutation_holm else "white", zorder=3)
        ax.set(title=TITLES[contrast], xlim=(-limits, limits), ylim=(len(d) - .5, -.5),
               xlabel="Hedges g_z" if contrast == "ON-OFF" else "Hedges g")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(names)), [f"{abbreviations.get(r.family, r.family)}  {r.state.replace('->', '→')}"
                                          for r in names.itertuples()], fontsize=9)
    fig.supxlabel("95% participant bootstrap intervals  ·  Filled markers pass both family Holm tests", fontsize=10)
    save(fig, destination)


def concordance(cross, match, destination):
    fig, (ax, right) = plt.subplots(1, 2, figsize=(11.5, 5.7), layout="constrained")
    fig.suptitle("Microstates and WEiDA on the same frozen scans", fontsize=18, fontweight="bold")
    use = cross[cross.status.isin(["comparable overall metric", "qualified state correspondence"])]
    cc = {"OFF-HC": COLORS["OFF"], "ON-HC": COLORS["HC"], "ON-OFF": COLORS["ON"]}
    markers = dict(zip(SHARED, "os^D"))
    extent = 1.02
    if len(use):
        extent = max(.5, np.nanmax(np.abs(use[["microstate_standardized_effect", "weida_standardized_effect"]].to_numpy()))) + .12
        for r in use.itertuples():
            if r.microstate_metric != "switching_rate":
                ax.scatter(r.weida_standardized_effect, r.microstate_standardized_effect, color=cc[r.contrast],
                           marker=markers.get(r.microstate_metric, "."), edgecolor="white", linewidth=.6, s=60, zorder=4)
        for r in use.loc[use.microstate_metric.eq("switching_rate")].itertuples():
            ax.scatter(r.weida_standardized_effect, r.microstate_standardized_effect, facecolor="none",
                       edgecolor=cc[r.contrast], marker="^", linewidth=1, s=146, zorder=5)
    else:
        ax.text(0, .1, "Verified WEiDA readouts unavailable", ha="center")
    ax.axhline(0, color="#CAD2DD", linewidth=.9)
    ax.axvline(0, color="#CAD2DD", linewidth=.9)
    ax.plot([-extent, extent], [-extent, extent], color="#7C889A", linestyle="--", linewidth=1)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="WEiDA standardized effect",
           ylabel="Microstate standardized effect", title="Comparable effect estimates", aspect="equal")
    if len(use):
        handles = [Line2D([], [], color=c, marker="o", linestyle="none", label=TITLES[g]) for g, c in cc.items()]
        handles += [Line2D([], [], color="#344156", marker=markers[m], linestyle="none", label=t,
                           markerfacecolor="none" if m == "switching_rate" else "#344156") for m, t in SHARED.items()]
        ax.legend(handles=handles, ncol=2, fontsize=8, frameon=False, loc="upper left")
    if len(match):
        matrix = match.pivot_table(index="microstate", columns="weida_state", values="abs_spatial_r", aggfunc="mean").to_numpy()
        image = right.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        for i, j in np.ndindex(matrix.shape):
            right.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < .53 else "#1A263A")
        fig.colorbar(image, ax=right, shrink=.71, label="Mean absolute spatial r across folds")
        quality = "State correspondence passed" if match.state_match_pass.all() else "State correspondence did not pass"
        right.set(title=quality, xlabel="WEiDA state", ylabel="Microstate", xticks=range(4), yticks=range(4),
                  xticklabels=[f"W{i}" for i in range(1, 5)], yticklabels=[f"M{i}" for i in range(1, 5)])
        right.tick_params(length=0)
    else:
        right.axis("off")
        right.text(.5, .5, "Template comparison unavailable", ha="center", transform=right.transAxes)
    save(fig, destination)


def write_figures(root, data, microstate, cross, match):
    style()
    group_comparison(data, root / "Group_Comparison.png")
    paired_figure(data, root / "Paired_ON_OFF.png")
    transition_figure(data, microstate, root / "Transition_Matrix.png")
    forest(microstate, root / "Effect_Size_Forest.png")
    concordance(cross, match, root / "Cross_Method_Concordance.png")
