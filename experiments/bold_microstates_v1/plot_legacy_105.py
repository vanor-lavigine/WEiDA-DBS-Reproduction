"""Compare FD-adjusted effects using the deidentified historical and SyN9 results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


def plot(source, destination):
    data = pd.read_csv(source)
    panels = [("coverage", "Coverage", "percentage points", 100),
              ("occurrence_per_min", "Occurrence", "events/min", 1),
              ("duration_sec", "Duration", "seconds", 1)]
    groups = [("Separate templates, 25 vs 26 pairs", "legacy_refit_25", "syn9_original_26"),
              ("Same 24 pairs, frames and templates", "legacy_common_24", "syn9_common_24")]
    styles = [("#006d77", "D", -0.13, "105 BIDS-fMRIpost"),
              ("#935125", "o", 0.13, "V100 SyN9")]
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.55), sharex="col", sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.095, top=0.78, hspace=0.54, wspace=0.24)
    fig.suptitle("STN-DBS ON minus OFF at equal mean head motion", y=0.985, fontsize=15)
    handles = [Line2D([], [], color=color, marker=marker, label=label) for color, marker, _, label in styles]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.945), frameon=False)
    for i, (title, old, syn) in enumerate(groups):
        axes[i, 0].text(0, 1.27 if i == 0 else 1.12, title, transform=axes[i, 0].transAxes,
                        fontsize=11, fontweight="semibold")
        for j, (metric, name, unit, scale) in enumerate(panels):
            ax = axes[i, j]
            for subset, (color, marker, offset, _) in zip((old, syn), styles):
                v = data.loc[data.analysis.eq(subset) & data.metric.eq(metric)].sort_values("state")
                assert len(v) == 4
                y = v.state + offset
                ax.hlines(y, v.adjusted_ci_low * scale, v.adjusted_ci_high * scale,
                          color=color, lw=1.6)
                ax.scatter(v.adjusted_delta * scale, y, color=color, marker=marker, s=28, zorder=3)
            ax.axvline(0, ls="--", lw=0.9, color="#777777", zorder=0)
            ax.set(ylim=(4.55, 0.45), yticks=range(1, 5), yticklabels=[f"M{k}" for k in range(1, 5)])
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(name)
            else:
                ax.set_xlabel(f"ON − OFF ({unit})")
    fig.savefig(destination, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stats", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    plot(args.stats, args.out)
