"""Plot original and FD-adjusted paired effects from the group-level CSV."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def plot(csv_path, out):
    table = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.9), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.72, wspace=0.23)
    series = (
        ("original", -0.13, "#596774", "o", "Original paired mean"),
        ("adjusted", 0.13, "#007974", "D", "Adjusted to equal mean FD"),
    )
    panels = (("coverage", "Coverage", "percentage points", 100),
              ("occurrence_per_min", "Occurrence", "events/min", 1),
              ("duration_sec", "Duration", "seconds", 1))
    for axis, (metric, title, unit, scale) in zip(axes, panels):
        data = table.loc[table["metric"].eq(metric)].sort_values("state")
        for prefix, offset, color, marker, _ in series:
            y = data["state"] + offset
            axis.hlines(y, data[f"{prefix}_ci_low"] * scale, data[f"{prefix}_ci_high"] * scale,
                        color=color, linewidth=1.8)
            axis.scatter(data[f"{prefix}_delta"] * scale, y, marker=marker, s=30,
                         color=color, zorder=3)
        axis.axvline(0, color="#909090", linewidth=1, linestyle="--", zorder=0)
        axis.set(xlabel=f"ON − OFF ({unit})", title=title, yticks=range(1, 5),
                 yticklabels=[f"M{i}" for i in range(1, 5)], ylim=(4.55, 0.45))
        axis.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker=marker, color=color, label=label)
               for _, _, color, marker, label in series]
    fig.suptitle("Paired STN-DBS effects and marginal 95% bootstrap intervals", y=0.98)
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.91))
    fig.savefig(out, dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plot(args.stats, args.out)
