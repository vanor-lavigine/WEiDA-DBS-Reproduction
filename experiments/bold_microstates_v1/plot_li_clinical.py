"""Plot the published EEG coefficients and group-level BOLD correlations for DBS OFF."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


# Li et al. (2021), doi:10.3389/fnins.2021.697909, Figure 4, rightmost panels.
EEG_REFERENCE = {"duration_sec": 0.68, "occurrence_per_sec": -0.76}


def draw(summary, destination):
    with summary.open(encoding="utf-8", newline="") as stream:
        rows = {
            (row["variant"], row["metric"]): row
            for row in csv.DictReader(stream)
            if row["contrast"] == "off_severity" and row["state"] == "all"
        }
    font_names = set(font_manager.get_font_names())
    chinese_font = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                         if name in font_names), None)
    chinese = chinese_font is not None
    style = {"font.family": chinese_font or "DejaVu Sans", "axes.unicode_minus": False,
             "font.size": 10, "text.color": "#243640", "axes.labelcolor": "#243640"}
    reference, peak, direct = "#5f513d", "#076c6b", "#526487"
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), sharey=True)
        fig.subplots_adjust(left=.23, right=.965, top=.78, bottom=.30, wspace=.16)
        fig.suptitle("关机时，运动评分与状态持续时间和出现率的相关性" if chinese else
                     "DBS OFF: motor severity and temporal state measures",
                     x=.035, y=.978, ha="left", fontsize=17, fontweight="bold")
        labels = (["EEG 论文（20 人）", "BOLD 峰间回填（24 人）", "BOLD 逐帧回填（24 人）"] if chinese else
                  ["Published EEG (n = 20)", "BOLD peak-based (n = 24)", "BOLD all-frame (n = 24)"])
        titles = (["全部状态段的平均持续时间", "全部状态每秒总出现率"] if chinese else
                  ["Mean episode duration", "Total occurrences per second"])
        for ax, (metric, reference_r), title in zip(axes, EEG_REFERENCE.items(), titles):
            ax.set_title(title, fontsize=12, loc="left", pad=8)
            ax.axvline(0, color="#78868b", lw=.9, ls="--", zorder=0)
            ax.grid(axis="x", color="#e4e9e9", lw=.8)
            ax.set_axisbelow(True)
            ax.plot(reference_r, 2, marker="D", ms=6, color=reference)
            ax.annotate(f"{reference_r:+.2f}", (reference_r, 2), xytext=(0, 8),
                        textcoords="offset points", ha="center", color=reference)
            for variant, y, color in (("peak", 1, peak), ("direct", 0, direct)):
                row = rows[variant, metric]
                if int(row["n"]) != 24:
                    raise ValueError("Unexpected clinical sample size")
                r, low, high = (float(row[name]) for name in ("r", "ci_low", "ci_high"))
                ax.errorbar(r, y, xerr=[[r - low], [high - r]], fmt="o", markersize=6,
                            color=color, capsize=3.5, elinewidth=1.7)
                ax.annotate(f"{r:+.3f}", (r, y), xytext=(0, 8), textcoords="offset points",
                            ha="center", color=color)
            ax.set_xlim(-1.04, 1.04)
            ax.set_ylim(-.46, 2.52)
            ax.set_xticks([-1, -.5, 0, .5, 1], ["−1", "−0.5", "0", "+0.5", "+1"])
            ax.set_yticks([2, 1, 0], labels)
            ax.tick_params(axis="y", length=0, pad=9)
            ax.tick_params(axis="x", colors="#51626a", length=3)
            for name in ("top", "left", "right"):
                ax.spines[name].set_visible(False)
            ax.spines["bottom"].set_color("#c2ccce")
            ax.set_xlabel("Pearson r", labelpad=3)
        caption = ("菱形取自 Li 等（2021）图 4。圆点为本次 BOLD 估计，横线为逐项 95% 置信区间。\n"
                   "两项研究的患者不同。两种 BOLD 回填使用同一批患者、同一套四状态模板。" if chinese else
                   "Diamonds: published EEG estimates (Li et al., 2021, Fig. 4). Circles and bars: BOLD estimates and pointwise 95% CIs.\n"
                   "EEG and BOLD come from different cohorts. Both BOLD methods use the same patients and four templates.")
        fig.text(.035, .055, caption, fontsize=9.5, linespacing=1.45, color="#465960")
        fig.savefig(destination, dpi=175, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stats", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(args.stats, args.output)
