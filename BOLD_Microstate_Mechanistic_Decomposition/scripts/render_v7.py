"""Create figures from real V7 aggregate and private paired readouts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

from frozen_inputs import PRIVATE, PROJECT, RESULTS

OFF, ON = "#556273", "#087F75"
ACCENT, VIOLET = "#B4452C", "#6D5490"
INK, MUTED, GRID = "#182734", "#4B5966", "#DEE5E9"
PUBLIC = PROJECT / "figures"
SEALED = PRIVATE / "figures"


def setup():
    candidates = (Path("C:/Windows/Fonts/msyh.ttc"), Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    for font in candidates:
        if font.is_file():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
            break
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "grid.color": GRID, "grid.linewidth": .65, "axes.axisbelow": True,
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def table(name):
    return pd.read_csv(RESULTS / f"{name}.csv").set_index("metric")


def save(fig, name, directory=PUBLIC):
    directory.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=.14)
    plt.close(fig)


def title(fig, headline, detail):
    fig.text(.055, .97, headline, fontsize=16, fontweight="bold", va="top")
    fig.text(.055, .900, detail, fontsize=9, color=MUTED, va="top")


def errors(value, low, high):
    return np.array([[max(value - low, 0)], [max(high - value, 0)]])


def two_conditions(ax, row, ylabel, heading, individuals=None, unit=1):
    if individuals is not None:
        ax.plot([0, 1], individuals.T * unit, color=MUTED, alpha=.29, lw=.65, zorder=1)
        ax.scatter(np.broadcast_to([0, 1], individuals.shape).ravel(),
                   (individuals * unit).ravel(), s=8, color=MUTED, alpha=.29, linewidths=0, zorder=2)
    ax.plot([0, 1], [row.off_mean, row.on_mean], color=ON, lw=2, zorder=3)
    for x, key, color in ((0, "off", OFF), (1, "on", ON)):
        value = row[f"{key}_mean"]
        low, high = row[f"{key}_ci95_low"], row[f"{key}_ci95_high"]
        ax.errorbar(x, value, yerr=errors(value, low, high), fmt="o", color=color, ms=7,
                    lw=1.4, capsize=3, markeredgecolor="white", markeredgewidth=.55, zorder=4)
    ax.set(xticks=[0, 1], xticklabels=["DBS-OFF", "DBS-ON"], xlim=(-.38, 1.38), ylabel=ylabel, title=heading)
    ax.grid(axis="y")


def effects(ax, rows, labels, xlabel, colors):
    ax.axvline(0, color=MUTED, ls=(0, (3, 3)), lw=1, zorder=1)
    for y, (row, color) in enumerate(zip(rows, colors, strict=True)):
        value, low, high = row.effect, row.ci95_low, row.ci95_high
        ax.errorbar(value, y, xerr=errors(value, low, high), fmt="o", color=color, ms=7, lw=1.5,
                    capsize=3, markeredgecolor="white", markeredgewidth=.55, zorder=3)
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel=xlabel, ylim=(len(labels) - .45, -.55))
    ax.grid(axis="x")


def render_gev():
    data = table("M1_GEV_Decomposition")
    fig, ax = plt.subplots(figsize=(7.1, 3.25))
    fig.subplots_adjust(left=.27, right=.96, bottom=.22, top=.72)
    title(fig, "M1 GEV 的精确分解", "25 名患者 · ON−OFF · 点为配对均值，误差线为患者 Bootstrap 95% 区间")
    rows = [data.loc[k] for k in ("gev_delta_complete_component_pairs", "amplitude_contribution", "fit_contribution")]
    effects(ax, rows, ["M1 GEV 总变化", "幅度权重贡献 W", "空间拟合贡献 F"], "GEV 变化（百分点）", [INK, ON, VIOLET])
    save(fig, "Figure_01_M1_GEV_Decomposition")

    for private in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(7.65, 3.50))
        fig.subplots_adjust(left=.1, right=.97, bottom=.16, top=.72, wspace=.44)
        detail = ("25 条线对应 25 名患者，粗线和区间为两条件的配对组级摘要" if private else
                  "25 名患者配对 · 条件均值及配对患者 Bootstrap 95% 区间")
        title(fig, "M1 幅度权重与空间拟合", detail)
        paired = pd.read_csv(PRIVATE / "results/M1_GEV_Decomposition.csv") if private else None
        for ax, field, ylabel, heading, raw, factor in [
            (axes[0], "amplitude_weight", "GFP² 权重（%）", "幅度权重 W", "m1_weight", 100),
            (axes[1], "weighted_spatial_fit", "加权空间相关平方", "空间拟合 F", "m1_fit", 1),
        ]:
            points = paired[[f"off_{raw}", f"on_{raw}"]].to_numpy(float) if private else None
            two_conditions(ax, data.loc[field], ylabel, heading, individuals=points, unit=factor)
        name = "Figure_02_M1_Paired_Individuals" if private else "Figure_02_M1_Weight_and_Fit"
        save(fig, name, SEALED if private else PUBLIC)


def render_duration():
    data = table("M4_Duration_Decomposition")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), gridspec_kw={"width_ratios": [.8, 1.2]})
    fig.subplots_adjust(left=.10, right=.97, bottom=.20, top=.70, wspace=.60)
    title(fig, "M4 平均持续时间的精确分解", "使用原有效窗口 · 点为患者均值，误差线为 95% 区间")
    two_conditions(axes[0], data.loc["duration"], "持续时间（秒）", "两条件的原量纲估计")
    keys = ("log_duration", "coverage_contribution", "occurrence_contribution")
    effects(axes[1], [data.loc[k] for k in keys], ["持续时间总变化", "占用率贡献", "出现率贡献"],
            "ON−OFF（自然对数比）", [INK, ON, VIOLET])
    axes[1].set_title("可加的对数分量")
    save(fig, "Figure_03_M4_Duration_Decomposition")


def render_exit():
    data = table("M4_Exit_Dynamics")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [.80, 1.20]})
    fig.subplots_adjust(left=.12, right=.97, bottom=.20, top=.70, wspace=.49)
    title(fig, "M4 的实际退出频率与去向", "25 名患者 · 退出频率按有效时间归一，去向先在每名患者内计算")
    two_conditions(axes[0], data.loc["exit_frequency"], "退出次数 / 有效分钟", "退出频率")
    axes[0].set_ylim(bottom=0)
    ax = axes[1]
    items = data.loc[["q41", "q42", "q43"]]
    for side, dx, color in (("off", -.105, OFF), ("on", .105, ON)):
        values = items[f"{side}_mean"].to_numpy()
        left, right = (items[f"{side}_ci95_{where}"].to_numpy() for where in ("low", "high"))
        ax.errorbar(np.arange(3) + dx, values, yerr=[values-left, right-values], fmt="o", capsize=3,
                    color=color, label=f"DBS-{side.upper()}", ms=6)
    ax.set(xticks=range(3), xticklabels=["M1", "M2", "M3"], ylabel="平均条件概率（%）",
           title="M4 退出后的去向", ylim=(0, 60))
    ax.legend(frameon=False, loc="upper right", ncol=2, fontsize=8)
    ax.grid(axis="y")
    save(fig, "Figure_04_M4_Exit_Dynamics")


def render_windows():
    samples = pd.read_csv(RESULTS / "Within_Scan_Sensitivity.csv")
    chosen = samples.loc[samples["sample"].eq("Eligible in this window")]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.subplots_adjust(left=.17, right=.97, bottom=.19, top=.73, wspace=.62)
    title(fig, "固定标签下的连续窗口取样", "各窗按预设的有效时间和状态支持筛选。不同窗的有效患者对可不同。")
    for ax, metric, heading, label, color in [
        (axes[0], "gev_1", "M1 GEV", "GEV ON−OFF（百分点）", ON),
        (axes[1], "duration_4", "M4 平均持续时间", "持续时间 ON−OFF（秒）", VIOLET),
    ]:
        rows = chosen.loc[chosen.metric.eq(metric)]
        labels = [f"{name}（{row.n_patients} 对）" for name, row in zip(
            ("全窗", "前半段", "后半段", "中间半段"), rows.itertuples(index=False), strict=True)]
        effects(ax, list(rows.itertuples(index=False)), labels, label, [color] * len(rows))
        ax.set_title(heading)
    save(fig, "Figure_07_Within_Scan_Sensitivity")


def main():
    setup()
    render_gev()
    render_duration()
    render_exit()
    render_windows()


if __name__ == "__main__":
    main()
