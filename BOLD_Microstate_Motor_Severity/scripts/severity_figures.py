"""Render the aggregate forest plot and a private page of patient-level scatter plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

INK = "#182f41"
MUTED = "#4b6273"
GRID = "#e1e9ef"
CONDITION = {"OFF": "#246187", "ON": "#ae4f2d"}


def configure():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei")
    font = next((name for name in preferred if name in installed), None)
    if font is None:
        raise RuntimeError("Install a Simplified Chinese font for the severity plots")
    plt.rcParams.update({
        "font.family": [font, "DejaVu Sans"], "font.size": 10.5,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False, "pdf.fonttype": 42,
    })


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def forest(stage, path):
    fig, ax = plt.subplots(figsize=(11.5, 7))
    fig.subplots_adjust(left=.255, right=.655, top=.76, bottom=.16)
    fig.text(.05, .945, "BOLD 微状态与同条件运动严重度", fontsize=21, weight="bold")
    fig.text(.05, .891, "冻结 A3  ·  历史运动评分  ·  每项按患者计算", fontsize=11.3, color=MUTED)
    positions = np.arange(len(stage))
    ylabels = [f"{r.condition}    {r.label}" for r in stage.itertuples(index=False)]
    ax.set(ylim=(len(stage) - .4, -.6), xlim=(-1, 1), xticks=[-1, -.5, 0, .5, 1])
    ax.set_yticks(positions, ylabels)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=.8)
    ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.set_xlabel("Spearman ρ")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#aebcc5")
    for p, label in enumerate(ax.get_yticklabels()):
        label.set_color(CONDITION[stage.condition.iloc[p]])
    cols = (1.035, 1.245, 1.50, 1.75)
    for x, name in zip(cols, ("ρ", "Holm11 P", "Holm15 P", "n"), strict=True):
        ax.text(x, -.82, name, transform=ax.get_yaxis_transform(), color=MUTED, weight="bold")
    for i, row in enumerate(stage.itertuples(index=False)):
        color = CONDITION[row.condition]
        if row.status == "OK":
            if np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high):
                ax.hlines(i, row.ci95_low, row.ci95_high, color=color, lw=2, zorder=2)
            ax.plot(row.rho, i, "o", color=color, ms=6, zorder=3)
            values = (f"{row.rho:+.2f}", f"{row.holm11_p:.3f}", f"{row.holm15_p:.3f}", str(row.n_patients))
        else:
            values = ("NA", "NA", "NA", str(row.n_patients))
        for x, value in zip(cols, values, strict=True):
            ax.text(x, i, value, transform=ax.get_yaxis_transform(), va="center", fontsize=9.8)
    fig.text(.05, .048, "线段为 95% 患者级 Bootstrap 区间。正相关表示指标较高的患者评分也较高。",
             color=MUTED)
    save(fig, path)


def scatter(registry, stage, observations, path):
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharey=True)
    fig.subplots_adjust(left=.07, right=.97, bottom=.09, top=.85, hspace=.56, wspace=.29)
    fig.text(.061, .945, "同一条件内的逐人运动评分与微状态指标", fontsize=18, weight="bold")
    fig.text(.061, .903, "私有研究图  ·  历史评分  ·  评分越高表示记录的症状越重", fontsize=10.8, color=MUTED)
    for i, (ax, spec) in enumerate(zip(axes.flat, registry.itertuples(index=False))):
        row = stage.iloc[i]
        d = observations.loc[observations.condition.eq(spec.condition), [spec.metric, "score"]].dropna()
        color = CONDITION[spec.condition]
        ax.grid(axis="y", color=GRID, lw=.65, zorder=0)
        ax.scatter(d[spec.metric] * spec.display_multiplier, d.score, c=color, s=25,
                   alpha=.84, edgecolors="white", linewidths=.48, zorder=3)
        ax.set_title(f"{spec.condition}　{spec.label}", fontsize=10.3, color=color, loc="left", pad=7)
        ax.set_xlabel(spec.display_unit)
        if i % 4 == 0:
            ax.set_ylabel("运动评分（分）")
        line = f"ρ={row.rho:+.2f}   n={row.n_patients}" if row.status == "OK" else "相关不可估计"
        ax.text(.97, .97, line, transform=ax.transAxes, ha="right", va="top", fontsize=8.6,
                color=INK, bbox=dict(facecolor="white", edgecolor="none", alpha=.8, pad=1.5))
    axes.flat[-1].set_visible(False)
    save(fig, path)


def draw_figures(registry, stage, observations, public_dir, private_dir):
    configure()
    forest(stage, public_dir / "Motor_Severity_Forest")
    scatter(registry, stage, observations, private_dir / "Motor_Severity_Patient_Scatter")
