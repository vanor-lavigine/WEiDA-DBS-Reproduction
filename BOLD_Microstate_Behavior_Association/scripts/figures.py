"""Render public effect summaries and private patient-level scatter plots."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from io_sources import ARCHIVES, ENDPOINTS, PRIVATE, PROJECT, REFERENCE


RESULT_FIGURES = PROJECT / "figures"
COLOR = "#116b71"
INK = "#172c3d"
MUTED = "#496171"
PALETTE = (COLOR, "#6d589c", "#b55a2a")
LABELS = {f"delta_{item['field']}": item["label"] for item in ENDPOINTS}
SOURCE_LABELS = dict(zip(ARCHIVES, ("4-1 参照", "3-42 核对", "4-3 备用"), strict=True))


def theme():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC", "SimHei")
    font = next((name for name in preferred if name in installed), None)
    if font is None:
        raise RuntimeError("Install a Simplified Chinese font for the figure labels")
    plt.rcParams.update({"font.family": [font, "DejaVu Sans"], "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "axes.titlecolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK,
                         "pdf.fonttype": 42, "axes.unicode_minus": False})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=360, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def rho_axis(ax, ys, show_labels=True):
    ax.axvline(0, color="#8096a2", lw=1, ls="--", zorder=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e0e9ed", lw=.8)
    ax.set(xlim=(-1, 1), ylim=(max(ys) + .55, min(ys) - .55),
           xticks=[-1, -.5, 0, .5, 1], xlabel="Spearman ρ")
    ax.set_yticks(ys, list(LABELS.values()) if show_labels else [""] * len(ys))
    ax.tick_params(axis="y", length=0, pad=10)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#aab9c1")


def draw_effects(verified, archive):
    ref = archive.loc[archive.score_source.eq(REFERENCE)]
    data = ref if ref.status.eq("OK").any() else verified
    historical = data.score_source.eq(REFERENCE).all()
    title = "微状态变化与运动评分变化"
    subtitle = (f"历史评分归档 4-1 · 当前访视未核实 · 每项最多 {int(data.n_pairs.max())} 对"
                if historical else "评分与扫描访视核实路径")
    fig, ax = plt.subplots(figsize=(9.9, 4.4))
    fig.subplots_adjust(left=.19, right=.72, top=.72, bottom=.22)
    fig.text(.055, .94, title, fontsize=19, weight="bold")
    fig.text(.055, .85, subtitle, color=MUTED, fontsize=11)
    indexes = np.arange(len(ENDPOINTS))
    rho_axis(ax, indexes)
    for y, r in enumerate(data.itertuples(index=False)):
        if np.isfinite(r.rho):
            if np.isfinite(r.ci95_low) and np.isfinite(r.ci95_high):
                ax.hlines(y, r.ci95_low, r.ci95_high, color=COLOR, lw=2.1, zorder=2)
            ax.plot(r.rho, y, "o", color=COLOR, ms=7, zorder=3)
            info = f"ρ {r.rho:+.2f}    Holm P {r.permutation_holm_p:.3f}    n={r.n_pairs}"
        else:
            info = f"无法估计    n={r.n_pairs}"
        ax.text(1.035, y, info, transform=ax.get_yaxis_transform(), va="center", fontsize=10.2, color=INK)
    footer = ("线段为 95% 患者级 Bootstrap 区间。Holm 在四项置换检验内校正。"
              + (" 访视核实合格对数为 0。" if not verified.status.eq("OK").any() and historical else ""))
    fig.text(.055, .044, footer, fontsize=9.8, color=MUTED)
    save(fig, RESULT_FIGURES / "Association_Effect_Summary")


def draw_source_sensitivity(sensitivity):
    from matplotlib.lines import Line2D

    full = sensitivity.loc[sensitivity.sensitivity_set.eq("all_evaluable_for_source")]
    present = [v for v in ARCHIVES if full.loc[full.score_source.eq(v), "status"].eq("OK").any()]
    if len(present) < 2:
        return
    full = full.loc[full.score_source.isin(present)]
    same = sensitivity.loc[sensitivity.sensitivity_set.eq("same_both_totals_across_readable_sources")
                           & sensitivity.score_source.isin(present)]
    fig, axs = plt.subplots(1, 2, figsize=(10.9, 4.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.95, top=.72, bottom=.23, wspace=.19)
    fig.text(.05, .94, "运动评分来源敏感性", fontsize=19, weight="bold")
    fig.text(.05, .86, "历史归档评分 · 患者配对 · 条件于冻结影像模板", color=MUTED)
    handles = []
    for i, (subset, heading) in enumerate(((full, "各版本有完整数据的配对"), (same, "两侧总分一致的配对"))):
        ax = axs[i]
        rho_axis(ax, np.arange(len(ENDPOINTS)))
        if i:
            ax.tick_params(labelleft=False)
        ax.set_title(heading, fontsize=11.5, pad=12)
        for k, source in enumerate(present):
            d = subset.loc[subset.score_source.eq(source)].set_index("endpoint").reindex(LABELS)
            color = PALETTE[k]
            offset = (k - (len(present) - 1) / 2) * .18
            for j, r in enumerate(d.itertuples(index=False)):
                if np.isfinite(r.rho):
                    ax.hlines(j + offset, r.ci95_low, r.ci95_high, color=color, lw=1.8)
                    ax.plot(r.rho, j + offset, "o", color=color, ms=5.5)
            if i == 0:
                handles.append(Line2D([0], [0], color=color, marker="o", lw=1.8, label=SOURCE_LABELS[source]))
        if not subset.status.eq("OK").any():
            ax.text(.5, .5, "当前配对不足或相关不可估计", transform=ax.transAxes, ha="center", va="center")
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(.56, .035))
    save(fig, RESULT_FIGURES / "Score_Source_Sensitivity")


def draw_scatter(name, data, estimates):
    fig, axs = plt.subplots(2, 2, figsize=(10.8, 7.4), sharey=True)
    fig.subplots_adjust(left=.09, right=.96, top=.79, bottom=.11, wspace=.20, hspace=.44)
    fig.text(.06, .947, "每名患者的运动评分变化与微状态变化", fontsize=18, weight="bold")
    fig.text(.06, .892, f"私有研究图 · {name} · 评分改善为 OFF 减 ON",
             color=MUTED)
    for k, (ax, item) in enumerate(zip(axs.flat, ENDPOINTS, strict=True)):
        field = f"delta_{item['field']}"
        d = data[[field, "improvement"]].dropna()
        row = estimates.loc[estimates.endpoint.eq(field)].iloc[0]
        ax.axvline(0, color="#a4b5be", lw=.9, ls="--")
        ax.axhline(0, color="#a4b5be", lw=.9, ls="--")
        ax.scatter(d[field] * item["display_multiplier"], d.improvement, s=29, alpha=.86,
                   color=COLOR, edgecolors="white", linewidths=.55, zorder=3)
        label = f"ρ={row.rho:+.2f}，Holm P={row.permutation_holm_p:.3f}" if row.status == "OK" else "相关不可估计"
        ax.set_title(f"{item['label']}    n={row.n_pairs}    {label}", loc="left", fontsize=10.9)
        ax.set_xlabel(f"ON−OFF（{item['display_unit']}）")
        if k % 2 == 0:
            ax.set_ylabel("运动评分改善（分）")
    save(fig, PRIVATE / "figures" / name / "Paired_Change_Scatter")


def draw_figures(verified, archived, sensitivity, joined, verified_data):
    theme()
    draw_effects(verified, archived)
    draw_source_sensitivity(sensitivity)
    for name, table in joined.items():
        estimate = archived.loc[archived.score_source.eq(name)]
        draw_scatter(name, table, estimate)
    if len(verified_data):
        draw_scatter("verified_visit", verified_data, verified)
