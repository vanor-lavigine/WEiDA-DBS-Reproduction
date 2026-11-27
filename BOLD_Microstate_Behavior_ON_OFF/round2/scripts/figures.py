"""Chinese PNG charts for aggregate publication and private trajectory review."""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


NAVY = "#162f49"
TEAL = "#147d72"
LIGHT_TEAL = "#a4d1ca"
GRAY = "#a5b3c2"
SLATE = "#546881"
ORANGE = "#bb6032"
BLUE = "#3872a6"
GRID = "#e3ebf2"
WHITE = "#ffffff"

LABELS = {
    "total": "运动总分",
    "rigidity": "强直",
    "bradykinesia": "运动徐缓",
    "tremor": "震颤",
    "gait_balance": "步态与平衡",
    "remaining_four": "其余四项",
}
ENDPOINTS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
COMPONENTS = ("bradykinesia", "tremor", "rigidity", "gait_balance", "remaining_four", "total")
PATTERNS = {
    "both_shorter": "M2、M4 均缩短",
    "only_m2_shorter": "只有 M2 缩短",
    "only_m4_shorter": "只有 M4 缩短",
    "both_longer": "M2、M4 均延长",
}


def _chinese_font(font_path=None):
    windows = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    if font_path:
        path = Path(font_path)
        if not path.is_file():
            raise ValueError("Specified Chinese font file does not exist")
        font_manager.fontManager.addfont(str(path))
        return font_manager.FontProperties(fname=str(path)).get_name()
    for name in ("msyh.ttc", "NotoSansSC-VF.ttf", "simhei.ttf"):
        candidate = windows / name
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
            return font_manager.FontProperties(fname=str(candidate)).get_name()
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei"):
        if name in installed:
            return name
    raise RuntimeError("Install a supported Chinese font or supply an explicit font_path")


def _theme():
    return {"figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
            "axes.unicode_minus": False, "axes.labelcolor": NAVY, "xtick.color": NAVY,
            "ytick.color": NAVY, "text.color": NAVY, "font.size": 10}


def _head(fig, title):
    fig.suptitle(title, x=.02, ha="left", fontsize=18, fontweight="bold")


def _axes(ax, grid="x"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=.75)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRAY)
    ax.tick_params(axis="y", length=0)


def _save(fig, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor=WHITE)
    plt.close(fig)


def clinical_response(profile, destination):
    rows = profile.set_index("endpoint", verify_integrity=True).loc[list(ENDPOINTS)]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 4.6), layout="constrained")
    _head(fig, "五项运动评分的个体改善分布")
    ax.set_title("方向按逐对 OFF−ON 判定，所有有效配对均计入分母", loc="left", color=SLATE, fontsize=10)
    offset = np.zeros(len(rows))
    for key, title, color, label_color in (
        ("improved", "改善", TEAL, WHITE),
        ("unchanged", "不变", GRAY, NAVY),
        ("worse", "变差", ORANGE, WHITE),
    ):
        width = 100 * rows[f"{key}_fraction"].to_numpy()
        ax.barh(y, width, left=offset, height=.57, label=title, color=color)
        for pos, total, start, span in zip(y, rows[f"n_{key}"], offset, width, strict=True):
            if span >= 10:
                ax.text(start + span/2, pos, str(total), va="center", ha="center", color=label_color)
        offset += width
    for pos, row in zip(y, rows.itertuples(), strict=True):
        ax.text(103, pos, f"{row.n_improved} / {row.n_unchanged} / {row.n_worse} 人（n={row.n}）", va="center",
                color=NAVY, fontsize=9)
    ax.set(yticks=y, yticklabels=[LABELS[key] for key in ENDPOINTS], xlim=(0, 133),
           xticks=[0, 25, 50, 75, 100], xticklabels=["0%", "25%", "50%", "75%", "100%"],
           xlabel="有效配对比例，右侧依次为改善／不变／变差人数")
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1.01))
    _axes(ax)
    _save(fig, destination)


def clinical_composition(composition, destination):
    rows = composition.set_index("component", verify_integrity=True).loc[list(COMPONENTS)]
    fig, ax = plt.subplots(figsize=(10.8, 4.9), layout="constrained")
    _head(fig, "运动总分改善的点数构成")
    ax.set_title(f"分母为总分平均改善量，{int(rows.loc['total', 'n'])} 对有效配对",
                 loc="left", color=SLATE, fontsize=10)
    steps = rows.improvement_mean.to_numpy()
    end = np.cumsum(steps[:-1])
    start = np.r_[0., end[:-1]]
    limits = np.r_[0., end, steps[-1]]
    padding = max(np.ptp(limits)*.08, .2)
    for i, (row, bottom, top) in enumerate(zip(rows.iloc[:-1].itertuples(), start, end, strict=True)):
        value = row.improvement_mean
        ax.bar(i, abs(value), bottom=min(bottom, top), width=.64, color=TEAL if value >= 0 else ORANGE)
        ax.plot([i+.32, i+.68], [top, top], color=SLATE, lw=.9, linestyle="--")
        frac = row.fraction_of_total_mean
        share = f"{frac:.1%}" if np.isfinite(frac) else "比例未定义"
        text = f"{value:+.2f} 分\n{share}"
        y = top + (padding*.2 if value >= 0 else -padding*.2)
        ax.text(i, y, text, ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.bar(len(steps)-1, steps[-1], color=NAVY, width=.64)
    ax.text(len(steps)-1, steps[-1]+(padding*.2 if steps[-1] >= 0 else -padding*.2), f"{steps[-1]:.2f} 分",
            ha="center", va="bottom" if steps[-1] >= 0 else "top")
    ax.axhline(0, color=SLATE, lw=.85)
    ax.set(xticks=range(len(steps)), xticklabels=[LABELS[key] for key in COMPONENTS],
           ylabel="累计平均改善（分，OFF−ON）", ylim=(limits.min()-padding, limits.max()+2*padding))
    _axes(ax, "y")
    _save(fig, destination)


def brain_patterns(patterns, states, destination):
    rows = patterns.set_index("pattern", verify_integrity=True).loc[list(PATTERNS)]
    zero = int(patterns.set_index("pattern").at["zero_in_either", "count"])
    n = int(rows.n_pairs.iloc[0])
    shifts = states.set_index("state", verify_integrity=True).loc[["M2", "M4"]]
    fig, (left, right) = plt.subplots(1, 2, layout="constrained", figsize=(12, 4.8),
                                    gridspec_kw={"width_ratios": [1.15, 1]})
    _head(fig, "M2 与 M4 持续时间的个体变化组合")
    left.set_title(f"{n} 对中的方向组合，任一态零差 {zero} 人", loc="left", fontsize=10)
    y = np.arange(len(rows))
    left.barh(y, rows["count"], color=[TEAL, BLUE, BLUE, ORANGE], height=.56)
    for pos, row in zip(y, rows.itertuples(), strict=True):
        left.text(row.count+.25, pos, f"{row.count} 人（{row.proportion:.0%}）", va="center")
    left.set(yticks=y, yticklabels=list(PATTERNS.values()), xlabel="人数",
             xlim=(0, max(rows["count"].max()+5, 5)))
    left.invert_yaxis()
    left.xaxis.set_major_locator(MaxNLocator(integer=True))
    right.set_title("持续时间差的中位数与四分位范围", loc="left", fontsize=10)
    right.axvline(0, color=SLATE, lw=.9)
    for i, row in enumerate(shifts.itertuples(index=False)):
        right.hlines(i, row.q1_s, row.q3_s, color=TEAL, lw=3)
        right.plot(row.median_s, i, "o", color=NAVY, ms=5)
    right.set(yticks=[0, 1], yticklabels=["M2", "M4"], xlabel="ON−OFF（秒）", ylim=(1.55, -.55))
    _axes(left)
    _axes(right)
    _save(fig, destination)


def hc_reference(reference, hc_durations, destination):
    (r,) = reference.itertuples(index=False)
    hcs = np.asarray(hc_durations, dtype=float)
    if hcs.shape != (r.n_hc,):
        raise ValueError("HC plot count differs from the released summary")
    np.testing.assert_allclose([r.hc_center_s, r.hc_q1_s, r.hc_median_s, r.hc_q3_s],
                               [hcs.mean(), *np.percentile(hcs, [25, 50, 75])], rtol=0, atol=1e-11)
    fig, (left, middle, right) = plt.subplots(1, 3, layout="constrained", figsize=(14.3, 4.55),
                                              gridspec_kw={"width_ratios": [1.15, 1, 1.2]})
    _head(fig, "M4 持续时间到健康参照的配对距离")
    left.set_title(f"原共同窗持续时间，HC n={r.n_hc}", loc="left", fontsize=10)
    left.axvline(r.hc_center_s, color=TEAL, lw=1.25, linestyle="--")
    left.hlines(0, r.hc_q1_s, r.hc_q3_s, lw=5, color=LIGHT_TEAL)
    left.plot(r.hc_median_s, 0, "o", color=TEAL, ms=6)
    left.plot([r.off_duration_mean_s, r.on_duration_mean_s], [1, 2], "D", color=NAVY, ms=5)
    left.set(yticks=range(3), yticklabels=["HC 中位数及四分位", "OFF 均值", "ON 均值"],
             xlabel=f"持续时间（秒），虚线为 HC 均值 {r.hc_center_s:.2f} 秒", ylim=(2.6, -.6))

    middle.set_title(f"到同一 HC 均值的个体距离，n={r.n_pairs}", loc="left", fontsize=10)
    for i, condition in enumerate(("off", "on")):
        low = getattr(r, f"{condition}_distance_q1_s")
        high = getattr(r, f"{condition}_distance_q3_s")
        middle.hlines(i, low, high, color=LIGHT_TEAL, lw=5)
        middle.plot(getattr(r, f"{condition}_distance_median_s"), i, "o", color=TEAL, ms=5)
        middle.plot(getattr(r, f"{condition}_distance_mean_s"), i, "D", color=NAVY, ms=5)
    middle.set(yticks=[0, 1], yticklabels=["OFF", "ON"], ylim=(1.55, -.55),
               xlabel="距离（秒），圆为中位数，菱形为均值")

    right.set_title(f"更近 {r.n_closer} 人／更远 {r.n_farther} 人／不变 {r.n_unchanged} 人", loc="left", fontsize=10)
    right.axvline(0, color=SLATE, lw=.9)
    right.hlines(0, r.ci95_low_s, r.ci95_high_s, color=TEAL, lw=2.4)
    right.plot(r.distance_change_mean_s, 0, "o", color=NAVY, ms=6)
    right.text(.5, .69, f"{r.distance_change_mean_s:+.2f} [{r.ci95_low_s:.2f}, {r.ci95_high_s:.2f}] 秒",
               transform=right.transAxes, ha="center")
    right.set(yticks=[0], yticklabels=["平均距离变化"], ylim=(-.8, .8),
              xlabel="OFF 距离−ON 距离（秒）及 95% 区间")
    for ax in (left, middle, right):
        _axes(ax)
    _save(fig, destination)


def clinical_trajectories(pairs, clinical_profile, destination):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout="constrained")
    _head(fig, "运动评分的逐人 ON／OFF 配对轨迹")
    summary = clinical_profile.set_index("endpoint", verify_integrity=True)
    for ax, endpoint in zip(axes.ravel()[:5], ENDPOINTS, strict=True):
        row = summary.loc[endpoint]
        vals = pairs[[f"{endpoint}_off", f"{endpoint}_on"]].to_numpy(float)
        for off, on in vals:
            color = TEAL if off > on else ORANGE if off < on else SLATE
            ax.plot([0, 1], [off, on], "o-", color=color, alpha=.42, lw=.85, ms=3, zorder=2)
        ax.plot([0, 1], [row.off_mean, row.on_mean], "o-", color=NAVY, lw=2, ms=5, zorder=3)
        ax.set(xticks=[0, 1], xticklabels=["OFF", "ON"], xlim=(-.25, 1.25), ylabel="评分（分）")
        ax.set_title(f"{LABELS[endpoint]}   n={int(row.n)}", loc="left", fontsize=10)
        _axes(ax, "y")
    last = axes.ravel()[5]
    values = np.sort(pairs["total_improvement"].to_numpy(float))
    last.vlines(np.arange(1, len(values)+1), 0, values, color=TEAL, lw=1.45)
    last.axhline(0, color=SLATE, lw=.85)
    last.set_title("运动总分的连续改善量", loc="left", fontsize=10)
    last.set(xlabel="按改善量排序的配对序号", ylabel="OFF−ON（分）")
    _axes(last, "y")
    _save(fig, destination)


def brain_trajectories(pairs, tolerance, destination):
    ordered = pairs.sort_values(["delta_m4_s", "delta_m2_s", "patient_id"], kind="stable")
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(9, max(6, len(ordered)*.24)), layout="constrained",
                             sharey=True, sharex=True)
    _head(fig, "逐人 M2／M4 持续时间差")
    for (ax, column, label) in zip(axes, ("delta_m2_s", "delta_m4_s"), ("M2", "M4"), strict=True):
        changes = ordered[column].to_numpy(float)
        colors = np.where(changes < -tolerance, TEAL, np.where(changes > tolerance, ORANGE, SLATE))
        ax.barh(y, changes, color=colors, height=.65)
        ax.axvline(0, color=SLATE, lw=.85)
        ax.set_title(f"{label}   n={len(ordered)}", loc="left", fontsize=11)
        ax.set_xlabel("ON−OFF（秒）")
        _axes(ax)
    ticks = np.unique(np.r_[0, np.arange(4, len(ordered), 5), len(ordered)-1])
    axes[0].set(yticks=ticks, yticklabels=ticks+1, ylabel="配对序号，按 M4 变化排序")
    axes[0].invert_yaxis()
    _save(fig, destination)


def distance_trajectories(pairs, reference, tolerance, destination):
    (r,) = reference.itertuples(index=False)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 5), layout="constrained")
    _head(fig, "逐人 M4 到健康参照的距离变化")
    for row in pairs.itertuples():
        color = TEAL if row.distance_change_s > tolerance else ORANGE if row.distance_change_s < -tolerance else SLATE
        left.plot([0, 1], [row.off_distance_s, row.on_distance_s], "o-", color=color, lw=.85, alpha=.42, ms=3)
    left.plot([0, 1], [r.off_distance_mean_s, r.on_distance_mean_s], "D-", color=NAVY, lw=2, ms=5)
    left.set(xticks=[0, 1], xticklabels=["OFF", "ON"], xlim=(-.3, 1.3), ylabel="到 HC 均值的距离（秒）")
    left.set_title(f"同一参照 {r.hc_center_s:.2f} 秒，n={r.n_pairs}", loc="left", fontsize=10)
    changes = np.sort(pairs.distance_change_s.to_numpy(float))
    colors = np.where(changes > tolerance, TEAL, np.where(changes < -tolerance, ORANGE, SLATE))
    right.bar(np.arange(1, len(changes)+1), changes, color=colors, width=.65)
    right.axhline(0, color=SLATE, lw=.85)
    right.set_title(f"更近 {r.n_closer} 人／更远 {r.n_farther} 人／不变 {r.n_unchanged} 人", loc="left", fontsize=10)
    right.set(xlabel="按距离变化排序的配对序号", ylabel="OFF 距离−ON 距离（秒）")
    _axes(left, "y")
    _axes(right, "y")
    _save(fig, destination)


def render_figures(clinical_profile, composition, clinical_private, patterns, states, reference,
                   brain_private, hc_durations, public_dir, private_dir, *, font_path=None,
                   zero_tolerance_s=1e-12):
    """Render explicitly separated public and private assets and return their filenames."""
    public_dir, private_dir = Path(public_dir).resolve(), Path(private_dir).resolve()
    if public_dir.is_relative_to(private_dir) or private_dir.is_relative_to(public_dir):
        raise ValueError("Public and private figure directories must be separate")
    font = _chinese_font(font_path)
    public_candidates = (
        ("Clinical_Response_Proportions.png", clinical_response, (clinical_profile,)),
        ("Clinical_Improvement_Composition.png", clinical_composition, (composition,)),
        ("M2_M4_Response_Patterns.png", brain_patterns, (patterns, states)),
        ("M4_HC_Reference.png", hc_reference, (reference, hc_durations)),
    )
    private_candidates = (
        ("Clinical_Paired_Trajectories.png", clinical_trajectories, (clinical_private, clinical_profile)),
        ("M2_M4_Paired_Changes.png", brain_trajectories, (brain_private, zero_tolerance_s)),
        ("M4_Patient_Distances.png", distance_trajectories, (brain_private, reference, zero_tolerance_s)),
    )
    public = [item for item in public_candidates if all(arg is not None for arg in item[2])]
    private = [item for item in private_candidates if all(arg is not None for arg in item[2])]
    with matplotlib.rc_context({"font.family": [font, "DejaVu Sans"], **_theme()}):
        for directory, assets in ((public_dir, public), (private_dir, private)):
            for name, render, args in assets:
                render(*args, directory / name)
    return {"public": [name for name, _, _ in public], "private": [name for name, _, _ in private],
            "font": font}
