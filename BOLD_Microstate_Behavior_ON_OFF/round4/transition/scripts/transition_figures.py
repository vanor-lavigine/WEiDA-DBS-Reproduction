"""Draw aggregate A3 transition figures and a private paired trajectory."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import numpy as np


STATES = ("M1", "M2", "M3", "M4")
CONDITIONS = ("HC", "OFF", "ON")
CONTRASTS = ("OFF-HC", "ON-OFF", "ON-HC")
CELLS = [(source, target) for source in STATES for target in STATES]
NON_DIAGONAL = [(source, target) for source, target in CELLS if source != target]
M4_EXITS = [("M4", target) for target in STATES[:-1]]
METADATA_KEY = "Transition_Annotations"

_style_spec = spec_from_file_location(
    "transition_figure_style", Path(__file__).resolve().parents[3] / "round2/scripts/figures.py"
)
style = module_from_spec(_style_spec)
_style_spec.loader.exec_module(style)


def _save(fig, destination, annotations=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = None
    if annotations is not None:
        metadata = {METADATA_KEY: json.dumps(annotations, ensure_ascii=True, separators=(",", ":"), allow_nan=False)}
    try:
        fig.savefig(destination, dpi=160, facecolor=style.WHITE, metadata=metadata)
    finally:
        plt.close(fig)


def _ordered(frame, slots):
    return frame.set_index(["from_state", "to_state"], verify_integrity=True).loc[slots].reset_index()


def _prob(value):
    return f"{value:.3g}" if np.isfinite(value) else "NA"


def _fixed(value, *, signed=False):
    if not np.isfinite(value):
        return "NA"
    return f"{value * 100:+.1f}" if signed else f"{value * 100:.1f}"


def _numeric(value):
    return float(value) if np.isfinite(value) else None


def _descriptives_summary(rows):
    return [
        {"from_state": r.from_state, "to_state": r.to_state, "condition": r.condition,
         "n": int(r.n), "mean": _numeric(r.mean)}
        for r in rows.itertuples(index=False)
    ]


def _comparison_summary(rows):
    return [
        {"from_state": r.from_state, "to_state": r.to_state, "contrast": r.contrast,
         "n_a": int(r.n_a), "n_b": int(r.n_b),
         "n_pairs": int(r.n_pairs) if np.isfinite(r.n_pairs) else None,
         "mean_diff": _numeric(r.mean_diff), "ci95_low": _numeric(r.ci95_low),
         "ci95_high": _numeric(r.ci95_high), "fdr_q": _numeric(r.fdr_q), "holm_p": _numeric(r.holm_p)}
        for r in rows.itertuples(index=False)
    ]


def _ink(rgba):
    lightness = np.dot(rgba[:3], [.2126, .7152, .0722])
    return style.NAVY if lightness > .58 else style.WHITE


def _matrix_layout(ax):
    ax.set_xticks(range(4), STATES)
    ax.set_yticks(range(4), STATES)
    ax.set(xlabel="下一状态", ylabel="当前状态")
    ax.set_xticks(np.arange(-.5, 4), minor=True)
    ax.set_yticks(np.arange(-.5, 4), minor=True)
    ax.tick_params(axis="both", which="both", length=0, pad=7)
    ax.grid(which="minor", color=style.WHITE, linewidth=2)
    ax.spines[:].set_visible(False)


def _matrix_legend(fig, ax, image, label):
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=.8, aspect=26, pad=.12)
    cbar.set_label(label)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)


def _group_matrix(rows, destination):
    condition = rows.condition.iloc[0]
    cmap = LinearSegmentedColormap.from_list("weida_probability", ["#f6fbfc", "#61b4b6", "#174065"])
    cmap.set_bad("#e9eef2")
    matrix = rows["mean"].to_numpy(float).reshape(4, 4)
    fig, ax = plt.subplots(figsize=(6.5, 6.3), layout="constrained")
    style._head(fig, f"{condition} 全转移概率矩阵")
    ax.set_title("WEiDA 主口径 · 包含自循环 · 先按扫描计算后平均", loc="left", color=style.SLATE)
    image = ax.imshow(np.ma.masked_invalid(matrix * 100), cmap=cmap, vmin=0, vmax=100)
    _matrix_layout(ax)
    for i, r in enumerate(rows.itertuples(index=False)):
        y, x = divmod(i, 4)
        ink = _ink(cmap(image.norm(r.mean * 100))) if np.isfinite(r.mean) else style.NAVY
        value = f"{_fixed(r.mean)}%" if np.isfinite(r.mean) else "NA"
        ax.text(x, y-.09, value, ha="center", va="center", fontweight="semibold", color=ink, fontsize=13)
        ax.text(x, y+.19, f"n={int(r.n)}", ha="center", va="center", color=ink, fontsize=9)
    _matrix_legend(fig, ax, image, "平均转移概率（%）")
    _save(fig, destination, _descriptives_summary(rows))


def _delta_matrix(rows, destination, color_limit):
    contrast = rows.contrast.iloc[0]
    paired = contrast == "ON-OFF"
    cmap = LinearSegmentedColormap.from_list("weida_delta", ["#214f83", "#fcfbfa", "#9f3b1c"])
    cmap.set_bad("#e9eef2")
    matrix = rows.mean_diff.to_numpy(float).reshape(4, 4)
    fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
    style._head(fig, f"{contrast.replace('-', '−')} 转移概率差")
    basis = "患者内整对" if paired else "两组独立"
    ax.set_title(f"WEiDA 主口径 · {basis} · 非对角 q 来自 BH36", loc="left", color=style.SLATE)
    image = ax.imshow(np.ma.masked_invalid(matrix * 100), cmap=cmap, vmin=-color_limit, vmax=color_limit)
    _matrix_layout(ax)
    for i, r in enumerate(rows.itertuples(index=False)):
        y, x = divmod(i, 4)
        ink = _ink(cmap(image.norm(r.mean_diff * 100))) if np.isfinite(r.mean_diff) else style.NAVY
        counts = f"n={int(r.n_pairs)} 对" if paired else f"n={int(r.n_a)}/{int(r.n_b)}"
        correction = "自循环描述" if x == y else f"q={_prob(r.fdr_q)}"
        ax.text(x, y-.20, _fixed(r.mean_diff, signed=True), ha="center", va="center", fontweight="semibold",
                color=ink, fontsize=12)
        ax.text(x, y+.04, counts, ha="center", va="center", color=ink, fontsize=8.5)
        ax.text(x, y+.26, correction, ha="center", va="center", color=ink, fontsize=8.5)
    _matrix_legend(fig, ax, image, "均值差（百分点）")
    _save(fig, destination, _comparison_summary(rows))


def _interval(ax, row, y, color):
    if np.isfinite(row.ci95_low) and np.isfinite(row.ci95_high):
        ax.hlines(y, 100 * row.ci95_low, 100 * row.ci95_high, color=color, linewidth=1.6)
    if np.isfinite(row.mean_diff):
        ax.plot(row.mean_diff * 100, y, "o", color=color, ms=5)


def _contrast_limits(rows):
    values = rows[["mean_diff", "ci95_low", "ci95_high"]].to_numpy(float)
    values = np.abs(values[np.isfinite(values)])
    return max(1., float(values.max(initial=0) * 112))


def _forest(comparisons, destination):
    rows = comparisons.loc[comparisons.from_state.ne(comparisons.to_state)]
    limit = _contrast_limits(rows)
    fig = plt.figure(figsize=(21.0, 7.2), layout="constrained")
    style._head(fig, "WEiDA 全部非对角转移的均值差与 95% Bootstrap 区间")
    grid = fig.add_gridspec(1, 6, width_ratios=[2.8, 2.15] * 3)
    labels = [f"{source}→{target}" for source, target in NON_DIAGONAL]
    for i, contrast in enumerate(CONTRASTS):
        paired = contrast == "ON-OFF"
        data = _ordered(rows.loc[rows.contrast.eq(contrast)], NON_DIAGONAL)
        ax = fig.add_subplot(grid[0, 2*i])
        detail = fig.add_subplot(grid[0, 2*i+1], sharey=ax)
        ax.set_title(contrast.replace("-", "−"), loc="left", fontweight="semibold", fontsize=13)
        columns = (.14, .47, .78)
        for position, heading in zip(columns, ("n（配对）" if paired else "n（前/后）", "BH36 q", "Holm"), strict=True):
            detail.text(position, -.78, heading, ha="center", va="bottom", color=style.SLATE, fontsize=9)
        ax.axvline(0, color=style.SLATE, lw=.85)
        for y, r in enumerate(data.itertuples(index=False)):
            _interval(ax, r, y, style.BLUE)
            counts = str(int(r.n_pairs)) if paired else f"{int(r.n_a)}/{int(r.n_b)}"
            for position, value in zip(columns, (counts, _prob(r.fdr_q), _prob(r.holm_p)), strict=True):
                detail.text(position, y, value, ha="center", va="center", fontsize=9, color=style.NAVY)
        ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.45, -.55),
               xlim=(-limit, limit), xlabel="均值差（百分点）")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        style._axes(ax)
        detail.set(xlim=(0, 1))
        detail.axis("off")
    _save(fig, destination, _comparison_summary(rows))


def _m4_exits(group, comparisons, destination):
    rows = comparisons.loc[comparisons.contrast.eq("ON-OFF")]
    rows = _ordered(rows.loc[rows.from_state.eq("M4")], M4_EXITS)
    prepared = [
        _ordered(group.loc[group.condition.eq(condition) & group.from_state.eq("M4")], M4_EXITS)
        for condition in CONDITIONS
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.05), layout="constrained",
                                    gridspec_kw={"width_ratios": [1.55, 1]})
    style._head(fig, "M4 离开后的去向分布")
    left.set_title("退出条件概率均值及 95% Bootstrap 区间", loc="left", color=style.SLATE)
    right.set_title("患者配对 ON−OFF · 退出条件概率", loc="left", color=style.SLATE)
    x = np.arange(3)
    for condition, data, color, marker, offset in zip(CONDITIONS, prepared, [style.TEAL, style.BLUE, style.ORANGE],
                                                      ["o", "s", "D"], [-.23, 0, .23], strict=True):
        means = data["mean"].to_numpy(float) * 100
        low = data.mean_ci95_low.to_numpy(float) * 100
        high = data.mean_ci95_high.to_numpy(float) * 100
        left.vlines(x+offset, low, high, color=color, linewidth=1.35)
        left.plot(x+offset, means, marker, color=color, label=condition, ms=5)
        for i, r in enumerate(data.itertuples(index=False)):
            value = f"{_fixed(r.mean)}%" if np.isfinite(r.mean) else "NA"
            y = max(high[i], means[i]) if np.isfinite(high[i]) else means[i]
            y = y if np.isfinite(y) else 0
            left.annotate(f"{value}\nn={int(r.n)}", (x[i]+offset, y), xytext=(0, 6), textcoords="offset points",
                          ha="center", va="bottom", fontsize=8, color=color)
    labels = [f"M4→{target}" for _, target in M4_EXITS]
    left.set(xlim=(-.5, 2.5), ylim=(0, 114), yticks=[0, 25, 50, 75, 100], ylabel="平均条件概率（%）")
    left.set_xticks(x, labels)
    left.legend(loc="upper center", frameon=False, ncols=3)
    style._axes(left, "y")
    limit = _contrast_limits(rows)
    right.axvline(0, color=style.SLATE, lw=.85)
    for i, row in enumerate(rows.itertuples(index=False)):
        _interval(right, row, i, style.BLUE)
        right.text(limit, i-.22, f"n={int(row.n_pairs)}，q={_prob(row.fdr_q)}",
                   ha="right", va="center", color=style.SLATE, fontsize=9)
    right.set(xlim=(-limit, limit), ylim=(2.55, -.55), xlabel="均值差（百分点）")
    right.set_yticks(x, labels)
    right.xaxis.set_major_locator(MaxNLocator(nbins=5))
    style._axes(right)
    _save(fig, destination, {"group": [entry for rows in prepared for entry in _descriptives_summary(rows)],
                             "exit_conditional_BH36": _comparison_summary(rows)})


def _private_pairs(matrices, row):
    selected = matrices.loc[matrices.condition.isin(("ON", "OFF")) & matrices.from_state.eq("M4")
                            & matrices.to_state.eq("M4")]
    points = selected.pivot(index="participant_id", columns="condition", values="p_weida")
    points = points[["OFF", "ON"]].dropna()
    if len(points) != row.n_pairs:
        raise ValueError("The plotted M4 self-loop pair count changed")
    np.testing.assert_allclose(
        [points.ON.mean(), points.OFF.mean(), (points.ON - points.OFF).mean()],
        [row.mean_a, row.mean_b, row.mean_diff], rtol=0, atol=1e-12, equal_nan=True,
    )
    return points.to_numpy(float) * 100


def _m4_self_loop(row, points, summary_path, individual_path):
    labels = ["OFF", "ON"]
    means = np.array([row.mean_b, row.mean_a]) * 100
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.2), layout="constrained")
    style._head(fig, "M4 自循环概率的配对变化")
    left.set_title(f"同一患者队列的组均值 · {int(row.n_pairs)} 个有效配对", loc="left", color=style.SLATE)
    left.plot([0, 1], means, "-o", color=style.BLUE, lw=2.1, ms=6)
    for x, value in enumerate(means):
        if np.isfinite(value):
            left.annotate(f"{value:.1f}%", (x, value), xytext=(0, 7), textcoords="offset points",
                          ha="center", color=style.NAVY)
    left.set(xlim=(-.4, 1.4), ylim=(0, 110), yticks=[0, 25, 50, 75, 100], ylabel="平均自循环概率（%）")
    left.set_xticks([0, 1], labels)
    style._axes(left, "y")

    right.set_title("ON−OFF 的平均差及 95% Bootstrap 区间", loc="left", color=style.SLATE)
    right.axvline(0, color=style.SLATE, lw=.85)
    _interval(right, row, 0, style.BLUE)
    limits = np.array([row.ci95_low, row.mean_diff, row.ci95_high]) * 100
    finite = limits[np.isfinite(limits)]
    limit = max(1., abs(finite).max(initial=0) * 1.3)
    right.set(xlim=(-limit, limit), ylim=(-.45, .55), yticks=[], xlabel="均值差（百分点）")
    if np.isfinite(row.mean_diff):
        right.annotate(f"{_fixed(row.mean_diff, signed=True)} 个百分点", (row.mean_diff * 100, 0),
                       xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                       color=style.NAVY, fontsize=9)
    right.text(0, -.32, f"增加 {int(row.paired_higher)} 人  ·  不变 {int(row.paired_same)} 人  ·  减少 {int(row.paired_lower)} 人",
               ha="center", va="center", color=style.SLATE, fontsize=9)
    style._axes(right)
    _save(fig, summary_path, {
        "method": "WEiDA", "from_state": "M4", "to_state": "M4", "n_pairs": int(row.n_pairs),
        "mean_on": _numeric(row.mean_a), "mean_off": _numeric(row.mean_b), "mean_diff": _numeric(row.mean_diff),
        "ci95_low": _numeric(row.ci95_low), "ci95_high": _numeric(row.ci95_high),
        "higher": int(row.paired_higher), "lower": int(row.paired_lower), "same": int(row.paired_same),
    })

    fig, ax = plt.subplots(figsize=(6.8, 5.4), layout="constrained")
    style._head(fig, "M4 自循环概率的逐人配对")
    ax.set_title(f"私有审阅 · n={len(points)} · 每线为一个患者配对", loc="left", color=style.SLATE)
    ax.plot([0, 1], points.T, "-o", color=style.BLUE, alpha=.30, lw=.85, ms=3)
    ax.plot([0, 1], means, "-o", color=style.NAVY, lw=2.5, ms=6, label="同一患者队列的均值")
    ax.set(xlim=(-.35, 1.35), ylim=(0, 100), ylabel="M4→M4 概率（%）")
    ax.set_xticks([0, 1], labels)
    ax.legend(frameon=False, loc="lower right")
    style._axes(ax, "y")
    _save(fig, individual_path)


def render_transition_figures(group_descriptives, group_comparisons, matrices, public_dir, private_dir, *,
                              font_path=None):
    """Render 9 released aggregate figures and one private patient trajectory."""
    public_dir, private_dir = Path(public_dir).resolve(), Path(private_dir).resolve()
    if private_dir == public_dir or private_dir.is_relative_to(public_dir):
        raise ValueError("Private plot output must be outside the public figures directory")
    group = group_descriptives.loc[group_descriptives.method.eq("WEiDA")]
    comparisons = group_comparisons.loc[group_comparisons.method.eq("WEiDA")]
    self_loop = comparisons.loc[
        comparisons.contrast.eq("ON-OFF") & comparisons.from_state.eq("M4") & comparisons.to_state.eq("M4")
    ]
    (row,) = self_loop.itertuples(index=False)
    points = _private_pairs(matrices, row)

    public = []
    private = "M4_SelfLoop_Individual_Paired.png"
    font = style._chinese_font(font_path)
    diff = comparisons.mean_diff.to_numpy(float)
    finite = np.abs(diff[np.isfinite(diff)])
    color_limit = max(1., np.ceil(finite.max(initial=0) * 100))
    with matplotlib.rc_context({"font.family": [font, "DejaVu Sans"], **style._theme()}):
        for condition in CONDITIONS:
            name = f"Transition_WEiDA_{condition}.png"
            _group_matrix(_ordered(group.loc[group.condition.eq(condition)], CELLS), public_dir / name)
            public.append(name)
        for contrast in CONTRASTS:
            name = f"Transition_WEiDA_{contrast.replace('-', '_minus_')}.png"
            _delta_matrix(_ordered(comparisons.loc[comparisons.contrast.eq(contrast)], CELLS), public_dir / name,
                          color_limit)
            public.append(name)
        name = "M4_Exit_Directions.png"
        _m4_exits(group_descriptives.loc[group_descriptives.method.eq("ExitConditional")],
                  group_comparisons.loc[group_comparisons.method.eq("ExitConditional")], public_dir / name)
        public.append(name)
        name = "M4_SelfLoop_Paired_Summary.png"
        _m4_self_loop(row, points, public_dir / name, private_dir / private)
        public.append(name)
        name = "Transition_WEiDA_Forest.png"
        _forest(comparisons, public_dir / name)
        public.append(name)
    return {"public": public, "private": [private], "font": font, "metadata_key": METADATA_KEY}
