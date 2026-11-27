"""White-background Chinese figures for round-three coverage associations."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np


STATES = ("M1", "M2", "M3", "M4")
LABELS = {
    "total": "运动总分",
    "rigidity": "强直",
    "bradykinesia": "运动徐缓",
    "tremor": "震颤",
    "gait_balance": "步态与平衡三项\n探索性合计",
}
KEYS = [(s, k) for s in STATES for k in LABELS]
FDR_LEVEL = .05
HEATMAP = "Coverage_Clinical_Rate_Heatmap.png"
SCATTER = "Coverage_Clinical_Rate_Scatter_Individual.png"
ANNOTATIONS_KEY = "Round3_Annotations"

# Use the exact Chinese font search and colors used in the preceding round.
_style_spec = spec_from_file_location("round2_coverage_figure_style",
                                     Path(__file__).resolve().parents[2] / "round2/scripts/figures.py")
style = module_from_spec(_style_spec)
_style_spec.loader.exec_module(style)


def _ordered_rows(main):
    data = main.set_index(["state", "clinical_endpoint"], verify_integrity=True)
    if len(data) != len(KEYS) or set(data.index) != set(KEYS):
        raise ValueError("Expected each of the twenty registered state–clinical slots")
    rows = data.loc[KEYS].reset_index()
    passing = rows.fdr_q.lt(FDR_LEVEL) & rows.fdr_q.notna()
    if not rows.passes_fdr.eq(passing).all():
        raise ValueError("Heatmap result flags do not match the registered q threshold")
    return rows


def _prob(value):
    return f"{value:.4g}" if np.isfinite(value) else "NA"


def _rho(value):
    return f"{value:+.3f}" if np.isfinite(value) else "NA"


def heatmap_annotations(main):
    """Encoded in public PNG metadata to make the displayed summary auditable."""
    return [
        dict(state=r.state, clinical_endpoint=r.clinical_endpoint, n=int(r.n),
             rho_display=_rho(r.rho), q_display=_prob(r.fdr_q), passes_fdr=bool(r.passes_fdr))
        for r in _ordered_rows(main).itertuples(index=False)
    ]


def _save(fig, destination, metadata=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(destination, dpi=160, facecolor=style.WHITE, metadata=metadata)
    finally:
        plt.close(fig)


def _ink(rgba):
    rgb = to_rgb(rgba)
    luminosity = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
    return style.NAVY if luminosity > .61 else style.WHITE


def _heatmap(rows, destination, annotations):
    rho = rows.rho.to_numpy(float).reshape(4, 5)
    cmap = LinearSegmentedColormap.from_list("coverage_correlation",
                                            ["#285681", "#f8fafc", "#a7421e"])
    cmap.set_bad("#e9eef2")
    fig, ax = plt.subplots(figsize=(10.3, 6.7), layout="constrained")
    style._head(fig, "状态占用率变化与运动评分改善率")
    image = ax.imshow(np.ma.masked_invalid(rho), cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5), list(LABELS.values()))
    ax.set_yticks(range(4), STATES)
    ax.set_title("双侧 Spearman · 四态 × 五项 · 方框标出本轮 BH20 q < 0.05", loc="left",
                 color=style.SLATE, fontsize=11, pad=13)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color=style.WHITE, linewidth=2)
    ax.tick_params(axis="both", which="both", length=0, pad=8)
    ax.spines[:].set_visible(False)

    for i, entry in enumerate(annotations):
        row, col = divmod(i, 5)
        bounded = rho[row, col]
        color = _ink(cmap(image.norm(bounded))) if np.isfinite(bounded) else style.NAVY
        if entry["passes_fdr"]:
            ax.add_patch(Rectangle((col-.475, row-.475), .95, .95, fill=False, linewidth=2.1,
                                   edgecolor=color))
        ax.text(col, row-.17, f'ρ {entry["rho_display"]}', ha="center", va="center", fontsize=13,
                fontweight="semibold", color=color)
        ax.text(col, row+.12, f'q {entry["q_display"]}  ·  n {entry["n"]}', ha="center", va="center",
                fontsize=9.6, color=color)

    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", shrink=.62, aspect=28, pad=.08,
                        ticks=[-1, -.5, 0, .5, 1])
    cbar.set_label("Spearman ρ")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    encoded = json.dumps(annotations, ensure_ascii=True, separators=(",", ":"))
    _save(fig, destination, metadata={ANNOTATIONS_KEY: encoded})


def _points(pairs, state, endpoint):
    x = pairs[f"delta_coverage_{STATES.index(state) + 1}"].to_numpy(float)
    y = pairs[f"{endpoint}_improvement_rate"].to_numpy(float)
    absolute = pairs[f"{endpoint}_improvement"].to_numpy(float)
    off = pairs[f"{endpoint}_off"].to_numpy(float)
    present = np.isfinite(x) & np.isfinite(y) & np.isfinite(absolute) & (off != 0)
    return x[present] * 100, y[present] * 100


def _paired_points(rows, pairs):
    # Compute all panel data before plotting and reconcile the inclusion mask with the main table.
    points = {}
    for r in rows.itertuples(index=False):
        key = (r.state, r.clinical_endpoint)
        points[key] = _points(pairs, *key)
        if len(points[key][0]) != r.n:
            raise ValueError(f"Private panel sample size differs for {r.state} × {r.clinical_endpoint}")
    return points


def _scatter(rows, points, destination):
    fig, axes = plt.subplots(4, 5, figsize=(17.5, 12.8), sharex="row", sharey="col", layout="constrained")
    style._head(fig, "状态占用率变化与运动评分改善率的逐人配对")
    fig.supxlabel("Coverage 的 ON−OFF 变化（百分点）", fontsize=13)
    fig.supylabel("运动评分改善率（%）", fontsize=13)
    for i, r in enumerate(rows.itertuples(index=False)):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        x, y = points[r.state, r.clinical_endpoint]
        ax.scatter(x, y, s=24, color=style.TEAL, alpha=.70, edgecolors=style.WHITE, linewidths=.42, zorder=4)
        ax.axhline(0, color=style.SLATE, linewidth=.68, zorder=2)
        ax.axvline(0, color=style.SLATE, linewidth=.68, zorder=2)
        label = LABELS[r.clinical_endpoint].replace("\n", " · ")
        summary = f"n={r.n}  ρ={_rho(r.rho)}  P={_prob(r.raw_p)}  q={_prob(r.fdr_q)}"
        ax.set_title(f"{r.state} · {label}\n{summary}", loc="left", color=style.NAVY, fontsize=8.7)
        ax.tick_params(axis="both", length=2, labelleft=True, labelbottom=True, labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_axisbelow(True)
        ax.grid(color=style.GRID, linewidth=.62)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color(style.GRAY)
    _save(fig, destination)


def render_figures(main, pairs, public_dir, private_dir, *, font_path=None):
    """Write one public aggregate heatmap and one private, unjittered scatter grid."""
    rows = _ordered_rows(main)
    points = _paired_points(rows, pairs)
    font = style._chinese_font(font_path)
    with matplotlib.rc_context({"font.family": [font, "DejaVu Sans"], **style._theme()}):
        _heatmap(rows, Path(public_dir) / HEATMAP, heatmap_annotations(main))
        _scatter(rows, points, Path(private_dir) / SCATTER)
    return {"public": [HEATMAP], "private": [SCATTER], "font": font, "metadata_key": ANNOTATIONS_KEY}
