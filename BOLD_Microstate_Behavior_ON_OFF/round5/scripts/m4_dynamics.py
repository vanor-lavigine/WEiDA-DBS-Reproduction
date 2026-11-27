"""Pair-preserving decomposition and plots for the frozen M4 dynamics."""

import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT.parents[1] / "V4_WEiDA_Statistical_Framework_Microstate/scripts"
sys.path.insert(0, str(V4))
try:
    from statistics_v4 import interval, standardized_effect
finally:
    sys.path.pop(0)


CONDITIONS = ("HC", "OFF", "ON")
STATES = ("M1", "M2", "M3", "M4")
CONTRASTS = ("OFF-HC", "ON-OFF", "ON-HC")
METRICS = {
    "duration_seconds": ("time", "seconds"),
    "coverage": ("time", "proportion"),
    "occurrence_per_minute": ("time", "runs_per_valid_minute"),
    "P44": ("maintenance", "proportion"),
    "exit_rate": ("exit", "proportion"),
    "pooled_entry": ("entry", "proportion"),
    "P41": ("exit", "proportion"),
    "P42": ("exit", "proportion"),
    "P43": ("exit", "proportion"),
    "P41_exit": ("exit_destination", "proportion"),
    "P42_exit": ("exit_destination", "proportion"),
    "P43_exit": ("exit_destination", "proportion"),
    "P14": ("entry", "proportion"),
    "P24": ("entry", "proportion"),
    "P34": ("entry", "proportion"),
    "enter_count": ("boundary", "transitions"),
    "exit_count": ("boundary", "transitions"),
    "episode_count": ("boundary", "runs"),
    "coverage_duration_component": ("coverage_component", "proportion"),
    "coverage_occurrence_component": ("coverage_component", "proportion"),
}
LABELS = {
    "duration_seconds": "平均驻留时间（秒）",
    "coverage": "占用率（%）",
    "occurrence_per_minute": "出现率（段/有效分钟）",
    "P44": "自循环 P44（%）",
    "exit_rate": "离开 M4 的概率（%）",
    "pooled_entry": "从非 M4 进入 M4（%）",
    "P41_exit": "M4→M1｜离开 M4（%）",
    "P42_exit": "M4→M2｜离开 M4（%）",
    "P43_exit": "M4→M3｜离开 M4（%）",
    "P14": "M1→M4（%）",
    "P24": "M2→M4（%）",
    "P34": "M3→M4（%）",
}
EDGE_SPECS = {
    "P44": ("WEiDA", "M4", "M4", "p_weida"),
    **{f"P4{i}": ("WEiDA", "M4", f"M{i}", "p_weida") for i in range(1, 4)},
    **{f"P4{i}_exit": ("ExitConditional", "M4", f"M{i}", "p_exit") for i in range(1, 4)},
    **{f"P{i}4": ("WEiDA", f"M{i}", "M4", "p_weida") for i in range(1, 4)},
}
STREAM_EQUIVALENTS = {
    "exit_rate": "P44",
    "coverage_duration_component": "coverage",
    "coverage_occurrence_component": "coverage",
}


def _assert_close(first, second, *, message):
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-12, equal_nan=True, err_msg=message)


def _ratio(numerator, denominator):
    return numerator.div(denominator.where(denominator.gt(0)))


def _scan_measurements(scans, matrices):
    keys = ["scan_id", "participant_id", "condition", "fold"]
    source = scans.set_index("scan_id", verify_integrity=True).sort_index()
    rows = matrices[keys].drop_duplicates().set_index("scan_id", verify_integrity=True).sort_index()
    if not rows.equals(source[["participant_id", "condition", "fold"]]):
        raise ValueError("Round4 scan identity, condition or fold differs between M4 inputs")
    expected = pd.MultiIndex.from_product([source.index, STATES, STATES],
                                          names=["scan_id", "from_state", "to_state"])
    matrix = matrices.set_index(["scan_id", "from_state", "to_state"], verify_integrity=True)
    if not matrix.index.sort_values().equals(expected):
        raise ValueError("M4 decomposition requires each scan's complete frozen 4x4 matrix")
    matrix = matrix.sort_index()
    counts = matrix["count"].unstack(["from_state", "to_state"])
    for first, second in product(STATES, repeat=2):
        _assert_close(counts[first, second], source[f"N_{first}_{second}"],
                      message="Round4 count files disagree")

    frame = source[["participant_id", "condition", "fold"]].copy()
    frame["duration_seconds"] = source.duration_seconds_M4
    frame["coverage"] = source.coverage_M4
    frame["occurrence_per_minute"] = 60 * source.occurrence_per_second_M4
    frame["enter_count"] = source.enters_M4
    frame["exit_count"] = source.exits_M4
    frame["episode_count"] = source.episodes_M4
    _assert_close(frame.coverage, frame.duration_seconds * frame.occurrence_per_minute / 60,
                  message="A scan's M4 coverage does not equal duration times occurrence")
    m4_origin = counts["M4"].sum(axis=1)
    non_m4_origin = counts.drop(columns="M4", level="from_state").sum(axis=1)
    enters = counts.loc[:, [(state, "M4") for state in STATES[:-1]]].sum(axis=1)
    exits = counts.loc[:, [("M4", state) for state in STATES[:-1]]].sum(axis=1)
    for left, right in (
        (m4_origin, source.frames_M4 - source.ends_M4),
        (counts["M4", "M4"], source.frames_M4 - source.episodes_M4),
        (enters, source.enters_M4),
        (exits, source.exits_M4),
        (enters - exits, source.ends_M4 - source.starts_M4),
    ):
        _assert_close(left, right, message="M4 transition, episode or boundary identity failed")
    for metric, (_, first, second, column) in EDGE_SPECS.items():
        frame[metric] = matrix[column].xs((first, second), level=["from_state", "to_state"])
        denominator = exits if column == "p_exit" else counts[first].sum(axis=1)
        _assert_close(frame[metric], _ratio(counts[first, second], denominator),
                      message=f"M4 frozen source denominator failed for {metric}")
    frame["exit_rate"] = _ratio(exits, m4_origin)
    frame["pooled_entry"] = _ratio(enters, non_m4_origin)
    _assert_close(frame.exit_rate, 1 - frame.P44, message="M4 total exit probability and self-loop are not complements")
    return frame.reset_index()


def _pair_measurements(frame):
    columns = ["scan_id", "fold", *frame.columns.intersection(METRICS)]
    on, off = (frame.loc[frame.condition.eq(condition), ["participant_id", *columns]]
               .set_index("participant_id", verify_integrity=True).sort_index()
               for condition in ("ON", "OFF"))
    if not on.index.equals(off.index) or not on.fold.equals(off.fold):
        raise ValueError("The frozen M4 ON/OFF identity or fold pairs are incomplete")
    pairs = on.add_suffix("_ON").join(off.add_suffix("_OFF"))
    for metric in frame.columns.intersection(METRICS):
        pairs[metric + "_change"] = on[metric] - off[metric]
    pairs["coverage_duration_component_change"] = (
        (on.occurrence_per_minute + off.occurrence_per_minute) / 120 * pairs.duration_seconds_change
    )
    pairs["coverage_occurrence_component_change"] = (
        (on.duration_seconds + off.duration_seconds) / 120 * pairs.occurrence_per_minute_change
    )
    _assert_close(pairs.coverage_change,
                  pairs.coverage_duration_component_change + pairs.coverage_occurrence_component_change,
                  message="Symmetric duration and occurrence terms do not add to M4 coverage change")
    return pairs.reset_index()


def _rng(seed, slot, metric):
    shared = STREAM_EQUIVALENTS.get(metric, metric)
    return np.random.default_rng([seed, 51, slot, tuple(METRICS).index(shared)])


def _group_summary(frame, seed, bootstrap):
    rows = []
    for slot, condition in enumerate(CONDITIONS):
        selected = frame.loc[frame.condition.eq(condition)].sort_values("participant_id")
        for metric in frame.columns.intersection(METRICS):
            section, unit = METRICS[metric]
            values = selected[metric].dropna().to_numpy()
            n = len(values)
            low, high = (np.nan, np.nan)
            if n:
                rng = _rng(seed, slot, metric)
                indices = rng.integers(n, size=(bootstrap, n))
                low, high = interval(values[indices].mean(axis=1))
            rows.append(dict(metric=metric, section=section, unit=unit, condition=condition,
                             n=n, n_total=len(selected), mean=values.mean() if n else np.nan,
                             median=np.median(values) if n else np.nan, sd=values.std(ddof=1) if n > 1 else np.nan,
                             ci95_low=low, ci95_high=high))
    return pd.DataFrame(rows)


def _effect(a, b, rng, bootstrap, *, paired):
    size_a, size_b = len(a), len(b)
    row = dict(n_a=size_a, n_b=size_b, n_pairs=size_a if paired else np.nan,
               mean_a=a.mean() if size_a else np.nan, mean_b=b.mean() if size_b else np.nan,
               ci95_low=np.nan, ci95_high=np.nan, hedges_g=np.nan, hedges_g_type="g_z" if paired else "g",
               hedges_g_ci95_low=np.nan, hedges_g_ci95_high=np.nan, hedges_g_bootstrap_valid=0)
    row["mean_diff"] = (a - b).mean() if paired and size_a else row["mean_a"] - row["mean_b"]
    if paired:
        signed = (a - b).round(12)
        row.update(n_decrease=int((signed < 0).sum()), n_unchanged=int((signed == 0).sum()),
                   n_increase=int((signed > 0).sum()))
    if not min(size_a, size_b):
        return row
    ia = rng.integers(size_a, size=(bootstrap, size_a))
    ib = ia if paired else rng.integers(size_b, size=(bootstrap, size_b))
    draw_a, draw_b = a[ia], b[ib]
    draws = (draw_a - draw_b).mean(axis=1) if paired else draw_a.mean(axis=1) - draw_b.mean(axis=1)
    row["ci95_low"], row["ci95_high"] = interval(draws)
    if min(size_a, size_b) >= 2:
        row["hedges_g"] = float(standardized_effect(a, b, paired))
        g_draws = standardized_effect(draw_a, draw_b, paired)
        row["hedges_g_ci95_low"], row["hedges_g_ci95_high"] = interval(g_draws)
        row["hedges_g_bootstrap_valid"] = int(np.isfinite(g_draws).sum())
    return row


def _inherit(frozen, metric, contrast, effect):
    """Expose unchanged Round4 tests beside new descriptive resampling intervals."""
    result = dict(round4_family="descriptive", round4_test="", round4_raw_p=np.nan,
                  round4_fdr_q=np.nan, round4_holm_p=np.nan)
    if metric not in EDGE_SPECS:
        return result
    method, first, second, _ = EDGE_SPECS[metric]
    row = frozen.loc[method, first, second, contrast]
    _assert_close(row.mean_diff, effect["mean_diff"], message=f"Round4 M4 point estimate drift for {metric}")
    if int(row.n_a) != effect["n_a"] or int(row.n_b) != effect["n_b"]:
        raise ValueError(f"Round4 M4 estimand denominator drift for {metric}")
    if row.scope == "prespecified":
        result.update(round4_family=f"{method}_36", round4_test=row.test, round4_raw_p=row.raw_p,
                      round4_fdr_q=row.fdr_q, round4_holm_p=row.holm_p)
    return result


def _m4_effects(frame, pairs, frozen_comparisons, seed, bootstrap):
    frozen = frozen_comparisons.set_index(["method", "from_state", "to_state", "contrast"], verify_integrity=True)
    tables = []
    for slot, contrast in enumerate(CONTRASTS, start=3):
        group_a, group_b = contrast.split("-")
        paired = contrast == "ON-OFF"
        all_metrics = METRICS if paired else {
            metric: spec for metric, spec in METRICS.items()
            if spec[0] in ("maintenance", "exit", "exit_destination", "entry")
        }
        for metric, (section, unit) in all_metrics.items():
            if section == "coverage_component":
                a = pairs[metric + "_change"].dropna().to_numpy()
                b = np.zeros_like(a)
            elif paired:
                a, b = pairs[[metric + "_ON", metric + "_OFF"]].dropna().to_numpy().T
            else:
                a, b = (frame.loc[frame.condition.eq(condition)].sort_values("participant_id")[metric]
                        .dropna().to_numpy() for condition in (group_a, group_b))
            effect = _effect(a, b, _rng(seed, slot, metric), bootstrap, paired=paired)
            result = dict(metric=metric, section=section, unit=unit, contrast=contrast,
                          design="paired" if paired else "independent", **effect,
                          **_inherit(frozen, metric, contrast, effect))
            if section == "coverage_component":
                result.update(mean_a=np.nan, mean_b=np.nan)
            if paired and metric == "P44":
                common = pairs[["duration_seconds_change", "P44_change"]].dropna().round(12)
                result.update(n_joint_duration_self_loop=len(common),
                              duration_self_loop_both_lower=int(common.lt(0).all(axis=1).sum()),
                              duration_self_loop_both_higher=int(common.gt(0).all(axis=1).sum()),
                              duration_self_loop_same_nonzero_direction=int(
                                  (common.duration_seconds_change * common.P44_change).gt(0).sum()))
            tables.append(result)
    return pd.DataFrame(tables)


def run_m4_dynamics(scans, matrices, frozen_comparisons, *, seed=20261127, bootstrap=9999):
    """Recheck the M4 denominators and summarize the already frozen contrasts."""
    frame = _scan_measurements(scans, matrices)
    pairs = _pair_measurements(frame)
    effects = _m4_effects(frame, pairs, frozen_comparisons, seed, bootstrap)
    paired = effects.loc[effects.design.eq("paired")].reset_index(drop=True)
    probability = effects.loc[effects.section.isin(("maintenance", "exit", "exit_destination", "entry"))]
    for contrast, fragment in probability.groupby("contrast"):
        by_metric = fragment.set_index("metric")
        loop, leave = by_metric.loc["P44"], by_metric.loc["exit_rate"]
        _assert_close([leave.mean_diff, leave.ci95_low, leave.ci95_high, leave.hedges_g,
                       leave.hedges_g_ci95_low, leave.hedges_g_ci95_high],
                      -np.array([loop.mean_diff, loop.ci95_high, loop.ci95_low, loop.hedges_g,
                                 loop.hedges_g_ci95_high, loop.hedges_g_ci95_low]),
                      message=f"Complementary M4 effects disagree for {contrast}")
    return {
        "M4_Dynamics_Decomposition": _group_summary(frame, seed, bootstrap),
        "M4_Dynamics_Paired_Effects": paired,
        "M4_Entry_Exit_Analysis": probability.reset_index(drop=True),
        "private_pairs": pairs,
    }


def _annotations(table, frame, columns):
    selected = frame[list(columns)].astype(object).where(frame[list(columns)].notna(), None)
    return dict(table=table, rows=selected.to_dict("records"))


def _plot_style():
    spec = importlib.util.spec_from_file_location("round5_m4_style", ROOT.parent / "round2/scripts/figures.py")
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    return library


def _export_plot(fig, path, data, style, plt):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Round5_Annotations": json.dumps(data, ensure_ascii=True, allow_nan=False)} if data else None
    try:
        fig.savefig(path, dpi=180, facecolor=style.WHITE, metadata=metadata)
    finally:
        plt.close(fig)


def _group_figures(output, figures, plt, style):
    summary_name = "M4_Dynamics_Decomposition"
    effects_name = "M4_Dynamics_Paired_Effects"
    summary = output[summary_name]
    effects = output[effects_name]
    group_columns = ("metric", "condition", "n", "mean", "ci95_low", "ci95_high")
    pair_columns = ("metric", "contrast", "n_pairs", "mean_diff", "ci95_low", "ci95_high", "round4_fdr_q")
    colors = [style.TEAL, style.ORANGE, style.BLUE]
    metrics = ("duration_seconds", "coverage", "occurrence_per_minute", "P44", "exit_rate")
    panel_rows = summary.loc[summary.metric.isin(metrics)]
    components = (
        "coverage", "coverage_duration_component", "coverage_occurrence_component",
    )
    figure, axes = plt.subplots(2, 3, figsize=(12.8, 7.9), layout="constrained")
    style._head(figure, "M4 驻留、占用与状态离开")
    for ax, metric in zip(axes.flat[:5], metrics, strict=True):
        rows = panel_rows.loc[panel_rows.metric.eq(metric)].set_index("condition").loc[list(CONDITIONS)]
        scale = 100 if METRICS[metric][1] == "proportion" else 1
        for position, (row, color) in enumerate(zip(rows.itertuples(), colors, strict=True)):
            ax.vlines(position, row.ci95_low * scale, row.ci95_high * scale, color=color, linewidth=2)
            ax.scatter(position, row.mean * scale, color=color, zorder=3)
            ax.annotate(f"{row.mean * scale:.2f}", (position, row.mean * scale), xytext=(6, 0),
                        textcoords="offset points", va="center", color=color, fontsize=9)
        ax.set(xticks=range(3), xticklabels=[f"{c}\nn={n}" for c, n in zip(CONDITIONS, rows.n, strict=True)],
               xlim=(-.35, 2.38))
        ax.set_title(LABELS[metric], loc="left", color=style.SLATE, fontsize=11)
        style._axes(ax, "y")
    ax = axes.flat[-1]
    component_rows = effects.set_index("metric").loc[list(components)].reset_index()
    component_labels = ["总变化", "驻留时间分量", "出现率分量"]
    for y, row in enumerate(component_rows.itertuples(index=False)):
        ax.hlines(y, 100 * row.ci95_low, 100 * row.ci95_high, color=colors[y], linewidth=2)
        ax.scatter(100 * row.mean_diff, y, color=colors[y], zorder=3)
    ax.set(yticks=range(3), yticklabels=component_labels, xlabel="ON−OFF（百分点）")
    ax.set_title("占用率的对称分解 · 95% CI", loc="left", color=style.SLATE, fontsize=11)
    ax.axvline(0, color=style.GRAY, lw=1)
    ax.invert_yaxis()
    style._axes(ax)
    path = figures / "M4_Dynamics_Decomposition.png"
    _export_plot(figure, path, [_annotations(summary_name, panel_rows, group_columns),
                                _annotations(effects_name, component_rows, pair_columns)], style, plt)

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.45), layout="constrained")
    style._head(figure, "M4 的进入概率与退出后的去向")
    paths = (("P14", "P24", "P34"), ("P41_exit", "P42_exit", "P43_exit"))
    labels = (("M1→M4", "M2→M4", "M3→M4"), ("M4→M1", "M4→M2", "M4→M3"))
    titles = ("按各自来源态的全部转移归一化", "限定离开 M4 后，按该态的非对角计数归一化")
    for side, (ax, metrics, name, title) in enumerate(zip(axes, paths, labels, titles, strict=True)):
        rows = summary.loc[summary.metric.isin(metrics)]
        differences = effects.set_index("metric").loc[list(metrics)]
        for i, (condition, color) in enumerate(zip(CONDITIONS, colors, strict=True)):
            series = rows.loc[rows.condition.eq(condition)].set_index("metric").loc[list(metrics)]
            x = np.arange(3) + (i - 1) * .20
            ax.vlines(x, 100 * series.ci95_low, 100 * series.ci95_high, color=color, lw=1.6)
            ax.plot(x, 100 * series["mean"], "o", color=color, label=f"{condition}（n={int(series.n.min())}）")
        ax.set(xticks=range(3), xticklabels=[
            f"{label}\nON−OFF {100 * row.mean_diff:+.1f} pp\nBH q={row.round4_fdr_q:.3f}"
            for label, row in zip(name, differences.itertuples(index=False), strict=True)
        ], xlim=(-.5, 2.5), ylabel="平均概率（%）")
        ax.set_title(title, loc="left", color=style.SLATE, fontsize=11)
        ax.legend(loc="best", frameon=False, ncol=1 if side else 3, fontsize=8)
        style._axes(ax, "y")
    path_two = figures / "M4_Entry_Exit_Comparison.png"
    edge_metrics = paths[0] + paths[1]
    _export_plot(figure, path_two,
                 [_annotations(summary_name, summary.loc[summary.metric.isin(edge_metrics)], group_columns),
                  _annotations(effects_name, effects.loc[effects.metric.isin(edge_metrics)], pair_columns)],
                 style, plt)
    return [path, path_two]


def _private_figure(pairs, destination, plt, style):
    metrics = ("duration_seconds", "coverage", "occurrence_per_minute", "P44", "exit_rate",
               "pooled_entry", "P41_exit", "P42_exit", "P43_exit", "P14", "P24", "P34")
    figure, axes = plt.subplots(3, 4, figsize=(14, 10), layout="constrained")
    style._head(figure, "M4 逐人 ON/OFF 配对变化")
    for ax, metric in zip(axes.flat, metrics, strict=True):
        values = pairs[[metric + "_OFF", metric + "_ON"]].dropna().to_numpy().T
        scale = 100 if METRICS[metric][1] == "proportion" else 1
        ax.plot([0, 1], scale * values, "-o", color=style.BLUE, lw=.75, ms=2.4, alpha=.3)
        if values.shape[1]:
            ax.plot([0, 1], scale * values.mean(axis=1), "-o", color=style.NAVY, lw=2.5, ms=5)
        ax.set(xticks=[0, 1], xticklabels=["OFF", "ON"], xlim=(-.22, 1.22))
        ax.set_title(f"{LABELS[metric]} · n={values.shape[1]}", loc="left", color=style.SLATE, fontsize=9)
        style._axes(ax, "y")
    _export_plot(figure, destination, None, style, plt)


def render_m4_dynamics(output, figures_dir, private_figures_dir):
    """Write aggregate figures publicly and reserve the trajectories for private review."""
    figures, private = Path(figures_dir).resolve(), Path(private_figures_dir).resolve()
    if private == figures or private.is_relative_to(figures):
        raise ValueError("Individual M4 trajectories require a separate private output directory")
    import matplotlib

    style = _plot_style()
    from matplotlib import pyplot as plt

    with matplotlib.rc_context({"font.family": [style._chinese_font(), "DejaVu Sans"], **style._theme()}):
        public = _group_figures(output, figures, plt, style)
        _private_figure(output["private_pairs"], private / "M4_Individual_Paired_Changes.png", plt, style)
    return public
