"""Describe M4, audit the original estimands, and compare with HC reference means."""

import numpy as np
import pandas as pd

from transition_io import CFG


STATES = CFG["states"]
CONDITIONS = list(CFG["cohort"])[:3]
EDGES = ["from_state", "to_state"]


def v6_parity(matrices, archived):
    """Cross-check all 16 ON/OFF sums and means against the published V6 table."""
    v6 = archived.rename(columns={"source": "from_state", "destination": "to_state"}).set_index(EDGES)
    on = matrices.loc[matrices.condition.eq("ON")].pivot(index="participant_id", columns=EDGES, values="p_weida")
    off = matrices.loc[matrices.condition.eq("OFF")].pivot(index="participant_id", columns=EDGES, values="p_weida")
    if set(on.index) != set(off.index) or len(v6) != 16 or set(on.columns) != set(v6.index):
        raise ValueError("V6 paired transition table or frozen paired scan keys are incomplete")
    off = off.reindex(index=on.index, columns=on.columns)
    use = on.notna() & off.notna()
    counts = matrices.groupby(["condition", *EDGES], sort=True)["count"].sum()
    discrepancies = []
    for label, observed in (("on", on.where(use).mean()), ("off", off.where(use).mean())):
        expected = v6.loc[observed.index, f"{label}_mean_probability"]
        discrepancies.append(float(np.nanmax(np.abs(observed - expected))))
        count = counts.loc[label.upper()].reindex(v6.index)
        sources = count.groupby(level=0).transform("sum")
        if not (count.eq(v6[label + "_count"]).all() and sources.eq(v6[label + "_source"]).all()):
            raise ValueError("Recounted transition or denominator totals differ from V6")
    change = (on - off).mean()
    discrepancies.append(float(np.nanmax(np.abs(change - v6.loc[change.index, "mean_difference"]))))
    if not np.array_equal(use.sum().to_numpy(), v6.loc[on.columns, "n_pairs"].to_numpy()) or max(discrepancies) > 1e-12:
        raise ValueError("Recounted paired probabilities differ from V6")
    return {"edges_verified": len(v6), "exact_count_and_denominator_parity": True,
            "maximum_absolute_probability_difference": max(discrepancies)}


def describe_quality(scans):
    rows = []
    for condition, part in scans.groupby("condition", sort=False):
        rows.append({
            "condition": condition,
            "n_scans": len(part),
            "valid_trs": int(part.valid_trs.sum()),
            "blocks": int(part.n_blocks.sum()),
            "valid_adjacent_pairs": int(part.n_adjacent.sum()),
            "min_valid_trs": int(part.valid_trs.min()),
            "max_valid_trs": int(part.valid_trs.max()),
        })
    return pd.DataFrame(rows)


def hc_reference(descriptives, comparisons):
    """Describe group means at the same paired patient denominator on each edge."""
    hc = descriptives.loc[descriptives.condition.eq("HC"), ["method", *EDGES, "n", "mean"]]
    hc = hc.rename(columns={"n": "n_hc", "mean": "mean_hc"})
    pairs = comparisons.loc[comparisons.contrast.eq("ON-OFF"), ["method", *EDGES, "n_pairs", "mean_a", "mean_b"]]
    pairs = pairs.rename(columns={"mean_a": "mean_on", "mean_b": "mean_off"})
    summary = pairs.merge(hc, on=["method", *EDGES], how="left", validate="one_to_one")
    summary["off_absolute_gap"] = (summary.mean_off - summary.mean_hc).abs()
    summary["on_absolute_gap"] = (summary.mean_on - summary.mean_hc).abs()
    summary["gap_reduction"] = summary.off_absolute_gap - summary.on_absolute_gap
    return summary


def summarize_m4(descriptives, comparisons, scans, matrices):
    group = descriptives.loc[descriptives.from_state.eq("M4") | descriptives.to_state.eq("M4")].copy()
    group["scope"] = "group"
    group["unit"] = "proportion"
    group["metric"] = "transition_probability"
    difference = comparisons.loc[comparisons.from_state.eq("M4") | comparisons.to_state.eq("M4")].copy()
    difference["scope"] = "contrast"
    difference["unit"] = "proportion"
    difference["metric"] = "transition_probability"
    self_change = difference.loc[difference.method.eq("WEiDA") & difference.contrast.eq("ON-OFF") &
                                 difference.from_state.eq("M4") & difference.to_state.eq("M4")].iloc[0]
    scans = scans.set_index("scan_id").copy()
    source_m4 = matrices.from_state.eq("M4")
    target_m4 = matrices.to_state.eq("M4")
    counts = matrices["count"]
    scan_keys = matrices.scan_id
    m4_origins = counts.where(source_m4, 0).groupby(scan_keys).sum()
    non_m4_origins = counts.where(~source_m4, 0).groupby(scan_keys).sum()
    enters_from_non_m4 = counts.where(~source_m4 & target_m4, 0).groupby(scan_keys).sum()
    diagonal = counts.where(source_m4 & target_m4, 0).groupby(scan_keys).sum()
    if not diagonal.reindex(scans.index).eq(scans.frames_M4 - scans.episodes_M4).all():
        raise ValueError("M4 self-loop count is inconsistent with M4 runs")
    if not m4_origins.reindex(scans.index).eq(scans.frames_M4 - scans.ends_M4).all():
        raise ValueError("M4 source denominator is inconsistent with block endings")
    if not enters_from_non_m4.reindex(scans.index).eq(scans.enters_M4).all():
        raise ValueError("M4 incoming transitions disagree with the segment entry count")

    def ratio(numerator, denominator):
        return numerator / denominator.where(denominator > 0)

    scans["m4_entry_given_other_origin"] = ratio(enters_from_non_m4, non_m4_origins)
    scans["m4_exit_given_m4_origin"] = ratio(m4_origins - diagonal, m4_origins)
    scans["occurrence_per_minute_M4"] = 60 * scans.occurrence_per_second_M4
    units = {
        "m4_entry_given_other_origin": "proportion", "m4_exit_given_m4_origin": "proportion",
        "duration_seconds_M4": "seconds", "coverage_M4": "proportion",
        "occurrence_per_minute_M4": "runs_per_valid_minute", "enters_M4": "count", "exits_M4": "count",
    }
    rows = []
    for metric_no, (metric, unit) in enumerate(units.items()):
        for condition in CONDITIONS:
            vec = scans.loc[scans.condition.eq(condition), metric].dropna()
            rows.append(dict(scope="group", method="M4_context", condition=condition, metric=metric,
                             unit=unit, n=len(vec), mean=vec.mean(), median=vec.median(), sd=vec.std()))
        matched = scans.loc[scans.condition.isin(("ON", "OFF"))].pivot(
            index="participant_id", columns="condition", values=metric,
        ).dropna()
        delta = matched.ON - matched.OFF
        if metric == "m4_exit_given_m4_origin":
            np.testing.assert_allclose(delta.mean(), -self_change.mean_diff, equal_nan=True, atol=1e-13)
            low, high = -self_change.ci95_high, -self_change.ci95_low
        elif len(delta):
            rng = np.random.default_rng([CFG["seed"], 401, metric_no])
            draw = rng.integers(len(delta), size=(CFG["bootstrap_draws"], len(delta)))
            low, high = np.quantile(delta.to_numpy()[draw].mean(axis=1), [.025, .975])
        else:
            low, high = np.nan, np.nan
        rows.append(dict(scope="contrast", method="M4_context", contrast="ON-OFF", metric=metric, unit=unit,
                         n_pairs=len(delta), mean_a=matched.ON.mean(), mean_b=matched.OFF.mean(),
                         mean_diff=delta.mean(), ci95_low=low, ci95_high=high))
    return pd.concat([group, difference, pd.DataFrame(rows)], ignore_index=True)
