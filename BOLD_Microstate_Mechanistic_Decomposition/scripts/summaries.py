"""Aggregate paired estimates, boundary diagnostics and window reliability."""

import numpy as np
import pandas as pd

from frozen_inputs import CFG, PRIVATE, RESULTS, STATES, TOL, write_csv
from inference import SENSITIVITY, adjusted_test, condition_summary, contributions, mean_bootstrap


def conditions(v6, paired, definitions, keep=None):
    return [dict(metric=metric, units=unit, **condition_summary(v6, paired, field, scale, keep))
            for metric, field, scale, unit in definitions]


def component_results(v6, paired, components, total, threshold, unit):
    keys = list(components)
    stats, shares = contributions(v6, paired, total, keys, threshold)
    rows = [dict(metric=components[key], units=unit, **stats[key]) for key in keys]
    rows += [dict(metric=f"{components[key]}_share", units="percent of total change", **shares[key]) for key in keys]
    return rows, stats[total]


def gev_summary(v6, paired):
    levels = [
        ("gev", "m1_gev", 100, "GEV percentage points"),
        ("amplitude_weight", "m1_weight", 100, "GFP squared weight percentage points"),
        ("weighted_spatial_fit", "m1_fit", 1, "weighted squared correlation"),
        ("ordinary_coverage", "coverage_1", 100, "coverage percentage points"),
    ]
    rows = conditions(v6, paired, levels)
    components = {"m1_shapley_w_pp": "amplitude_contribution", "m1_shapley_f_pp": "fit_contribution"}
    additions, total = component_results(v6, paired, components, "m1_gev_delta_pp",
                                         CFG["share_min_abs_gev_pp"], "GEV percentage points")
    rows += [dict(metric="gev_delta_complete_component_pairs", units="GEV percentage points", **total), *additions]
    write_csv(pd.DataFrame(rows), RESULTS / "M1_GEV_Decomposition.csv")
    cols = [c for c in paired.columns if "m1_" in c or "coverage_1" in c or "frames_1" in c]
    write_csv(paired[cols].reset_index(), PRIVATE / "results/M1_GEV_Decomposition.csv")


def duration_summary(v6, paired):
    for complete in (False, True):
        suffix = "_edge" if complete else ""
        values = ("dwell_edge_mean_4", "m4_complete_coverage", "m4_complete_occurrence") if complete else (
            "duration_4", "coverage_4", "occurrence_4")
        table = "M4_Boundary_Sensitivity.csv" if complete else "M4_Duration_Decomposition.csv"
        fields = [
            ("duration", values[0], 1, "seconds"), ("coverage", values[1], 100, "coverage percentage points"),
            ("occurrence", values[2], 60, "episodes per minute"),
        ]
        if complete:
            fields.append(("complete_episodes", "complete_episodes_4", 1, "episodes per scan"))
        keep = paired[f"m4_log_duration{suffix}"].notna().to_numpy()
        rows = conditions(v6, paired, fields, keep)
        labels = {f"m4_log_coverage{suffix}": "coverage_contribution",
                  f"m4_negative_log_occurrence{suffix}": "occurrence_contribution"}
        parts, total = component_results(v6, paired, labels, f"m4_log_duration{suffix}",
                                         CFG["share_min_abs_log_duration"], "natural log ratio")
        rows.extend([dict(metric="log_duration", units="natural log ratio", **total), *parts])
        if complete:
            qualified = [("duration", f"delta_{values[0]}", "seconds"),
                         ("log_duration", "m4_log_duration_edge", "natural log ratio"),
                         *[(name, key, "natural log ratio") for key, name in labels.items()]]
            for model, column in SENSITIVITY.items():
                for metric, key, unit in qualified:
                    adjusted = adjusted_test(v6, paired[key].where(keep).to_numpy(float), paired[column].to_numpy(float))
                    used = ["n_patients", "effect", "ci95_low", "ci95_high", "bootstrap_valid", "status", "reason"]
                    rows.append(dict(metric=f"{metric}_{model}", units=unit, **{p: adjusted[p] for p in used}))
        write_csv(pd.DataFrame(rows), RESULTS / table)
    keys = [c for c in paired.columns if any(x in c for x in (
        "duration_4", "coverage_4", "occurrence_4", "m4_log_", "m4_negative_log_", "dwell_edge", "m4_complete",
        "episodes_4", "valid_seconds", "m4_boundary_status"))]
    write_csv(paired[keys].reset_index(), PRIVATE / "results/M4_Duration_Decomposition.csv")


def exit_summary(v6, paired, full):
    valid_dest = paired[["off_m4_ilr_1", "on_m4_ilr_1"]].notna().all(axis=1).to_numpy()
    rates = [
        ("observed_exits", "m4_n_exits", 1, "exits per scan"),
        ("exit_frequency", "m4_exit_rate_min", 1, "exits per minute of valid scan time"),
        ("exit_intensity_in_state", "m4_exit_intensity_min", 1, "exits per minute of M4 occupancy"),
        ("all_m4_episodes", "episodes_4", 1, "episodes per scan"),
        ("right_truncated", "right_boundary_episodes_4", 1, "episodes per scan"),
    ]
    paths = [(f"q4{j}", f"m4_q4{j}", 100, "probability percentage points") for j in (1, 2, 3)]
    coordinates = [(f"ilr_{j}", f"m4_ilr_{j}", 1, "Helmert ILR coordinates") for j in (1, 2)]
    rows = conditions(v6, paired, rates) + conditions(v6, paired, paths + coordinates, valid_dest)
    write_csv(pd.DataFrame(rows), RESULTS / "M4_Exit_Dynamics.csv")
    cols = [c for c in paired.columns if any(x in c for x in (
        "m4_exit", "m4_n_exits", "m4_q4", "m4_ilr", "episodes_4", "valid_seconds", "duration_4", "occurrence_4",
        "coverage_4"))]
    visits = full.loc[full.condition.isin(("OFF", "ON"))].copy()
    visits["reciprocal_occurrence_s"] = 1 / visits.occurrence_4.where(visits.occurrence_4 > 0)
    visits["reciprocal_minus_duration_s"] = visits.reciprocal_occurrence_s - visits.duration_4
    visits["duration_times_occurrence"] = visits.duration_4 * visits.occurrence_4
    visits["duration_equals_reciprocal"] = np.isclose(visits.reciprocal_occurrence_s, visits.duration_4, atol=TOL, rtol=0)
    estimated = visits.occurrence_4 > 0
    np.testing.assert_allclose(visits.loc[estimated, "duration_times_occurrence"],
                               visits.loc[estimated, "coverage_4"], atol=TOL, rtol=0)
    write_csv(paired[cols].reset_index(), PRIVATE / "results/M4_Exit_Dynamics.csv")
    write_csv(visits[["subject", "condition", "coverage_4", "occurrence_4", "duration_4", "reciprocal_occurrence_s",
                      "reciprocal_minus_duration_s", "duration_times_occurrence", "duration_equals_reciprocal"]],
              PRIVATE / "results/M4_Reciprocal_Check.csv")
    out = []
    for condition, group in visits.groupby("condition", sort=False):
        eligible = group.occurrence_4 > 0
        out.append(dict(
            condition=condition, n_scans=len(group), n_scans_with_exit=int(group.m4_n_exits.gt(0).sum()),
            total_actual_exits=int(group.m4_n_exits.sum()), total_right_truncated_segments=int(group.right_boundary_episodes_4.sum()),
            clock_minutes=float(group.valid_seconds.sum() / 60), exact_exit_count_identity=int(
                (group.m4_n_exits == group.episodes_4 - group.right_boundary_episodes_4).sum()),
            duration_reciprocal_evaluable=int(eligible.sum()),
            duration_reciprocal_equal_count=int(group.duration_equals_reciprocal.sum()),
            exact_ratio_identity=int(np.isclose(group.duration_times_occurrence, group.coverage_4, atol=TOL, rtol=0).sum()),
            reciprocal_mean_seconds=group.reciprocal_occurrence_s.mean(), mean_duration_seconds=group.duration_4.mean(),
            mean_reciprocal_minus_duration_seconds=group.reciprocal_minus_duration_s.mean(),
        ))
    write_csv(pd.DataFrame(out), RESULTS / "M4_Exit_Eligibility_and_Reciprocal.csv")


def window_support(windows):
    supported = windows.copy()
    duration_ok = supported.valid_seconds >= CFG["window_min_valid_seconds"]
    supported["m1_evaluable"] = supported.gev_1.notna()
    supported["m4_evaluable"] = supported.duration_4.notna()
    supported["m1_eligible"] = (duration_ok & supported.m1_evaluable &
                                supported.frames_1.ge(CFG["window_min_m1_frames"]))
    supported["m4_eligible"] = (duration_ok & supported.m4_evaluable &
                                supported.complete_episodes_4.ge(CFG["window_min_complete_m4_episodes"]))
    supported["m1_window_status"] = np.select(
        [~supported.m1_evaluable, ~duration_ok, ~supported.m1_eligible],
        ["Not estimable", "Below valid time requirement", "Too few M1 frames"], default="Eligible")
    supported["m4_window_status"] = np.select(
        [~supported.m4_evaluable, ~duration_ok, ~supported.m4_eligible],
        ["Not estimable", "Below valid time requirement", "Too few complete M4 episodes"], default="Eligible")
    records = []
    for (window, condition), group in supported.groupby(["window", "condition"], sort=False):
        row = dict(window=window, condition=condition, n_scans=len(group),
                   valid_time_mean_seconds=group.valid_seconds.mean(),
                   valid_time_min_seconds=group.valid_seconds.min(), valid_time_max_seconds=group.valid_seconds.max(),
                   valid_time_below_threshold=int(group.valid_seconds.lt(CFG["window_min_valid_seconds"]).sum()),
                   n_m1_evaluable=int(group.m1_evaluable.sum()), n_m4_evaluable=int(group.m4_evaluable.sum()),
                   n_m1_eligible=int(group.m1_eligible.sum()), n_m4_eligible=int(group.m4_eligible.sum()))
        for state in STATES:
            row[f"mean_state_{state}_frames"] = group[f"frames_{state}"].mean()
            row[f"mean_complete_state_{state}_episodes"] = group[f"complete_episodes_{state}"].mean()
            for edge in ("left", "right"):
                row[f"sum_{edge}_truncated_state_{state}_episodes"] = int(group[f"{edge}_boundary_episodes_{state}"].sum())
        records.append(row)
    write_csv(pd.DataFrame(records), RESULTS / "Window_Support.csv")
    write_csv(supported, PRIVATE / "results/Scan_Window_Eligibility.csv")
    return supported


def window_summary(v6, windows, roi, frozen_pairs):
    supported = window_support(windows)
    names = [w["window"] for w in CFG["windows"]]
    records = []
    for state, metric, scale, unit in [("m1", "gev_1", 100, "GEV percentage points"),
                                       ("m4", "duration_4", 1, "seconds")]:
        patients = supported.loc[supported.condition.isin(("OFF", "ON"))]
        levels = patients.pivot(index="subject", columns=["window", "condition"], values=metric)
        eligibility = patients.pivot(index="subject", columns=["window", "condition"], values=f"{state}_eligible")
        levels, eligibility = levels.loc[frozen_pairs], eligibility.loc[frozen_pairs]
        common = eligibility.all(axis=1).to_numpy()
        for name in names:
            frame = levels[name].rename(columns={"OFF": f"off_{metric}", "ON": f"on_{metric}"})
            window_eligible = eligibility[name].all(axis=1).to_numpy()
            for label, keep in [("Eligible in this window", window_eligible), ("Eligible in all windows", common)]:
                summary = condition_summary(v6, frame, metric, scale, keep)
                delta = summary["effect"]
                width = summary["ci95_high"] - summary["ci95_low"]
                direction = "Undetermined" if not np.isfinite(delta) else ("ON higher" if delta > 0 else "ON lower" if delta < 0 else "Tie")
                status = "Eligible estimate" if np.isfinite(delta) else "Fewer than three eligible pairs"
                records.append(dict(window=name, metric=metric, sample=label, units=unit,
                                    n_on_eligible=int(eligibility[name].ON.sum()),
                                    n_off_eligible=int(eligibility[name].OFF.sum()),
                                    direction=direction, ci_width=width, status=status, **summary))
    out = pd.DataFrame(records)
    write_csv(out, RESULTS / "Within_Scan_Sensitivity.csv")
    by_region = (roi.groupby(["condition", "roi"], as_index=False)
                 .agg(n_scans=("scan", "nunique"), mean_lag1_acf=("acf_lag1", "mean"),
                      median_lag1_acf=("acf_lag1", "median"), n_efold_observed=("efold_observed", "sum"),
                      median_observed_efold_seconds=("first_efold_seconds", "median")))
    by_scan = (roi.groupby(["condition", "scan"], as_index=False)
               .agg(median_lag1_acf=("acf_lag1", "median"),
                    median_observed_efold_seconds=("first_efold_seconds", "median"),
                    fraction_efold_observed=("efold_observed", "mean")))
    overall_roi = (by_scan.groupby("condition", as_index=False)
                   .agg(n_scans=("scan", "nunique"), median_scan_lag1_acf=("median_lag1_acf", "median"),
                        median_scan_efold_seconds=("median_observed_efold_seconds", "median"),
                        mean_scan_fraction_efold_observed=("fraction_efold_observed", "mean")))
    write_csv(by_region, RESULTS / "ROI_Temporal_Autocorrelation.csv")
    write_csv(overall_roi, RESULTS / "ROI_Temporal_Overview.csv")
    return out
