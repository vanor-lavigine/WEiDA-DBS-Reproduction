"""Run the fixed and, if eligible, independently calibrated V5 endpoints."""

import argparse
import json

import numpy as np
import pandas as pd

from v5_data import CFG, HERE, RESULT, STATE, csv, dump, load_data
from occurrence_models import models
from v5_stats import P_COLUMNS, RESAMPLE, adjustment, ci, groups, omnibus, restoration, scalar_tests
from statistics_v4 import corrections


HC_CONTRASTS = [c for c in CFG["contrasts"] if c.endswith("-HC")]
ILR = [f"ilr_{i}" for i in range(1, 4)]
MATRIX = [f"trans_ilr_{i}" for i in range(1, 13)]
COVERAGE = [(f"M{i}", f"coverage_{i}") for i in STATE]
EDGES = [(f"M{i}->M{j}", f"transition_{i}_{j}") for i in STATE for j in STATE if i != j]
DURATION = {
    f"{label} dwell, {window}": [(f"M{i}", f"dwell_{window}_{quantile}_{i}") for i in STATE]
    for window in ("window", "edge") for quantile, label in (("median", "Median"), ("p75", "P75"))
}


def describe(data):
    coverage, transition, duration, occurrence, exposures = [], [], [], [], []
    for condition, group in data.groupby("condition", sort=False):
        exposures.append(dict(condition=condition, n=len(group), min_frames=group.valid_frames.min(),
                              median_frames=group.valid_frames.median(), max_frames=group.valid_frames.max(),
                              total_valid_seconds=group.exposure.sum(), total_adjacent_transitions=group.filter(regex=r"^source_count").sum().sum()))
        for i in STATE:
            c = group[f"coverage_{i}"]
            coverage.append(dict(condition=condition, state=f"M{i}", n=len(c), mean=c.mean(), sd=c.std(),
                                 q1=c.quantile(.25), median=c.median(), q3=c.quantile(.75)))
            episodes = group[f"count_{i}"]
            rate = 60*episodes/group.exposure
            occurrence.append(dict(condition=condition, state=f"M{i}", n=len(rate), total_episodes=episodes.sum(),
                                   mean_rate_per_min=rate.mean(), sd_rate_per_min=rate.std(),
                                   q1=rate.quantile(.25), median=rate.median(), q3=rate.quantile(.75)))
            source = group[f"source_count_{i}"].sum()
            for j in STATE:
                p = group[f"transition_{i}_{j}"].dropna()
                n = group[f"count_{i}_{j}"].sum()
                transition.append(dict(condition=condition, source=f"M{i}", destination=f"M{j}", scans=len(p),
                                       raw_count=n, source_count=source, pooled_probability=n/source if source else np.nan,
                                       mean_probability=p.mean(), sd_probability=p.std()))
        for label, keys in DURATION.items():
            for state, key in keys:
                x = group[key].dropna()
                duration.append(dict(condition=condition, state=state, definition=label, n=len(x),
                                     n_undefined=len(group)-len(x), mean=x.mean(), sd=x.std(), q1=x.quantile(.25),
                                     median=x.median(), q3=x.quantile(.75), minimum=x.min(), maximum=x.max()))
    return tuple(pd.DataFrame(x) for x in (coverage, transition, duration, occurrence, exposures))


def coordinate_effects(data):
    # The sign of each coordinate is first condition minus second condition.
    rows = []
    for kind, keys, contrast_set in [("Coverage ILR", ILR, HC_CONTRASTS), ("Transition ILR", MATRIX, CFG["contrasts"])]:
        for contrast in contrast_set:
            a, b, *_ = groups(data, keys, contrast)
            x, y = RESAMPLE.resample(len(a), len(b), contrast == "ON-OFF")
            draws = a[x].mean(axis=1)-b[y].mean(axis=1)
            for j, key in enumerate(keys):
                low, high, n = ci(draws[:, j])
                rows.append(dict(matrix=kind, coordinate=key, contrast=contrast, n_a=len(a), n_b=len(b),
                                 effect_size=a[:, j].mean()-b[:, j].mean(), ci95_low=low, ci95_high=high,
                                 bootstrap_valid=n, bootstrap_requested=CFG["bootstrap"]))
    return pd.DataFrame(rows)


def conjunction(scalar):
    rows = []
    for (family, endpoint), block in scalar.groupby(["family", "endpoint"], sort=False):
        by_group = block.set_index("contrast")
        off, on = (by_group.loc[c] for c in HC_CONTRASTS)
        same = np.sign(off.effect_size) == np.sign(on.effect_size) and np.sign(off.effect_size) in (-1, 1)
        agrees_fd = np.sign(off.fd_effect) == np.sign(on.fd_effect) and np.sign(off.fd_effect) in (-1, 1)
        def maximum(key, agree):
            p = np.asarray([off[key], on[key]])
            return p.max() if agree and np.isfinite(p).all() else 1.

        rows.append(dict(family="Shared direction IUT", source_family=family, endpoint=endpoint,
                         feature=off.feature, contrast="OFF-HC & ON-HC", effect_units=off.effect_units,
                         effect_size=np.nan, off_hc_effect=off.effect_size, on_hc_effect=on.effect_size,
                         off_hc_ci95_low=off.ci95_low, off_hc_ci95_high=off.ci95_high,
                         on_hc_ci95_low=on.ci95_low, on_hc_ci95_high=on.ci95_high,
                         off_hc_fd_effect=off.fd_effect, on_hc_fd_effect=on.fd_effect,
                         n_a=off.n_a, n_b=off.n_b, paired=False, defined=bool(off.defined and on.defined),
                         raw_p=maximum("raw_p", same), permutation_p=maximum("permutation_p", same),
                         fd_p=maximum("fd_p", agrees_fd), raw_p_basis="maximum of the two two-sided constituent P values",
                         direction="higher" if same and off.effect_size>0 else "lower" if same else "discordant"))
    return pd.DataFrame(rows)


def run(method, penalty):
    data = load_data(penalty)
    print(f"Computing six primary tests with {method}", flush=True)
    primary = [omnibus(data, ILR if endpoint == "coverage_energy" else MATRIX, c, endpoint)
               for endpoint in ("coverage_energy", "transition_wald") for c in CFG["primary"][endpoint]]
    rest, distances = restoration(data)
    primary = pd.DataFrame([*primary, rest]).assign(family="Primary six", method=method)
    dump(dict(n_hc=15, n_pairs=len(distances), n_closer=int(np.sum(distances[:, 2]>0)), n_farther=int(np.sum(distances[:, 2]<0)),
              n_equal=int(np.sum(distances[:, 2]==0))), HERE / "private" / f"{method}_restoration_counts.json")
    coverage_aux = pd.DataFrame([omnibus(data, ILR, c, key) for key in ("coverage_center", "coverage_dispersion")
                                 for c in HC_CONTRASTS])
    print(f"Computing state and episode outcomes with {method}", flush=True)
    coverage_local = scalar_tests(data, COVERAGE, "Coverage localization", HC_CONTRASTS)
    edge_results = scalar_tests(data, EDGES, "Transition localization")
    duration = [scalar_tests(data, keys, label) for label, keys in DURATION.items()]
    occurrence, diagnostics, rate = models(data)
    common = conjunction(pd.concat([coverage_local, edge_results, *duration, rate], ignore_index=True))
    full = pd.concat([primary, coverage_aux, coverage_local, edge_results, *duration, occurrence, common], ignore_index=True)
    for col, description in [("raw_p_basis", "normality-guided scalar test named in test column"),
                             ("fd_p_basis", "Freedman-Lane participant-level permutation"),
                             ("permutation_p_basis", "participant-level permutation")]:
        full[col] = full.get(col, pd.Series(index=full.index, dtype=object)).fillna(description)
    count = full.family.eq("Count Occurrence")
    full.loc[count, ["raw_p_basis", "fd_p_basis"]] = "participant-clustered GLM t (rate permutation if fallback)"
    full.loc[count, "permutation_p_basis"] = "participant-level permutation on episode rates"
    full.loc[full.family.eq("Shared direction IUT"), ["fd_p_basis", "permutation_p_basis"]] = "maximum of constituent P values for matching directions"
    full["method"] = method
    return full, dict(descriptives=describe(data), diagnostics=diagnostics, rate=adjustment(rate), coordinates=coordinate_effects(data))


def correct(all_tests):
    s = all_tests.copy()
    calibration = s.method.nunique()>1
    corrections(s, P_COLUMNS, ["method", "family"])
    s["family_size"] = s.groupby(["method", "family"]).endpoint.transform("size")
    together = s.assign(new_hypotheses="All new hypotheses")
    corrections(together, {p: "joint_"+prefix for p, prefix in P_COLUMNS.items()}, ["new_hypotheses"])
    s = together.drop(columns="new_hypotheses")
    primary = s.loc[s.family.eq("Primary six")].copy()
    primary["primary_hypotheses"] = "Shared primary hypotheses"
    corrections(primary, {p: "cross_method_"+prefix for p, prefix in P_COLUMNS.items()}, ["primary_hypotheses"])
    transition = primary.loc[primary.endpoint.eq("transition_wald")].copy()
    corrections(transition, {p: "transition_three_"+prefix for p, prefix in P_COLUMNS.items()}, ["method"])
    for table, prefix in [(primary, "cross_method_"), (transition, "transition_three_")]:
        for col in table.filter(regex=f"^{prefix}.*_p$"):
            s.loc[table.index, col] = table[col]
    by_test = primary.set_index(["method", "endpoint", "contrast"])
    for suffix in ("permutation", "fd", "raw"):
        stem = "" if suffix == "raw" else f"{suffix}_"
        pcol = f"cross_method_{stem}holm_p" if calibration else f"{stem}holm_p"
        column = f"{suffix}_gate_open"
        s[column] = True
        for group, key in [("Coverage localization", "coverage_energy"), ("Transition localization", "transition_wald")]:
            indices = s.index[s.family.eq(group)]
            for i in indices:
                row = s.loc[i]
                s.loc[i, column] = by_test.loc[(row.method, key, row.contrast), pcol] < CFG["alpha"]
            for i in s.index[s.source_family.eq(group)]:
                s.loc[i, column] = all(by_test.loc[(s.loc[i, "method"], key, c), pcol]<CFG["alpha"] for c in HC_CONTRASTS)
        s[f"{suffix}_holm_supported"] = s[column] & s[f"{stem}holm_p"].lt(CFG["alpha"])
        s[f"{suffix}_fdr_supported"] = s[column] & s[f"{stem}fdr_p"].lt(CFG["alpha"])
        s[f"{suffix}_joint_holm_supported"] = s[column] & s[f"joint_{stem}holm_p"].lt(CFG["alpha"])
    # A candidate's primary evidence must also pass the explicit across-method primary correction.
    if calibration:
        for i in primary.index:
            for suffix, stem in (("raw", ""), ("permutation", "permutation_"), ("fd", "fd_")):
                for correction in ("holm", "fdr"):
                    s.loc[i, f"{suffix}_{correction}_supported"] &= s.loc[i, f"cross_method_{stem}{correction}_p"]<CFG["alpha"]
    return s


def export(all_tests, details):
    csv(all_tests, RESULT / "All_V5_Statistical_Results.csv")
    merged = []
    for method, detail in details.items():
        r = all_tests.loc[all_tests.method.eq(method)]
        target = RESULT if method == "V4_A3" else RESULT / "alternate" / method
        cov, transition, duration, occurrence, exposure = detail["descriptives"]
        out = {
            "Primary_Six_Results": r.loc[r.family.eq("Primary six")],
            "coverage/Coverage_Composition": cov,
            "coverage/Coverage_Energy": r.loc[r.endpoint.eq("coverage_energy")],
            "coverage/Coverage_Center_and_Dispersion": r.loc[r.endpoint.isin(["coverage_center", "coverage_dispersion"])],
            "coverage/Coverage_Localization": r.loc[r.family.eq("Coverage localization")],
            "transition/Transition_Matrices": transition,
            "transition/Transition_Omnibus": r.loc[r.endpoint.eq("transition_wald")],
            "transition/Transition_Localization": r.loc[r.family.eq("Transition localization")],
            "duration/Duration_Results": r.loc[r.feature.str.startswith("dwell_", na=False) & r.family.ne("Shared direction IUT")],
            "duration/Duration_Summaries": duration,
            "occurrence/Occurrence_Count_Models": r.loc[r.family.eq("Count Occurrence")],
            "occurrence/Occurrence_Diagnostics": detail["diagnostics"],
            "occurrence/Occurrence_Rate_Sensitivity": detail["rate"],
            "occurrence/Occurrence_Summaries": occurrence,
            "restoration/Restoration_Results": r.loc[r.endpoint.eq("restoration")],
            "covariates/Exposure_Audit": exposure,
            "bootstrap/ILR_Coordinate_Intervals": detail["coordinates"],
            "Shared_PD_Direction_Results": r.loc[r.family.eq("Shared direction IUT")],
        }
        for name, frame in out.items():
            csv(frame, target / f"{name}.csv")
        boot = r.loc[r.bootstrap_requested.notna(), ["method", "family", "endpoint", "feature", "contrast", "n_a", "n_b",
                     "effect_size", "ci95_low", "ci95_high", "bootstrap_requested", "bootstrap_valid", "bootstrap_failure_reason",
                     "fd_bootstrap_valid", "fd_bootstrap_failure_reason"]]
        csv(boot, target / "bootstrap/Fixed_Template_Bootstrap_Audit.csv")
        merged.append(boot)
    csv(pd.concat(merged, ignore_index=True), RESULT / "bootstrap/All_Methods_Bootstrap_Audit.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-only", action="store_true")
    args = parser.parse_args()
    methods = [("V4_A3", None)]
    selection = RESULT / "simulation/Penalty_Selection.json"
    if not args.frozen_only:
        if not selection.is_file():
            raise FileNotFoundError("Run independent noise calibration first")
        new = json.loads(selection.read_text(encoding="utf-8"))["selected_lambda"]
        if new is not None and new != CFG["baseline"]["lambda"]:
            methods.append((f"V5_A3_lambda_{new:g}", new))
    tables, lookup = [], {}
    for method, penalty in methods:
        table, lookup[method] = run(method, penalty)
        tables.append(table)
    complete = correct(pd.concat(tables, ignore_index=True))
    export(complete, lookup)
    print(f"V5 participant analysis complete: {len(complete)} rows from {len(methods)} method(s).", flush=True)


if __name__ == "__main__":
    main()
