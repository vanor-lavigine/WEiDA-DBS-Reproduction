"""Secondary PD–HC analysis using V4's fixed scan features and statistics.

Use every primary state/edge metric from each method. The patient-level
average is half ON and half OFF, with both visits required for that outcome.
For a shared direction across visits, use the larger of the two archived
two-sided P values, assigning P=1 if the directions disagree. Apply BH and
Holm across all primary metrics within each method and analysis, separately
for the raw, permutation, and FD-adjusted permutation P values.
"""

import json
import sys

import numpy as np
import pandas as pd

from features import validate_roster
from run_v4 import HERE, V3, save_csv
from statistics_v4 import Resampling, corrections, freedman_lane, measures, outcome_test, select_samples


def participant_mean(scans, columns):
    groups = scans[scans.condition.isin(["ON", "OFF"])].pivot(index="subject", columns="condition", values=columns)
    on = groups.xs("ON", axis=1, level="condition")
    off = groups.xs("OFF", axis=1, level="condition")
    patient = ((on + off) / 2).reset_index().assign(condition="PD")
    hc = scans.loc[scans.condition.eq("HC"), ["subject", "condition", *columns]]
    return pd.concat([hc, patient], ignore_index=True)


def shared_p(on, off, p_col, effect_col):
    p = np.array([on[p_col], off[p_col]])
    if not np.isfinite(p).all():
        return np.nan
    same = on[effect_col] * off[effect_col] > 0
    return float(p.max()) if same else 1.


def run_method(method, cfg, resampling, archive):
    from inference_core import scalar

    results = HERE / "results"
    filename = "Microstate" if method == "Microstate" else "WEiDA"
    original = pd.read_csv(results / f"{filename}_Results.csv").set_index(["metric", "contrast"])
    features = pd.read_csv(HERE / "private" / f"{filename}_Subject_Features.csv")
    validate_roster(features)
    metrics = [m for m in measures(method) if m.scope == "primary"]
    averaged = participant_mean(features, ["fd", *(m.metric for m in metrics)])
    rows = []
    for metric in metrics:
        key = metric.metric
        on, off = (original.loc[(key, contrast)] for contrast in ("ON-HC", "OFF-HC"))
        x, y, fd_x, fd_y = select_samples(averaged, key, "PD", "HC", False)
        if min(len(x), len(y)) < 3:
            raise ValueError(f"Too few defined participant values for {method} {key}")
        unc = outcome_test(x, y, False, cfg, resampling)
        fl = freedman_lane(x, y, fd_x, fd_y, False, resampling)
        pooled = scalar(features.loc[np.isfinite(features[key])], key)["C4"]
        old = archive.get(key) if method == "Microstate" else None
        if old is not None:
            np.testing.assert_allclose([pooled[n] for n in ("Effect_Size", "CI_low", "CI_high", "P_raw")],
                                       [old[n] for n in ("Effect_Size", "CI_low", "CI_high", "P_raw")],
                                       atol=1e-11, rtol=1e-8)
        row = dict(method=method, metric=key, state=metric.state, family=metric.family, unit=metric.unit,
                   n_pd=len(x), n_hc=len(y), pooled_test=unc["test"],
                   pooled_mean_pd=x.mean(), pooled_mean_hc=y.mean(), pooled_mean_difference=unc["effect_size"],
                   pooled_ci95_low=unc["ci95_low"], pooled_ci95_high=unc["ci95_high"],
                   pooled_hedges_g=unc["standardized_effect"], pooled_raw_p=unc["raw_p"],
                   pooled_permutation_p=unc["permutation_p"], pooled_fd_permutation_p=fl["fd_permutation_p"],
                   pooled_fd_effect=fl["fd_permutation_effect"], c4_cr1_effect=pooled["Effect_Size"],
                   c4_cr1_ci95_low=pooled["CI_low"], c4_cr1_ci95_high=pooled["CI_high"], c4_cr1_raw_p=pooled["P_raw"],
                   v3_c4_archive_verified=old is not None, v3_c4_archived_holm_p=old["P_Holm"] if old else np.nan,
                   same_mean_direction=off.effect_size * on.effect_size > 0,
                   both_raw_p=shared_p(on, off, "raw_p", "effect_size"),
                   both_permutation_p=shared_p(on, off, "permutation_p", "effect_size"),
                   both_fd_permutation_p=shared_p(on, off, "fd_permutation_p", "fd_permutation_effect"))
        for prefix, visited in (("off_hc", off), ("on_hc", on)):
            for name in ("effect_size", "ci95_low", "ci95_high", "raw_p", "holm_p", "permutation_p",
                         "permutation_holm_p", "fd_permutation_p", "fd_permutation_holm_p"):
                row[f"{prefix}_{name}"] = visited[name]
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    sys.path.insert(0, str(V3 / "scripts"))
    results = HERE / "results"
    cfg = json.loads((HERE / "configs" / "protocol.json").read_text(encoding="utf-8"))
    archive = pd.read_csv(results / "V3_FD_Adjusted_Archive.csv")
    previous = archive.loc[archive.experiment.eq(cfg["v3_state_archive"]) & archive.contrast.eq("C4") & archive.model.eq("CR1")]
    previous = previous.set_index("metric").to_dict("index")
    resampling = Resampling(cfg["seed"], cfg["permutations"], cfg["bootstrap_draws"])
    table = pd.concat([run_method(name, cfg, resampling, previous) for name in ("Microstate", "WEiDA")], ignore_index=True)
    mapping = {f"{target}_{measure}_p": f"{target}_{'' if measure == 'raw' else measure + '_'}"
               for target in ("pooled", "both") for measure in ("raw", "permutation", "fd_permutation")}
    corrections(table, mapping | {"c4_cr1_raw_p": "c4_cr1_"}, ["method"])
    table["n_metrics_in_family"] = table.groupby("method").metric.transform("size")
    assert table.n_metrics_in_family.value_counts().to_dict() == {32: 32, 28: 28}
    for target in ("pooled", "both"):
        table[f"{target}_both_holm_pass"] = table[f"{target}_holm_p"].lt(cfg["alpha"]) & table[f"{target}_permutation_holm_p"].lt(cfg["alpha"])
    save_csv(table, results, "PD_Disease_Associations")
    print(f"Disease analysis: {len(table)} metrics, {table.v3_c4_archive_verified.sum()} archived V3 C4 estimates independently reproduced.")


if __name__ == "__main__":
    main()
