"""Replay the original V3 tests and independently recount archived simulations."""

import json

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from micro_inputs import HERE, RELIABILITY, SETTINGS, V3
from inference_core import adjust, composition, scalar
from inference_correspondence import correspondence, reindex_coverage, state_family


def read_original_tables():
    root = V3 / "results"
    archives = pd.read_csv(root / "All_Methods_Comparison.csv", low_memory=False)
    for name, code in [("HC_vs_OFF_All_Results", "C2"), ("HC_vs_ON_All_Results", "C1"),
                       ("ON_vs_OFF_All_Results", "C3"), ("HC_Both_Conditions_Significant", "both")]:
        actual = pd.read_csv(root / f"{name}.csv", low_memory=False)
        expected = archives.loc[archives.contrast.eq(code)].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=False)
    return archives


def v3_tests(out):
    archive = read_original_tables()
    private = pd.read_csv(V3 / "private/Subject_Level_Features.csv", low_memory=False)
    raw_maps = pd.read_csv(V3 / "results/Group_Templates.csv")
    maps = raw_maps.loc[raw_maps.experiment.eq("E04")]
    match_rows, order, passed = correspondence(maps)
    assert passed and match_rows.qualified.all()
    peak = raw_maps.loc[raw_maps.experiment.eq("E02"), maps.columns[1:]].reset_index(drop=True)
    np.testing.assert_allclose(peak.to_numpy(float), maps.iloc[:, 1:].to_numpy(float), atol=1e-12, rtol=0)
    observed = private.loc[private.experiment.eq("E04") & private.fit.eq("crossfit")]
    assert len(observed) == 65
    aligned = reindex_coverage(observed, order)
    coverage = [f"coverage_{j}" for j in range(1, 5)]
    values = {"switching_rate": scalar(observed, "switching_rate"),
              "coverage_ilr": composition(aligned, coverage, draws=9999)}
    primary = archive.loc[archive.experiment.eq("E04") & archive.fit.eq("crossfit") & archive.P_primary_Holm4.notna()].copy()
    recomputed = [values[r.metric][r.contrast] for r in primary.itertuples()]
    np.testing.assert_allclose([r["P_raw"] for r in recomputed], primary.P_raw, atol=1e-12, rtol=0)
    np.testing.assert_allclose([r["Effect_Size"] for r in recomputed], primary.Effect_Size, atol=1e-12, rtol=0)
    np.testing.assert_allclose(adjust(primary.P_raw, "holm"), primary.P_primary_Holm4, atol=1e-12, rtol=0)
    primary.assign(reproduced=True).to_csv(out / "V3_Primary_Reproduction.csv", index=False)
    reference = archive.loc[archive.experiment.isin(["E02", "E04", "E04_R", "E09"]) & archive.fit.eq("crossfit")].copy()
    reference.to_csv(out / "V3_Reference_Comparisons.csv", index=False)
    rows = []
    for (experiment, model, contrast), g in reference.groupby(["experiment", "model", "contrast"]):
        rows.append(dict(experiment=experiment, model=model, contrast=contrast, archived_rows=len(g),
                         defined=g.P_raw.notna().sum(), raw_p_lt_05=g.P_raw.lt(.05).sum(),
                         fdr_p_lt_05=g.P_FDR.lt(.05).sum(), holm_p_lt_05=g.P_Holm.lt(.05).sum(),
                         across_method_holm_lt_05=g.P_across_methods.lt(.05).sum()))
    pd.DataFrame(rows).to_csv(out / "V3_Archive_Threshold_Counts.csv", index=False)
    # Recompute every archived CR1 state contrast for the aligned E04_R point fit.
    d = state_family(observed, "E04", order)
    states = reference.loc[reference.experiment.eq("E04_R") & reference.model.eq("CR1")]
    replayed = 0
    for key, g in states.groupby("metric"):
        estimates = scalar(d.loc[d[key].notna()], key)
        for r in g.itertuples():
            est = estimates[r.contrast]
            np.testing.assert_allclose([est[k] for k in ("Effect_Size", "P_raw", "CI_low", "CI_high")],
                                       [r.Effect_Size, r.P_raw, r.CI_low, r.CI_high], atol=1e-11, rtol=1e-8)
            replayed += 1
    boot = pd.read_csv(V3 / "results/Supplement_Bootstrap_formal.csv")
    boot = pd.concat([boot, pd.read_csv(V3 / "results/all_R_Bootstrap_formal.csv")], ignore_index=True)
    boot.to_csv(out / "V3_Bootstrap_Archive.csv", index=False)
    return dict(all_v3_rows_read=len(archive), subject_feature_rows_read=len(private),
                original_primary_tests_reproduced=len(primary), e04_r_cr1_reproduced=replayed,
                e02_and_e04_same_template=True, e04_min_abs_r=float(match_rows.abs_correlation.min()),
                e04_min_margin=float(match_rows.bidirectional_margin.min()),
                archived_bootstrap_rows=len(boot))


def v3_simulation(out):
    rows = []
    for s in ("S0", "S5"):
        pieces = sorted((V3 / "private/simulation").glob(f"{s}_*.csv"))
        trials = pd.concat([pd.read_csv(p) for p in pieces], ignore_index=True)
        only = trials.loc[trials.metric.eq("switching_rate")]
        for method in ("F1_TR_common", "F1_peak_common"):
            g = only.loc[only.method.eq(method)]
            assert len(g) == g.rep.nunique() == g.p.notna().sum() == 500
            n = g.p.lt(.05).sum()
            lower, upper = proportion_confint(n, len(g), method="wilson")
            rows.append(dict(scenario=s, method=method, n=len(g), rejections=n, rate=n/len(g), wilson_low=lower, wilson_high=upper))
    table = pd.DataFrame(rows)
    table.to_csv(out / "V3_Peak_Null_Recount.csv", index=False)
    assert table.loc[table.scenario.eq("S5"), "rejections"].tolist() == [213, 43]


def v4_simulation(out):
    r = RELIABILITY / "results"
    paths = sorted((RELIABILITY / "private/simulation/formal").glob("scenario_*_cohort_*.json"))
    assert len(paths) == 1200
    records, family_tests = [], []
    for p in paths:
        trial = json.loads(p.read_text(encoding="utf-8"))
        tests = pd.DataFrame(trial["inference"])
        for q in trial["quality"]:
            t = tests.loc[tests.method.eq(q["method"])]
            core = t.loc[t.metric.isin(["coverage_ilr", "global_gev", "mean_margin"]), "p"]
            assert len(core) == 6
            any_reject = bool(np.any(adjust(core, "holm") < .05))
            assert any_reject == q["core_any_reject"]
            records.append(dict(scenario=q["scenario"], method=q["method"], repeat=q["repeat"], rejected=any_reject))
            for metric in ("coverage", "duration", "occurrence"):
                pvalues = t.loc[t.metric.isin([f"{metric}_{i}" for i in range(1, 5)]), "p"]
                assert len(pvalues) == 8
                family_tests.append(dict(scenario=q["scenario"], method=q["method"], family=metric,
                                         estimable_all=pvalues.notna().all(),
                                         reject_holm=bool(np.any(adjust(pvalues, "holm") < .05)),
                                         reject_bh=bool(np.any(adjust(pvalues, "fdr_bh") < .05))))
    recount = pd.DataFrame(records)
    agg = recount.groupby(["scenario", "method"]).rejected.agg(["size", "sum"]).reset_index()
    core = pd.read_csv(r / "Simulation_Core_Calibration.csv")
    cross = core.merge(agg, on=["scenario", "method"], validate="one_to_one")
    assert len(cross) == 24 and cross.replicates.eq(cross["size"]).all() and cross.core_any_reject.eq(cross["sum"]).all()
    lower, upper = proportion_confint(cross["sum"], cross["size"], method="wilson")
    np.testing.assert_allclose(upper, cross.wilson_hi, atol=1e-12, rtol=0)
    cross.drop(columns=["size", "sum"]).to_csv(out / "Reliability_Core_Calibration_Recount.csv", index=False)
    rows = []
    for (scenario, method, family), d in pd.DataFrame(family_tests).groupby(["scenario", "method", "family"]):
        n = int(d.reject_holm.sum())
        a, b = proportion_confint(n, len(d), method="wilson")
        rows.append(dict(scenario=scenario, method=method, family=family, n=len(d), all_8_estimable=d.estimable_all.sum(),
                         holm_rejects=n, holm_any_rate=n/len(d), wilson_low=a, wilson_high=b,
                         bh_rejects=d.reject_bh.sum(), bh_any_rate=d.reject_bh.mean(), source_test="archived FD-adjusted CR1"))
    pd.DataFrame(rows).to_csv(out / "Reliability_Traditional_Family_Recount.csv", index=False)
    choice = json.loads((r / "Calibration_Selection.json").read_text(encoding="utf-8"))
    assert choice["preferred_method"] is None and not choice["eligible_methods"] and choice["lambda"] == SETTINGS["lambda"]
    rank = core.loc[core.scenario.str.startswith("S5")].groupby("method").wilson_hi.max().rename("worst_s5_wilson_upper")
    blinded = pd.read_csv(r / "Disjoint_Reliability.csv").set_index("method")
    comparison = blinded.join(rank).sort_values(["worst_s5_wilson_upper", "ari_mean"], ascending=[True, False])
    assert comparison.index[0] == SETTINGS["main_method"]
    comparison.assign(prospective_gate_passed=False).to_csv(out / "Fallback_Method_Selection.csv")
    peak = recount.loc[recount.scenario.eq("S5_high") & recount.method.isin(["A0", "A1"])].rejected.sum()
    return dict(formal_synthetic_cohorts=len(paths), core_simulation_rows_replayed=len(recount),
                eligible_methods=choice["eligible_methods"], fallback_method=comparison.index[0], peak_high_noise_rejections_total=int(peak))


def main():
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    v3_simulation(out)
    results = v4_simulation(out) | v3_tests(out)
    (out / "Historical_Audit_Verification.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
