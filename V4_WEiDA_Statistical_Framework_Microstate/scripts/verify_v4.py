"""Independently validate the result families and the resampling implementation."""

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
import statsmodels.api as sm

from features import EDGES, validate_roster
from run_v4 import HERE, ROOT
from statistics_v4 import Resampling, freedman_lane, hc3_t, outcome_test, tail_probability


def holm(p):
    p = np.asarray(p, float)
    valid = np.isfinite(p)
    order = np.argsort(np.where(valid, p, 1))
    out = np.empty(len(p))
    out[order] = np.minimum(1., np.maximum.accumulate(np.where(valid, p, 1)[order] * np.arange(len(p), 0, -1)))
    return np.where(valid, out, np.nan)


def verify_p(data, base, bh, hm):
    p = data[base].to_numpy()
    corrected = stats.false_discovery_control(np.where(np.isfinite(p), p, 1), method="bh")
    np.testing.assert_allclose(data[bh], np.where(np.isfinite(p), corrected, np.nan), atol=1e-12)
    np.testing.assert_allclose(data[hm], holm(p), atol=1e-12)


def exact_permutation_checks():
    x, y = np.array([4., 8., 9., 13.]), np.array([1., 2., 5., 8.])

    class Complete(Resampling):
        def independent(self, nx, ny):
            candidates = list(itertools.combinations(range(nx + ny), nx))
            return np.array([list(a) + sorted(set(range(nx + ny)) - set(a)) for a in candidates])

        def signs(self, n):
            return np.array(list(itertools.product([-1., 1.], repeat=n)))

    resampling = Complete(12, 9999, 500)
    cfg = {"normality_alpha": .05}
    order = resampling.independent(len(x), len(y))
    pooled = np.r_[x, y]
    observed = abs(stats.ttest_ind(x, y, equal_var=False).statistic)
    null = [abs(stats.ttest_ind(pooled[row[:len(x)]], pooled[row[len(x):]], equal_var=False).statistic) for row in order]
    p = (1 + sum(t >= observed - 1e-12 for t in null)) / (1 + len(null))
    np.testing.assert_allclose(outcome_test(x, y, False, cfg, resampling)["permutation_p"], p)
    observed = abs(stats.ttest_rel(x, y).statistic)
    null = [abs(stats.ttest_1samp((x - y) * signs, 0).statistic) for signs in resampling.signs(len(x))]
    p = (1 + sum(t >= observed - 1e-12 for t in null)) / (1 + len(null))
    np.testing.assert_allclose(outcome_test(x, y, True, cfg, resampling)["permutation_p"], p)


def fd_checks():
    rng = np.random.default_rng(547)
    fd_x, fd_y, n = rng.uniform(.03, .5, 15), rng.uniform(.01, .4, 15), 15
    x, y = 2 * fd_x + rng.normal(size=n), 2 * fd_y + rng.normal(size=n)
    resampling = Resampling(1673, 127, 100)
    for paired in (True, False):
        if paired:
            d = fd_x - fd_y
            z, full, effect_index, outcome = d[:, None], np.c_[np.ones(n), d], 0, x - y
        else:
            d = np.r_[fd_x, fd_y]
            z = np.c_[np.ones(2 * n), d - d.mean()]
            full, effect_index, outcome = np.c_[z, np.repeat([1, 0], n)], 2, np.r_[x, y]
        fit = sm.OLS(outcome, z).fit()
        if paired:
            draws = fit.fittedvalues + resampling.signs(n) * fit.resid
        else:
            draws = fit.fittedvalues + fit.resid[resampling.independent(n, n)]
        expected = sm.OLS(outcome, full).fit(cov_type="HC3")
        null = np.array([sm.OLS(row, full).fit(cov_type="HC3").tvalues[effect_index] for row in draws])
        actual, t = hc3_t(full, outcome, effect_index)
        np.testing.assert_allclose([actual, t], [expected.params[effect_index], expected.tvalues[effect_index]], atol=1e-10)
        full_result = freedman_lane(x, y, fd_x, fd_y, paired, resampling)
        np.testing.assert_allclose(full_result["fd_permutation_p"], tail_probability(null, expected.tvalues[effect_index]))


def verify_outputs():
    results = HERE / "results"

    def get(name):
        return pd.read_csv(results / f"{name}.csv")

    audit = json.loads((results / "Input_Verification.json").read_text(encoding="utf-8"))
    config = json.loads((HERE / "configs/protocol.json").read_text(encoding="utf-8"))
    for name, sha in audit["input_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    for name, key in [("configs/protocol.json", "config_sha256"), ("reports/WEIDA_STATISTICAL_PROTOCOL.md", "protocol_sha256")]:
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == audit[key], name
    assert audit["cohort"]["effective_frames"] == 13647 and audit["cohort"]["participants"] == 40
    assert audit["permutations"] == config["permutations"] == 9999
    original = get("Microstate_Results")
    ms_data = pd.read_csv(HERE / "private/Microstate_Subject_Features.csv")
    validate_roster(ms_data)
    core = original[original.scope.eq("primary")].reset_index(drop=True)
    assert len(core) == 96 and core.groupby("contrast").size().eq(32).all()
    assert core.status.eq("ok").all()
    for row in core.itertuples():
        test = ms_data.loc[ms_data.condition.isin([row.group_a, row.group_b])].pivot(index="subject", columns="condition", values=row.metric)
        if row.paired:
            test = test.dropna()
            assert row.n_pairs == len(test) == 25
        np.testing.assert_allclose(row.effect_size, (test[row.group_a].mean() - test[row.group_b].mean()), atol=1e-12)
        if row.test == "Mann-Whitney U":
            p = stats.mannwhitneyu(test[row.group_a].dropna().round(12), test[row.group_b].dropna().round(12), method="asymptotic").pvalue
        elif row.test == "Wilcoxon signed-rank":
            p = stats.wilcoxon((test[row.group_a] - test[row.group_b]).round(12), method="approx").pvalue
        elif row.test == "paired t":
            p = stats.ttest_rel(test[row.group_a], test[row.group_b]).pvalue
        else:
            assert row.test == "Welch t"
            p = stats.ttest_ind(test[row.group_a].dropna(), test[row.group_b].dropna(), equal_var=False).pvalue
        np.testing.assert_allclose(row.raw_p, p, atol=1e-12)
    for dataset in (original, get("WEiDA_Results")):
        for _, family in dataset.groupby("family"):
            assert family.family_size.eq(len(family)).all()
            for p, prefix in (("raw_p", ""), ("permutation_p", "permutation_"), ("fd_permutation_p", "fd_permutation_")):
                verify_p(family, p, prefix + "fdr_p", prefix + "holm_p")
    verify_p(core, "raw_p", "all_primary_fdr_p", "all_primary_holm_p")
    verify_p(core, "permutation_p", "all_primary_permutation_fdr_p", "all_primary_permutation_holm_p")
    transition = get("Transition_Results")
    assert set(transition.metric) == {f"transition_{i}_{j}" for i, j in EDGES}
    assert len(transition) == 36 and transition.family_size.eq(36).all()
    for a, b in [("HC_vs_OFF", "OFF-HC"), ("HC_vs_ON", "ON-HC"), ("ON_vs_OFF", "ON-OFF")]:
        pd.testing.assert_frame_equal(get(a), core[core.contrast.eq(b)].reset_index(drop=True))
    for name in ["FDR_Results", "Holm_Results", "Permutation_Results"]:
        read = get(name)
        pd.testing.assert_frame_equal(read, core[read.columns])
    closeness = get("Towards_HC")
    for _, d in closeness.groupby("scope"):
        verify_p(d, "toward_p", "toward_fdr_p", "toward_holm_p")
    mapping = pd.read_csv(HERE / "private/Participant_Scan_Mapping.csv")
    assert len(mapping) == 65 and mapping.exact_roi_values.all() and mapping.fo_dt_transition_verified.all()
    validate_roster(mapping)
    match = get("State_Matching")
    assert len(match) == 80 and match.hungarian_selected.sum() == 20
    cross = get("Cross_Method_Comparison")
    if not audit["cross_method_state_match"]:
        assert cross.weida_metric.notna().sum() == 12
        assert cross.loc[cross.weida_metric.notna(), "status"].eq("comparable overall metric").all()
    assert original.v3_fd_archive_status.eq("ok").sum() == audit["v3_fd_reproduced_rows"]
    np.testing.assert_allclose(core["permutation_p"] * (config["permutations"] + 1),
                               np.rint(core["permutation_p"] * (config["permutations"] + 1)), atol=1e-10)
    for name in ["Group_Comparison", "Paired_ON_OFF", "Transition_Matrix", "Effect_Size_Forest", "Cross_Method_Concordance"]:
        with Image.open(HERE / "figures" / f"{name}.png") as picture:
            assert picture.format == "PNG" and min(picture.size) >= 600
            picture.verify()
    print("Output checks passed: 96 microstate comparisons, 36 transitions, 65 scan links, 5 figures and all correction families.")


if __name__ == "__main__":
    exact_permutation_checks()
    fd_checks()
    verify_outputs()
    print("Permutation checks passed against exhaustive small examples and independent statsmodels HC3 fits.")
