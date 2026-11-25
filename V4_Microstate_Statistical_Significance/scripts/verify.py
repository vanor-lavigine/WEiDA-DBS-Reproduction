"""Verify the complete publication set, numerical families and CI semantics."""

from hashlib import sha256
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from statsmodels.stats.multitest import multipletests

from micro_inputs import EDGE, FEATURES, HERE, SETTINGS, scheme
from micro_inference import P_VALUES, UniqueResampling, hc3_wald, independent_wald


R = HERE / "results"


def csv(name):
    return pd.read_csv(R / f"{name}.csv")


def compare_adjustment(d, p, fdr, holm):
    array = d[p].to_numpy()
    keep = np.isfinite(array)
    for label, method in ((fdr, "fdr_bh"), (holm, "holm")):
        expected = np.where(keep, multipletests(np.where(keep, array, 1), method=method)[1], np.nan)
        np.testing.assert_allclose(d[label], expected, atol=1e-12, equal_nan=True)


def unit_checks():
    labels = np.array([0, 0, 1, -1, 2, 2, 1, 1, 3, 3])
    q = scheme((labels, np.full((len(labels), 4), .6), np.ones(len(labels))), np.arange(4))
    m = dict(zip(FEATURES, q))
    assert m["transition_1_1"] == .5 and m["transition_1_2"] == .5
    assert m["transition_2_3"] == 0 and m["transition_3_2"] == .5 and m["transition_4_4"] == 1
    assert m["switching_rate"] == 3 / (9 * SETTINGS["tr_seconds"])
    assert m["duration_2"] == 1.5 * SETTINGS["tr_seconds"]

    rs = UniqueResampling(SETTINGS["seed"], 10000, 10000)
    permutation = rs.independent(25, 15)
    allocations = np.sort(permutation[:, :25], axis=1)
    assert np.unique(allocations, axis=0).shape == (10000, 25)
    assert not np.all(allocations == np.arange(25), axis=1).any()
    signs = rs.signs(25)
    assert np.unique(signs, axis=0).shape == (10000, 25) and not np.all(signs == 1, axis=1).any()

    rng = np.random.default_rng(211)
    x, y = rng.normal(size=(25, 3)), rng.normal(size=(15, 3))
    diff = x.mean(0) - y.mean(0)
    cov = np.cov(x.T) / len(x) + np.cov(y.T) / len(y)
    np.testing.assert_allclose(independent_wald(x, y), diff @ np.linalg.solve(cov, diff), atol=1e-12)
    d = np.c_[np.ones(40), rng.normal(size=40), np.r_[np.ones(25), np.zeros(15)]]
    out = np.r_[x, y]
    bread = np.linalg.inv(d.T @ d)
    fit = bread @ d.T @ out
    resid = out - d @ fit
    c = np.array([0, 0, 1.])
    score = np.stack([r * (c @ bread @ row) / (1 - row @ bread @ row) for row, r in zip(d, resid)])
    vc = score.T @ score
    np.testing.assert_allclose(hc3_wald(d, out), fit[-1] @ np.linalg.solve(vc, fit[-1]), atol=1e-12)


def output_checks():
    tests, comp, shared, extra = [csv(n) for n in ("All_Scalar_Results", "Coverage_ILR_Results", "Shared_PD_Direction_Results", "Exploratory_Global_Corrections")]
    assert len(tests) == 99 and len(comp) == 4 and len(shared) == 33
    for files in (tests, comp):
        assert files.status.eq("ok").all()
        assert files.permutation_draws.eq(10000).all() and files.fd_permutation_draws.eq(10000).all()
        for key in ("permutation_p", "fd_permutation_p"):
            multiples = (10001 * files[key]).to_numpy()
            np.testing.assert_allclose(multiples, np.round(multiples), atol=1e-10, rtol=0)
    for data in (tests, comp, shared):
        for _, d in data.groupby("correction_family"):
            assert d.family_size.eq(len(d)).all()
            for raw, prefix in P_VALUES.items():
                if raw in d:
                    compare_adjustment(d, raw, prefix + "fdr_p", prefix + "holm_p")
    for raw, prefix in P_VALUES.items():
        compare_adjustment(tests, raw, f"all_scalar_{prefix}fdr_p", f"all_scalar_{prefix}holm_p")
        compare_adjustment(extra, raw, f"global_{prefix}fdr_p", f"global_{prefix}holm_p")
    keys = list(P_VALUES)
    grid = extra[keys].to_numpy()
    slots = np.ones(grid.shape, bool)
    for p in ("raw_p", "fd_cr1_p"):
        slots[extra.family.eq("Sorted Coverage ILR"), keys.index(p)] = False
    for name, how in (("fdr", "fdr_bh"), ("holm", "holm")):
        observed = extra[[f"global_model_union_{p}{name}_p" for p in P_VALUES.values()]].to_numpy()[slots]
        defined = np.isfinite(grid[slots])
        expected = np.where(defined, multipletests(np.where(defined, grid[slots], 1.), method=how)[1], np.nan)
        np.testing.assert_allclose(expected, observed, atol=1e-12, equal_nan=True)

    ds = pd.read_csv(HERE / "private/Subject_Level_Microstate_Features.csv")
    assert len(ds) == ds.scan.nunique() == 65 and ds.subject.nunique() == 40
    assert ds.groupby("subject").fold.nunique().max() == 1
    patients = ds.loc[ds.condition.isin(["ON", "OFF"])].pivot(index="subject", columns="condition", values="switching_rate")
    assert patients.shape == (25, 2) and patients.notna().all().all()
    np.testing.assert_allclose(ds[[f"coverage_{i}" for i in range(1,5)]].sum(axis=1), 1, atol=1e-12)
    for i in range(1, 5):
        p = ds[[f"transition_{i}_{j}" for j in range(1, 5)]]
        np.testing.assert_allclose(p.dropna().sum(axis=1), 1, atol=1e-12)
    t = csv("Transition_Results")
    assert len(t) == 36 and set(t.metric) == {f"transition_{i}_{j}" for i, j in EDGE}
    assert len(csv("Transition_Matrix_Descriptive")) == 48
    assert len(csv("Reference_Templates")) == 4 and len(csv("Fold_Templates")) == 20
    assert len(csv("State_Matching")) == 40

    boot = csv("Bootstrap_Results")
    record = json.loads((R / "Bootstrap_Verification.json").read_text(encoding="utf-8"))
    assert len(boot) == len(FEATURES)*3*2
    assert record["fully_refitted"] == record["draws"] == 2000 and record["historical_state_counts_equal"]
    absent = boot.defined.lt(2000)
    assert boot.loc[absent, ["ci95_low", "ci95_high"]].isna().all().all()
    assert boot.loc[~absent, ["ci95_low", "ci95_high"]].notna().all().all()
    consistent = boot[boot.scheme.eq("full") & boot.full_interval_excludes_zero]
    assert consistent.defined.eq(2000).all()
    assert consistent.metric.str.startswith(("switching_rate", "sorted_ilr_")).all()

    for name in ("Coverage_Comparison", "Duration_Comparison", "Occurrence_Comparison", "Transition_Matrix", "GEV_Comparison",
                 "Effect_Size_Forest", "Significance_Summary", "State_Templates"):
        with Image.open(HERE / "figures" / f"{name}.png") as figure:
            assert figure.format == "PNG" and min(figure.size) >= 640
            figure.verify()
    required = [HERE / "README.md", *sorted((HERE / "reports").glob("*.md"))]
    assert len(required) == 5
    for path in required:
        contents = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\((\.{1,2}/[^)]+|(?:results|reports|figures)/[^)]+)\)", contents):
            assert (path.parent / link).exists(), (path, link)
    files = sorted([*R.glob("*.csv"), *(HERE / "reports").glob("*.md"), *(HERE / "figures").glob("*.png")])
    summary = dict(scalar_rows=len(tests), composition_rows=len(comp), shared_rows=len(shared), transition_rows=len(t),
                   participants=40, scans=65, distinct_permutations_per_test=10000, full_bootstrap_draws=2000,
                   checked_artifacts=len(files), hashes={p.relative_to(HERE).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files})
    (R / "Final_Verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(files)} artifacts, identical-cohort units, all correction families, distinct permutations and 2,000 full refits.")


if __name__ == "__main__":
    unit_checks()
    output_checks()
