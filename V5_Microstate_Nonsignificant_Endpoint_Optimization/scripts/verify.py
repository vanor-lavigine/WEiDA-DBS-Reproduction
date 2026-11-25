"""Verify numerical invariants, frozen inputs and the public V5 deliverable."""

from hashlib import sha256
from importlib.metadata import version
import json
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from v5_data import CFG, HERE, RESULT, V4_NAME, counts_episodes, dump, features, git, load_data, verify_v4
from v5_stats import RESAMPLE, energy, regression_wald


def check_geometry():
    labels = np.array([-1,0,0,1,0,-1,2,2,3,-1])
    counts, all_runs, internal = counts_episodes(labels)
    np.testing.assert_array_equal(counts, [[1,1,0,0],[1,0,0,0],[0,0,1,1],[0,0,0,0]])
    assert all_runs == [[4.2,2.1],[2.1],[4.2],[2.1]] and internal == [[],[2.1],[],[]]
    absent = features(labels)
    assert np.isnan(absent["dwell_edge_median_4"]) and np.isnan(absent["trans_ilr_1"])
    assert absent["transition_3_4"] == .5 and absent["valid_frames"] == 7
    theoretical = np.arange(27,dtype=float).reshape(9,3)
    a,b = theoretical[:4],theoretical[4:]
    expected = 2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean()
    np.testing.assert_allclose(energy(a,b),expected,rtol=1e-14)
    np.testing.assert_allclose(energy(a,a),0,atol=1e-14)
    orthogonal,_ = np.linalg.qr(np.array([[1.,2,3],[3,1,4],[2,-3,0]]))
    np.testing.assert_allclose(energy(a@orthogonal,b@orthogonal),expected,rtol=1e-14)

    rng = np.random.default_rng(72)
    x = np.c_[np.ones(40),np.r_[np.ones(25),np.zeros(15)],rng.normal(size=40)]
    y = rng.normal(size=(40,3))
    inv = np.linalg.inv(x.T@x)
    beta = inv@x.T@y
    resid = y-x@beta
    lever = np.diag(x@inv@x.T)
    weights = (inv@x.T)[1]/(1-lever)
    var = sum(weights[i]**2*np.outer(resid[i],resid[i]) for i in range(len(x)))
    np.testing.assert_allclose(regression_wald(x,y,1,0), beta[1]@np.linalg.inv(var)@beta[1],rtol=1e-12)


def check_resampling():
    count = CFG["permutations"]
    order = RESAMPLE.independent(25,15)
    signs = RESAMPLE.signs(25)
    assert len(order) == len(signs) == count
    assert len(np.unique(np.sort(order[:, :25],axis=1),axis=0)) == count
    assert len(np.unique(signs,axis=0)) == count and not np.all(signs==1.,axis=1).any()
    a,b = RESAMPLE.resample(25,25,True)
    np.testing.assert_array_equal(a,b)
    assert a.shape == (CFG["bootstrap"],25)


def check_adjustments(frame, raw, output, how):
    p = frame[raw].to_numpy()
    q = frame[output].to_numpy()
    valid = np.isfinite(p)
    assert np.array_equal(valid,np.isfinite(q))
    if valid.any():
        np.testing.assert_allclose(q[valid],multipletests(p[valid],method=how)[1],atol=1e-13,rtol=0)


def read(path):
    return pd.read_csv(RESULT / f"{path}.csv")


def check_outputs():
    a = read("All_V5_Statistical_Results")
    assert not a.duplicated(["method","family","feature","contrast","source_family"]).any()
    assert len(a) == a.method.nunique()*150
    primary = a.loc[a.family.eq("Primary six")]
    assert set(zip(primary.endpoint,primary.contrast)) == {(ep,c) for ep, contrasts in CFG["primary"].items() for c in contrasts}
    for raw,prefix in [("raw_p",""),("fd_p","fd_"),("permutation_p","permutation_")]:
        for how, suffix in [("holm","holm"),("fdr_bh","fdr")]:
            for _, g in a.groupby(["method","family"]):
                check_adjustments(g,raw,f"{prefix}{suffix}_p",how)
            check_adjustments(a,raw,f"joint_{prefix}{suffix}_p",how)
            check_adjustments(primary,raw,f"cross_method_{prefix}{suffix}_p",how)
            for _,g in primary.loc[primary.endpoint.eq("transition_wald")].groupby("method"):
                check_adjustments(g,raw,f"transition_three_{prefix}{suffix}_p",how)
    for method in a.method.unique():
        parent = RESULT if method=="V4_A3" else RESULT / "alternate" / method
        group = a.loc[a.method.eq(method)]
        t = pd.read_csv(parent / "transition/Transition_Matrices.csv")
        assert len(t) == 48 and (t.scans>0).all() and (t.raw_count>=0).all()
        for _,row in t.groupby(["condition","source"]):
            assert row.raw_count.sum() == row.source_count.iloc[0]
            np.testing.assert_allclose([row.pooled_probability.sum(),row.mean_probability.sum()],[1,1],atol=1e-12)
        edges = group.loc[group.family.eq("Transition localization")]
        assert len(edges) == 36 and all(edge[:2]!=edge[-2:] for edge in edges.endpoint)
        assert not edges.loc[~edges.permutation_gate_open,"permutation_holm_supported"].any()
        assert not edges.loc[~edges.fd_gate_open,"fd_holm_supported"].any()
        b = pd.read_csv(parent / "bootstrap/Fixed_Template_Bootstrap_Audit.csv")
        assert len(b)==114 and b.bootstrap_requested.eq(CFG["bootstrap"]).all()
        assert b.bootstrap_valid.eq(CFG["bootstrap"]).all()
        assert b.fd_bootstrap_valid.dropna().eq(CFG["bootstrap"]).all()
        assert (b.ci95_low<=b.ci95_high).all()
    studies = read("power/Sample_Size_Planning")
    assert studies.achieved_power.ge(studies.intended_power).all()
    np.testing.assert_array_equal(studies.recruit_scans,studies.recruit_hc+2*studies.recruit_pd_pairs)
    return a


def check_calibration():
    s = CFG["simulation"]
    d = read("simulation/Formal_Synthetic_Cohorts")
    method = "switch_penalty"
    assert not d.duplicated([method,"scenario","repeat"]).any()
    assert sorted(d[method].unique()) == sorted(s["candidates"])
    assert d.groupby([method,"scenario"]).size().eq(s["draws_per_setting"]).all()
    total = s["draws_per_setting"]*len(s["settings"])
    assert d.groupby(method).size().eq(total).all() and total>=1000
    assert d.groupby(["scenario","repeat"])[list(d.filter(regex=r"_truth$"))].nunique().le(1).all().all()
    selection = json.loads((RESULT / "simulation/Penalty_Selection.json").read_text(encoding="utf-8"))
    assert selection["uses_real_group_p"] is False and selection["fixed_templates"] is True
    table = read("simulation/Penalty_Selection_Table")
    expected = None if not table.eligible.any() else table.loc[table.eligible,"penalty"].iloc[0]
    assert expected == selection["selected_lambda"]
    q = read("simulation/Calibration_by_Scenario")
    low,high = proportion_confint(q.rejected,q.n,method="wilson")
    np.testing.assert_allclose(q.wilson_low,low,atol=1e-12)
    np.testing.assert_allclose(q.wilson_upper,high,atol=1e-12)
    return len(d)


def check_public_files():
    figures = tuple((HERE / "figures").glob("*.png"))
    reports = tuple((HERE / "reports").glob("*.md"))
    assert len(figures)==9 and len(reports)==6
    for path in figures:
        with Image.open(path) as img:
            assert img.width>=1000 and img.height>=600
            img.verify()
    for path in (*reports,HERE / "README.md"):
        for target in re.findall(r"\]\(([^ )]+)\)",path.read_text(encoding="utf-8")):
            if "://" not in target:
                assert (path.parent / target).exists() or target=="logs/Run_Verification.json", (path.name,target)
    for path in (HERE / "results").rglob("*.csv"):
        assert not {"subject", "scan", "patient_id", "research_id"} & set(pd.read_csv(path,nrows=0).columns), path.name
        assert not re.search(r"\b(?:sub-(?:STN|HC)|DBS-P\d+)",path.read_text(encoding="utf-8"),re.I), path.name
    return len(figures),len(reports)


def main():
    files = verify_v4(export=False)
    assert not git("diff", "--name-only", CFG["v4_commit"], "--", V4_NAME)
    check_geometry()
    check_resampling()
    a = check_outputs()
    draws = check_calibration()
    figures,reports = check_public_files()
    d = load_data(save_private=False)
    assert len(d)==65 and d.valid_frames.sum()==12574 and d.trans_ilr_1.notna().all()
    entries = sorted([*HERE.glob("*.md"), *HERE.glob("V4_*"), *HERE.glob("results/**/*.csv"),
                      *HERE.glob("results/**/*.json"), *HERE.glob("manifests/*"), *HERE.glob("configs/*"),
                      *HERE.glob("scripts/*.py"), *HERE.glob("figures/*.png"), *HERE.glob("reports/*.md")])
    dump(dict(v4_commit=CFG["v4_commit"], v4_original_run_hashes_verified=41, v4_git_files_verified=len(files),
              v4_workspace_unchanged=True, scans=len(d), participants=int(d.subject.nunique()), valid_frames=int(d.valid_frames.sum()),
              participant_permutation_draws=CFG["permutations"], bootstrap_draws=CFG["bootstrap"],
              all_new_test_rows=len(a), primary_rows=int(a.family.eq("Primary six").sum()), simulated_method_by_cohort_rows=draws,
              figure_count=figures,report_count=reports, tests={"transition_gaps_and_episodes":"passed","energy_and_hc3_reference":"passed",
                  "distinct_resampling_and_paired_bootstrap":"passed","independent_multiplicity_recalculation":"passed",
                  "calibration_replicates_and_wilson":"passed","report_links_and_figure_files":"passed","participant_columns_and_labels":"passed"},
              versions={"python":sys.version.split()[0], **{p:version(p) for p in ("numpy","pandas","scipy","statsmodels","numba","openpyxl","matplotlib","pillow")}},
              v5_sha256={p.relative_to(HERE).as_posix():sha256(p.read_bytes()).hexdigest() for p in entries if p.is_file()}),
         HERE / "logs/Run_Verification.json")
    print(f"Verified {len(files)} frozen V4 files, {len(a)} V5 test rows, {draws} synthetic rows, {figures} figures, {reports} reports.")


if __name__ == "__main__":
    main()
