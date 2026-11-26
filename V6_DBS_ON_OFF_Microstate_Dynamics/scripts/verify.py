"""Verify the released outputs, attribution to frozen inputs and privacy boundary."""

from importlib.metadata import version
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

from prepare import verify_releases
from v6_common import CFG, HERE, PRIVATE, RESULT, digest, git, read_json, write_json


def csv(name):
    return pd.read_csv(RESULT / f"{name}.csv")


def verify_adjustment(block,p,stem):
    data = block.loc[block[p].notna()].sort_values(p)
    if not len(data):
        return
    weights = np.arange(len(data),0,-1)
    ordered = data[p].to_numpy()
    manual_holm = np.minimum(np.maximum.accumulate(ordered*weights),1.)
    manual_bh = np.minimum(np.minimum.accumulate((ordered*len(data)/np.arange(1,len(data)+1))[::-1])[::-1],1.)
    np.testing.assert_allclose(data[f"{stem}holm_p"],manual_holm,rtol=0,atol=1e-13)
    np.testing.assert_allclose(data[f"{stem}fdr_p"],manual_bh,rtol=0,atol=1e-13)


def audit_statistics():
    summary = csv("ON_OFF_Feature_Summary")
    omnibus = csv("ON_OFF_Omnibus_Results")
    health = csv("Health_Reference_Results")
    assert len(summary)==41 and len(omnibus)==2 and len(health)==2
    assert summary.endpoint.is_unique and omnibus.endpoint.is_unique and health.endpoint.is_unique
    assert set(summary.loc[summary.family.eq("Primary"),"endpoint"])==set(CFG["primary"])
    assert set(summary.loc[summary.family.eq("Secondary"),"endpoint"])==set(CFG["secondary"])
    assert summary.n_pairs.eq(25).all() and omnibus.n_pairs.eq(25).all()
    for table in (summary,omnibus,health):
        for _,block in table.groupby("family"):
            for key,prefix in [("raw_p",""),("permutation_p","permutation_"),("fd_p","fd_"),
                               ("dvars_p","dvars_"),("order_subset_p","order_subset_")]:
                verify_adjustment(block,key,prefix)
    p = summary.set_index("endpoint")
    np.testing.assert_allclose(p.mean_difference,p.mean_on-p.mean_off,atol=1e-12)
    np.testing.assert_allclose(p.dz,p.mean_difference/p.sd_difference,atol=1e-12)
    assert (summary.n_positive+summary.n_negative+summary.n_tied).eq(summary.n_pairs).all()
    gate = omnibus.set_index("endpoint").loc["transition_omnibus"]
    edges = csv("Transition_Path_Results")
    assert len(edges)==12
    for name in ("", "permutation_", "fd_", "dvars_", "order_subset_"):
        assert edges[f"{name}gate_open"].eq(gate[f"{name}holm_p"]<CFG["alpha"]).all()
    transitions = csv("Transition_Matrices")
    for _, row in transitions.groupby("source"):
        assert len(row)==4 and row.self_transition.sum()==1
        for cond in ("on","off"):
            assert row[f"{cond}_count"].sum()==row[f"{cond}_source"].iloc[0]
            np.testing.assert_allclose(row[f"{cond}_mean_probability"].sum(),1.,atol=1e-12)
        np.testing.assert_allclose(row.mean_difference.sum(),0.,atol=1e-12)
    bootstrap = csv("Paired_Bootstrap_Audit")
    assert len(bootstrap)==44
    for pfx in ("", "fd_", "dvars_"):
        assert bootstrap[f"{pfx}bootstrap_valid"].eq(CFG["bootstrap"]).all()
        assert bootstrap[f"{pfx}bootstrap_failed"].eq(0).all()
        assert bootstrap[f"{pfx}ci95_low"].le(bootstrap[f"{pfx}ci95_high"]).all()
    power = csv("Paired_Sample_Size_Planning")
    assert (power.planned_scans==2*power.planned_to_recruit).all() and power.achieved_power.ge(power.power_target).all()
    return summary,omnibus,health


def audit_private():
    pairs = pd.read_csv(PRIVATE / "manifests/ON_OFF_Pair_Manifest.csv")
    feats = pd.read_csv(PRIVATE / "results/ON_OFF_Subject_Features.csv")
    delta = pd.read_csv(PRIVATE / "results/ON_OFF_Paired_Differences.csv")
    assert pairs.participant_id.is_unique and len(pairs)==len(delta)==25 and len(feats)==50
    assert sets_match(feats, "ON", pairs.on_scan_id) and sets_match(feats, "OFF", pairs.off_scan_id)
    scores = delta.Distance_OFF_to_HC - delta.Distance_ON_to_HC
    np.testing.assert_allclose(scores,delta.Restoration_Score,atol=1e-12)
    table = csv("Restoration_Results").iloc[0]
    np.testing.assert_allclose([scores.mean(),scores.median()],[table.effect_size,table.median_improvement],atol=1e-12)
    assert (scores.gt(0).sum()==table.n_closer) and (scores.lt(0).sum()==table.n_farther)
    raw_boot = pd.read_csv(PRIVATE / "results/Paired_Bootstrap_Draws.csv")
    assert raw_boot.endpoint.nunique()==44 and raw_boot.groupby("endpoint").replicate.nunique().eq(CFG["bootstrap"]).all()
    assert not raw_boot.duplicated(["endpoint","replicate"]).any()
    clinical = read_json(RESULT / "UPDRS_Input_Audit.json")
    association = csv("UPDRS_Association_Results")
    assert len(association)==4
    if clinical["status"]=="BLOCKED":
        assert association.status.eq("BLOCKED").all() and association.rho.isna().all() and association.raw_p.isna().all()
    return len(pairs),clinical["status"]


def sets_match(feature,condition,scans):
    return set(feature.loc[feature.condition.eq(condition),"scan"])==set(scans)


def audit_artifacts():
    pattern = re.compile(r"\b(?:sub-(?:STN|HC)[a-z0-9]+|DBS-P\d+|github_pat_|ghp_\w+)",re.I)
    required_ext = {".png",".svg",".pdf"}
    for directory in (HERE / "figures",PRIVATE / "figures"):
        attachments = list(directory.glob("Figure_*.*"))
        assert len(attachments)==24
        for stem in {p.stem for p in attachments}:
            assert {p.suffix for p in attachments if p.stem==stem}==required_ext
        for p in attachments:
            if p.suffix==".png":
                with Image.open(p) as image:
                    np.testing.assert_allclose(image.info["dpi"],[600,600],atol=.01)
                    assert image.width>2000 and image.height>1500
                    image.verify()
            elif p.suffix==".svg":
                assert ET.parse(p).getroot().tag.endswith("svg")
                if directory==HERE / "figures":
                    assert not pattern.search(p.read_text(encoding="utf-8")),p.name
            else:
                data = p.read_bytes()
                assert data.startswith(b"%PDF-") and b"/Type /Page" in data
    for path in RESULT.glob("*.csv"):
        cols = pd.read_csv(path,nrows=0).columns
        assert not {"scan","subject","participant_id","on_scan_id","off_scan_id"} & set(cols),path.name
    docs = tuple((HERE / "reports").glob("*.md"))
    assert len(docs)==7
    for p in (*docs,HERE / "README.md",*RESULT.glob("*.csv"),*RESULT.glob("*.json")):
        data = p.read_text(encoding="utf-8")
        assert not pattern.search(data),p.name
        if p.suffix==".md":
            for link in re.findall(r"\]\(([^ )]+)\)",data):
                assert "://" in link or (p.parent/link).exists() or link=="logs/Run_Verification.json",(p.name,link)
    return len(docs)


def pytest_count():
    path = PRIVATE / "verification/pytest.xml"
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = root.findall("testsuite") if root.tag=="testsuites" else [root]
    assert all(s.attrib.get("failures")=="0" and s.attrib.get("errors")=="0" for s in suites)
    return sum(int(s.attrib["tests"]) for s in suites)


def main():
    history = verify_releases(export=False)
    inputs = read_json(HERE / "manifests/Input_Audit.json")
    assert inputs["protocol_sha256"]==digest(HERE / "configs/protocol.json") and inputs["plan_sha256"]==digest(HERE / "ANALYSIS_PLAN.md")
    order = read_json(RESULT / "Scan_Order_Audit.json")
    assert (order["total_pairs"],order["verified_pairs"],order["on_first"])==(25,24,24)
    assert order["stimulus_order_separable_in_verified_pairs"] is False
    paired,omnibus,health = audit_statistics()
    n,status = audit_private()
    reports = audit_artifacts()
    n_tests = pytest_count()
    candidates = sorted([*HERE.glob("*.md"),*HERE.glob("*.json"),*HERE.glob(".git*"),*HERE.glob("configs/*"),
                         *HERE.glob("manifests/*"),*HERE.glob("scripts/*.py"),*HERE.glob("tests/*.py"),
                         *HERE.glob("reports/*.md"),*HERE.glob("results/*"),*HERE.glob("figures/*")])
    write_json(dict(history_files_verified=len(history), original_v4_runtime_hashes_verified=41,
                    history_v4_git=CFG["v4_commit"],history_v5_git=CFG["v5_commit"],paired_patients=n,
                    all_paired_scalar_rows=len(paired),paired_omnibus_rows=len(omnibus),health_reference_rows=len(health),
                    complete_fixed_template_bootstrap_draws=CFG["bootstrap"],distinct_sign_flips=CFG["permutations"],
                    reports=reports,figures_per_release=8,formats_per_figure=3,clinical_status=status,
                    pytest_passed=n_tests, input_parity_verified=True, p_adjustments_verified_independently=True,
                    aggregate_only_public_content_checked=True, working_branch=git("branch","--show-current").decode().strip(),
                    dependency_versions={"python":sys.version.split()[0],**{k:version(k) for k in (
                        "numpy","pandas","scipy","statsmodels","scikit-learn","numba","openpyxl","matplotlib","pillow","pytest")}},
                    v6_sha256={p.relative_to(HERE).as_posix():digest(p) for p in candidates if p.is_file()}),
               HERE / "logs/Run_Verification.json")
    print(f"Verified {len(history)} history files, {len(paired)} scalar tests, {len(omnibus)} omnibus tests, {reports} reports and both eight-figure exports.")


if __name__ == "__main__":
    main()
