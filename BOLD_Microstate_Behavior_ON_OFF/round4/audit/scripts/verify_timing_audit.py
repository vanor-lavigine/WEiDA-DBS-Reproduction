"""Verify source receipts, aggregate counts, links and publication boundaries."""

import json
from pathlib import Path
import re

import pandas as pd

from run_timing_audit import BASE, CFG, MODULE, PRIVATE, PROTOCOL, ROOT, check_inputs, digest, verify_baseline
from timing_outputs import summaries
from timing_rules import STATUSES


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv(folder, name):
    return pd.read_csv(folder / (name + ".csv"))


def verify_private_data(manifest):
    visits = csv(PRIVATE, "Patient_Condition_Evidence")
    pairs = csv(PRIVATE, "Clinical_Scan_Visit_Matching")
    groups = csv(PRIVATE, "Patient_Groups")
    orders = csv(PRIVATE, "Scan_Order_Audit")
    hc = csv(PRIVATE, "HC_Clinical_Score_Audit")
    n_pairs, n_hc = CFG["cohort"]["STN_pairs"], CFG["cohort"]["HC_scans"]
    require(len(visits) == n_pairs * 2 and visits.scan_id.is_unique, "Wrong private STN scan inventory")
    require(len(groups) == len(orders) == len(pairs) == n_pairs and pairs.patient_id.is_unique,
            "Wrong private patient-pair inventory")
    require(len(hc) == n_hc and hc.scan_id.is_unique and set(hc.scan_id).isdisjoint(visits.scan_id),
            "Wrong private HC scan inventory")
    cohort = visits.pivot(index="patient_id", columns="condition", values="visit_match_status")
    require(cohort.notna().all().all(), "Missing an ON/OFF status")
    expected_a = set(cohort.index[cohort.eq(STATUSES[0]).all(axis=1)])
    expected_c = set(cohort.index[cohort.eq(STATUSES[5]).any(axis=1)])
    for label, ids in (("A", expected_a), ("B", set(cohort.index) - expected_a - expected_c), ("C", expected_c)):
        require(set(groups.loc[groups.group.eq(label), "patient_id"]) == ids, "Patient group evidence mismatch")
        require(len(ids) == manifest["patient_group_counts"][label], "Manifest patient count mismatch")
    hour_rows = visits.loc[visits.date_difference_hours.notna()]
    require(hour_rows[["score_datetime", "scan_datetime"]].notna().all().all(),
            "An exact difference used a missing clinical or acquisition clock")
    require(visits.loc[visits.score_date_unavailable_reason.notna(), "score_date"].isna().all(),
            "A scan date was reused as a clinical assessment date")
    missing = ~hc.motor_assessment_documented
    require(hc.loc[missing, "motor_score"].isna().all() and not hc.loc[missing, "measured_zero_verified"].any(),
            "An unobserved HC score was used as a measured zero")
    require(set(orders.loc[orders.scan_order.eq("ON first"), "patient_id"]) ==
            set(orders.loc[orders.OFF_minus_ON_hours.gt(0), "patient_id"]), "Scan interval signs disagree")
    for name, aggregate in summaries(visits, groups, orders, hc).items():
        published = pd.read_csv(ROOT / name)
        pd.testing.assert_frame_equal(published, aggregate, check_dtype=False, check_exact=False)
    return visits, hc


def verify_public_data(manifest, visits, hc):
    private_ids = set(visits.scan_id) | set(visits.patient_id) | set(hc.scan_id)
    private_ids |= {str(code) for code in visits.study_code.dropna()
                    if len(str(code)) >= 5 and not str(code).isnumeric()}
    public_files = [ROOT / name for name in manifest["public_output_sha256"]]
    public_files += [ROOT / "Clinical_Timing_Audit_Manifest.json", ROOT / "ANALYSIS_PLAN.md", PROTOCOL]
    for path in public_files:
        body = path.read_text(encoding="utf-8")
        require(not re.search(r"(?:[A-Za-z]:[/\\]|\\\\[^\s]+\\)|\b\d{4}-\d\d-\d\d(?:T\d\d:\d\d)?\b", body),
                f"Person-level date or absolute path in {path.name}")
        require(not any(re.search(r"(?<![\w-])" + re.escape(key) + r"(?![\w-])", body, re.I)
                        for key in private_ids), f"An individual key appears in {path.name}")
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", body):
                if "://" not in target:
                    require((path.parent / target).is_file(), f"Broken relative link in {path.name}")


def main():
    verify_baseline()
    manifest = read_json(ROOT / "Clinical_Timing_Audit_Manifest.json")
    require(manifest["source_commit"] == BASE, "Wrong frozen source commit")
    require(digest(PROTOCOL, normalize=True) == manifest["protocol_sha256"], "Protocol digest mismatch")
    require(digest(ROOT / "ANALYSIS_PLAN.md", normalize=True) == manifest["plan_sha256"], "Plan digest mismatch")
    for field, parent in (
        ("script_sha256", ROOT / "scripts"), ("public_output_sha256", ROOT),
        ("previous_evidence_sha256", MODULE),
    ):
        for name, expected in manifest[field].items():
            require(digest(parent / name, normalize=True) == expected, f"Digest mismatch in {field}: {Path(name).name}")
    inputs = read_json(PRIVATE / "Clinical_Timing_Input_Hashes.json")["file_sha256"]
    require(len(check_inputs(inputs)) == manifest["raw_input_files"], "Private input receipt count mismatch")
    visits, hc = verify_private_data(manifest)
    verify_public_data(manifest, visits, hc)
    print(f"Verified {len(inputs)} read-only source files, visit logic, private/public linkage and frozen-tree preservation.")


if __name__ == "__main__":
    main()
