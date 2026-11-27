"""Read original evidence, classify visits, and write de-identified audit summaries."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from hc_sources_audit import load_hc_scores
from scan_sources_audit import load_hc_scan_times, load_scan_times
from score_sources_audit import load_score_audit
from timing_outputs import patient_worksheets, summaries
from timing_report import evidence_inventory, write_report
from timing_rules import link_visits, scan_order


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT.parents[1]
REPO = MODULE.parent
PRIVATE = MODULE / "private/round4/audit"
PROTOCOL = ROOT / "configs/protocol.json"
CFG = json.loads(PROTOCOL.read_text(encoding="utf-8"))
BASE = CFG["source_commit"]
PREVIOUS = (
    "reports/Data_and_Score_Audit.md",
    "ANALYSIS_PLAN.md",
    "results/Run_Manifest.json",
    "results/Clinical_Paired_Results.csv",
    "round2/results/Clinical_Response_Profile.csv",
    "round3/ANALYSIS_PLAN.md",
    "round3/results/Round3_Run_Manifest.json",
)


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
        check=True, capture_output=True, encoding="utf-8",
    ).stdout.strip()


def verify_baseline():
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("diff", "--exit-code", BASE, "--", ".", f":(exclude){ROOT.relative_to(REPO).as_posix()}/**")
    git("check-ignore", PRIVATE.relative_to(REPO).as_posix() + "/patient.csv")


def digest(path, *, normalize=False):
    raw = Path(path).read_bytes()
    return sha256(raw.replace(b"\r\n", b"\n") if normalize else raw).hexdigest()


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(text, encoding="utf-8", newline="\n")
    pending.replace(path)


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=as_json) + "\n")


def as_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unexpected manifest type: {type(value).__name__}")


def write_csv(path, table):
    atomic_write(path, table.to_csv(index=False, lineterminator="\n", na_rep="NA", float_format="%.12g"))


def check_inputs(*receipts):
    inputs = {}
    for receipt in receipts:
        for path, expected in receipt.items():
            path = Path(path).resolve()
            if path in inputs and inputs[path] != expected:
                raise ValueError("Input digests disagree across audit evidence loaders")
            if digest(path) != expected:
                raise ValueError("A source changed during the audit")
            inputs[path] = expected
    return {str(path): expected for path, expected in sorted(inputs.items())}


def same_visit_family(groups):
    """Use the frozen round-three definitions on the confirmed A subset only."""
    patients = groups.loc[groups.group.eq("A"), "patient_id"]
    if len(patients) < CFG["same_visit_association"]["minimum_valid_pairs"]:
        return None, {}

    sys.path.insert(0, str(MODULE / "round3/scripts"))
    from coverage_inputs import load_inputs
    from coverage_stats import adjust, spearman
    from round3_io import CFG as COVERAGE_CONFIG, ENDPOINTS, STATES

    inputs = load_inputs(REPO.parent, MODULE, coverage_tolerance=COVERAGE_CONFIG["imaging"]["closure_tolerance"],
                         expected_patients=CFG["cohort"]["STN_pairs"])
    selected = inputs.pairs.loc[inputs.pairs.patient_id.isin(patients)]
    if len(selected) != len(patients):
        raise ValueError("Confirmed visits did not join the frozen Coverage and score pairs")
    rows = []
    for index, state in enumerate(STATES, start=1):
        for endpoint in ENDPOINTS:
            x = selected[f"delta_coverage_{index}"].to_numpy(float)
            y = selected[f"{endpoint}_improvement_rate"].to_numpy(float)
            use = np.isfinite(x) & np.isfinite(y)
            rho, p, reason = spearman(x[use], y[use])
            rows.append({"state": state, "clinical_endpoint": endpoint, "n": int(use.sum()),
                         "rho": rho, "raw_p": p, "reason": reason})
    return adjust(pd.DataFrame(rows)), inputs.input_hashes


def main():
    verify_baseline()
    workspace = REPO.parent
    scans, scan_facts, scan_hashes = load_scan_times(workspace, MODULE, PRIVATE)
    pairs = scans.pivot(index="patient_id", columns="condition", values="scan_id")
    pairs = pairs.rename(columns={"ON": "on_scan_id", "OFF": "off_scan_id"}).reset_index()
    scores, score_facts, score_hashes = load_score_audit(workspace, MODULE, pairs)
    hc, hc_facts, hc_hashes = load_hc_scores(workspace, MODULE, PRIVATE)
    hc_times, hc_time_facts, hc_time_hashes = load_hc_scan_times(workspace, MODULE, PRIVATE)
    hc = hc.merge(hc_times, on="scan_id", how="outer", validate="one_to_one", indicator=True,
                  suffixes=("_clinical", "_imaging"))
    if len(hc) != CFG["cohort"]["HC_scans"] or not hc.pop("_merge").eq("both").all():
        raise ValueError("HC clinical lookup and frozen imaging keys did not agree")
    hc["scan_identity_confirmed"] = hc.scan_identity_confirmed_clinical & hc.scan_identity_confirmed_imaging
    visits, groups = link_visits(scores, scans, CFG["cohort"]["STN_pairs"])
    orders = scan_order(scans)
    tables = summaries(visits, groups, orders, hc)
    facts = {"assessment_sources": score_facts, "scan_sources": scan_facts,
             "hc_assessment_sources": hc_facts, "hc_scan_sources": hc_time_facts}
    tables["Timing_Evidence_Inventory.csv"] = evidence_inventory(facts)
    association, association_hashes = same_visit_family(groups)
    if association is not None:
        tables["Confirmed_Visit_Associations.csv"] = association
    input_hashes = check_inputs(scan_hashes, score_hashes, hc_hashes, hc_time_hashes, association_hashes)

    comparison, medication = patient_worksheets(visits, groups)
    private_tables = {
        "Clinical_Scan_Visit_Matching.csv": comparison,
        "Patient_Condition_Evidence.csv": visits,
        "HC_Clinical_Score_Audit.csv": hc,
        "Scan_Order_Audit.csv": orders,
        "Medication_Stimulation_Audit.csv": medication,
        "Patient_Groups.csv": groups,
    }
    for name, table in private_tables.items():
        write_csv(PRIVATE / name, table)
    write_json(PRIVATE / "Clinical_Timing_Input_Hashes.json", {"file_sha256": input_hashes})
    for name, table in tables.items():
        write_csv(ROOT / name, table)
    report = "Clinical_Timing_Audit_Report.md"
    atomic_write(ROOT / report, write_report(tables, facts, visits, groups, orders, association))

    verify_baseline()
    manifest = {
        "source_commit": BASE,
        "source_tree": git("rev-parse", BASE + "^{tree}"),
        "branch": git("branch", "--show-current"),
        "protocol": CFG,
        "protocol_sha256": digest(PROTOCOL, normalize=True),
        "plan_sha256": digest(ROOT / "ANALYSIS_PLAN.md", normalize=True),
        "command": f"python -B -X utf8 {ROOT.relative_to(REPO).as_posix()}/scripts/run_timing_audit.py",
        "verify_command": f"python -B -X utf8 {ROOT.relative_to(REPO).as_posix()}/scripts/verify_timing_audit.py",
        "previous_evidence_sha256": {name: digest(MODULE / name, normalize=True) for name in PREVIOUS},
        "raw_input_files": len(input_hashes),
        "source_facts": facts,
        "patient_group_counts": {key: int(groups.group.eq(key).sum()) for key in ("A", "B", "C")},
        "same_visit_association": "insufficient_confirmed_pairs" if association is None else "computed",
        "private_receipt": "private/round4/audit/Clinical_Timing_Input_Hashes.json",
        "script_sha256": {path.name: digest(path, normalize=True) for path in sorted((ROOT / "scripts").glob("*.py"))},
        "public_output_sha256": {name: digest(ROOT / name, normalize=True) for name in sorted([*tables, report])},
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
    }
    write_json(ROOT / "Clinical_Timing_Audit_Manifest.json", manifest)
    print(f"Audited {len(groups)} STN pairs and {len(hc)} HC scans. Group counts {manifest['patient_group_counts']}.")


if __name__ == "__main__":
    main()
