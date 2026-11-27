"""Authenticate the Round 4 scientific archive and separate publication from identifiers."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT.parent
REPO = MODULE.parent
WORKSPACE = REPO.parent
PRIVATE = ROOT / "private"
RESULTS = ROOT / "results"
CONFIG = ROOT / "configs/protocol.json"
R4 = MODULE / "round4/transition"
R4_PRIVATE = MODULE / "private/round4/transition"
V6 = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics"
EDGES = ["from_state", "to_state"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


CFG = read_json(CONFIG)
BASE = CFG["source_commit"]
STATES = tuple(CFG["states"])
CELLS = [(source, target) for source in STATES for target in STATES]
NONDIAGONAL = [(source, target) for source, target in CELLS if source != target]


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
        check=True, capture_output=True, encoding="utf-8",
    ).stdout.strip()


def verify_base():
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("diff", "--exit-code", BASE, "--", ".", f":(exclude){ROOT.relative_to(REPO).as_posix()}/**")
    if git("branch", "--show-current") != CFG["branch"]:
        raise ValueError("Run Round 5 on the dedicated analysis branch")
    git("check-ignore", (PRIVATE / "results/private.csv").relative_to(REPO).as_posix())


def digest(path, *, text=False):
    b = Path(path).read_bytes()
    return sha256(b.replace(b"\r\n", b"\n") if text else b).hexdigest()


def collection_digest(lock):
    return sha256("".join(value + "\n" for value in sorted(lock.values())).encode("ascii")).hexdigest()


def confirm_receipts(*locks):
    output = {}
    for lock in locks:
        for name, expected in lock.items():
            file = Path(name).resolve()
            if not file.is_relative_to(WORKSPACE):
                raise ValueError("An input source is outside the research workspace")
            prior = output.get(str(file))
            if (prior is not None and prior != expected) or (prior is None and digest(file) != expected):
                raise ValueError("An input fingerprint is inconsistent")
            output[str(file)] = expected
    return dict(sorted(output.items()))


def normalize_json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported manifest type {type(value).__name__}")


def write_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".pending")
    temp.write_text(value, encoding="utf-8", newline="\n")
    temp.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=normalize_json) + "\n")


def write_csv(path, table):
    write_text(path, table.to_csv(index=False, na_rep="NA", float_format="%.17g", lineterminator="\n"))


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def assert_tables(expected, observed):
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), observed.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=1e-13, rtol=1e-12)


@dataclass(frozen=True)
class FrozenInputs:
    scans: pd.DataFrame
    matrices: pd.DataFrame
    pairs: pd.DataFrame
    scores: pd.DataFrame
    comparisons: pd.DataFrame
    associations: pd.DataFrame
    m4_summary: pd.DataFrame
    round4_protocol: dict
    v6_omnibus: pd.DataFrame
    v6_paths: pd.DataFrame
    hashes: dict
    audit: dict


def load_frozen_inputs():
    sys.path.insert(0, str(R4 / "scripts"))
    sys.path.insert(0, str(MODULE / "round3/scripts"))
    from coverage_inputs import load_inputs
    from transition_sources import load_transition_sources
    from transition_summaries import v6_parity

    report = R4 / "results/Transition_Run_Manifest.json"
    frozen = read_json(report)
    locked = R4_PRIVATE / "Transition_Input_Hashes.json"
    origin_lock = read_json(locked)["file_sha256"]
    if (len(origin_lock) != frozen["verified_input_files"] or
            collection_digest(origin_lock) != frozen["input_collection_sha256"]):
        raise ValueError("Round 4 source receipt no longer matches the published receipt")
    for root, paths in ((R4, frozen["public_outputs_sha256"]), (R4 / "scripts", frozen["scripts_sha256"])):
        for name, expected in paths.items():
            path = root / name
            if digest(path, text=path.suffix != ".png") != expected:
                raise ValueError(f"Frozen Round 4 artifact fingerprint differs: {path.name}")
    for filename, name in (("ANALYSIS_PLAN.md", "plan"), ("configs/protocol.json", "protocol"),
                           ("configs/test_registry.csv", "registry")):
        if digest(R4 / filename, text=True) != frozen[name + "_sha256"]:
            raise ValueError("Frozen Round 4 plan or protocol changed")

    source = load_transition_sources(WORKSPACE, MODULE)
    raw_dir = R4_PRIVATE / "results"
    matrices = read_csv(raw_dir / "Transition_Matrix_All_Scans.csv")
    scans = read_csv(raw_dir / "Transition_Counts_All_Scans.csv")
    assert_tables(source.matrices, matrices)
    assert_tables(source.scan_metrics, scans[source.scan_metrics.columns])
    for method, column in (("WEiDA", "p_weida"), ("ExitConditional", "p_exit")):
        separate = read_csv(raw_dir / f"Transition_Probability_{method}.csv")
        assert_tables(matrices[[*separate.columns]], separate)
        aggregate = read_csv(R4 / f"results/Transition_Probability_{method}.csv")
        actual = matrices.groupby(["condition", *EDGES], sort=False)[column].mean()
        lookup = aggregate.set_index(["condition", *EDGES])["mean"]
        compare = actual.loc[lookup.index]
        np.testing.assert_allclose(compare, lookup, atol=1e-13, rtol=0)
    clinical = load_inputs(WORKSPACE, MODULE)
    pair_keys = ["patient_id", "fold", "on_scan_id", "off_scan_id"]
    pairs = source.pairs
    scores = clinical.pairs[pair_keys + ["total_on", "total_off"] +
                            [name + "_improvement" for name in frozen["protocol"]["clinical_endpoints"]]].copy()
    assert_tables(pairs.rename(columns={"participant_id": "patient_id"})[pair_keys].sort_values("patient_id"),
                  scores[pair_keys].sort_values("patient_id"))
    score_copy = read_csv(raw_dir / "Frozen_Clinical_Pairs.csv")
    assert_tables(scores[score_copy.columns].sort_values("patient_id"), score_copy.sort_values("patient_id"))

    v6_matrix = V6 / "results/Transition_Matrices.csv"
    v6_omnibus = V6 / "results/ON_OFF_Omnibus_Results.csv"
    v6_paths = V6 / "results/Transition_Path_Results.csv"
    v6_report = V6 / "reports/V6_Transition_Dynamics.md"
    baseline_protocol = REPO / "V4_WEiDA_Statistical_Framework_Microstate/reports/WEIDA_STATISTICAL_PROTOCOL.md"
    prior_files = [
        report, locked, *[raw_dir / (filename + ".csv") for filename in (
            "Transition_Matrix_All_Scans", "Transition_Counts_All_Scans", "Transition_Probability_WEiDA",
            "Transition_Probability_ExitConditional", "Frozen_Clinical_Pairs",
        )], v6_matrix, v6_omnibus, v6_paths, v6_report, baseline_protocol,
        V6 / "configs/protocol.json", R4 / "ANALYSIS_PLAN.md", R4 / "reports/A3_Transition_Dynamics_Report.md",
        MODULE / "round4/audit/Clinical_Timing_Audit_Report.md",
    ]
    for name in ("Transition_Group_Comparisons", "Transition_Clinical_Associations", "M4_Transition_Summary",
                 "Transition_Probability_WEiDA", "Transition_Probability_ExitConditional"):
        prior_files.append(R4 / f"results/{name}.csv")
    receipt = confirm_receipts(origin_lock, source.input_hashes, clinical.input_hashes,
                               {str(path): digest(path) for path in prior_files})
    parity = v6_parity(matrices, read_csv(v6_matrix))
    audit = {**source.audit, "round4_private_matrix_recounted": True, "round4_public_receipt_verified": True,
             "round4_input_receipt_count": len(origin_lock), "v6_archive_transition_parity": parity}
    return FrozenInputs(
        scans, matrices, pairs, scores, read_csv(R4 / "results/Transition_Group_Comparisons.csv"),
        read_csv(R4 / "results/Transition_Clinical_Associations.csv"), read_csv(R4 / "results/M4_Transition_Summary.csv"),
        frozen["protocol"], read_csv(v6_omnibus), read_csv(v6_paths), receipt, audit,
    )
