"""Verify and lock source files before the four V9 tests are evaluated."""

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parent
PRIVATE = MODULE / "private"
V7 = REPO / "BOLD_Microstate_Mechanistic_Decomposition"
V8 = REPO / "BOLD_Microstate_STN_Original_Scores"
V6 = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics"
V4 = REPO / "V4_Microstate_Statistical_Significance"
V4_FRAMEWORK = REPO / "V4_WEiDA_Statistical_Framework_Microstate"
V5 = REPO / "V5_Microstate_Nonsignificant_Endpoint_Optimization"
PROTOCOL = MODULE / "configs/protocol.json"
PLAN = MODULE / "ANALYSIS_PLAN.md"
LOCK_PATH = PRIVATE / "manifests/Input_Hashes.json"
V7_TABLES = (
    V7 / "results/Mechanistic_Primary_Statistics.csv",
    V7 / "results/Mechanistic_Motion_Sensitivity.csv",
)
V8_TABLES = tuple(V8 / "results/date_unrestricted" / name for name in (
    "Primary_Associations.csv",
    "Exploratory_Domain_Associations.csv",
    "Motion_and_Visit_Sensitivity.csv",
))

sys.path[:0] = [str(V7 / "scripts"), str(V8 / "scripts")]
from frozen_inputs import CFG as V7_CFG
from frozen_inputs import digest, load_module, read_json, require, verify_template_tables, write_csv, write_json
from pipeline_io import CFG as V8_CFG
from pipeline_io import PAIR_KEYS, load_imaging

CFG = read_json(PROTOCOL)
TOL = CFG["tolerance"]
COHORT = CFG["cohort"]


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()


def verify_code():
    v4_verifier = V4 / "scripts/verify_freeze.py"
    frozen_manifest = V4 / "workflow/Freeze_Manifest.json"
    keys = ("seed", "permutations", "bootstrap", "alpha")
    v6_cfg = read_json(V6 / "configs/protocol.json")
    for source in [V7_CFG, v6_cfg]:
        require(source["baseline"] == CFG["baseline"], "Frozen imaging method differs")
    for key in keys:
        require(all(old[key] == CFG[key] for old in (v6_cfg, V7_CFG, V8_CFG)),
                f"Different resampling settings in a prior module: {key}")
    require(V7_CFG["share_min_abs_gev_pp"] == CFG["component_share_min_abs_gev_pp"],
            "The V7 contribution-share threshold differs")
    require(V7_CFG["wald_shrinkage"] == v6_cfg["wald_ridge"], "Frozen V6/V7 shrinkage differs")
    git("diff", "--exit-code", CFG["v7_imaging_commit"], "--", *(
        p.name for p in (V4_FRAMEWORK, V4, V5, V6, V7)
    ))
    git("diff", "--exit-code", CFG["v8_archive_commit"], "--", V8.name)
    git("diff", "--exit-code", "HEAD", "--", str(PROTOCOL.relative_to(REPO)), str(PLAN.relative_to(REPO)))
    verifier = load_module("_v9_v4_verifier", v4_verifier)
    frozen = read_json(frozen_manifest)
    verifier.check_files(V4, frozen["public_files"])
    return verifier, frozen


def check_path(path, expected, role):
    require(path.is_file(), f"Required frozen input unavailable: {role}")
    require(digest(path) == expected, f"Frozen input checksum differs: {role}")


@dataclass
class Inputs:
    cohort: pd.DataFrame
    roster: pd.DataFrame
    decomposition: pd.DataFrame
    clinical: pd.DataFrame
    frames: dict
    mapping: dict
    acquired_frames: int
    manifest: dict


def read_v8_lock(workspace, v8_source):
    original_lock_path = v8_source / "private/manifests/Prespecified_Inference_Lock.json"
    current_lock_path = v8_source / "private/date_unrestricted/manifests/Inference_Lock.json"
    original_lock, current_lock = (read_json(path) for path in (original_lock_path, current_lock_path))
    source_protocol = v8_source / "configs/protocol.json"
    source_current = v8_source / "configs/date_unrestricted.json"
    check_path(source_protocol, original_lock["protocol_sha256"], "V8 protocol")
    check_path(source_current, current_lock["config_sha256"], "V8 date-unrestricted protocol")
    require(
        current_lock["parent_protocol_sha256"] == original_lock["protocol_sha256"]
        and current_lock["inputs"] == original_lock["input_files"]
        and current_lock["seed"] == original_lock["seed"] == CFG["seed"]
        and current_lock["visit_status_counts"] == CFG["clinical_visit_counts"],
        "V8 date-unrestricted analysis and original source lock disagree",
    )
    require(read_json(source_protocol) == V8_CFG, "V8 source protocol differs from archived protocol")
    paths = {}
    for role, entry in current_lock["inputs"].items():
        path = Path(entry["path"]).resolve()
        require(path.is_relative_to(workspace), f"Indexed source lies outside the authorized workspace: {role}")
        check_path(path, entry["sha256"], role)
        paths[role] = path
    score_pairs = v8_source / "private/date_unrestricted/results/STN_Analysis_Pairs_25.csv"
    check_path(score_pairs, current_lock["patient_input_sha256"], "V8 date-unrestricted score pairs")
    references = (original_lock_path, current_lock_path, source_protocol, source_current, score_pairs)
    return paths, references


def read_v7_lock(workspace, v7_index, paths, verifier, frozen):
    ledger = read_json(v7_index)
    require(Path(ledger["prior_input_index"]).resolve() == paths["prior_index"], "V7 and V8 prior indices differ")
    known = {Path(path).resolve(): sha for path, sha in ledger["file_sha256"].items()}
    collection, = [row for row in frozen["local_archive_collections"]
                   if row["role"] == "A3 archived leave-out backfits"]
    folder = (workspace / collection["path"]).resolve()
    all_frames = {path.stem: path for path in known if path.parent == folder and path.match(collection["glob"])}
    require(len(all_frames) == collection["count"] == COHORT["frame_archives"],
            "Indexed frame archives differ in number from V4's freeze receipt")
    frame_paths = set(all_frames.values())
    missing = {path for path in frame_paths if not path.is_file()}
    for path, expected in known.items():
        require(path.is_relative_to(workspace), "V7 indexed source lies outside the authorized workspace")
        if path not in missing:
            check_path(path, expected, "A3 frame archive" if path in frame_paths else path.name)
    if not missing:
        count, checksum = verifier.collection_digest(workspace, collection["path"], collection["glob"])
        require(count == collection["count"] and checksum == collection["sha256"],
                "V4 and V7 archive collection receipts disagree")
    for role in ("v6_manifest", "v6_cohort", "v6_scans", "v6_differences"):
        require(paths[role] in known, f"V7 and V8 disagree on the frozen imaging archive: {role}")

    def asset(name):
        path, = [path for path in known if path.name == name]
        return path

    with np.load(asset("V4_Maps.npz"), allow_pickle=False) as maps:
        mapping, _ = verify_template_tables(maps["G1"], asset("Fold_Templates.csv"), asset("Reference_Templates.csv"))
    frame_manifest = read_csv(asset("input_manifest.csv"), usecols=["scan", "condition", "frames"])
    return all_frames, missing, mapping, frame_manifest, known


def joined_tables(paths, score_pairs, frame_manifest, all_frames):
    original = load_imaging(paths)
    roster = original.roster.reset_index()
    require(len(roster) == COHORT["image_pairs"], "Frozen participant count differs")
    cohort = read_csv(paths["v6_cohort"]).sort_values("scan", ignore_index=True)
    raw = frame_manifest.set_index("scan", verify_integrity=True)
    require(set(all_frames) == set(raw.index) == set(cohort.scan), "V4 and V6 scan keys disagree")
    raw = raw.loc[cohort.scan]
    np.testing.assert_array_equal(raw.condition.map({1: "HC", 2: "OFF", 3: "ON"}), cohort.condition)
    acquired = V7_CFG["cohort"]["acquired_frames"]
    require(pd.api.types.is_integer_dtype(raw.frames) and raw.frames.between(1, acquired).all(),
            "Source frame count exceeds the archived window")
    cohort["source_frames"] = raw.frames.to_numpy()

    decomposition = read_csv(paths["v7_paired"]).sort_values("participant_id", ignore_index=True)
    np.testing.assert_array_equal(decomposition.participant_id, roster.participant_id)
    clinical = read_csv(score_pairs, usecols=PAIR_KEYS + [
        "visit_status", "off_motor_total", "on_motor_total", "motor_total_improvement", "fd", "mean_std_dvars",
    ]).sort_values("participant_id", ignore_index=True)
    require(len(clinical) == COHORT["clinical_pairs"], "V8 date-unrestricted pair count differs")
    require(clinical.visit_status.value_counts().to_dict() == CFG["clinical_visit_counts"], "Visit group counts differ")
    np.testing.assert_array_equal(roster[PAIR_KEYS], clinical[PAIR_KEYS])
    totals = clinical[["off_motor_total", "on_motor_total", "motor_total_improvement"]].to_numpy(float)
    require(np.isfinite(totals).all(), "V8 score pairs contain missing totals")
    np.testing.assert_array_equal(totals[:, 0] - totals[:, 1], totals[:, 2])
    np.testing.assert_allclose(clinical[["fd", "mean_std_dvars"]], decomposition[["fd", "mean_std_dvars"]],
                               atol=TOL, rtol=0, equal_nan=True)
    return cohort, roster, decomposition, clinical, acquired


def lock_inputs(workspace, v7_index, v8_source):
    verifier, frozen = verify_code()
    paths, v8_references = read_v8_lock(workspace, v8_source)
    all_frames, missing, mapping, frame_manifest, v7_files = read_v7_lock(
        workspace, v7_index, paths, verifier, frozen,
    )
    tables = joined_tables(paths, v8_references[-1], frame_manifest, all_frames)
    indexed = {str(path): digest(path) for path in (
        PROTOCOL, PLAN, v7_index, *v8_references, *paths.values(), *V7_TABLES, *V8_TABLES,
        V4 / "workflow/Freeze_Manifest.json",
    )}
    indexed.update({str(path): sha for path, sha in v7_files.items() if path not in missing})
    snapshot = {
        "report_base": CFG["report_base"],
        "v8_source_commit": CFG["v8_source_commit"],
        "v8_archive_commit": CFG["v8_archive_commit"],
        "v7_imaging_commit": CFG["v7_imaging_commit"],
        "hypotheses": CFG["hypotheses"],
        "seed": CFG["seed"],
        "file_sha256": indexed,
        "missing_indexed_frame_archives": sorted(str(path) for path in missing),
        "v4_archive_collection_checksum_verified": not missing,
    }
    if LOCK_PATH.exists():
        require(read_json(LOCK_PATH) == snapshot, "Current input hashes differ from the saved V9 analysis lock")
    else:
        write_json(snapshot, LOCK_PATH)
    frames = {scan: path for scan, path in all_frames.items() if path not in missing}
    return Inputs(*tables[:4], frames, mapping, tables[4], snapshot)


def check_lock_unchanged(snapshot):
    require(read_json(LOCK_PATH) == snapshot, "V9 input lock changed during analysis")
    for path, checksum in snapshot["file_sha256"].items():
        check_path(Path(path), checksum, "V9 locked input")
