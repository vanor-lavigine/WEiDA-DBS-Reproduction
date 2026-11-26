"""Resolve frozen inputs from the original indices and audit the FD clock."""

from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
REPO = MODULE.parent
PRIVATE = MODULE / "private"
RESULTS = MODULE / "results"
PROTOCOL = MODULE / "configs/protocol.json"
V9 = REPO / "BOLD_Microstate_Expression_Followup"
V8 = REPO / "BOLD_Microstate_STN_Original_Scores"
sys.path.insert(0, str(V9 / "scripts"))
import v9_inputs as prior

CFG = prior.read_json(PROTOCOL)
TOL = CFG["identity_tolerance"]
PARAM = CFG["imaging"]
V9_CSVS = (
    "M1_Component_Statistics.csv",
    "M1_Component_Motion_Sensitivity.csv",
    "M1_Component_Shares.csv",
    "V7_V9_Joint_Correction.csv",
    "Continuous_Expression_Group_Summary.csv",
    "Continuous_Expression_Motor_Associations.csv",
    "V8_V9_Clinical_Joint_Correction.csv",
)


@dataclass
class Frozen:
    cohort: pd.DataFrame
    roster: pd.DataFrame
    decomposition: pd.DataFrame
    frames: dict
    mapping: dict
    banks: np.ndarray
    source_scans: dict
    fd: dict
    old_qc: pd.DataFrame
    decoder: ModuleType
    fingerprints: dict


def asset(index, name):
    matched = [path for path in index if path.name == name]
    prior.require(len(matched) == 1, f"Expected one locked source for {name}")
    return matched[0]


def path_free_hash(hashes):
    lines = "".join(f"{value}\n" for value in sorted(hashes))
    return {"files": len(hashes), "sha256": sha256(lines.encode("ascii")).hexdigest()}


def checked_v9_index(index_path, workspace):
    record = prior.read_json(index_path)
    index = {Path(path).resolve(): sha for path, sha in record["file_sha256"].items()}
    for path, sha in index.items():
        prior.require(path.is_relative_to(workspace), "V9 input lies outside the authorized workspace")
        prior.check_path(path, sha, "Original V9 index entry")
    for key in ("report_base", "v8_source_commit", "v8_archive_commit", "v7_imaging_commit", "seed"):
        prior.require(record[key] == prior.CFG[key], f"Original V9 receipt differs at {key}")
    prior.require(not record["missing_indexed_frame_archives"], "Incomplete original V9 A3 archive")
    prior.git("merge-base", "--is-ancestor", CFG["archive_restore"], "HEAD")
    prior.git("diff", "--exit-code", CFG["v9_results_source"], "--", V8.name, f"{V9.name}/results",
              f"{V9.name}/configs", f"{V9.name}/ANALYSIS_PLAN.md")
    prior.git("diff", "--exit-code", "HEAD", "--", str(PROTOCOL.relative_to(REPO)))
    return index


def open_fd(v4_root, v9_index, profile):
    os.environ["NUMBA_CACHE_DIR"] = str(PRIVATE / "numba")
    sys.path.insert(0, str(v4_root / "scripts"))
    import v4_models as decoder
    import input_data as source
    import audit_dvars

    prior.require(Path(source.__file__).resolve() == asset(v9_index, "input_data.py")
                  and Path(decoder.__file__).resolve() == v4_root / "scripts/v4_models.py",
                  "A different implementation of the V4 decoder or V3 clock was loaded")
    prior.require(decoder.TR == PARAM["tr_s"] and decoder.METHODS["A3"] == ("G1", "S1"),
                  "Frozen model or TR differs from V10")
    prior.require(prior.read_json(v4_root / "results/Lambda_Selection.json")["lambda"] == PARAM["lambda"],
                  "V4 frozen lambda differs from V10")
    quality_archive = v4_root / "private/frozen_dvars.json"
    receipt = prior.read_json(v4_root / "results/Frame_Quality_Input_Provenance.json")
    prior.check_path(source.AUDIT, receipt["fd_source_archive_sha256"], "V4 audited FD")
    prior.check_path(quality_archive, receipt["dvars_payload_sha256"], "Verified fMRIPrep confounds copy")
    prior.require(receipt["scans"] == receipt["source_confounds_sha256_matches"] == sum(CFG["cohort"].values()),
                  "V4 confounds receipt has incomplete coverage")
    quality = prior.read_json(quality_archive)
    fd = audit_dvars.get_motion_fd()
    with gzip.open(source.AUDIT, "rt", encoding="utf-8") as stream:
        evidence = json.load(stream)
    acquisition = {row["label"]: row for row in evidence["scans"]}
    rows, folds = decoder.cohort()
    source_scans = {row["scan"]: dict(row, fold=int(fold)) for row, fold in zip(rows, folds, strict=True)}
    prior.require(set(source_scans) == set(quality) == set(profile.scan) and set(profile.scan) <= set(fd),
                  "Source scan and confounds keys disagree")
    scan_fd = {}
    roi_hashes = {}
    for frozen in profile.itertuples(index=False):
        item = source_scans[frozen.scan]
        paper = acquisition[frozen.scan]
        v = fd[frozen.scan]
        q = quality[frozen.scan]
        n = PARAM["original_clock_frames"]
        prior.require(item["confounds_sha256"] == paper["confounds"]["sha256"]
                      and item["condition"] == frozen.condition == paper["condition"]
                      and item["fold"] == frozen.fold, "Scan linkage, condition or fold changed")
        prior.require(v.shape == (n,) and np.isnan(v[0]) and np.all(np.isfinite(v[1:]) & (v[1:] >= 0)),
                      "FD is not the original nonnegative 210-TR sequence")
        starts = paper["confounds"]["initial_outliers"]
        prior.require(starts == list(range(frozen.source_start)) and item["x"].shape == (n, 78)
                      and np.isnan(item["x"][:frozen.source_start]).all()
                      and np.isfinite(item["x"][frozen.source_start:]).all(),
                      "ROI clock does not match the initial fMRIPrep outlier record")
        np.testing.assert_allclose(v, np.asarray(q["framewise_displacement"], float), rtol=0, atol=TOL, equal_nan=True)
        np.testing.assert_allclose(v[1:].mean(), frozen.fd, rtol=0, atol=TOL)
        np.testing.assert_allclose(np.nanmean(np.asarray(q["std_dvars"], float)), frozen.mean_std_dvars, rtol=0, atol=TOL)
        scan_fd[frozen.scan] = v
        relative = f"PD_DBS_zcc/derivatives/fmripost/{frozen.scan}/{frozen.scan}_BOLD_AAL1ctx_MNI_basic36_sm6.csv"
        roi_hashes[str((source.ARCHIVE / relative).resolve())] = item["roi_sha256"]
    return decoder, source_scans, scan_fd, roi_hashes, [source.AUDIT, quality_archive]


def load(workspace, v9_lock):
    indexed = checked_v9_index(v9_lock, workspace)
    v7_locks = [path for path in indexed if path.name == "Input_Hashes.json" and path.parent.parent.name ==
                "BOLD_Microstate_Mechanistic_Decomposition"]
    prior.require(len(v7_locks) == 1, "The original V9 index lacks a unique V7 input lock")
    v7_lock, = v7_locks
    v7 = prior.read_json(v7_lock)
    roles = {
        "v6_manifest": "ON_OFF_Pair_Manifest.csv",
        "v6_cohort": "Frozen_Cohort_Features.csv",
        "v6_scans": "ON_OFF_Subject_Features.csv",
        "v6_differences": "ON_OFF_Paired_Differences.csv",
        "v7_paired": "Paired_Decompositions.csv",
    }
    paths = {role: asset(indexed, name) for role, name in roles.items()}
    paths["prior_index"] = Path(v7["prior_input_index"]).resolve()
    prior.require(paths["prior_index"] in indexed, "V7 prior receipt is outside the locked V9 index")
    verifier, manifest = prior.verify_code()
    verifier.check_files(workspace, manifest["local_dependencies"])
    verifier.check_files(workspace, manifest["local_archive_files"])
    frames, missing, mapping, clock, _ = prior.read_v7_lock(workspace, v7_lock, paths, verifier, manifest)
    prior.require(not missing, "Original frame archives required for the formal quality analysis are missing")
    original = prior.load_imaging(paths)
    cohort = prior.read_csv(paths["v6_cohort"]).merge(clock, on="scan", validate="one_to_one",
                                                    suffixes=("", "_source")).sort_values("scan", ignore_index=True)
    prior.require(len(cohort) == len(frames) and cohort.condition.value_counts().to_dict() == CFG["cohort"],
                  "V4 and V6 cohorts are incomplete")
    np.testing.assert_array_equal(cohort.condition, cohort.condition_source.map({1: "HC", 2: "OFF", 3: "ON"}))
    cohort = cohort.rename(columns={"frames": "source_frames", "source_index_zero_based_start": "source_start"})
    prior.require(cohort.source_start.isin([0, 1]).all()
                  and (cohort.source_start + cohort.source_frames).eq(PARAM["original_clock_frames"]).all(),
                  "Source ROI clock length is invalid")
    original_pairs = prior.read_csv(paths["v7_paired"]).sort_values("participant_id", ignore_index=True)
    roster = original.roster.reset_index()
    np.testing.assert_array_equal(original_pairs.participant_id, roster.participant_id)
    maps = asset(indexed, "V4_Maps.npz")
    v4_root = maps.parent.parent
    decoder, source_scans, fd, roi_hashes, fd_files = open_fd(v4_root, indexed, cohort)
    with np.load(maps, allow_pickle=False) as stored:
        banks = stored[PARAM["learner"]]
    metrics_path, = [(workspace / row["path"]).resolve() for row in manifest["local_archive_files"]
                     if Path(row["path"]).name == "V4_Scan_Metrics.csv"]
    old_qc = prior.read_csv(metrics_path).query("method == 'A3' and window == 'fd_censor'")
    old_qc = old_qc.set_index("scan", verify_integrity=True)
    prior.require(set(old_qc.index) == set(cohort.scan), "V4 FD sensitivity cache is incomplete")
    locked = {str(path): sha for path, sha in indexed.items()}
    locked.update(roi_hashes)
    locked.update({str(path): prior.digest(path) for path in (
        PROTOCOL, MODULE / "ANALYSIS_PLAN.md", v9_lock, metrics_path, *fd_files,
        *(V9 / "results" / name for name in V9_CSVS),
        *(workspace / row["path"] for row in manifest["local_dependencies"]),
    )})
    lock_path = PRIVATE / "manifests/Input_Hashes.json"
    if lock_path.exists():
        prior.require(prior.read_json(lock_path)["file_sha256"] == locked, "V10 locked input hashes have changed")
    else:
        prior.write_json({"file_sha256": locked}, lock_path)

    v8_locks = [path for path in indexed if path.name in {"Prespecified_Inference_Lock.json", "Inference_Lock.json"}
                and "BOLD_Microstate_STN_Original_Scores" in path.parts]
    prior.require(len(v8_locks) == 2, "Missing paired original V8 input receipts")
    v8_index = prior.read_json(next(path for path in v8_locks if path.name == "Inference_Lock.json"))
    v8_hashes = [item["sha256"] for item in v8_index["inputs"].values()] + [v8_index["patient_input_sha256"]]
    public = [
        V8 / "configs/protocol.json", V8 / "configs/date_unrestricted.json",
        V8 / "results/Score_Reconciliation_Summary.csv", V8 / "results/STN_Cohort_Flow.csv",
        V9 / "configs/protocol.json", *(V9 / "results" / name for name in V9_CSVS),
    ]
    fingerprints = {
        "public_text_lf_normalized_sha256": {p.relative_to(REPO).as_posix(): prior.digest(p, normalise=True)
                                              for p in public},
        "private_source_collection_scheme": "Sort lowercase file SHA-256 hex strings, concatenate each with LF, then SHA-256 the ASCII bytes. Counts retain multiplicity.",
        "private_source_collections": {
            "v8_locked_sources_and_date_unrestricted_table": path_free_hash(v8_hashes),
            "v9_locked_source_files": path_free_hash(list(indexed.values())),
            "v10_roi_clock_series": path_free_hash(list(roi_hashes.values())),
            "v10_fd_archive_and_verified_confounds": path_free_hash([prior.digest(path) for path in fd_files]),
        },
    }
    return Frozen(cohort, roster, original_pairs, frames, mapping, banks, source_scans, fd,
                  old_qc, decoder, fingerprints)


def verify_inputs_unchanged():
    locked = prior.read_json(PRIVATE / "manifests/Input_Hashes.json")["file_sha256"]
    for path, digest in locked.items():
        prior.check_path(Path(path), digest, "V10 locked input during analysis")
    return len(locked)
