"""Read and verify the V4/V6 archive without fitting, filtering or decoding MRI."""

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
PRIVATE = PROJECT / "private"
RESULTS = PROJECT / "results"
LOGS = PROJECT / "logs"
V4 = REPO / "V4_Microstate_Statistical_Significance"
V6 = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics"
PROTOCOL_PATH = PROJECT / "configs/protocol.json"
PLAN_COMMIT = "740e4f3c65f601a03ba6171aff5989f7982f9793"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


CFG = read_json(PROTOCOL_PATH)
TR = CFG["baseline"]["tr"]
STATES = range(1, CFG["baseline"]["k"] + 1)
TOL = CFG["identity_tolerance"]
STIMULATION = ("OFF", "ON")
COUNT_COLUMNS = [f"count_{i}_{j}" for i in STATES for j in STATES]
FEATURES = [f"{family}_{i}" for family in ("coverage", "duration", "occurrence", "gev") for i in STATES]
REQUIRED_SCAN_FIELDS = ["scan", "subject", "condition", "fold", "fd", "mean_std_dvars", "valid_frames",
                        "valid_blocks", "dwell_edge_mean_4", *FEATURES, *COUNT_COLUMNS]
V6_NAMES = {
    "v6_manifest": "ON_OFF_Pair_Manifest.csv",
    "v6_cohort": "Frozen_Cohort_Features.csv",
    "v6_scans": "ON_OFF_Subject_Features.csv",
    "v6_differences": "ON_OFF_Paired_Differences.csv",
    "transition_counts": "ON_OFF_Transition_Matrices.csv",
}
NUMERICAL_DELTA_COLUMNS = ("gev_1", "duration_4", "coverage_4", "fd", "mean_std_dvars", "valid_frames")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path, normalise=False):
    content = path.read_bytes()
    if normalise and path.suffix.lower() in {".csv", ".json", ".md", ".py", ".yml"}:
        content = content.replace(b"\r\n", b"\n")
    return sha256(content).hexdigest()


def write_json(item, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", na_rep="NA")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FrozenInputs:
    workspace: Path
    scans: pd.DataFrame
    pairs: pd.DataFrame
    frame_files: dict
    frame_order: dict
    bank: np.ndarray
    roi_names: list
    v3_loader: Path
    original_files: dict


def validate_roster(cohort, patients, roster, deltas, transitions):
    sizes = CFG["cohort"]
    counts = {condition: sizes["pd_pairs"] for condition in STIMULATION}
    counts["HC"] = sizes["hc"]
    require(cohort.scan.is_unique and len(cohort) == sizes["scans"], "Duplicate or missing frozen scan")
    require(cohort.condition.value_counts().to_dict() == counts, "Frozen cohort counts disagree")
    require(cohort.subject.nunique() == sizes["hc"] + sizes["pd_pairs"], "Frozen participant count disagrees")
    require(cohort.groupby("subject").fold.nunique().eq(1).all(), "A patient's scans are in different folds")
    require(set(cohort.fold) == set(range(5)), "The five frozen folds are incomplete")
    require(cohort.groupby(["fold", "condition"]).size().unstack().reindex(columns=["HC", "OFF", "ON"])
            .eq(pd.Series({"HC": 3, "OFF": 5, "ON": 5})).all().all(), "Frozen folds do not have 3 HC and 5 pairs")
    pd.testing.assert_frame_equal(cohort.loc[cohort.condition.isin(STIMULATION)].reset_index(drop=True),
                                  patients.reset_index(drop=True), check_exact=True)
    observed = roster.set_index("participant_id", verify_integrity=True).sort_index()
    require(len(observed) == sizes["pd_pairs"], "Frozen pair count disagrees")
    sorted_index = observed.index
    require(deltas.index.equals(sorted_index) and transitions.index.equals(sorted_index),
            "Frozen paired outputs do not have identical participant keys")
    sides = {}
    for side in STIMULATION:
        key = side.lower()
        frozen = patients.loc[patients.condition.eq(side)].set_index("subject", verify_integrity=True).sort_index()
        require(frozen.index.equals(sorted_index), "Frozen scan and pair participant keys differ")
        np.testing.assert_array_equal(frozen.scan, observed[f"{key}_scan_id"])
        np.testing.assert_array_equal(frozen.valid_frames, observed[f"{key}_valid_frames"])
        np.testing.assert_allclose(frozen.fd, observed[f"{key}_mean_fd"], atol=TOL, rtol=0)
        np.testing.assert_allclose(frozen.mean_std_dvars, observed[f"{key}_std_dvars"], atol=TOL, rtol=0)
        np.testing.assert_array_equal(frozen[COUNT_COLUMNS], transitions[[f"{key}_{c}" for c in COUNT_COLUMNS]])
        sides[side] = frozen
    fields = list(NUMERICAL_DELTA_COLUMNS)
    np.testing.assert_allclose(sides["ON"][fields] - sides["OFF"][fields], deltas[fields],
                               atol=TOL, rtol=0, equal_nan=True)
    return observed


def verify_template_tables(bank, fold_path, reference_path):
    folds = pd.read_csv(fold_path)
    reference = pd.read_csv(reference_path)
    require(len(folds) == 20 and set(folds.fold) == set(range(5)), "Frozen state maps are incomplete")
    require(folds.stable.all(), "Unexpected change to the original state eligibility flag")
    pd.testing.assert_frame_equal(folds.loc[folds.fold.eq(0)].reset_index(drop=True), reference, check_exact=True)
    names = list(folds.columns[4:])
    require(len(names) == len(set(names)) == 78 and bank.shape == (5, 4, 78), "Frozen ROI axis differs")
    mapping = {}
    for fold, group in folds.groupby("fold"):
        ordered = group.sort_values("original_state")
        np.testing.assert_array_equal(ordered.original_state, list(STATES))
        np.testing.assert_array_equal(np.sort(ordered.state), list(STATES))
        polarity = np.where(np.sum(bank[fold] * ordered[names].to_numpy(), axis=1) >= 0, 1, -1)
        np.testing.assert_allclose(bank[fold] * polarity[:, None], ordered[names], atol=TOL, rtol=0)
        mapping[fold] = ordered.state.to_numpy() - 1
    return mapping, names


def archive_sources(workspace, prior_index):
    frozen = read_json(V4 / "workflow/Freeze_Manifest.json")
    prior = read_json(prior_index)
    require(prior["parent_commit"] == CFG["motor_severity_reference"], "Unexpected prior targeted base")
    indexed = prior["inputs"]
    by_key = {key: Path(indexed[key]["file"]).resolve(strict=True) for key in V6_NAMES}
    for key, path in by_key.items():
        require(path.name == V6_NAMES[key] and digest(path) == indexed[key]["sha256"],
                f"V6 frozen file does not match its verified index ({key})")

    local = frozen["local_archive_files"] + frozen["local_dependencies"]
    wanted = ("private/V4_Maps.npz", "scripts/input_data.py", "/input_manifest.csv")
    assets = {}
    for suffix in wanted:
        entry, = (row for row in local if row["path"].endswith(suffix))
        path = workspace / entry["path"]
        require(digest(path, normalise=True) == entry["sha256"], f"Frozen receipt mismatch ({suffix})")
        assets[Path(suffix).name] = path
    assets["Fold_Templates.csv"] = V4 / "results/Fold_Templates.csv"
    assets["Reference_Templates.csv"] = V4 / "results/Reference_Templates.csv"
    verifier = load_module("_v7_v4_verifier", V4 / "scripts/verify_freeze.py")
    verifier.check_files(V4, frozen["public_files"])
    collection, = (row for row in frozen["local_archive_collections"]
                   if row["role"] == "A3 archived leave-out backfits")
    count, fingerprint = verifier.collection_digest(workspace, collection["path"], collection["glob"])
    require(count == collection["count"] == CFG["cohort"]["scans"] and fingerprint == collection["sha256"],
            "A3 archive does not match the V4 freeze receipt")
    frames = {path.stem: path for path in (workspace / collection["path"]).glob(collection["glob"])}
    return by_key, assets, frames, len(frozen["public_files"])


def load_inputs(workspace, prior_index):
    by_key, assets, frames, verified_v4_files = archive_sources(workspace, prior_index)
    require(read_json(V6 / "configs/protocol.json")["baseline"] == CFG["baseline"], "V6 method profile changed")
    v4_scans = pd.read_csv(assets["input_manifest.csv"], usecols=["scan", "condition", "frames"])
    cohort = pd.read_csv(by_key["v6_cohort"], usecols=REQUIRED_SCAN_FIELDS).sort_values("scan").reset_index(drop=True)
    patients = pd.read_csv(by_key["v6_scans"], usecols=REQUIRED_SCAN_FIELDS).sort_values("scan").reset_index(drop=True)
    require(set(cohort.scan) == set(v4_scans.scan) == set(frames), "V4, V6 and frame archive scan keys differ")
    reference_condition = v4_scans.set_index("scan").loc[cohort.scan].condition.map({1: "HC", 2: "OFF", 3: "ON"})
    np.testing.assert_array_equal(cohort.condition, reference_condition)
    roster = pd.read_csv(by_key["v6_manifest"], usecols=["participant_id", "scan_order", *[
        f"{sid}_{key}" for sid in ("off", "on") for key in ("scan_id", "valid_frames", "mean_fd", "std_dvars")
    ]])
    deltas = pd.read_csv(by_key["v6_differences"], usecols=["participant_id", *NUMERICAL_DELTA_COLUMNS])
    trans = pd.read_csv(by_key["transition_counts"], usecols=["participant_id", *[
        f"{side}_{c}" for side in ("off", "on") for c in COUNT_COLUMNS
    ]])
    roster = validate_roster(cohort, patients, roster,
                             deltas.set_index("participant_id", verify_integrity=True).sort_index(),
                             trans.set_index("participant_id", verify_integrity=True).sort_index())
    with np.load(assets["V4_Maps.npz"], allow_pickle=False) as maps:
        bank = maps["G1"]
    mapping, roi_names = verify_template_tables(bank, assets["Fold_Templates.csv"], assets["Reference_Templates.csv"])
    frozen_files = {str(path): digest(path) for path in [*by_key.values(), *assets.values(), *frames.values()]}
    write_json({"file_sha256": frozen_files, "prior_input_index": str(prior_index)}, PRIVATE / "Input_Hashes.json")
    write_json({
        "status": "verified", "retained_cohort": CFG["cohort"], "independent_participants": int(cohort.subject.nunique()),
        "frozen_v6_tables_verified": list(V6_NAMES.values()), "v4_public_files_verified": verified_v4_files,
        "frozen_frame_archives_verified": len(frames), "v4_and_v6_scan_keys_exact": True,
        "v6_feature_difference_and_transition_tables_exact": True, "on_and_off_in_same_fold": True,
        "distinct_folds": len(mapping), "frozen_roi_axis_count": len(roi_names),
        "verified_scan_order_pairs": int(roster.scan_order.ne("unverified").sum()),
        "assessment_data_read": False,
    }, LOGS / "V7_Input_Audit.json")
    v6_receipt = read_json(V6 / "V4_V5_FROZEN_REFERENCE.json")
    write_json({
        "report_base": CFG["report_base"], "motor_severity_reference": CFG["motor_severity_reference"],
        "v6_release": CFG["frozen_v6"], "v5_release": v6_receipt["v5_commit"],
        "v4_release": v6_receipt["v4_commit"], "v4_frame_collection_against_freeze_receipt": True,
        "g1_private_maps_match_public_fold_templates": True, "public_template_rows": 20,
    }, LOGS / "V7_Frozen_Reference.json")
    write_json({
        "protocol_commit": PLAN_COMMIT, "protocol": CFG,
        "labels": "Archived A3 common-window frame arrays, mapped to the frozen reference index",
        "head_motion": "Independent V6 Freedman–Lane residual sign flips with separate delta FD and delta DVARS models",
        "imaging_reprocessing": False, "template_refit": False, "segmentation_rerun": False,
    }, LOGS / "V7_Method_Manifest.json")
    return FrozenInputs(workspace, cohort, roster, frames, mapping, bank, roi_names, assets["input_data.py"], frozen_files)
