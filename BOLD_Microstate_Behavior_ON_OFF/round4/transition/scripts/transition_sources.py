"""Load the original frozen A3 labels and count only consecutive valid TRs."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATES = ("M1", "M2", "M3", "M4")
CONDITIONS = {"HC": 15, "OFF": 25, "ON": 25}
ACQUIRED_TRS = 210
TR = 2.1
ATOL = 1e-12
LABEL_ROLE = "A3 archived leave-out backfits"
V6 = "V6_DBS_ON_OFF_Microstate_Dynamics"
V4 = "V4_Microstate_Statistical_Significance"
V6_BASELINE = {
    "method": "A3", "atlas": "AAL78", "k": 4, "tr": TR, "band_hz": [0.01, 0.08],
    "band": "F1", "normalization": "N0", "learner": "G1", "decoder": "S1", "lambda": 0.04,
}


@dataclass(frozen=True)
class TransitionSources:
    scan_metrics: pd.DataFrame
    matrices: pd.DataFrame
    pairs: pd.DataFrame
    input_hashes: dict
    audit: dict


@dataclass(frozen=True)
class ScanTransitions:
    counts: np.ndarray
    p_weida: np.ndarray
    p_exit: np.ndarray
    frames: np.ndarray
    episodes: np.ndarray
    enters: np.ndarray
    exits: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    block_starts: np.ndarray
    block_ends: np.ndarray
    durations: np.ndarray
    coverages: np.ndarray
    occurrences: np.ndarray


def require(condition, message):
    if not condition:
        raise ValueError(message)


def measure_scan(labels):
    """Calculate the 4×4 matrices from one canonical, zero-based state series."""
    labels = np.asarray(labels)
    require(labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
            and np.isin(labels, [-1, 0, 1, 2, 3]).all(),
            "A transition series must be a one-dimensional M1–M4 label vector with -1 gaps")
    keep = labels >= 0
    adjacent = keep[:-1] & keep[1:]
    counts = np.bincount(4 * labels[:-1][adjacent] + labels[1:][adjacent], minlength=16).reshape(4, 4)
    frames = np.bincount(labels[keep], minlength=4)
    block_starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]]) if len(keep) else np.array([], dtype=int)
    block_ends = np.flatnonzero(keep & ~np.r_[keep[1:], False]) if len(keep) else np.array([], dtype=int)
    starts = np.bincount(labels[block_starts], minlength=4)
    ends = np.bincount(labels[block_ends], minlength=4)
    sources, destinations = counts.sum(axis=1), counts.sum(axis=0)
    diagonal = counts.diagonal()
    exits, enters = sources - diagonal, destinations - diagonal
    episodes = starts + enters
    n_valid = int(keep.sum())
    require(counts.sum() == adjacent.sum() == n_valid - len(block_starts)
            and len(block_starts) == len(block_ends),
            "Adjacent pairs disagree with the independent valid-window length identity")
    require(np.array_equal(episodes, ends + exits) and np.array_equal(frames, diagonal + episodes)
            and np.array_equal(frames, sources + ends) and np.array_equal(frames, destinations + starts),
            "State frame, segment, entry, exit and boundary conservation failed")

    p_weida = np.divide(counts, sources[:, None], out=np.full((4, 4), np.nan), where=sources[:, None] > 0)
    p_exit = np.divide(counts, exits[:, None], out=np.full((4, 4), np.nan),
                       where=(exits[:, None] > 0) & ~np.eye(4, dtype=bool))
    np.testing.assert_allclose(p_weida[sources > 0].sum(axis=1), 1, rtol=0, atol=ATOL)
    np.testing.assert_allclose(np.nansum(p_exit[exits > 0], axis=1), 1, rtol=0, atol=ATOL)
    require(np.isnan(p_weida[sources == 0]).all() and np.isnan(p_exit[exits == 0]).all()
            and np.isnan(p_exit.diagonal()).all(), "Undefined rows and exit diagonals must be NA")
    durations = np.divide(TR * frames, episodes, out=np.full(4, np.nan), where=episodes > 0)
    coverages = frames / n_valid if n_valid else np.full(4, np.nan)
    occurrences = episodes / (TR * n_valid) if n_valid else np.full(4, np.nan)
    return ScanTransitions(
        counts, p_weida, p_exit, frames, episodes, enters, exits, starts, ends,
        block_starts, block_ends, durations, coverages, occurrences,
    )


def read_source(path, workspace, hashes, expected=None, canonical_csv=False):
    path = path.resolve()
    if not path.is_relative_to(workspace) or not path.is_file():
        raise FileNotFoundError(f"Missing local source within workspace: {path.name}")
    content = path.read_bytes()
    actual = sha256(content).hexdigest()
    if expected is not None and actual != expected:
        require(canonical_csv and sha256(content.replace(b"\r\n", b"\n")).hexdigest() == expected,
                f"Frozen source SHA-256 differs: {path.name}")
    hashes[str(path)] = actual
    return content


def check_scan_keys(cohort, original, clock, pairs):
    require(cohort.condition.value_counts().to_dict() == CONDITIONS and len(cohort) == 65
            and cohort.scan.is_unique and cohort.subject.nunique() == 40
            and not cohort[["scan", "subject", "condition", "fold"]].isna().any().any(),
            "Frozen A3 cohort must have 65 unique scans, 40 participants, and the locked conditions")
    require(cohort.groupby("subject").fold.nunique().eq(1).all()
            and cohort.groupby("condition").fold.nunique().eq(5).all(),
            "Frozen five-fold assignment differs across conditions or paired visits")
    original = original.set_index("scan")
    require(original.index.is_unique and set(original.index) == set(cohort.scan),
            "Original A3 feature scan keys differ from the frozen V6 cohort")
    columns = ["subject", "condition", "fold", "valid_frames"]
    np.testing.assert_array_equal(cohort[columns].to_numpy(), original.loc[cohort.scan, columns].to_numpy())
    require(clock.index.is_unique and set(clock.index) == set(cohort.scan),
            "Original 210-TR manifest scan keys differ from the frozen V6 cohort")
    timing = clock.loc[cohort.scan]
    np.testing.assert_array_equal(cohort.condition, timing.condition.map({1: "HC", 2: "OFF", 3: "ON"}))
    np.testing.assert_array_equal(cohort.input_frames, timing.frames)
    np.testing.assert_array_equal(cohort.input_missing_initial_frames, timing.source_index_zero_based_start)
    np.testing.assert_array_equal(timing.frames, ACQUIRED_TRS - timing.source_index_zero_based_start)
    require(timing.source_index_zero_based_start.between(0, ACQUIRED_TRS - 1).all(),
            "Invalid original TR source offset")
    patient_scans = cohort.loc[cohort.condition.isin(("ON", "OFF"))]
    np.testing.assert_array_equal(patient_scans.subject, clock.loc[patient_scans.scan, "pair_id"])
    require(len(pairs) == 25 and pairs.participant_id.is_unique and not pairs.isna().any().any(),
            "The frozen pairing manifest must contain 25 unique complete patients")
    for condition in ("ON", "OFF"):
        prefix = condition.lower()
        side = cohort.loc[cohort.condition.eq(condition)].set_index("scan")
        scan_keys = pairs[f"{prefix}_scan_id"]
        require(scan_keys.is_unique and set(scan_keys) == set(side.index),
                "The frozen pair must contain exactly the original scans for each DBS condition")
        np.testing.assert_array_equal(pairs.participant_id, side.loc[scan_keys, "subject"])
        np.testing.assert_array_equal(pairs[f"{prefix}_valid_frames"], side.loc[scan_keys, "valid_frames"])
    return pairs[["participant_id", "on_scan_id", "off_scan_id"]].assign(
        fold=pairs.participant_id.map(cohort.groupby("subject").fold.first())
    )


def verify_features(metrics, row, original):
    """Cross-check independent saved V6 counts and original V4 state summaries."""
    rows = metrics.counts.sum(axis=1)
    valid_trs = int(metrics.frames.sum())
    n_blocks = len(metrics.block_starts)
    qa_window = {
        "valid_frames": valid_trs,
        "valid_blocks": n_blocks,
        "leading_unlabeled": int(metrics.block_starts[0]),
        "trailing_unlabeled": ACQUIRED_TRS - 1 - int(metrics.block_ends[-1]),
        "interior_unlabeled": int(metrics.block_ends[-1] - metrics.block_starts[0] + 1 - valid_trs),
    }
    for key, value in qa_window.items():
        require(getattr(row, key) == value, f"Original valid-window definition changed ({key})")
    require(qa_window["leading_unlabeled"] >= row.input_missing_initial_frames and valid_trs <= row.input_frames,
            "Frozen valid labels extend beyond the original input time grid")
    for i, j in np.ndindex(4, 4):
        require(metrics.counts[i, j] == getattr(row, f"count_{i+1}_{j+1}"), "Frozen V6 transition counts disagree")
    compare = {
        "switching_rate": metrics.exits.sum() / (TR * valid_trs),
        **{f"{prefix}_{index}": values[index - 1]
           for prefix, values in (("coverage", metrics.coverages), ("duration", metrics.durations),
                                  ("occurrence", metrics.occurrences), ("source", rows))
           for index in range(1, 5)},
        **{f"transition_{i+1}_{j+1}": metrics.p_weida[i, j] for i, j in np.ndindex(4, 4)},
    }
    checked = []
    for key, value in compare.items():
        expected = getattr(row, key)
        np.testing.assert_allclose(value, expected, rtol=0, atol=ATOL, equal_nan=True, err_msg=key)
        if key in original.index:
            np.testing.assert_allclose(value, original[key], rtol=0, atol=ATOL, equal_nan=True, err_msg=key)
        if np.isfinite(value):
            checked.append(abs(value - expected))
    return max(checked, default=0.0)


def load_transition_sources(workspace, module):
    """Find and authenticate the frozen archive, without fitting or writing to it."""
    workspace = Path(workspace).resolve(strict=True)
    module = Path(module).resolve(strict=True)
    repo = module.parent
    historical = workspace / "WEiDA-DBS-V4"
    v6_private = workspace / "WEiDA-DBS-V6" / V6 / "private"
    hashes = {}
    receipts = {}

    def read(role, path, expected=None, canonical=False):
        data = read_source(path, workspace, hashes, expected, canonical)
        receipts[role] = sha256(data).hexdigest()
        return data

    def table(role, path, expected=None, columns=None):
        return pd.read_csv(BytesIO(read(role, path, expected)), usecols=columns, float_precision="round_trip")

    run1 = json.loads(read("round1_run_manifest", module / "results/Run_Manifest.json"))
    run2 = json.loads(read("round2_run_manifest", module / "round2/results/Round2_Run_Manifest.json"))
    known = run2["checks"]["v6_sources"]["verified_sha256"]
    for key in ("Frozen_Cohort_Features.csv", "ON_OFF_Pair_Manifest.csv"):
        require(run1["imaging_fingerprints"][key] == known[key], "Frozen V6 input receipts disagree by round")
    protocol = json.loads(read("v6_protocol", repo / V6 / "configs/protocol.json",
                               known["V6 protocol.json"], canonical=True))
    v6_receipt = json.loads(read("v6_input_audit", repo / V6 / "manifests/Input_Audit.json",
                                 known["V6 Input_Audit.json"], canonical=True))
    require(protocol["baseline"] == V6_BASELINE and protocol["cohort"] ==
            {"hc": 15, "pd_pairs": 25, "scans": 65, "acquired_frames": ACQUIRED_TRS}
            and run1["imaging_fingerprints"]["v6_protocol_sha256"] == v6_receipt["protocol_sha256"] ==
            known["V6 protocol.json"] and v6_receipt["template_sha256"] == known["Fold_Templates.csv"]
            and v6_receipt["frozen_mapping_reused"] and v6_receipt["original_metric_parity"],
            "Frozen A3 protocol, cohort or template receipt mismatch")
    freeze = json.loads(read("v4_archive_receipt", repo / V4 / "workflow/Freeze_Manifest.json"))
    archive, = [row for row in freeze["local_archive_collections"] if row["role"] == LABEL_ROLE]
    require({key: archive[key] for key in ("count", "sha256")} == v6_receipt["labels"]
            and archive["count"] == v6_receipt["scans"] == 65 and archive["glob"] == "*.npz",
            "Original A3 label archive receipts disagree")

    cohort = table("v6_cohort_features", v6_private / "results/Frozen_Cohort_Features.csv",
                   known["Frozen_Cohort_Features.csv"])
    pairs = table("v6_pair_manifest", v6_private / "manifests/ON_OFF_Pair_Manifest.csv",
                  known["ON_OFF_Pair_Manifest.csv"],
                  ["participant_id", "on_scan_id", "off_scan_id", "on_valid_frames", "off_valid_frames"])
    original = table("v4_original_features", historical / V4 / "private/Subject_Level_Microstate_Features.csv",
                     v6_receipt["original_feature_table_sha256"])
    clock = table("v4_scan_intake_manifest", workspace / "WEiDA_STN25_HC15_K4_20261117/input_manifest.csv",
                  known["input_manifest.csv"]).set_index("scan")
    template = table("v4_frozen_fold_templates", repo / V4 / "results/Fold_Templates.csv",
                     known["Fold_Templates.csv"], ["fold", "state", "original_state", "stable"])
    pairs = check_scan_keys(cohort, original, clock, pairs)
    original = original.set_index("scan")

    fold_mapping = {}
    for fold, rows in template.groupby("fold", sort=True):
        rows = rows.sort_values("original_state")
        require(rows.original_state.tolist() == [1, 2, 3, 4]
                and sorted(rows.state.tolist()) == [1, 2, 3, 4] and rows.stable.eq(True).all(),
                "Incomplete or unstable published state correspondence")
        fold_mapping[int(fold)] = rows.state.to_numpy(dtype=int) - 1
    require(list(fold_mapping) == list(range(5)) and np.array_equal(fold_mapping[0], np.arange(4)),
            "Original five-fold M1–M4 correspondence differs")
    require(set(cohort.fold) == set(fold_mapping), "A scan is missing its published hold-out fold")

    label_root = (workspace / archive["path"]).resolve()
    require(label_root.is_relative_to(workspace) and label_root.is_dir(), "Missing original A3 backfit archive")
    files = sorted(p for p in label_root.glob(archive["glob"]) if p.is_file())
    require(len(files) == archive["count"] and {p.stem for p in files} == set(cohort.scan),
            "The original A3 label archive does not contain exactly the 65 frozen scan keys")
    joined = cohort.set_index("scan", drop=False)
    reference_roi = workspace / "STN-DBS_105_AAL78_20261117/PD_DBS_zcc/derivatives/fmripost"
    collection_digest = sha256()
    scan_rows = []
    matrix_rows = []
    errors = []
    for path in files:
        scan_id = path.stem
        row = next(joined.loc[[scan_id]].itertuples(index=False))
        roi_path = reference_roi / scan_id / f"{scan_id}_BOLD_AAL1ctx_MNI_basic36_sm6.csv"
        read_source(roi_path, workspace, hashes, clock.at[scan_id, "sha256"])
        label_bytes = read_source(path, workspace, hashes)
        collection_digest.update(path.name.encode("utf-8") + b"\0")
        collection_digest.update(sha256(label_bytes).digest())
        with np.load(BytesIO(label_bytes), allow_pickle=False) as source:
            labels = source["labels"]
        require(labels.shape == (ACQUIRED_TRS,) and np.issubdtype(labels.dtype, np.integer)
                and np.isin(labels, [-1, 0, 1, 2, 3]).all(),
                "Original A3 label shape or label domain differs")
        valid = labels >= 0
        canonical = labels.copy()
        canonical[valid] = fold_mapping[row.fold][labels[valid]]
        metrics = measure_scan(canonical)
        require(metrics.frames.sum() > 0, "A frozen scan has no valid observations")
        errors.append(verify_features(metrics, row, original.loc[scan_id]))
        keys = {
            "scan_id": scan_id, "participant_id": row.subject, "condition": row.condition, "fold": int(row.fold),
        }
        scan_row = dict(
            keys, acquired_trs=ACQUIRED_TRS, source_input_trs=int(row.input_frames),
            valid_trs=int(metrics.frames.sum()), n_blocks=len(metrics.block_starts),
            n_adjacent=int(metrics.counts.sum()), interior_unlabeled=int(row.interior_unlabeled),
            block_start_tr_zero_based=json.dumps(metrics.block_starts.tolist()),
            block_end_tr_inclusive_zero_based=json.dumps(metrics.block_ends.tolist()),
        )
        for stem, values in (
            ("frames", metrics.frames), ("episodes", metrics.episodes),
            ("enters", metrics.enters), ("exits", metrics.exits), ("starts", metrics.starts), ("ends", metrics.ends),
            ("duration_seconds", metrics.durations), ("coverage", metrics.coverages),
            ("occurrence_per_second", metrics.occurrences),
        ):
            scan_row.update({f"{stem}_{state}": value for state, value in zip(STATES, values)})
        scan_rows.append(scan_row)
        matrix_rows.extend([
            dict(keys, from_state=STATES[i], to_state=STATES[j], count=int(metrics.counts[i, j]),
                 p_weida=metrics.p_weida[i, j], p_exit=metrics.p_exit[i, j])
            for i, j in np.ndindex(4, 4)
        ])
    label_sha256 = collection_digest.hexdigest()
    require(label_sha256 == archive["sha256"], "Original A3 label collection SHA-256 differs")
    scans = pd.DataFrame(scan_rows)
    matrices = pd.DataFrame(matrix_rows)
    require(int(scans.valid_trs.sum()) == v6_receipt["valid_trs"], "Total valid TRs differ from the frozen receipt")
    require((scans.n_adjacent == scans.valid_trs - scans.n_blocks).all(),
            "Valid adjacent pairs differ from the sum of every block length minus one")

    audit = {
        "scans": len(scans),
        "scan_counts": {condition: int(scans.condition.eq(condition).sum()) for condition in CONDITIONS},
        "patient_pairs": len(pairs),
        "folds": len(fold_mapping),
        "scans_per_condition_and_fold": {
            condition: scans.loc[scans.condition.eq(condition)].groupby("fold").size().tolist()
            for condition in CONDITIONS
        },
        "archive_sha256": label_sha256,
        "archive_files": len(files),
        "roi_input_sha256_matches": len(files),
        "reference_input_role_sha256": receipts,
        "original_fold_mapping_reused": True,
        "valid_trs": int(scans.valid_trs.sum()),
        "valid_blocks": int(scans.n_blocks.sum()),
        "valid_adjacent_pairs": int(scans.n_adjacent.sum()),
        "scans_with_internal_gaps": int(scans.interior_unlabeled.gt(0).sum()),
        "window_and_transition_count_parity_scans": len(scans),
        "duration_coverage_occurrence_parity_scans": len(scans),
        "max_abs_numeric_error_vs_v6": float(max(errors)),
        "undefined_weida_source_rows": int(matrices.loc[matrices.from_state.eq(matrices.to_state), "p_weida"].isna().sum()),
        "undefined_exit_source_rows": int(matrices.groupby(["scan_id", "from_state"]).p_exit.count().eq(0).sum()),
    }
    return TransitionSources(scans, matrices, pairs, hashes, audit)
