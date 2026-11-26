"""Read only the locked ON/OFF A3 scan features and their fixed quality counterpart."""

from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parent
V10 = REPO / "BOLD_Microstate_M1_Robustness"
V6 = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics"
V9_LOCK = "WEiDA-DBS-V9/BOLD_Microstate_Expression_Followup/private/manifests/Input_Hashes.json"
BASELINE = {
    "method": "A3", "atlas": "AAL78", "k": 4, "tr": 2.1, "band_hz": [0.01, 0.08],
    "band": "F1", "normalization": "N0", "learner": "G1", "decoder": "S1", "lambda": 0.04,
}
N_PAIRS = 25
LABELS = tuple(range(1, BASELINE["k"] + 1))
TOLERANCE_SECONDS = 1e-12


@dataclass(frozen=True)
class Imaging:
    original: pd.DataFrame
    quality: pd.DataFrame
    audit: dict
    input_hashes: dict


def _pair_roster(frozen):
    """Link each visit by the locked scan and participant identifiers."""
    scans = frozen.cohort.loc[frozen.cohort.condition.isin(["ON", "OFF"])].copy()
    lookup = scans.set_index("scan", verify_integrity=True)
    roster = frozen.roster.set_index("participant_id", verify_integrity=True).sort_index()
    if (len(roster) != N_PAIRS or len(lookup) != 2 * len(roster) or roster.index.hasnans
            or roster[["on_scan_id", "off_scan_id"]].isna().any().any()):
        raise ValueError("The frozen patient scan inventory is incomplete")
    folded = {}
    for condition in ("ON", "OFF"):
        keys = roster[f"{condition.lower()}_scan_id"]
        expected = lookup.loc[lookup.condition.eq(condition)]
        if not keys.is_unique or set(keys) != set(expected.index):
            raise ValueError("The frozen manifest and scan keys disagree")
        visits = lookup.loc[keys]
        if (not visits.condition.eq(condition).all()
                or not np.array_equal(visits.subject.to_numpy(), roster.index.to_numpy())):
            raise ValueError("The frozen patient and scan condition keys disagree")
        folded[condition] = visits.fold.to_numpy(int)
    if not np.array_equal(folded["ON"], folded["OFF"]):
        raise ValueError("A patient uses different frozen folds across ON/OFF")
    return scans, roster.assign(fold=folded["ON"])


def _duration_seconds(labels, mapping, segments):
    """Average full observed episodes, separating all gaps on the TR clock."""
    durations = [[] for _ in LABELS]
    for start, stop, raw_state in segments(labels):
        durations[mapping[raw_state]].append((stop - start) * BASELINE["tr"])
    return np.array([np.mean(values) if values else np.nan for values in durations])


def _archive_duration(row, mapping=None):
    raw = np.asarray([row[f"duration_{state}"] for state in LABELS], dtype=float)
    if mapping is None:
        return raw
    aligned = np.full_like(raw, np.nan)
    aligned[mapping] = raw
    return aligned


def _assert_duration(measured, archived, window):
    if not np.allclose(measured, archived, atol=TOLERANCE_SECONDS, rtol=0, equal_nan=True):
        raise ValueError(f"Replay duration differs from the locked {window} archive")
    valid = np.isfinite(archived)
    if not valid.any():
        return 0.0
    if np.any(archived[valid] < BASELINE["tr"] - TOLERANCE_SECONDS):
        raise ValueError("An archived mean duration is below one TR")
    return float(np.abs(measured[valid] - archived[valid]).max())


def _assemble_pairs(roster, scans):
    rows = roster.reset_index().rename(columns={"participant_id": "patient_id"})
    for state in LABELS:
        endpoint = f"duration_{state}"
        on = rows.on_scan_id.map(scans[endpoint])
        off = rows.off_scan_id.map(scans[endpoint])
        rows[f"{endpoint}_on"] = on
        rows[f"{endpoint}_off"] = off
        rows[f"{endpoint}_change"] = on - off
    return rows


def load_imaging(workspace=None, private=None, v9_lock=None):
    """Return patient-keyed ON/OFF duration tables and path-free measurement audit.

    The returned input_hashes map contains local paths for private verification only.
    """
    workspace = Path(workspace or REPO.parent).resolve()
    private = Path(private or (EXPERIMENT / "private")).resolve()
    if not private.is_relative_to(EXPERIMENT / "private"):
        raise ValueError("The V10 replay cache must be inside this experiment's private directory")
    runtime = private / "imaging_v10"
    cache = runtime / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(cache)
    # Historical Numba helpers may have been imported by the caller.
    if "numba" in sys.modules:
        sys.modules["numba"].config.CACHE_DIR = str(cache)

    v10_scripts = V10 / "scripts"
    sys.path.insert(0, str(v10_scripts))
    import v10_inputs as v10
    from v10_frames import FORMAL, ORIGINAL, motion_mask, replay

    if Path(v10.__file__).resolve() != v10_scripts / "v10_inputs.py":
        raise ValueError("A V10 module outside the independent worktree was loaded")
    v10.PRIVATE = runtime
    v6_protocol = V6 / "configs/protocol.json"
    if v10.prior.read_json(v6_protocol)["baseline"] != BASELINE:
        raise ValueError("The frozen V6 A3 protocol differs from the registered imaging definition")
    expected = {key: value for key, value in BASELINE.items() if key not in ("tr", "band_hz")}
    expected.update(tr_s=BASELINE["tr"], original_clock_frames=210, folds=5)
    if v10.PARAM != expected:
        raise ValueError("The frozen V10 A3 protocol differs from the registered imaging definition")
    rule = v10.CFG["fd"]
    if (rule["strictly_greater_than"], rule["preceding_frames"], rule["following_frames"]) != (0.5, 1, 2):
        raise ValueError("The archived V10 FD rule differs from the fixed QC protocol")

    frozen = v10.load(workspace, Path(v9_lock or (workspace / V9_LOCK)).resolve())
    patient_scans, roster = _pair_roster(frozen)
    # The V10 loader verifies the full shared freeze. Only the patient scans are replayed here.
    qc, evidence = replay(replace(frozen, cohort=patient_scans))
    log = qc.set_index(["mode", "scan"], verify_integrity=True)
    patient_lookup = patient_scans.set_index("scan")
    records = {ORIGINAL: [], FORMAL: []}
    max_error = dict.fromkeys(records, 0.0)

    for scan, meta in patient_lookup.iterrows():
        with np.load(frozen.frames[scan], allow_pickle=False) as archive:
            labels, gfp, scores = (archive[name] for name in ("labels", "gfp", "corr"))
        mapping = frozen.mapping[meta.fold]
        original = _archive_duration(meta)
        retained = (labels >= 0) & motion_mask(frozen.fd[scan])
        replay_labels = frozen.decoder.temporal_labels(
            scores, gfp, retained, BASELINE["lambda"],
            normalization=log.at[(ORIGINAL, scan), "normalization_gfp_squared"],
        )
        if not np.array_equal(replay_labels >= 0, retained):
            raise ValueError("The quality replay did not preserve the frozen FD clock mask")
        quality_meta = frozen.old_qc.loc[scan]
        if (quality_meta.fold != meta.fold or quality_meta.condition != meta.condition
                or quality_meta.subject != meta.subject or int(retained.sum()) != quality_meta.n_labeled_frames):
            raise ValueError("An archived QC record does not match its frozen patient scan")
        quality = _archive_duration(quality_meta, mapping)
        for mode, clock_labels, archive_values in ((ORIGINAL, labels, original), (FORMAL, replay_labels, quality)):
            replayed = _duration_seconds(clock_labels, mapping, frozen.decoder.segments)
            max_error[mode] = max(max_error[mode], _assert_duration(replayed, archive_values, mode))
            records[mode].append({"scan": scan, **dict(zip(
                [f"duration_{state}" for state in LABELS], archive_values, strict=True,
            ))})

    paired = {mode: _assemble_pairs(roster, pd.DataFrame(rows).set_index("scan", verify_integrity=True))
              for mode, rows in records.items()}
    n_checked = v10.verify_inputs_unchanged()
    locks = v10.prior.read_json(runtime / "manifests/Input_Hashes.json")["file_sha256"]
    order_path = V6 / "results/Scan_Order_Audit.json"
    order = v10.prior.read_json(order_path)
    locks.update({str(path): v10.prior.digest(path) for path in
                  (v6_protocol, order_path, v10_scripts / "v10_inputs.py", v10_scripts / "v10_frames.py")})
    release_hashes = {name: next(value for path, value in locks.items() if Path(path).name == name)
                      for name in ("Frozen_Cohort_Features.csv", "ON_OFF_Pair_Manifest.csv",
                                   "V4_Scan_Metrics.csv", "V4_Maps.npz")}
    release_hashes["v6_protocol_sha256"] = locks[str(v6_protocol)]
    audit = dict(
        baseline=BASELINE, paired_patients=len(roster), patient_scans=len(patient_scans),
        folds=int(roster.fold.nunique()), states=[f"M{state}" for state in LABELS], duration_unit="s",
        duration_definition="2.1 seconds times mean episode length in TR; gaps split episodes, window-boundary episodes are retained",
        original_source="V6 Frozen_Cohort_Features.csv, verified against frozen aligned A3 labels",
        quality_source="V4 A3 fd_censor archive durations, verified against the V10 fd_censor_s1 replay",
        quality_rule={"fd_strictly_greater_than_mm": 0.5, "exclude_preceding_tr": 1, "exclude_following_tr": 2,
                      "independent_blocks_for_s1": True},
        max_duration_error_seconds=max_error,
        complete_duration_pairs={mode: {f"M{k}": int(table[f"duration_{k}_change"].notna().sum())
                                        for k in LABELS} for mode, table in paired.items()},
        number_of_short_roi_scans=int(patient_scans.source_start.sum()),
        archived_inputs_rechecked=n_checked,
        input_fingerprints=v10.path_free_hash(list(locks.values())),
        source_sha256=release_hashes,
        aligned_g1_mapping_sha256=sha256(np.stack([frozen.mapping[fold] for fold in range(5)])
                                         .astype("<i8").tobytes()).hexdigest(),
        scan_order={key: order[key] for key in ("verified_pairs", "on_first", "off_first", "unknown_order")},
        **evidence,
    )
    return Imaging(paired[ORIGINAL], paired[FORMAL], audit, locks)
