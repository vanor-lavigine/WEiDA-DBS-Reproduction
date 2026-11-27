"""Read and reconcile the frozen clinical visits and V6 Coverage features."""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


STATES = (1, 2, 3, 4)
ENDPOINTS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
PAIR_KEYS = ("patient_id", "on_scan_id", "off_scan_id")
BASELINE = {
    "method": "A3", "atlas": "AAL78", "k": 4, "tr": 2.1, "band_hz": [0.01, 0.08], "band": "F1",
    "normalization": "N0", "learner": "G1", "decoder": "S1", "lambda": 0.04,
}
ROUND2_LOCK = "private/round2/Round2_Input_Hashes.json"
MAIN_PAIRS = "private/results/Main_Pairs.csv"
FIRST_LOCK = "private/Input_Hashes.json"
V6 = "V6_DBS_ON_OFF_Microstate_Dynamics"


class UnverifiedInput(ValueError):
    """A required frozen input could not be located or reconciled."""


@dataclass(frozen=True)
class CoverageInputs:
    pairs: pd.DataFrame
    audit: dict
    input_hashes: dict[str, str]


def _digest(path, *, text=False):
    data = path.read_bytes()
    return sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _family_paths(workspace, module, relative):
    """Try the active checkout first, then exact relative paths in sibling worktrees."""
    candidates = [module / relative, *sorted(workspace.glob(f"*/{module.name}/{relative}"))]
    for path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if path.is_relative_to(workspace) and path.is_file():
            yield path


def _collection(hashes):
    return sha256("".join(value + "\n" for value in sorted(hashes.values())).encode("ascii")).hexdigest()


def _read_round2_lock(workspace, module, receipt):
    """Authenticate the old private receipt against the released path-free collection hash."""
    expected = receipt["input_collection_sha256"]
    for candidate in _family_paths(workspace, module, ROUND2_LOCK):
        try:
            entries = _json(candidate)["file_sha256"]
            if (isinstance(entries, dict) and all(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)
                                                  for value in entries.values())
                    and _collection(entries) == expected):
                return candidate, entries
        except (OSError, UnicodeError, ValueError, KeyError):
            continue
    raise UnverifiedInput("Missing or unmatched private/round2/Round2_Input_Hashes.json from the released round-two receipt")


def _locked_source(entries, relative, workspace, module=None):
    """Resolve an archived pathname or an identical copy in the known worktree family."""
    suffix = Path(relative).parts
    refs = [(Path(path).resolve(), value) for path, value in entries.items()
            if Path(path).parts[-len(suffix):] == suffix]
    hashes = {digest for _, digest in refs}
    if len(hashes) != 1:
        raise UnverifiedInput(f"Expected a unique prior SHA-256 for {relative}")
    expected, = hashes
    if module is None:
        aliases = [workspace / relative, *sorted(workspace.glob(f"*/{relative}"))]
    else:
        aliases = list(_family_paths(workspace, module, relative))
    candidates = [*aliases, *(path for path, _ in refs)]
    for path in dict.fromkeys(candidate.resolve() for candidate in candidates):
        if path.is_relative_to(workspace) and path.is_file() and _digest(path) == expected:
            return path, expected
    raise UnverifiedInput(f"Missing or unmatched locked input {relative}")


def _tracked_source(path, expected, used):
    """Match original bytes or their LF-canonical form after a Windows checkout."""
    actual = _digest(path)
    if actual != expected and _digest(path, text=True) != expected:
        raise UnverifiedInput(f"Frozen {path.name} differs from its existing SHA-256 receipt")
    used[str(path)] = actual


def _read_clinical(path):
    score_columns = [f"{metric}_{measure}" for metric in ENDPOINTS
                     for measure in ("off", "on", "improvement", "improvement_rate")]
    scores = pd.read_csv(path, usecols=[*PAIR_KEYS, "fold", *score_columns], float_precision="round_trip")
    by_endpoint = {}
    for metric in ENDPOINTS:
        off, on = (scores[f"{metric}_{side}"].to_numpy(float) for side in ("off", "on"))
        for values in (off, on):
            finite = values[np.isfinite(values)]
            if np.isinf(values).any() or (finite < 0).any() or not np.equal(finite, np.floor(finite)).all():
                raise UnverifiedInput(f"Invalid frozen integer score in {metric}")
        paired = np.isfinite(off) & np.isfinite(on)
        difference = off-on
        rate = np.divide(difference, off, out=np.full(len(scores), np.nan), where=paired & (off > 0))
        if (not np.allclose(difference, scores[f"{metric}_improvement"], atol=1e-12, rtol=0, equal_nan=True)
                or not np.allclose(rate, scores[f"{metric}_improvement_rate"], atol=1e-12, rtol=0, equal_nan=True)):
            raise UnverifiedInput(f"OFF-minus-ON values disagree with saved {metric} score changes")
        scores[f"{metric}_improvement"] = difference
        scores[f"{metric}_improvement_rate"] = rate
        zero = off == 0
        by_endpoint[metric] = {
            "absolute_available": int(paired.sum()), "rate_available": int(np.isfinite(rate).sum()),
            "off_zero": int(zero.sum()), "other_missing_rate": int((~np.isfinite(rate) & ~zero).sum()),
            "negative_rate": int((rate < 0).sum()), "zero_rate": int((rate == 0).sum()),
        }
    return scores, by_endpoint


def _unique_pairs(pairs, expected_patients):
    return (len(pairs) == expected_patients
            and all(pairs[col].notna().all() and pairs[col].is_unique for col in PAIR_KEYS)
            and not set(pairs.on_scan_id).intersection(pairs.off_scan_id))


def _audit_clock(cohort, clock, receipt):
    if not clock.index.is_unique or set(clock.index) != set(cohort.scan):
        raise UnverifiedInput("V6 scans disagree with the original 210-TR inventory")
    source = clock.loc[cohort.scan]
    if (not np.array_equal(cohort.condition, source.condition.map({1: "HC", 2: "OFF", 3: "ON"}))
            or not np.array_equal(cohort.input_frames, source.frames)
            or not np.array_equal(cohort.input_missing_initial_frames, source.source_index_zero_based_start)
            or not np.array_equal(source.frames, 210-source.source_index_zero_based_start)
            or not (cohort.valid_frames.gt(0) & cohort.valid_frames.le(cohort.input_frames)).all()
            or int(cohort.valid_frames.sum()) != receipt["valid_trs"]):
        raise UnverifiedInput("V6 common window, scan conditions or original TR counts differ from the archived audit")


def _connect(scores, cohort, roster, *, expected_patients, tolerance):
    if not _unique_pairs(scores, expected_patients) or not _unique_pairs(roster, expected_patients):
        raise UnverifiedInput("The frozen clinical and V6 rosters must contain complete unique ON/OFF pairs")
    patients = cohort.loc[cohort.condition.isin(("ON", "OFF"))]
    if (len(patients) != 2*expected_patients or not cohort.scan.is_unique
            or cohort[["scan", "subject", "condition", "fold"]].isna().any().any()
            or set(patients.scan) != set(roster.on_scan_id).union(roster.off_scan_id)):
        raise UnverifiedInput("V6 feature rows do not exactly match the frozen patient roster")
    joined = scores.merge(roster, on=list(PAIR_KEYS), how="left", validate="one_to_one", indicator=True)
    if not joined.pop("_merge").eq("both").all():
        raise UnverifiedInput("The clinical patient and condition-specific scan keys disagree with the V6 pair roster")
    coverage = [f"coverage_{i}" for i in STATES]
    for side in ("off", "on"):
        count = f"{side}_valid_frames"
        visit = patients.loc[patients.condition.eq(side.upper()), ["subject", "scan", "fold", "valid_frames", *coverage]]
        visit = visit.rename(columns={"subject": "patient_id", "scan": side+"_scan_id", "valid_frames": count,
                                      **{col: col+"_"+side for col in coverage}})
        keys = ["patient_id", side+"_scan_id", "fold", count]
        joined = joined.merge(visit, on=keys, how="left", validate="one_to_one", indicator=True)
        if not joined.pop("_merge").eq("both").all():
            raise UnverifiedInput(f"The {side.upper()} V6 patient, scan, fold or valid frame count does not reconcile")

    sides = [joined[[c+"_"+side for c in coverage]].to_numpy(float) for side in ("off", "on")]
    observed = np.concatenate(sides)
    if not np.isfinite(observed).all() or (observed < 0).any() or (observed > 1).any():
        raise UnverifiedInput("Frozen Coverage is not complete and bounded within 0–1")
    max_visit_error = float(np.max(np.abs(observed.sum(axis=1)-1)))
    delta = sides[1] - sides[0]
    max_delta_error = float(np.max(np.abs(delta.sum(axis=1))))
    if max(max_visit_error, max_delta_error) > tolerance:
        raise UnverifiedInput("The four Coverage fractions or paired changes violate compositional closure")
    for idx, col in enumerate(coverage):
        joined["delta_"+col] = delta[:, idx]
    return joined.sort_values("patient_id", ignore_index=True), max_visit_error, max_delta_error


def load_inputs(workspace, module, *, coverage_tolerance=1e-12, expected_patients=25):
    """Return private paired values, release-safe audit counts, and the raw file lock."""
    workspace, module = Path(workspace).resolve(), Path(module).resolve()
    if not module.is_relative_to(workspace):
        raise UnverifiedInput("The active module is outside the authorized workspace")
    if not np.isfinite(coverage_tolerance) or coverage_tolerance <= 0:
        raise ValueError("Coverage tolerance must be positive and finite")

    first_receipt_path = module / "results/Run_Manifest.json"
    round2_receipt_path = module / "round2/results/Round2_Run_Manifest.json"
    first, round2 = _json(first_receipt_path), _json(round2_receipt_path)
    used = {str(path): _digest(path) for path in (first_receipt_path, round2_receipt_path)}
    path, round2_lock = _read_round2_lock(workspace, module, round2)
    used[str(path)] = _digest(path)
    pair_file, pair_hash = _locked_source(round2_lock, MAIN_PAIRS, workspace, module)
    first_lock_file, first_lock_hash = _locked_source(round2_lock, FIRST_LOCK, workspace, module)
    used.update({str(pair_file): pair_hash, str(first_lock_file): first_lock_hash})
    first_lock = _json(first_lock_file)

    v6 = module.parent / V6
    old_hashes = first["imaging_fingerprints"]
    originals = {
        "Frozen_Cohort_Features.csv": V6 + "/private/results/Frozen_Cohort_Features.csv",
        "ON_OFF_Pair_Manifest.csv": V6 + "/private/manifests/ON_OFF_Pair_Manifest.csv",
        "input_manifest.csv": "WEiDA_STN25_HC15_K4_20261117/input_manifest.csv",
        "Fold_Templates.csv": "WEiDA-DBS-V7/V4_Microstate_Statistical_Significance/results/Fold_Templates.csv",
    }
    sources = {key: _locked_source(first_lock, rel, workspace) for key, rel in originals.items()}
    used.update({str(path): digest for path, digest in sources.values()})
    old_round2 = round2["checks"]["v6_sources"]["verified_sha256"]
    if any(old_round2[k] != digest for k, (_, digest) in sources.items()):
        raise UnverifiedInput("The frozen V6 sources no longer match the released round-two provenance")
    if any(old_hashes[key] != sources[key][1] for key in ("Frozen_Cohort_Features.csv", "ON_OFF_Pair_Manifest.csv")):
        raise UnverifiedInput("V6 feature and patient roster hashes differ from the first-round receipt")
    protocol_path, audit_path = v6 / "configs/protocol.json", v6 / "manifests/Input_Audit.json"
    _tracked_source(protocol_path, old_hashes["v6_protocol_sha256"], used)
    _tracked_source(audit_path, old_round2["V6 Input_Audit.json"], used)
    dictionary_path = module / "results/Score_Item_Dictionary.csv"
    _tracked_source(dictionary_path, first["public_outputs_sha256"]["results/Score_Item_Dictionary.csv"], used)

    protocol, receipt = _json(protocol_path), _json(audit_path)
    if (protocol["baseline"] != BASELINE or protocol["cohort"]["pd_pairs"] != expected_patients
            or protocol["cohort"]["acquired_frames"] != 210
            or receipt["protocol_sha256"] != old_hashes["v6_protocol_sha256"]
            or receipt["template_sha256"] != sources["Fold_Templates.csv"][1]
            or not receipt["frozen_mapping_reused"] or not receipt["original_metric_parity"]
            or receipt["duplicated_scan_ids"] != 0 or receipt["labels"]["count"] != receipt["scans"]):
        raise UnverifiedInput("V6 recorded protocol, already-aligned state axis or original-window receipt differs")
    templates = pd.read_csv(sources["Fold_Templates.csv"][0], usecols=["fold", "state", "original_state", "stable"])
    groups = templates.groupby("fold", sort=True)
    if (list(groups.groups) != list(range(5)) or not templates.stable.eq(True).all()
            or any(tuple(sorted(group[key])) != STATES for _, group in groups for key in ("state", "original_state"))):
        raise UnverifiedInput("V6 already-aligned five-fold state-numbering archive is incomplete")

    scores, eligibility = _read_clinical(pair_file)
    dictionary = pd.read_csv(dictionary_path, dtype=str)
    if len(dictionary) != 33 or not dictionary["item"].is_unique or not dictionary["column"].is_unique:
        raise UnverifiedInput("The original 33-item manual score dictionary differs")
    columns = ["scan", "subject", "condition", "fold", "valid_frames", "input_frames", "input_missing_initial_frames",
               *[f"coverage_{i}" for i in STATES]]
    cohort = pd.read_csv(sources["Frozen_Cohort_Features.csv"][0], usecols=columns, float_precision="round_trip")
    counts = cohort.condition.value_counts().to_dict()
    if (counts != {"ON": expected_patients, "OFF": expected_patients, "HC": protocol["cohort"]["hc"]}
            or len(cohort) != receipt["scans"] or len(cohort) != protocol["cohort"]["scans"]):
        raise UnverifiedInput("The V6 cohort inventory differs from the original frozen receipt")
    clock = pd.read_csv(sources["input_manifest.csv"][0],
                        usecols=["scan", "condition", "frames", "source_index_zero_based_start"]).set_index("scan")
    _audit_clock(cohort, clock, receipt)
    roster = pd.read_csv(sources["ON_OFF_Pair_Manifest.csv"][0], usecols=[
        "participant_id", "on_scan_id", "off_scan_id", "on_valid_frames", "off_valid_frames",
    ]).rename(columns={"participant_id": "patient_id"})
    pairs, max_visit_error, max_delta_error = _connect(scores, cohort, roster, expected_patients=expected_patients,
                                                       tolerance=coverage_tolerance)
    if set(pairs.fold) != set(range(5)):
        raise UnverifiedInput("The paired patient folds do not cover the frozen five-fold state axis")

    audit = {
        "patient_pairs": len(pairs), "verified_patient_scans": 2*len(pairs),
        "different_scan_keys_across_conditions": True, "paired_patient_scan_and_fold_keys_verified": True,
        "hc_in_correlations": 0, "source_alignment_reused_without_permutation": True,
        "original_common_window_and_clock_verified": True, "coverage_input_unit": "fraction 0–1",
        "coverage_closure_tolerance": coverage_tolerance, "max_abs_coverage_sum_error": max_visit_error,
        "max_abs_delta_coverage_sum_error": max_delta_error, "manual_total_item_count": len(dictionary),
        "clinical_eligibility": eligibility, "verified_raw_source_count": len(used),
        "input_collection_sha256": _collection(used),
        "verified_v6_sha256": {k: digest for k, (_, digest) in sources.items()},
    }
    return CoverageInputs(pairs, audit, used)
