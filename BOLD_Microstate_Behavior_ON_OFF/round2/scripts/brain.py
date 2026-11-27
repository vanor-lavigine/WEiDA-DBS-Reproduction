"""Audit frozen duration sources and summarize M2/M4 paired changes."""

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATES = (2, 4)
PATTERNS = {
    (-1, -1): "both_shorter",
    (-1, 1): "only_m2_shorter",
    (1, -1): "only_m4_shorter",
    (1, 1): "both_longer",
}
KEYS = ["patient_id", "on_scan_id", "off_scan_id"]
TR_SECONDS = 2.1
SECS_TOL = 1e-12


class UnverifiedDurationInput(ValueError):
    """A needed local frozen-duration source is missing or cannot be verified."""


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _locked(input_hashes, workspace, relative_suffix):
    suffix = Path(relative_suffix).parts
    candidates = [(Path(path).resolve(), expected) for path, expected in input_hashes.items()
                  if Path(path).parts[-len(suffix):] == suffix]
    if len(candidates) != 1:
        raise UnverifiedDurationInput(f"Expected one legacy hash entry for {suffix[-1]}")
    path, expected = candidates[0]
    if not path.is_relative_to(workspace) or not path.is_file():
        raise UnverifiedDurationInput(f"Locked source unavailable in the permitted worktree family: {suffix[-1]}")
    if _digest(path) != expected:
        raise UnverifiedDurationInput(f"Frozen source digest differs from prior manifest: {suffix[-1]}")
    return path, expected


def _audit_pairs(pairs, cohort, manifest):
    if not all(pairs[key].notna().all() and pairs[key].is_unique for key in KEYS):
        raise UnverifiedDurationInput("Private patient and condition keys must be populated and unique")
    if not all(manifest[key].notna().all() and manifest[key].is_unique for key in KEYS):
        raise UnverifiedDurationInput("Frozen V6 roster keys must be populated and unique")
    if not pairs["fold"].notna().all():
        raise UnverifiedDurationInput("Private pair records are missing a frozen fold")
    supplied = pairs.set_index("patient_id", verify_integrity=True).sort_index()
    expected = manifest.set_index("patient_id", verify_integrity=True).sort_index()
    if not supplied[KEYS[1:]].equals(expected[KEYS[1:]]):
        raise UnverifiedDurationInput("Private pair records do not exactly match the frozen condition visits")
    visits = cohort.set_index("scan", verify_integrity=True)
    frozen_folds = []
    for side in ("on", "off"):
        ids = expected[f"{side}_scan_id"].tolist()
        archived = visits.loc[ids]
        selected = cohort.loc[cohort.condition.eq(side.upper())]
        if (set(ids) != set(selected.scan) or not archived.condition.eq(side.upper()).all()
                or not np.array_equal(archived.subject, expected.index)
                or not np.array_equal(archived.valid_frames, expected[f"{side}_valid_frames"])):
            raise UnverifiedDurationInput("Private pair roster disagrees with frozen visit conditions or clock")
        frozen_folds.append(archived.fold)
        if not np.array_equal(archived.fold, supplied.fold):
            raise UnverifiedDurationInput("Private or frozen visit folds disagree")
        for state in STATES:
            actual = supplied[f"duration_{state}_{side}"].to_numpy(float)
            if not np.allclose(actual, archived[f"duration_{state}"], atol=SECS_TOL, rtol=0):
                raise UnverifiedDurationInput(f"Private M{state} duration does not match the original V6 common window")
    if not np.array_equal(*frozen_folds):
        raise UnverifiedDurationInput("ON/OFF visits do not share the same frozen fold")


def load_hc(input_hashes, workspace, module, *, pairs=None, private_lock=None):
    """Read the locked V6 reference and verify source, condition, state and clock."""
    workspace, module = Path(workspace).resolve(), Path(module).resolve()
    specs = {
        "Frozen_Cohort_Features.csv": "V6_DBS_ON_OFF_Microstate_Dynamics/private/results/Frozen_Cohort_Features.csv",
        "ON_OFF_Pair_Manifest.csv": "V6_DBS_ON_OFF_Microstate_Dynamics/private/manifests/ON_OFF_Pair_Manifest.csv",
        "input_manifest.csv": "WEiDA_STN25_HC15_K4_20261117/input_manifest.csv",
        "Fold_Templates.csv": "WEiDA-DBS-V7/V4_Microstate_Statistical_Significance/results/Fold_Templates.csv",
    }
    sources = {key: _locked(input_hashes, workspace, name) for key, name in specs.items()}
    v6 = module.parent / "V6_DBS_ON_OFF_Microstate_Dynamics"
    protocol_path = v6 / "configs/protocol.json"
    receipt_path = v6 / "manifests/Input_Audit.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    prior = json.loads((module / "results/Run_Manifest.json").read_text(encoding="utf-8-sig"))
    protocol_digest = _digest(protocol_path)
    expected_digests = prior["imaging_fingerprints"]
    for name in ("Frozen_Cohort_Features.csv", "ON_OFF_Pair_Manifest.csv"):
        if sources[name][1] != expected_digests[name]:
            raise UnverifiedDurationInput(f"Existing public run receipt disagrees with the private {name} digest")
    if (protocol_digest != receipt["protocol_sha256"] or protocol_digest != expected_digests["v6_protocol_sha256"]
            or sources["Fold_Templates.csv"][1] != receipt["template_sha256"]
            or not receipt["frozen_mapping_reused"] or not receipt["original_metric_parity"]
            or receipt["duplicated_scan_ids"] != 0 or receipt["labels"]["count"] != 65):
        raise UnverifiedDurationInput("Frozen V6 source or aligned label audit differs from its receipt")
    required_baseline = {
        "method": "A3", "atlas": "AAL78", "k": 4, "tr": TR_SECONDS, "band_hz": [0.01, 0.08], "band": "F1",
        "normalization": "N0", "learner": "G1", "decoder": "S1", "lambda": 0.04,
    }
    if protocol["baseline"] != required_baseline or protocol["cohort"] != {
        "hc": 15, "pd_pairs": 25, "scans": 65, "acquired_frames": 210
    }:
        raise UnverifiedDurationInput("V6 protocol no longer matches the locked A3 common-window cohort")

    raw_templates = pd.read_csv(sources["Fold_Templates.csv"][0], usecols=["fold", "state", "original_state", "stable"])
    axis = raw_templates.groupby("fold", sort=True)
    if (list(axis.groups) != list(range(5)) or len(raw_templates) != 20 or not raw_templates.stable.eq(True).all()
            or any(tuple(sorted(frame[field])) != (1, 2, 3, 4)
                   for _, frame in axis for field in ("state", "original_state"))):
        raise UnverifiedDurationInput("The frozen five-fold alignment is incomplete")
    fold_zero = raw_templates.loc[raw_templates.fold.eq(0)].sort_values("original_state")
    if not np.array_equal(fold_zero.state, fold_zero.original_state):
        raise UnverifiedDurationInput("Aligned state anchor fold has changed")
    local_templates = module.parent / "V4_Microstate_Statistical_Significance/results/Fold_Templates.csv"
    if _digest(local_templates) != sources["Fold_Templates.csv"][1]:
        raise UnverifiedDurationInput("Worktree template map differs from the already aligned V6 template")

    cols = ["scan", "subject", "condition", "fold", "valid_frames", "input_frames", "input_missing_initial_frames",
            "coverage_4", "occurrence_4", "duration_2", "duration_4"]
    cohort = pd.read_csv(sources["Frozen_Cohort_Features.csv"][0], usecols=cols, float_precision="round_trip")
    counts = cohort.condition.value_counts().to_dict()
    if (counts != {"ON": 25, "OFF": 25, "HC": 15} or len(cohort) != receipt["scans"]
            or not cohort.scan.notna().all() or not cohort.scan.is_unique
            or not cohort.subject.notna().all() or cohort.subject.nunique() != 40
            or not cohort.groupby("subject").fold.nunique().eq(1).all()
            or set(cohort.fold) != set(range(5))):
        raise UnverifiedDurationInput("The frozen HC and patient cohort inventory differs")
    fold_counts = cohort.groupby(["condition", "fold"]).size()
    if (any(fold_counts.get(("HC", fold), 0) != 3 for fold in range(5))
            or any(fold_counts.get((condition, fold), 0) != 5 for condition in ("ON", "OFF") for fold in range(5))):
        raise UnverifiedDurationInput("The frozen per-fold cohort composition differs")

    clock = pd.read_csv(sources["input_manifest.csv"][0], usecols=[
        "scan", "condition", "frames", "source_index_zero_based_start"
    ]).set_index("scan", verify_integrity=True)
    if set(clock.index) != set(cohort.scan):
        raise UnverifiedDurationInput("Frozen scan set differs from the original 210-TR inventory")
    ordered = clock.loc[cohort.scan]
    if (not np.array_equal(cohort.condition, ordered.condition.map({1: "HC", 2: "OFF", 3: "ON"}))
            or not np.array_equal(cohort.input_missing_initial_frames, ordered.source_index_zero_based_start)
            or not np.array_equal(cohort.input_frames, ordered.frames)
            or not np.array_equal(ordered.frames, 210-ordered.source_index_zero_based_start)
            or not ((cohort.valid_frames > 0) & (cohort.valid_frames <= cohort.input_frames)).all()
            or int(cohort.valid_frames.sum()) != receipt["valid_trs"]):
        raise UnverifiedDurationInput("The frozen condition keys or valid TR counts disagree")
    if not ((cohort.occurrence_4 > 0) & (cohort.coverage_4 > 0)).all():
        raise UnverifiedDurationInput("A frozen M4 record has no observed episodes")
    seconds = cohort.coverage_4 / cohort.occurrence_4
    if (not np.allclose(seconds, cohort.duration_4, atol=SECS_TOL, rtol=0)
            or (cohort.duration_4 < TR_SECONDS-SECS_TOL).any()):
        raise UnverifiedDurationInput("M4 seconds disagree with the locked occupancy/rate identity")
    hc = cohort.loc[cohort.condition.eq("HC")].sort_values("scan")
    if not hc.subject.is_unique:
        raise UnverifiedDurationInput("Each HC must contribute one independent measurement")
    if pairs is not None:
        manifest = pd.read_csv(sources["ON_OFF_Pair_Manifest.csv"][0], usecols=[
            "participant_id", "on_scan_id", "off_scan_id", "on_valid_frames", "off_valid_frames",
        ]).rename(columns={"participant_id": "patient_id"})
        _audit_pairs(pairs, cohort, manifest)

    if private_lock is not None:
        private_lock.update({str(path): expected for path, expected in sources.values()})
        private_lock.update({str(path): _digest(path) for path in (protocol_path, receipt_path, local_templates)})
    audit = {
        "source": "V6 Frozen_Cohort_Features.csv, condition HC, aligned duration_4 on the original common valid window",
        "unit": "seconds",
        "n_hc": len(hc),
        "n_verified_patient_visit_pairs": len(pairs) if pairs is not None else None,
        "original_clock_tr": 210,
        "folds": 5,
        "hc_per_fold": 3,
        "original_alignment_checked_without_second_mapping": True,
        "max_abs_duration_seconds_algebra_error": float(np.max(np.abs(seconds-cohort.duration_4))),
        "verified_sha256": {key: expected for key, (_, expected) in sources.items()}
        | {"V6 protocol.json": protocol_digest, "V6 Input_Audit.json": _digest(receipt_path)},
    }
    return hc.duration_4.to_numpy(float), audit


def _summary(values, prefix):
    median, q1, q3 = np.percentile(values, [50, 25, 75])
    return {
        f"{prefix}mean_s": float(np.mean(values)),
        f"{prefix}median_s": float(median),
        f"{prefix}q1_s": float(q1),
        f"{prefix}q3_s": float(q3),
    }


def _sign(values, tolerance):
    return np.where(values > tolerance, 1, np.where(values < -tolerance, -1, 0))


def duration_patterns(pairs, duration_results, tolerance):
    """Verify first-round duration changes and expose all registered direction combinations."""
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("A nonnegative finite zero tolerance is required")
    if len(pairs) != 25 or not all(pairs[key].notna().all() and pairs[key].is_unique for key in KEYS):
        raise ValueError("Expected exactly 25 unique frozen patient and visit keys")
    if set(pairs.on_scan_id) & set(pairs.off_scan_id):
        raise ValueError("A scan occurs in both condition columns")
    ordered = pairs.sort_values("patient_id", ignore_index=True)
    baseline = duration_results.set_index("endpoint", verify_integrity=True)
    private = ordered[KEYS].copy()
    stats_rows, signs = [], []
    for state in STATES:
        row = baseline.loc[f"M{state}"]
        off = ordered[f"duration_{state}_off"].to_numpy(float)
        on = ordered[f"duration_{state}_on"].to_numpy(float)
        change = on-off
        stored = ordered[f"duration_{state}_change"].to_numpy(float)
        if (not np.isfinite(np.r_[on, off]).all()
                or not np.allclose(change, stored, rtol=0, atol=SECS_TOL)):
            raise ValueError("Original private duration entries are incomplete or change direction differs")
        direction = _sign(change, tolerance)
        shorter, same, longer = (int((direction == code).sum()) for code in (-1, 0, 1))
        if (row.n != len(change) or row.unit != "seconds" or row.direction != "ON minus OFF"
                or not np.allclose([on.mean(), off.mean(), change.mean()], [row.on_mean, row.off_mean, row.mean_change],
                                   rtol=0, atol=SECS_TOL)
                or (shorter, same, longer) != (row.n_decreased, row.n_unchanged, row.n_increased)):
            raise ValueError("Private duration signs, means or units differ from the public frozen result")
        label = f"m{state}"
        private[f"{label}_off_s"], private[f"{label}_on_s"], private[f"delta_{label}_s"] = off, on, change
        private[f"{label}_direction"] = np.select([direction < 0, direction > 0], ["shorter", "longer"], "unchanged")
        signs.append(direction)
        stats_rows.append(dict(state=f"M{state}", n_pairs=len(change), **_summary(change, ""),
                               shorter=shorter, unchanged=same, longer=longer))
    private["pattern"] = [PATTERNS.get(key, "zero_in_either") for key in zip(*signs, strict=True)]
    totals = private.pattern.value_counts()
    patterns = pd.DataFrame([
        dict(pattern=key, count=int(totals.get(key, 0)), proportion=totals.get(key, 0)/len(private), n_pairs=len(private))
        for key in [*PATTERNS.values(), "zero_in_either"]
    ])
    states = pd.DataFrame(stats_rows)
    if patterns["count"].sum() != len(private):
        raise ValueError("Change combinations do not sum to the number of pairs")
    zero_subset = private.loc[private.pattern.eq("zero_in_either")]
    for i, row in enumerate(states.itertuples()):
        for code, label in ((-1, "shorter"), (0, "unchanged"), (1, "longer")):
            category_total = sum(totals.get(name, 0) for condition, name in PATTERNS.items() if condition[i] == code)
            zero_subset_total = zero_subset[f"{row.state.lower()}_direction"].eq(label).sum()
            if category_total + zero_subset_total != getattr(row, label):
                raise ValueError("Joint patterns and marginal direction counts disagree")
    return patterns, states, private


def distance_change(off, on, center):
    """Positive within-patient difference means reduction in absolute distance."""
    before, after = np.abs(off-center), np.abs(on-center)
    return before, after, before-after


def bootstrap_distance(hc, off, on, *, draws, seed):
    """Resample independent controls and patient pairs and re-estimate one center per draw."""
    if (not len(hc) or not len(off) or off.shape != on.shape or draws < 1
            or not np.isfinite(np.r_[hc, off, on]).all()):
        raise ValueError("Bootstrap requires finite complete patient pairs and independent controls")
    rng = np.random.default_rng(seed)
    hc_draws = rng.integers(0, len(hc), (draws, len(hc)))
    pair_draws = rng.integers(0, len(off), (draws, len(off)))
    center = hc[hc_draws].mean(axis=1, keepdims=True)
    _, _, values = distance_change(off[pair_draws], on[pair_draws], center)
    return values.mean(axis=1)


def m4_reference(private, hc_durations, *, tolerance, draws, seed):
    """Estimate the M4 absolute distance change and maintain a separate protected audit."""
    hc = np.asarray(hc_durations, float)
    if hc.shape != (15,) or not np.isfinite(hc).all():
        raise ValueError("Exactly 15 finite frozen HC durations are required")
    off, on = (private[f"m4_{side}_s"].to_numpy(float) for side in ("off", "on"))
    center = float(np.mean(hc))
    before, after, change = distance_change(off, on, center)
    direction = _sign(change, tolerance)
    bootstrap = bootstrap_distance(hc, off, on, draws=draws, seed=seed)
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    row = {
        "n_hc": len(hc), "n_pairs": len(off), "hc_center_s": center, **_summary(hc, "hc_"),
        "off_duration_mean_s": float(np.mean(off)), "on_duration_mean_s": float(np.mean(on)),
        "off_mean_duration_distance_to_hc_s": float(abs(np.mean(off)-center)),
        "on_mean_duration_distance_to_hc_s": float(abs(np.mean(on)-center)),
        **_summary(before, "off_distance_"), **_summary(after, "on_distance_"),
        **_summary(change, "distance_change_"), "n_closer": int((direction > 0).sum()),
        "n_unchanged": int((direction == 0).sum()), "n_farther": int((direction < 0).sum()),
        "ci95_low_s": float(lo), "ci95_high_s": float(hi),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
    }
    individual = private.assign(
        off_distance_s=before, on_distance_s=after, distance_change_s=change,
        distance_direction=np.select([direction > 0, direction < 0], ["closer", "farther"], "unchanged"),
    )
    audit = {
        "distance_definition": "abs(M4_OFF - HC mean) minus abs(M4_ON - same HC mean), seconds",
        "primary_estimand": "mean within-patient distance reduction",
        "distance_zero_tolerance_seconds_only_for_counts": tolerance,
        "n_closer_and_m4_shorter": int(((direction > 0) & (private.delta_m4_s < -tolerance)).sum()),
        "n_m4_shorter": int((private.delta_m4_s < -tolerance).sum()),
        "interval": "95% bootstrap percentile with joint patient pairs and resampled HC mean in each draw",
        "draws": draws,
        "seed": seed,
    }
    return pd.DataFrame([row]), individual, audit


def brain(pairs, duration_results, hc_durations, *, tolerance, draws, seed):
    patterns, states, private = duration_patterns(pairs, duration_results, tolerance)
    reference, individual, audit = m4_reference(private, hc_durations, tolerance=tolerance, draws=draws, seed=seed)
    return patterns, states, reference, individual, audit
