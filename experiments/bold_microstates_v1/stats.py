"""Participant-level, exploratory statistics for BOLD spatial states.

Only independently documented DBS ON/OFF pairs enter inference.  The three
prespecified comparison families are all K x 3 condition effects, all K x 3
clinical-change associations, and K OFF-duration/clinical-score associations.
State numbers are computational labels, not EEG A--D identities.

The module deliberately does not discover subject identities, pick significant
states, combine differently named clinical scales, or treat TRs as participants.
Sign-flipping requires paired-label exchangeability (or symmetric null
differences); it does not remove confounding by fixed acquisition order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata


METRICS = ("coverage", "occurrence_per_min", "duration_sec")
METADATA_REQUIRED = (
    "run_id", "subject_id", "cohort", "condition", "mapping_source"
)
FAMILY_ON_OFF = "on_off"
FAMILY_CLINICAL_CHANGE = "clinical_change"
FAMILY_CLINICAL_OFF = "clinical_off_duration"


def _string(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _finite_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _reasons(values: Sequence[str]) -> str:
    return ";".join(dict.fromkeys(value for value in values if value))


def _is_constant(values: np.ndarray) -> bool:
    """Ignore rank variation created solely by floating-point roundoff."""
    if len(values) < 2:
        return True
    scale = float(np.max(np.abs(values)))
    return bool(np.ptp(values) <= 16 * np.finfo(float).eps * scale)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm-adjust a *fixed* family; unavailable tests occupy a p=1 slot.

    Returned entries for unavailable tests remain NaN.  Their planned slots are
    still included in every multiplier, so missing endpoints never shrink the
    comparison family.  Arbitrary dependence between endpoints is allowed.
    """
    original = np.asarray(p_values, dtype=float)
    if original.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if not len(original):
        return original.copy()
    available = np.isfinite(original)
    if np.any((original[available] < 0) | (original[available] > 1)):
        raise ValueError("finite p-values must lie between 0 and 1")
    filled = np.where(available, original, 1.0)
    order = np.argsort(filled, kind="stable")
    adjusted_ordered = np.minimum(
        1.0,
        np.maximum.accumulate(filled[order] * np.arange(len(filled), 0, -1)),
    )
    adjusted = np.empty_like(filled)
    adjusted[order] = adjusted_ordered
    adjusted[~available] = np.nan
    return adjusted


def signflip_pvalue(
    differences: Sequence[float],
    n_permutations: int = 10000,
    seed: int | np.random.Generator = 20261121,
) -> float:
    """Two-sided Monte Carlo sign-flip p for the paired mean difference.

    This numerical helper accepts any nonempty, finite vector; analyze() applies
    the inferential minimum of three independent participant pairs.  All-zero
    differences have p=1.  Constant nonzero differences also have a defined
    sign-flip test even though paired Cohen's dz is undefined.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 1 or not len(d) or not np.all(np.isfinite(d)):
        raise ValueError("differences must be a nonempty finite vector")
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    scale = float(np.max(np.abs(d)))
    if scale == 0:
        return 1.0
    # Scaling avoids overflow and makes the rounding tolerance scale invariant.
    d = d / scale
    observed = abs(float(np.mean(d)))
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    count = 0
    tolerance = 16 * np.finfo(float).eps
    for start in range(0, n_permutations, 4096):
        size = min(4096, n_permutations - start)
        signs = rng.integers(0, 2, size=(size, len(d)), dtype=np.int8) * 2 - 1
        simulated = np.abs(np.mean(signs * d, axis=1))
        count += int(np.count_nonzero(simulated >= observed - tolerance))
    return float((count + 1) / (n_permutations + 1))


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if _is_constant(x) or _is_constant(y):
        return np.nan
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    return float(np.clip(np.dot(rx, ry) / denominator, -1, 1)) if denominator else np.nan


def _spearman_permutation_p(
    x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> float:
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator == 0:
        return np.nan
    observed = abs(float(np.dot(rx, ry) / denominator))
    count = 0
    tolerance = 16 * np.finfo(float).eps
    for start in range(0, n_permutations, 2048):
        size = min(2048, n_permutations - start)
        # Independent random orders shuffle the clinical outcomes across people.
        order = np.argsort(rng.random((size, len(y))), axis=1)
        simulated = np.abs(np.sum(ry[order] * rx, axis=1) / denominator)
        count += int(np.count_nonzero(simulated >= observed - tolerance))
    return float((count + 1) / (n_permutations + 1))


def _bootstrap_mean(
    d: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, int]:
    if n_bootstrap == 0:
        return np.nan, np.nan, 0
    indices = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    values = np.mean(d[indices], axis=1)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), n_bootstrap


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[float, float, int]:
    if n_bootstrap == 0:
        return np.nan, np.nan, 0
    # Resample each participant's x/y pair together, never individual TRs.
    indices = rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    rx = rankdata(x[indices], axis=1).astype(float)
    ry = rankdata(y[indices], axis=1).astype(float)
    rx -= np.mean(rx, axis=1, keepdims=True)
    ry -= np.mean(ry, axis=1, keepdims=True)
    denominator = np.linalg.norm(rx, axis=1) * np.linalg.norm(ry, axis=1)
    usable = denominator > 0
    if not np.any(usable):
        return np.nan, np.nan, 0
    values = np.sum(rx[usable] * ry[usable], axis=1) / denominator[usable]
    values = np.clip(values, -1, 1)
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high), int(np.count_nonzero(usable))


def _rng(seed: int, family: int, state: int, metric: int, purpose: int) -> np.random.Generator:
    # Separate streams make p-values invariant to n_bootstrap and other families.
    return np.random.default_rng(np.random.SeedSequence([seed, family, state, metric, purpose]))


def _validate_frame(frame: pd.DataFrame, required: Sequence[str], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    result = frame.copy(deep=True)
    missing = [column for column in required if column not in result.columns]
    if missing and len(result):
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    # Empty input remains auditable and does not require fabricated metadata.
    for column in missing:
        result[column] = pd.Series(dtype=object)
    return result


def _prepare_pairs(
    metrics_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    k: int,
    clinical_scale: str | None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, float]], pd.DataFrame, dict[str, Any]]:
    metrics = _validate_frame(metrics_df, ("run_id", "state", *METRICS), "metrics_df")
    metadata = _validate_frame(metadata_df, METADATA_REQUIRED, "metadata_df").reset_index(drop=True)
    for column in METADATA_REQUIRED:
        metadata[column] = metadata[column].map(_string)
    metadata["cohort"] = metadata["cohort"].str.upper()
    metadata["condition"] = metadata["condition"].str.upper()
    for column in ("clinical_scale", "clinical_score"):
        if column not in metadata:
            metadata[column] = "" if column == "clinical_scale" else np.nan
    metadata["clinical_scale"] = metadata["clinical_scale"].map(_string)
    metadata["clinical_score"] = metadata["clinical_score"].map(_finite_number)
    metrics["run_id"] = metrics["run_id"].map(_string)
    state_numbers = pd.to_numeric(metrics["state"], errors="coerce")
    state_valid = (
        np.isfinite(state_numbers)
        & state_numbers.between(1, k)
        & (state_numbers == np.floor(state_numbers))
    )
    metrics["state"] = state_numbers
    for metric in METRICS:
        metrics[metric] = metrics[metric].map(_finite_number)

    metric_runs = set(metrics["run_id"]) - {""}
    metadata_runs = set(metadata["run_id"]) - {""}
    duplicated_metadata = set(metadata.loc[
        metadata["run_id"].ne("") & metadata["run_id"].duplicated(keep=False), "run_id"
    ])
    valid_states = metrics.loc[state_valid]
    duplicate_metric_runs = set(valid_states.loc[
        valid_states.duplicated(["run_id", "state"], keep=False), "run_id"
    ])
    invalid_state_runs = set(metrics.loc[~state_valid, "run_id"])
    lookup: dict[tuple[str, int], dict[str, float]] = {}
    for row in valid_states.to_dict("records"):
        if row["run_id"] and row["run_id"] not in duplicate_metric_runs:
            lookup[(row["run_id"], int(row["state"]))] = {
                metric: row[metric] for metric in METRICS
            }

    metric_counts = metrics.groupby("run_id").size().to_dict()
    available_counts: dict[str, int] = {}
    for run_id in metric_runs:
        available_counts[run_id] = sum(
            np.isfinite(lookup.get((run_id, state), {}).get(metric, np.nan))
            for state in range(1, k + 1) for metric in METRICS
        )
    audit_rows: list[dict[str, Any]] = []
    initial_reasons: dict[int, list[str]] = {}
    for index, row in metadata.iterrows():
        reasons: list[str] = []
        run_id = row["run_id"]
        if not run_id:
            reasons.append("missing_run_id")
        if row["cohort"] != "DBS":
            reasons.append("not_dbs_cohort" if row["cohort"] else "missing_cohort")
        else:
            if not row["subject_id"]:
                reasons.append("missing_subject_id")
            if row["condition"] not in ("ON", "OFF"):
                reasons.append("unexpected_condition")
            if not row["mapping_source"]:
                reasons.append("missing_mapping_source")
        if run_id in duplicated_metadata:
            reasons.append("duplicate_run_id")
        if run_id not in metric_runs:
            reasons.append("no_metric_rows")
        if run_id in duplicate_metric_runs:
            reasons.append("duplicate_metric_state")
        if run_id in invalid_state_runs:
            reasons.append("invalid_metric_state")
        initial_reasons[index] = reasons
        audit_rows.append({
            "metadata_row": index,
            **{column: row[column] for column in (*METADATA_REQUIRED, "clinical_scale", "clinical_score")},
            "n_metric_rows": int(metric_counts.get(run_id, 0)),
            "n_available_metric_values": int(available_counts.get(run_id, 0)),
            "n_expected_metric_values": k * len(METRICS),
            "pair_eligible": False,
            "pair_status": "excluded",
            "pair_reason": _reasons(reasons),
            "paired_on_run_id": "",
            "paired_off_run_id": "",
            "clinical_eligible": False,
            "clinical_reason": "pair_not_eligible",
            "selected_clinical_scale": "",
        })

    pairs: list[dict[str, Any]] = []
    dbs = metadata.loc[metadata["cohort"].eq("DBS") & metadata["subject_id"].ne("")]
    for subject, rows in dbs.groupby("subject_id", sort=True):
        on_rows = rows.loc[rows["condition"].eq("ON")]
        off_rows = rows.loc[rows["condition"].eq("OFF")]
        reasons = [reason for index in rows.index for reason in initial_reasons[index]]
        if len(on_rows) != 1:
            reasons.append(f"on_run_count_{len(on_rows)}")
        if len(off_rows) != 1:
            reasons.append(f"off_run_count_{len(off_rows)}")
        if len(rows) != 2:
            reasons.append(f"dbs_metadata_row_count_{len(rows)}")
        on_id = on_rows.iloc[0]["run_id"] if len(on_rows) == 1 else ""
        off_id = off_rows.iloc[0]["run_id"] if len(off_rows) == 1 else ""
        if on_id and on_id == off_id:
            reasons.append("same_run_for_on_and_off")
        reason_string = _reasons(reasons)
        eligible = not reason_string
        for index in rows.index:
            audit_rows[index].update({
                "pair_eligible": eligible,
                "pair_status": "eligible" if eligible else "excluded",
                "pair_reason": reason_string,
                "paired_on_run_id": on_id,
                "paired_off_run_id": off_id,
            })
        if eligible:
            on = on_rows.iloc[0]
            off = off_rows.iloc[0]
            pairs.append({
                "subject_id": subject,
                "on_run_id": on_id,
                "off_run_id": off_id,
                "on_scale": on["clinical_scale"],
                "off_scale": off["clinical_scale"],
                "on_score": on["clinical_score"],
                "off_score": off["clinical_score"],
                "audit_indices": list(rows.index),
            })

    candidates = sorted({
        pair[field] for pair in pairs for field in ("on_scale", "off_scale") if pair[field]
    })
    if clinical_scale is not None:
        selected_scale = _string(clinical_scale)
        if not selected_scale:
            raise ValueError("clinical_scale must be a nonempty exact scale name or None")
        scale_status = "specified" if selected_scale in candidates else "requested_scale_not_observed"
    elif len(candidates) == 1:
        selected_scale = candidates[0]
        scale_status = "single_scale_selected"
    else:
        selected_scale = None
        scale_status = "ambiguous_clinical_scale" if candidates else "missing_clinical_scale"

    for pair in pairs:
        reasons = []
        if selected_scale is None:
            reasons.append(scale_status)
        else:
            if not pair["on_scale"] or not pair["off_scale"]:
                reasons.append("clinical_scale_missing")
            elif pair["on_scale"] != selected_scale or pair["off_scale"] != selected_scale:
                reasons.append("clinical_scale_mismatch")
        if not np.isfinite(pair["on_score"]) or not np.isfinite(pair["off_score"]):
            reasons.append("clinical_score_missing")
        pair["clinical_eligible"] = not reasons
        pair["clinical_reason"] = _reasons(reasons)
        for index in pair["audit_indices"]:
            audit_rows[index].update({
                "clinical_eligible": pair["clinical_eligible"],
                "clinical_reason": pair["clinical_reason"],
                "selected_clinical_scale": selected_scale or "",
            })

    for run_id in sorted(set(metrics["run_id"]) - metadata_runs):
        audit_rows.append({
            "metadata_row": np.nan,
            "run_id": run_id,
            "subject_id": "", "cohort": "", "condition": "", "mapping_source": "",
            "clinical_scale": "", "clinical_score": np.nan,
            "n_metric_rows": int(metric_counts.get(run_id, 0)),
            "n_available_metric_values": int(available_counts.get(run_id, 0)),
            "n_expected_metric_values": k * len(METRICS),
            "pair_eligible": False,
            "pair_status": "excluded",
            "pair_reason": "metadata_missing" if run_id else "missing_run_id",
            "paired_on_run_id": "", "paired_off_run_id": "",
            "clinical_eligible": False,
            "clinical_reason": "pair_not_eligible",
            "selected_clinical_scale": selected_scale or "",
        })
    audit_columns = [
        "metadata_row", *METADATA_REQUIRED, "clinical_scale", "clinical_score",
        "n_metric_rows", "n_available_metric_values", "n_expected_metric_values",
        "pair_eligible", "pair_status", "pair_reason", "paired_on_run_id", "paired_off_run_id",
        "clinical_eligible", "clinical_reason", "selected_clinical_scale",
    ]
    audit = pd.DataFrame(audit_rows, columns=audit_columns)
    scale_info = {
        "clinical_scale": selected_scale,
        "clinical_scale_status": scale_status,
        "clinical_scale_candidates": candidates,
    }
    return pairs, lookup, audit, scale_info


def _base_result(
    family: str, state: int, metric: str, family_size: int,
    n_pool: int, scale: str | None,
) -> dict[str, Any]:
    return {
        "family": family,
        "state": state,
        "metric": metric,
        "family_size": family_size,
        "clinical_scale": scale or "",
        "n_pool": n_pool,
        "n_subjects": 0,
        "n_missing_endpoint": n_pool,
        "subject_ids": "",
        "estimate_type": "mean_difference" if family == FAMILY_ON_OFF else "spearman_rho",
        "estimate": np.nan,
        "mean_on": np.nan,
        "mean_off": np.nan,
        "mean_delta": np.nan,
        "effect_dz": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "ci_method": "participant_bootstrap_percentile_95_marginal",
        "n_bootstrap_valid": 0,
        "p_raw": np.nan,
        "p_holm": np.nan,
        "n_permutations": 0,
        "status": "not_estimable",
        "reason": "",
        "comparison": (
            "ON - OFF" if family == FAMILY_ON_OFF
            else "metric(ON-OFF) vs score(OFF-ON)" if family == FAMILY_CLINICAL_CHANGE
            else "OFF duration vs OFF clinical score"
        ),
    }


def _correlation_result(
    row: dict[str, Any], x: np.ndarray, y: np.ndarray, subject_ids: list[str],
    n_permutations: int, n_bootstrap: int,
    perm_rng: np.random.Generator, boot_rng: np.random.Generator,
) -> dict[str, Any]:
    row.update({
        "n_subjects": len(x),
        "n_missing_endpoint": row["n_pool"] - len(x),
        "subject_ids": "|".join(subject_ids),
    })
    if len(x) < 3:
        row["reason"] = "insufficient_subjects"
        return row
    rho = _spearman_rho(x, y)
    if not np.isfinite(rho):
        row["reason"] = "constant_variable"
        return row
    low, high, valid_boot = _bootstrap_spearman(x, y, n_bootstrap, boot_rng)
    row.update({
        "estimate": rho,
        "p_raw": _spearman_permutation_p(x, y, n_permutations, perm_rng),
        "ci_low": low,
        "ci_high": high,
        "n_bootstrap_valid": valid_boot,
        "n_permutations": n_permutations,
        "status": "ok",
        "reason": "",
    })
    return row


def analyze(
    metrics_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str | Path,
    k: int = 4,
    n_permutations: int = 10000,
    n_bootstrap: int = 2000,
    seed: int = 20261121,
    clinical_scale: str | None = None,
) -> dict[str, Any]:
    """Write stats.csv and pairing_audit.tsv, returning a JSON-safe summary.

    Each DBS participant must have exactly one evidenced ON and one OFF run.
    Duplicate run IDs or duplicate run/state metrics invalidate the affected
    pair; this function never silently deduplicates or aggregates them.

    Clinical analyses require the same explicitly named scale and finite scores
    in both conditions.  If multiple scale names occur, clinical_scale must be
    selected explicitly; UPDRS is not silently renamed to UPDRS-III.  Missing
    clinical data never exclude otherwise valid pairs from condition effects.

    Family 3 uses the clinically complete, validated pair pool but only requires
    OFF duration to be finite.  Its actual n can therefore differ from family 2.
    All endpoint-specific sample sizes and exclusions are reported.
    """
    if not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be a nonnegative integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    k, n_permutations, n_bootstrap, seed = map(int, (k, n_permutations, n_bootstrap, seed))
    pairs, lookup, audit, scale_info = _prepare_pairs(metrics_df, metadata_df, k, clinical_scale)
    clinical_pairs = [pair for pair in pairs if pair["clinical_eligible"]]
    selected_scale = scale_info["clinical_scale"]
    rows: list[dict[str, Any]] = []

    def value(pair: dict[str, Any], condition: str, state: int, metric: str) -> float:
        return lookup.get((pair[f"{condition}_run_id"], state), {}).get(metric, np.nan)

    for state in range(1, k + 1):
        for metric_index, metric in enumerate(METRICS):
            row = _base_result(FAMILY_ON_OFF, state, metric, k * 3, len(pairs), None)
            complete = [
                pair for pair in pairs
                if np.isfinite(value(pair, "on", state, metric))
                and np.isfinite(value(pair, "off", state, metric))
            ]
            on = np.asarray([value(pair, "on", state, metric) for pair in complete], dtype=float)
            off = np.asarray([value(pair, "off", state, metric) for pair in complete], dtype=float)
            d = on - off
            row.update({
                "n_subjects": len(complete),
                "n_missing_endpoint": len(pairs) - len(complete),
                "subject_ids": "|".join(pair["subject_id"] for pair in complete),
            })
            if len(complete):
                row.update({
                    "mean_on": float(np.mean(on)),
                    "mean_off": float(np.mean(off)),
                    "mean_delta": float(np.mean(d)),
                    "estimate": float(np.mean(d)),
                })
            if len(complete) < 3:
                row["reason"] = "insufficient_pairs"
            else:
                low, high, valid_boot = _bootstrap_mean(d, n_bootstrap, _rng(seed, 1, state, metric_index, 1))
                sd = float(np.std(d, ddof=1))
                all_zero = bool(np.all(d == 0))
                constant = _is_constant(d)
                row.update({
                    "p_raw": signflip_pvalue(d, n_permutations, _rng(seed, 1, state, metric_index, 0)),
                    "ci_low": low,
                    "ci_high": high,
                    "n_bootstrap_valid": valid_boot,
                    "effect_dz": float(np.mean(d) / sd) if sd > 0 and not constant else np.nan,
                    "n_permutations": n_permutations,
                    "status": "ok",
                    "reason": "all_zero_difference" if all_zero else "constant_nonzero_difference" if constant else "",
                })
            rows.append(row)

    for state in range(1, k + 1):
        for metric_index, metric in enumerate(METRICS):
            row = _base_result(FAMILY_CLINICAL_CHANGE, state, metric, k * 3, len(clinical_pairs), selected_scale)
            complete = [
                pair for pair in clinical_pairs
                if np.isfinite(value(pair, "on", state, metric))
                and np.isfinite(value(pair, "off", state, metric))
            ]
            x = np.asarray([
                value(pair, "on", state, metric) - value(pair, "off", state, metric)
                for pair in complete
            ], dtype=float)
            y = np.asarray([pair["off_score"] - pair["on_score"] for pair in complete], dtype=float)
            row = _correlation_result(
                row, x, y, [pair["subject_id"] for pair in complete], n_permutations, n_bootstrap,
                _rng(seed, 2, state, metric_index, 0), _rng(seed, 2, state, metric_index, 1),
            )
            if not clinical_pairs:
                row["reason"] = scale_info["clinical_scale_status"] if selected_scale is None else "no_complete_clinical_pairs"
            rows.append(row)

    for state in range(1, k + 1):
        row = _base_result(FAMILY_CLINICAL_OFF, state, "duration_sec", k, len(clinical_pairs), selected_scale)
        complete = [pair for pair in clinical_pairs if np.isfinite(value(pair, "off", state, "duration_sec"))]
        x = np.asarray([value(pair, "off", state, "duration_sec") for pair in complete], dtype=float)
        y = np.asarray([pair["off_score"] for pair in complete], dtype=float)
        row = _correlation_result(
            row, x, y, [pair["subject_id"] for pair in complete], n_permutations, n_bootstrap,
            _rng(seed, 3, state, 2, 0), _rng(seed, 3, state, 2, 1),
        )
        if not clinical_pairs:
            row["reason"] = scale_info["clinical_scale_status"] if selected_scale is None else "no_complete_clinical_pairs"
        rows.append(row)

    results = pd.DataFrame(rows)
    for family in (FAMILY_ON_OFF, FAMILY_CLINICAL_CHANGE, FAMILY_CLINICAL_OFF):
        index = results.index[results["family"].eq(family)]
        results.loc[index, "p_holm"] = holm_adjust(results.loc[index, "p_raw"].to_numpy(dtype=float))

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    stats_path = destination / "stats.csv"
    audit_path = destination / "pairing_audit.tsv"
    results.to_csv(stats_path, index=False)
    audit.to_csv(audit_path, sep="\t", index=False)
    families = {
        family: {
            "planned_tests": int(len(group)),
            "estimable_tests": int(group["p_raw"].notna().sum()),
            "holm_significant_tests": int(group["p_holm"].lt(0.05).sum()),
        }
        for family, group in results.groupby("family", sort=False)
    }
    warnings = []
    if not pairs:
        warnings.append("No DBS subject had an eligible, explicitly documented ON/OFF pair; see pairing_audit.tsv.")
    if not clinical_pairs:
        warnings.append("No clinical association was estimable from a complete same-scale clinical pair pool.")
    if scale_info["clinical_scale_status"] == "ambiguous_clinical_scale":
        warnings.append("Multiple clinical-scale names were observed; select one exact name with clinical_scale.")
    return {
        "status": "complete" if pairs else "no_eligible_dbs_pairs",
        "k": k,
        "seed": seed,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "n_eligible_pairs": len(pairs),
        "n_clinically_eligible_pairs": len(clinical_pairs),
        **scale_info,
        "families": families,
        "stats_path": str(stats_path),
        "pairing_audit_path": str(audit_path),
        "warnings": warnings,
    }
