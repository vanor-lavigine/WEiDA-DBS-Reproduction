"""Numerical core for exploratory, microstate-inspired BOLD spatial states.

The signed analysis keeps opposite BOLD spatial configurations distinct. The
unsigned analysis is an optional EEG-style, polarity-invariant comparison; its
classes are axes, not directed BOLD patterns. Neither analysis identifies EEG
microstates or establishes that a selected number of clusters is biologically
unique. Labels are zero-based internally; reported state numbers are one-based.

Only NumPy and SciPy are required. These functions do not modify their inputs.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.linalg import eigh


def _matrix(values: Any, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError("{} must be a nonempty two-dimensional array".format(name))
    return out


def _polarity(value: str) -> str:
    if value not in ("signed", "unsigned"):
        raise ValueError("polarity must be 'signed' or 'unsigned'")
    return value


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("{} must be a positive integer".format(name))
    if value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _canonical_axis(vector: np.ndarray) -> np.ndarray:
    """Choose a reproducible global sign, without claiming to fix degeneracy."""
    result = vector.copy()
    pivot = int(np.argmax(np.abs(result)))
    if result[pivot] < 0:
        result *= -1.0
    return result


def prepare_bold(X: Any) -> Dict[str, np.ndarray]:
    """Temporally z-score each ROI, then give each nonzero frame unit L2 norm.

    Temporal standard deviations use ``ddof=1``. No spatial mean is subtracted.
    A frame whose standardized values are all zero remains a zero row and is
    marked invalid. Nonfinite input and zero-variance ROIs are errors rather
    than silently dropped columns. At least two time points are required.

    ``frame_norm`` is the norm *after* temporal standardization and *before*
    frame normalization. Returned means and standard deviations allow the
    transformation to be reproduced without changing the source time series.
    """
    data = _matrix(X, "X")
    if data.shape[0] < 2:
        raise ValueError("X needs at least two time points for ddof=1 standardization")
    if not np.isfinite(data).all():
        bad_columns = np.flatnonzero(~np.isfinite(data).all(axis=0)).tolist()
        raise ValueError("X contains nonfinite values in ROI columns {}".format(bad_columns))
    with np.errstate(over="ignore", invalid="ignore"):
        temporal_mean = data.mean(axis=0)
        temporal_std = data.std(axis=0, ddof=1)
    bad_std = (~np.isfinite(temporal_std)) | (temporal_std <= 0)
    if not np.isfinite(temporal_mean).all() or bad_std.any():
        bad_columns = np.flatnonzero(bad_std | ~np.isfinite(temporal_mean)).tolist()
        raise ValueError(
            "Zero-variance or numerically invalid ROI columns {}; "
            "check these ROIs before analysis".format(bad_columns)
        )
    standardized = (data - temporal_mean) / temporal_std
    frame_norm = np.linalg.norm(standardized, axis=1)
    if not np.isfinite(frame_norm).all():
        raise ValueError("Temporal standardization produced nonfinite frame norms")
    valid = frame_norm > 0.0
    unit_vectors = np.zeros_like(standardized)
    unit_vectors[valid] = standardized[valid] / frame_norm[valid, None]
    return {
        "unit_vectors": unit_vectors,
        "valid": valid,
        "frame_norm": frame_norm,
        "temporal_mean": temporal_mean,
        "temporal_std": temporal_std,
    }


def assign_states(
    unit_vectors: Any,
    centers: Any,
    valid: Optional[Any] = None,
    polarity: str = "signed",
) -> Dict[str, np.ndarray]:
    """Assign each usable frame to the closest shared spatial template.

    Inputs are normalized defensively. Invalid/nonfinite/zero frames receive
    label -1, similarity NaN and polarity_sign 0. A supplied valid mask can
    exclude additional frames; it cannot make a zero/nonfinite frame usable.

    In signed mode similarity is signed cosine. In unsigned mode similarity is
    its absolute value and ``polarity_sign`` retains the original orientation.
    Ties follow NumPy's first-maximum rule. No temporal smoothing is performed.
    """
    polarity = _polarity(polarity)
    data = _matrix(unit_vectors, "unit_vectors")
    templates = _matrix(centers, "centers")
    if data.shape[1] != templates.shape[1]:
        raise ValueError("unit_vectors and centers must have the same number of ROIs")
    if not np.isfinite(templates).all():
        raise ValueError("centers must contain only finite values")
    center_norm = np.linalg.norm(templates, axis=1)
    if (~np.isfinite(center_norm)).any() or (center_norm <= 0).any():
        raise ValueError("centers must have finite, nonzero norms")
    templates = templates / center_norm[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(data, axis=1)
    usable = np.isfinite(data).all(axis=1) & np.isfinite(norms) & (norms > 0)
    if valid is not None:
        provided = np.asarray(valid)
        if provided.shape != (data.shape[0],) or provided.dtype.kind != "b":
            raise ValueError("valid must be a boolean vector with one value per frame")
        usable &= provided
    labels = np.full(data.shape[0], -1, dtype=np.int64)
    similarity = np.full(data.shape[0], np.nan, dtype=np.float64)
    polarity_sign = np.zeros(data.shape[0], dtype=np.int8)
    if usable.any():
        vectors = data[usable] / norms[usable, None]
        signed_scores = np.clip(vectors @ templates.T, -1.0, 1.0)
        match_scores = signed_scores if polarity == "signed" else np.abs(signed_scores)
        best = np.argmax(match_scores, axis=1)
        rows = np.arange(best.size)
        labels[usable] = best
        similarity[usable] = match_scores[rows, best]
        polarity_sign[usable] = np.sign(signed_scores[rows, best]).astype(np.int8)
    return {"labels": labels, "similarity": similarity, "polarity_sign": polarity_sign}


def _scores_and_loss(
    vectors: np.ndarray, centers: np.ndarray, polarity: str
) -> Any:
    signed = np.clip(vectors @ centers.T, -1.0, 1.0)
    scores = signed if polarity == "signed" else signed * signed
    labels = np.argmax(scores, axis=1)
    loss = np.maximum(0.0, 1.0 - scores[np.arange(vectors.shape[0]), labels])
    return labels, loss


def _initialize_centers(
    vectors: np.ndarray,
    k: int,
    weights: np.ndarray,
    rng: np.random.Generator,
    polarity: str,
) -> np.ndarray:
    """Weighted k-means++-style seeding in the chosen directional geometry."""
    first = int(rng.choice(vectors.shape[0], p=weights / weights.sum()))
    centers = [vectors[first].copy()]
    chosen = {first}
    for _ in range(1, k):
        _, losses = _scores_and_loss(vectors, np.asarray(centers), polarity)
        probabilities = weights * losses
        probabilities[list(chosen)] = 0.0
        total = probabilities.sum()
        if total <= np.finfo(np.float64).eps:
            # Duplicate directions can make further geometric seeding impossible.
            # Keep the requested shape; n_occupied_clusters exposes degeneracy.
            candidates = np.array(
                [i for i in np.flatnonzero(weights > 0) if i not in chosen], dtype=int
            )
            probability = weights[candidates] / weights[candidates].sum()
            picked = int(rng.choice(candidates, p=probability))
        else:
            picked = int(rng.choice(vectors.shape[0], p=probabilities / total))
        centers.append(vectors[picked].copy())
        chosen.add(picked)
    out = np.asarray(centers)
    if polarity == "unsigned":
        out = np.asarray([_canonical_axis(center) for center in out])
    return out


def spherical_kmeans(
    X: Any,
    k: int = 4,
    n_init: int = 100,
    max_iter: int = 300,
    seed: Optional[int] = 20261121,
    weights: Optional[Any] = None,
    polarity: str = "signed",
) -> Dict[str, Any]:
    """Fit shared unit-norm templates, retaining the best of multiple starts.

    X contains usable frame vectors; pass ``unit_vectors[valid]`` from
    prepare_bold. Row norms are removed defensively. Nonnegative sample weights
    affect seeding, centroids and the objective. Zero-weight rows are labeled
    but do not influence the fit. Weights are internally rescaled to sum to 1.

    Signed loss is weighted mean ``1 - cosine``; each centroid is a normalized
    weighted vector sum. Unsigned loss is weighted mean ``1 - cosine**2``;
    each centroid is the principal eigenvector of its weighted scatter matrix.
    Unsigned axes have their largest-magnitude element made positive. This sign
    rule does not remove nonuniqueness of a degenerate leading eigenspace.

    ``run_objectives`` records the final loss of every initialization. Labels
    are zero-based. ``n_occupied_clusters`` counts positive-weight occupied
    clusters: a value below k signals duplicate/degenerate directions and must
    not be presented as a successful k-state solution. K=4 is merely the
    default exploratory baseline, not an estimate of a true state count.
    """
    polarity = _polarity(polarity)
    k = _positive_integer(k, "k")
    n_init = _positive_integer(n_init, "n_init")
    max_iter = _positive_integer(max_iter, "max_iter")
    data = _matrix(X, "X")
    if not np.isfinite(data).all():
        raise ValueError("X must contain only finite values")
    norms = np.linalg.norm(data, axis=1)
    if (~np.isfinite(norms)).any() or (norms <= 0).any():
        raise ValueError("X contains zero or invalid rows; train only on valid frame vectors")
    vectors = data / norms[:, None]
    if weights is None:
        sample_weights = np.ones(vectors.shape[0], dtype=np.float64)
    else:
        sample_weights = np.asarray(weights, dtype=np.float64).copy()
        if sample_weights.shape != (vectors.shape[0],):
            raise ValueError("weights must have one value per input row")
    if not np.isfinite(sample_weights).all() or (sample_weights < 0).any():
        raise ValueError("weights must be finite and nonnegative")
    if np.count_nonzero(sample_weights > 0) < k:
        raise ValueError("k cannot exceed the number of positive-weight samples")
    # Max-rescaling avoids overflow when callers supply large but finite weights.
    sample_weights /= sample_weights.max()
    sample_weights /= sample_weights.sum()
    if np.count_nonzero(sample_weights > 0) < k:
        raise ValueError("Weight rescaling leaves fewer than k numerically positive samples")
    rng = np.random.default_rng(seed)
    run_objectives = np.empty(n_init, dtype=np.float64)
    best = None
    positive_indices = np.flatnonzero(sample_weights > 0)
    for run in range(n_init):
        centers = _initialize_centers(vectors, k, sample_weights, rng, polarity)
        for iteration in range(1, max_iter + 1):
            labels, losses = _scores_and_loss(vectors, centers, polarity)
            updated = centers.copy()
            empty = []
            for state in range(k):
                mask = (labels == state) & (sample_weights > 0)
                if not mask.any():
                    empty.append(state)
                    continue
                cluster_vectors = vectors[mask]
                cluster_weights = sample_weights[mask]
                # Local rescaling leaves the direction/eigenvector unchanged
                # while avoiding unnecessarily tiny cluster scatter matrices.
                cluster_weights = cluster_weights / cluster_weights.sum()
                if polarity == "signed":
                    direction = np.sum(cluster_weights[:, None] * cluster_vectors, axis=0)
                    length = np.linalg.norm(direction)
                    if length > np.finfo(np.float64).eps * cluster_weights.sum():
                        updated[state] = direction / length
                    # An exactly cancelling cluster has no preferred direction;
                    # retaining its prior unit center preserves the objective.
                else:
                    scatter = cluster_vectors.T @ (cluster_weights[:, None] * cluster_vectors)
                    _, eigvecs = eigh(
                        scatter,
                        subset_by_index=(scatter.shape[0] - 1, scatter.shape[0] - 1),
                        check_finite=False,
                    )
                    direction = eigvecs[:, 0]
                    direction /= np.linalg.norm(direction)
                    updated[state] = _canonical_axis(direction)
            if empty:
                # Split poorly represented samples without adding random jitter
                # or modifying observed directions. Distinct indices are used.
                ranking = positive_indices[
                    np.argsort(-(sample_weights[positive_indices] * losses[positive_indices]), kind="stable")
                ]
                for state, picked in zip(empty, ranking):
                    updated[state] = vectors[picked]
                    if polarity == "unsigned":
                        updated[state] = _canonical_axis(updated[state])
            new_labels, new_losses = _scores_and_loss(vectors, updated, polarity)
            centers = updated
            if np.array_equal(new_labels, labels):
                break
        objective = float(np.dot(sample_weights, new_losses))
        run_objectives[run] = objective
        occupied = int(np.unique(new_labels[sample_weights > 0]).size)
        if best is None or objective < best["objective"] - 1e-14 or (
            abs(objective - best["objective"]) <= 1e-14
            and occupied > best["n_occupied_clusters"]
        ):
            best = {
                "centers": centers.copy(),
                "labels": new_labels.copy(),
                "objective": objective,
                "n_iter": iteration,
                "n_occupied_clusters": occupied,
                "cluster_weight": np.bincount(new_labels, weights=sample_weights, minlength=k),
                "polarity": polarity,
            }
    best["run_objectives"] = run_objectives
    return best


def _row_probabilities(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.full(counts.shape, np.nan, dtype=np.float64)
    np.divide(counts, totals, out=probabilities, where=totals > 0)
    return probabilities


def sequence_metrics(labels: Any, tr: float, k: int) -> Dict[str, Any]:
    """Summarize a single run without connecting states across missing frames.

    Input labels are integers in [-1, k-1]; -1 means unobserved/unassigned.
    Exposure is valid frame count times TR. Episodes are *observed* contiguous
    segments, including segments meeting run boundaries or gaps; no censoring
    correction is implied. Duration is their mean observed duration in seconds.

    Frame transitions include self-transitions between adjacent valid frames.
    Episode transitions count only changes at adjacent valid frames, excluding
    self-transitions. Neither matrix bridges -1 gaps. Rows with no observed
    outgoing transitions have all-NaN probabilities, not zero probabilities.
    An absent state's coverage and occurrence are zero when exposure is nonzero;
    its duration is undefined (NaN). With no valid exposure all three are NaN.
    """
    k = _positive_integer(k, "k")
    if not np.isscalar(tr) or not np.isfinite(tr) or tr <= 0:
        raise ValueError("tr must be a finite positive number of seconds")
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional integer sequence")
    if values.size and values.dtype.kind not in "iu":
        raise ValueError("labels must contain integers; use -1 for invalid frames")
    if values.size and ((values < -1).any() or (values >= k).any()):
        raise ValueError("labels must lie between -1 and k-1")
    values = values.astype(np.int64, copy=False)
    valid = values >= 0
    n_valid = int(valid.sum())
    exposure_sec = float(n_valid * tr)
    frame_counts = np.bincount(values[valid], minlength=k)
    episode_start = valid.copy()
    if values.size > 1:
        episode_start[1:] &= (~valid[:-1]) | (values[1:] != values[:-1])
    episode_counts = np.bincount(values[episode_start], minlength=k)
    states = []
    for state in range(k):
        n_frames = int(frame_counts[state])
        n_episodes = int(episode_counts[state])
        states.append({
            "state": state + 1,
            "coverage": n_frames / float(n_valid) if n_valid else float("nan"),
            "occurrence_per_min": n_episodes * 60.0 / exposure_sec if n_valid else float("nan"),
            "duration_sec": n_frames * float(tr) / n_episodes if n_episodes else float("nan"),
            "n_episodes": n_episodes,
            "n_state_frames": n_frames,
        })
    frame_transition_counts = np.zeros((k, k), dtype=np.int64)
    episode_transition_counts = np.zeros((k, k), dtype=np.int64)
    if values.size > 1:
        adjacent = valid[:-1] & valid[1:]
        source = values[:-1][adjacent]
        target = values[1:][adjacent]
        np.add.at(frame_transition_counts, (source, target), 1)
        changes = source != target
        np.add.at(episode_transition_counts, (source[changes], target[changes]), 1)
    return {
        "states": states,
        "n_valid_frames": n_valid,
        "n_invalid_frames": int(values.size - n_valid),
        "valid_duration_sec": exposure_sec,
        "frame_transition_counts": frame_transition_counts,
        "frame_transition_probabilities": _row_probabilities(frame_transition_counts),
        "episode_transition_counts": episode_transition_counts,
        "episode_transition_probabilities": _row_probabilities(episode_transition_counts),
    }
