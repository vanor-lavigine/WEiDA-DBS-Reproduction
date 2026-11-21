"""Spatial GFP peaks, polarity-invariant AAHC, group maps, and peak backfitting."""

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.optimize import linear_sum_assignment

from core import prepare_bold


K = 4
PERMUTATIONS = np.asarray(list(permutations(range(K))))


def oriented(axis):
    """Give unsigned axes a deterministic display orientation."""
    return axis if axis[np.argmax(np.abs(axis))] >= 0 else -axis


def principal_axis(maps):
    return oriented(np.linalg.svd(maps, full_matrices=False)[2][0])


def aahc(maps, k=K):
    """Atomize the least-fitted cluster by summed absolute projections (unnormalized input)."""
    maps = np.asarray(maps, dtype=float)
    if maps.ndim != 2 or not 1 <= k <= len(maps) or not np.isfinite(maps).all():
        raise ValueError("AAHC needs finite maps and at least K observations")
    lengths = np.linalg.norm(maps, axis=1)
    if np.any(lengths == 0):
        raise ValueError("A zero map cannot define a cluster direction")
    centers = maps / lengths[:, None]
    contribution = lengths.copy()
    owners = np.arange(len(maps))
    active = np.ones(len(maps), dtype=bool)
    for _ in range(len(maps) - k):
        removed = contribution.argmin()
        active[removed] = False
        contribution[removed] = np.inf
        keep = np.flatnonzero(active)
        lost = owners == removed
        recipients = keep[np.abs(maps[lost] @ centers[keep].T).argmax(axis=1)]
        owners[lost] = recipients
        for recipient in np.unique(recipients):
            members = maps[owners == recipient]
            axis = principal_axis(members)
            centers[recipient] = axis
            contribution[recipient] = np.abs(members @ axis).sum()
    return np.stack([oriented(c) for c in centers[active]])


@dataclass
class PeakScan:
    units: np.ndarray
    gfp: np.ndarray
    peaks: np.ndarray
    individual: np.ndarray
    fd: float


def prepare_scan(values, fd):
    scaled = prepare_bold(values)
    z = (values - scaled["temporal_mean"]) / scaled["temporal_std"]
    centered = z - z.mean(axis=1, keepdims=True)
    gfp = np.sqrt(np.mean(centered ** 2, axis=1))
    usable = gfp > 0
    peaks = np.flatnonzero(usable[:-2] & usable[2:] & (gfp[1:-1] > gfp[:-2])
                           & (gfp[1:-1] > gfp[2:])) + 1
    units = np.zeros_like(centered)
    units[usable] = centered[usable] / np.linalg.norm(centered[usable], axis=1, keepdims=True)
    return PeakScan(units, gfp, peaks, aahc(centered[peaks]), fd)


def group_maps(individual, weights):
    """Fit one-to-one group maps, trying every individual map set as a start."""
    maps = np.asarray(individual, dtype=float)
    w = np.asarray(weights, dtype=float)
    if maps.ndim != 3 or maps.shape[:2] != (len(w), K) or not np.all(w > 0):
        raise ValueError("Expected four individual maps and a positive weight per scan")
    np.testing.assert_allclose(np.linalg.norm(maps, axis=2), 1, atol=1e-12)
    best = None
    best_score = -np.inf
    objectives = []
    for initial in maps:
        centers = initial
        previous = None
        for _ in range(300):
            similarity = (maps @ centers.T) ** 2
            scores = similarity[:, PERMUTATIONS, np.arange(K)].sum(axis=2)
            assignment = PERMUTATIONS[scores.argmax(axis=1)]
            if np.array_equal(assignment, previous):
                break
            aligned = maps[np.arange(len(w))[:, None], assignment]
            centers = np.stack([principal_axis(aligned[:, j] * np.sqrt(w)[:, None]) for j in range(K)])
            previous = assignment
        else:
            raise ValueError("Individual-to-group spatial alignment did not converge")
        objective = np.average(np.sum(aligned * centers, axis=2) ** 2, axis=0, weights=w).mean()
        objectives.append(objective)
        if objective > best_score + 1e-12:
            best, best_score = centers.copy(), objective
    return best, {"starts": len(maps), "best_mean_squared_r": best_score,
                  "lowest_start_mean_squared_r": min(objectives)}


def match_to_signed(reference, new):
    """Name new axes by one-to-one absolute spatial Pearson r to the old signed maps."""
    ref = reference - reference.mean(axis=1, keepdims=True)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    corr = ref @ new.T
    i, order = linear_sum_assignment(-np.abs(corr))
    np.testing.assert_array_equal(i, np.arange(K))
    aligned = new[order] * np.where(corr[i, order] < 0, -1, 1)[:, None]
    aligned_corr = ref @ aligned.T
    totals = np.abs(aligned_corr)[np.arange(K), PERMUTATIONS].sum(axis=1)
    gap = np.sort(totals)[-1] - np.sort(totals)[-2]
    return aligned, aligned_corr, float(gap)


def interpolate_peaks(n, peaks, labels, valid, midpoint_similarity=None):
    """Linear indicator interpolation with optional spatial tie-breaking at exact midpoints."""
    out = np.full(n, -1, dtype=int)
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    ties = 0
    for start, end in boundaries.reshape(-1, 2):
        take = (peaks >= start) & (peaks < end)
        p, v = peaks[take], labels[take]
        if len(p) < 2:
            continue
        frames = np.arange(p[0], p[-1] + 1)
        right = np.clip(np.searchsorted(p, frames), 1, len(p) - 1)
        left = right - 1
        dl, dr = frames - p[left], p[right] - frames
        segment = np.where(dl < dr, v[left], v[right])
        tied = (dl == dr) & (v[left] != v[right])
        segment[tied] = np.minimum(v[left[tied]], v[right[tied]])
        if midpoint_similarity is not None and tied.any():
            lstate, rstate = v[left[tied]], v[right[tied]]
            lfit = midpoint_similarity[frames[tied], lstate]
            rfit = midpoint_similarity[frames[tied], rstate]
            segment[tied] = np.where(lfit > rfit, lstate, np.where(rfit > lfit, rstate, segment[tied]))
        ties += int(tied.sum())
        changes = np.flatnonzero(segment[1:] != segment[:-1]) + 1
        if len(changes) >= 2:
            inside = slice(changes[0], changes[-1])
            out[frames[inside]] = segment[inside]
    return out, ties


def backfit(scan, centers):
    similarity = np.abs(scan.units @ centers.T)
    direct = similarity.argmax(axis=1)
    valid = scan.gfp > 0
    direct[~valid] = -1
    peak_labels = direct[scan.peaks]
    interpolated, ties = interpolate_peaks(len(direct), scan.peaks, peak_labels, valid)
    peaks = scan.peaks
    max_r = similarity[peaks, peak_labels]
    kept = interpolated >= 0
    diagnostic = {
        "peaks": len(peaks), "peaks_per_min": 60 * len(peaks) / (len(direct) * 2.1),
        "median_gap_s": np.median(np.diff(peaks)) * 2.1,
        "retained_fraction": kept.mean(), "n_retained": int(kept.sum()),
        "peak_gev": np.average(max_r ** 2, weights=scan.gfp[peaks] ** 2),
        "direct_agreement_on_retained": np.mean(interpolated[kept] == direct[kept]) if kept.any() else np.nan,
        "differing_label_midpoint_ties": ties,
    }
    return interpolated, direct, diagnostic
