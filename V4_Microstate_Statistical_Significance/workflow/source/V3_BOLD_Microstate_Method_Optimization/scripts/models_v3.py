"""GFP representations, participant-weighted training and spatial correspondence."""
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

LEGACY = Path(__file__).resolve().parents[2] / "experiments/bold_microstates_v1"
sys.path.insert(0, str(LEGACY))
from peak_states import aahc, group_maps, oriented, principal_axis

SEED = 20261125
GROUPS = ("HC", "ON", "OFF")
MIN_MATCH, MIN_MARGIN = 0.60, 0.05


@dataclass
class Prepared:
    vectors: np.ndarray
    units: np.ndarray
    amplitude: np.ndarray
    valid: np.ndarray
    peaks: np.ndarray


def prepare(values, norm="N0"):
    """Historical ddof=1 scaling on usable rows, with physical gaps retained."""
    if norm not in {"N0", "N1", "N2"}:
        raise ValueError(norm)
    values = np.asarray(values, float)
    valid = np.isfinite(values).all(axis=1)
    x = values[valid].copy()
    x -= x.mean(axis=0)
    if norm != "N2":
        scale = x.std(axis=0, ddof=1)
        if np.any(scale <= 0):
            raise ValueError("Constant ROI")
        x /= scale
    if norm != "N1":
        x -= x.mean(axis=1, keepdims=True)
    vectors = np.zeros_like(values)
    vectors[valid] = x
    lengths = np.linalg.norm(vectors, axis=1)
    valid &= lengths > 0
    units = np.zeros_like(values)
    units[valid] = vectors[valid] / lengths[valid, None]
    amplitude = lengths / np.sqrt(values.shape[1])
    peaks = np.flatnonzero(valid[:-2] & valid[1:-1] & valid[2:]
                           & (amplitude[1:-1] > amplitude[:-2])
                           & (amplitude[1:-1] > amplitude[2:])) + 1
    return Prepared(vectors, units, amplitude, valid, peaks)


def individual(prepared, k=4):
    return aahc(prepared.vectors[prepared.peaks], k)


def align(reference, maps, signed=False):
    """One-to-one correspondence with row and column ambiguity diagnostics."""
    scores = reference @ maps.T
    if not signed:
        scores = np.abs(scores)
    rows, order = linear_sum_assignment(-scores)
    if len(rows) != len(maps) or not np.array_equal(rows, np.arange(len(maps))):
        raise ValueError("Equal-size map banks required")
    best = scores[rows, order]
    other = scores.copy()
    other[rows, order] = -np.inf
    margins = best - np.maximum(other.max(axis=1), other.max(axis=0)[order])
    return maps[order], best, margins


def group_fit(individuals, weights):
    if individuals.shape[1] == 4:
        return group_maps(individuals, weights)[0]
    # Use the first training scan (input sorted by scan) as the deterministic
    # seed for general K; Hungarian prevents the factorial assignment cost.
    centers = individuals[0].copy()
    prev = None
    for _ in range(300):
        assignments = np.stack([linear_sum_assignment(-(centers @ bank.T)**2)[1]
                                for bank in individuals])
        if np.array_equal(prev, assignments):
            return centers
        aligned = individuals[np.arange(len(individuals))[:, None], assignments]
        centers = np.stack([principal_axis(col * np.sqrt(weights)[:, None])
                            for col in aligned.transpose(1, 0, 2)])
        prev = assignments
    raise RuntimeError("General-K group alignment did not converge")


def spherical_kmeans(units, weights, k, seed=SEED, starts=50):
    """Maximize the weighted signed cosine objective. No polarity flips."""
    rng = np.random.default_rng(seed)
    best_score = -np.inf
    best = None
    for _ in range(starts):
        centers = units[rng.choice(len(units), size=k, replace=False,
                                   p=weights / weights.sum())].copy()
        for iteration in range(300):
            scores = units @ centers.T
            labels = scores.argmax(axis=1)
            new = np.zeros_like(centers)
            for j in range(k):
                members = labels == j
                if not members.any():
                    # Restart an empty cluster at the worst represented vector.
                    new[j] = units[scores.max(axis=1).argmin()]
                else:
                    new[j] = weights[members] @ units[members]
            lengths = np.linalg.norm(new, axis=1)
            if np.any(lengths < 1e-12):
                break
            new /= lengths[:, None]
            if np.max(np.abs(centers - new)) < 1e-9:
                objective = np.average((units @ new.T).max(axis=1), weights=weights)
                if objective > best_score + 1e-12:
                    best, best_score = new.copy(), objective
                break
            centers = new
    if best is None:
        raise RuntimeError("No converged signed spherical k-means initialization")
    return best


@dataclass
class Fitted:
    maps: np.ndarray
    family: np.ndarray
    stable: np.ndarray
    match_rows: list
    polarity: str
    canonical_k: int


def train(scans, prepared, individuals, template="T1", k=4, polarity="P0",
          reference=None, multiplicity=None):
    """Fit with training scans only. Multiplicity counts unique-source draws.

    For T2_HC_ON/OFF, the caller supplies exactly the specified groups.
    The same naming reference, if supplied, is used only to permute states.
    """
    conditions = np.array([s["condition"] for s in scans])
    subject = np.array([s["subject"] for s in scans])
    mult = np.ones(len(scans)) if multiplicity is None else np.asarray(multiplicity, float)
    if len(mult) != len(scans) or not np.all(mult > 0):
        raise ValueError("Expected positive multiplicities for all training scans")
    groups = [g for g in GROUPS if g in conditions]
    weights = mult.copy()
    if template == "T3":
        for person in set(subject):
            weights[subject == person] /= np.count_nonzero(subject == person)
    else:
        for g in groups:
            weights[conditions == g] /= mult[conditions == g].sum()
    signed = polarity == "P2"
    diag = []
    if signed:
        units = np.vstack([p.units[p.peaks] for p in prepared])
        w = np.concatenate([np.full(len(p.peaks), w / len(p.peaks))
                            for p, w in zip(prepared, weights)])
        common = spherical_kmeans(units, w, k)
    else:
        individuals = np.asarray(individuals)
        common = group_fit(individuals, weights)
    stable = np.ones(k, bool)
    if reference is not None:
        common, sim, gap = align(reference, common, signed=signed)
        stable = (sim >= MIN_MATCH) & (gap >= MIN_MARGIN)
        diag += [dict(stage="naming", group="common", state=j+1,
                      similarity=s, margin=m, stable=bool(ok))
                 for j, (s, m, ok) in enumerate(zip(sim, gap, stable))]
    if not signed:
        common = np.stack([oriented(c) for c in common])
    if template.startswith("T2"):
        variants = []
        reliable = np.ones(k, bool)
        for g in groups:
            sel = conditions == g
            bank = group_fit(individuals[sel], mult[sel])
            # Group-to-common and pairwise one-to-one matching must agree.
            aligned, sim, gap = align(common, bank)
            ok = (sim >= MIN_MATCH) & (gap >= MIN_MARGIN)
            reliable &= ok
            diag += [dict(stage="group_to_common", group=g, state=j+1,
                          similarity=s, margin=m, stable=bool(a))
                     for j, (s,m,a) in enumerate(zip(sim,gap,ok))]
            variants.append(aligned)
        for i, left in enumerate(variants):
            for j in range(i+1, len(variants)):
                matrix = np.abs(left @ variants[j].T)
                sim = np.diag(matrix)
                other = matrix.copy()
                np.fill_diagonal(other, -np.inf)
                gap = sim - np.maximum(other.max(axis=1), other.max(axis=0))
                ok = (sim >= MIN_MATCH) & (gap >= MIN_MARGIN)
                reliable &= ok
                diag += [dict(stage="pairwise", group=groups[i]+"_"+groups[j], state=h+1,
                              similarity=s, margin=m, stable=bool(a))
                         for h,(s,m,a) in enumerate(zip(sim,gap,ok))]
        # Unmatched variants receive fixed source-group-specific IDs. Merged
        # canonical labels are used only when every variant passed the gate.
        family = np.concatenate([np.where(reliable, np.arange(k),
                                 k + GROUPS.index(g)*k + np.arange(k)) for g in groups])
        n = int(family.max()) + 1
        stable_out = np.zeros(n, bool)
        stable_out[:k] = stable & reliable
        return Fitted(np.vstack(variants), family, stable_out, diag, polarity, k)
    if polarity == "P1":
        if reference is not None:
            oriented_ref = np.stack([oriented(c) for c in reference])
            signed_dot = np.einsum("ij,ij->i", common, oriented_ref)
            stable &= signed_dot >= MIN_MATCH
            diag += [dict(stage="polarity_orientation", group="common", state=j+1,
                          similarity=r, margin=np.nan, stable=bool(ok))
                     for j, (r,ok) in enumerate(zip(signed_dot,stable))]
        common = np.stack([common, -common], axis=1).reshape(-1, common.shape[1])
        stable = np.repeat(stable,2)
    return Fitted(common, np.arange(len(common)), stable, diag, polarity, k)
