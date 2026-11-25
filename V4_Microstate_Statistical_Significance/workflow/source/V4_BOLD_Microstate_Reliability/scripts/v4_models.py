"""Fixed F1/N0/K4 models and gap-aware segmentation for the V4 comparison."""
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from numba import njit
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / "V3_BOLD_Microstate_Method_Optimization"
sys.path.insert(0, str(V3 / "scripts"))
from features_v3 import blocks, measure, segments
from fit_v3 import folds
from input_data import load_scans, transform, TR
from models_v3 import individual, prepare
from peak_states import interpolate_peaks
from resample_fit import fit4, orient_axes

CONFIG = json.loads((ROOT / "configs/protocol.json").read_text(encoding="utf8"))
METHODS = {"A0": ("G0", "S0"), "A1": ("G1", "S0"), "A2": ("G0", "S1"), "A3": ("G1", "S1")}
K = 4


@dataclass
class Prepared:
    units: np.ndarray
    valid: np.ndarray
    amplitude: np.ndarray
    peaks: np.ndarray
    maps: np.ndarray


@dataclass
class Fit:
    maps: np.ndarray
    n_converged: int | None = None
    objective: float | None = None
    n_distinct_solutions: int | None = None


def cohort():
    """Use the original V3 input validation and preserve original acquisition clocks."""
    gate = json.loads((V3 / "manifests/provenance_gate.json").read_text(encoding="utf8"))
    if not gate["pd_comparable"] or not gate["hc_comparable"]:
        raise ValueError("Missing frozen V3 provenance decision")
    scans = sorted(load_scans(), key=lambda row: row["scan"])
    split = folds(scans, seed=CONFIG["fivefold_seed"])
    for s in range(5):
        groups = [row["condition"] for row, f in zip(scans, split) if f == s]
        if [groups.count(c) for c in ("HC", "ON", "OFF")] != [3, 5, 5]:
            raise ValueError("Frozen roster differs from V3")
    return scans, split


def prepare_scans(scans):
    result = []
    for scan in scans:
        p = prepare(transform(scan["x"], "F1"))
        result.append(Prepared(p.units, p.valid, p.amplitude, p.peaks, individual(p)))
    return result


def peak_pool(prepared, train, weight):
    """Unit vectors; the weights of every scan's peaks sum to its scan mass."""
    counts = np.array([len(prepared[i].peaks) for i in train])
    if min(counts) < K:
        raise ValueError("Too few GFP peaks")
    matrix = np.vstack([prepared[i].units[prepared[i].peaks] for i in train])
    return matrix, np.repeat(weight / counts, counts)


@njit(cache=True)
def _projection_updates(activity, weights, prev):
    """Weighted power-step right-hand sides for all active starts."""
    n, starts, k = activity.shape
    labels = np.empty((n, starts), dtype=np.int16)
    counts = np.zeros((starts, k), dtype=np.int32)
    objectives = np.zeros(starts, dtype=np.float64)
    changes = np.zeros(starts, dtype=np.int32)
    direction = np.zeros((n, starts, k), dtype=np.float32)
    for i in range(n):
        for s in range(starts):
            jj = 0
            best = abs(activity[i, s, 0])
            for j in range(1, k):
                score = abs(activity[i, s, j])
                if score > best:
                    jj = j
                    best = score
            labels[i, s] = jj
            counts[s, jj] += 1
            value = weights[i] * activity[i, s, jj]
            direction[i, s, jj] = value
            objectives[s] += value * activity[i, s, jj]
            changes[s] += int(jj != prev[i, s])
    return direction, labels, counts, objectives, changes


def _partition(labels):
    """Canonicalize labels by first occurrence to deduplicate equivalent starts."""
    first = np.full(K, len(labels))
    np.minimum.at(first, labels, np.arange(len(labels)))
    order = np.argsort(first)
    return np.argsort(order)[labels]


def _exact_axes(x, weights, labels, max_iter):
    """Polish with exact first eigenvectors of each assigned weighted covariance."""
    for _ in range(max_iter):
        axes = []
        for k in range(K):
            mask = labels == k
            if not mask.any():
                return None
            z = x[mask] * np.sqrt(weights[mask, None])
            cov = z.T @ z
            _, u = eigh(cov, subset_by_index=[x.shape[1]-1, x.shape[1]-1], check_finite=False)
            axes.append(u[:, 0])
        axes = np.array(axes)
        r2 = (x @ axes.T)**2
        next_labels = r2.argmax(axis=1)
        if np.array_equal(labels, next_labels):
            objective = float(weights @ r2.max(axis=1) / weights.sum())
            return axes, objective
        labels = next_labels
    return None


def modified_kmeans(x, w, *, settings=None):
    """Weighted modified K-means, 50 fixed starts and a train-only objective.

    Normalize power updates of the assigned weighted second moment in parallel.
    Polish each distinct converged partition using exact leading eigenvectors.
    A failed start cannot compete for the maximizing final training solution.
    """
    cfg = CONFIG["g1"] if settings is None else settings
    max_iter = cfg["max_iter"]
    rng = np.random.default_rng(cfg["seed"])
    prob = w / w.sum()
    seeds = np.array([rng.choice(len(x), K, p=prob, replace=False) for _ in range(cfg["n_init"])])
    xs = np.ascontiguousarray(x, dtype=np.float32)
    weights = np.asarray(prob, np.float32)
    starts = len(seeds)
    centers = xs[seeds]
    previous = np.full((len(x), starts), -1, np.int16)
    obj_old = np.full(starts, np.nan)
    candidate_keys = set()
    candidates = []
    converged = 0
    for _ in range(max_iter):
        n_active = len(centers)
        if n_active == 0:
            break
        projections = (xs @ centers.reshape(-1, xs.shape[1]).T).reshape(len(xs), n_active, K)
        directions, labels, counts, obj, changed = _projection_updates(projections, weights, previous)
        update = (directions.reshape(len(xs), -1).T @ xs).reshape(centers.shape)
        norm = np.linalg.norm(update, axis=2, keepdims=True)
        good = (counts > 0).all(1) & (norm[:, :, 0] > 0).all(1)
        update /= np.maximum(norm, np.finfo(np.float32).tiny)
        diff = np.linalg.norm(update - centers, axis=2).max(axis=1)
        close = np.abs(obj - obj_old) <= cfg["relative_objective_tolerance"] * np.maximum(obj, 1e-12)
        done = good & ((changed == 0) | ((diff <= cfg["axis_tolerance"]) & close))
        converged += int(done.sum())
        for index in np.flatnonzero(done):
            canonical = _partition(labels[:, index])
            key = canonical.tobytes()
            if key not in candidate_keys:
                candidate_keys.add(key)
                candidates.append(canonical)
        active = good & ~done
        if not active.any():
            break
        centers, previous, obj_old = update[active], labels[:, active].copy(), obj[active]
    best = None
    best_score = -np.inf
    polished = 0
    for labels in candidates:
        exact = _exact_axes(x, w, labels, max_iter)
        if exact is not None:
            polished += 1
            maps, score = exact
            if score > best_score + 1e-12:
                best, best_score = maps, score
    if best is None:
        raise RuntimeError(f"G1 had no converged weighted solution ({converged}/{len(seeds)} power starts)")
    return Fit(orient_axes(best), converged, best_score, polished)


def fit_bank(g, prepared, conditions, train_indices, weights=None):
    train = np.asarray(train_indices)
    if train.dtype == bool:
        train = np.flatnonzero(train)
    conditions = np.asarray(conditions)
    all_weight = np.array([CONFIG["participant_weight"][condition] for condition in conditions])
    if weights is not None:
        all_weight *= np.asarray(weights)
    train = train[all_weight[train] > 0]
    weight = all_weight[train]
    if len(train) < K:
        raise ValueError("Too few distinct training scans")
    if g == "G0":
        maps = np.stack([prepared[i].maps for i in train])
        return Fit(fit4(maps, weight))
    if g != "G1":
        raise ValueError(g)
    matrix, w = peak_pool(prepared, train, weight)
    return modified_kmeans(matrix, w)


def fit_pair(prepared, conditions, train_indices, weights=None):
    return {g: fit_bank(g, prepared, conditions, train_indices, weights) for g in ("G0", "G1")}


def pair_match(reference, target):
    """Unsigned Hungarian match and a bidirectional margin without sign flipping."""
    sim = np.abs(reference @ target.T)
    src, order = linear_sum_assignment(-sim)
    assigned = sim[src, order]
    others = sim.copy()
    others[src, order] = -np.inf
    margins = assigned - np.maximum(others.max(axis=1), others.max(axis=0)[order])
    gate = CONFIG["identification"]
    passed = (assigned >= gate["min_abs_cosine"]) & (margins >= gate["min_row_and_column_gap"])
    return order, assigned, margins, passed


def match_banks(banks):
    """State-level correspondence across all folds in fold-zero reference order."""
    banks = dict(enumerate(banks)) if not isinstance(banks, dict) else banks
    keys = sorted(banks)
    first = keys[0]
    order = {first: np.arange(K)}
    for dest in keys[1:]:
        order[dest] = pair_match(banks[first], banks[dest])[0]
    stable = np.ones(K, bool)
    rows, nearest_rows = [], []
    for i, j in combinations(keys, 2):
        direct, similarity, margin, passed = pair_match(banks[i], banks[j])
        consistent = direct[order[i]] == order[j]
        stable &= passed[order[i]] & consistent
        raw = np.abs(banks[i] @ banks[j].T)
        candidates = raw.argmax(axis=1)
        reverse = raw.argmax(axis=0)
        for k, s in enumerate(order[i]):
            d = direct[s]
            rows.append(dict(fold_i=i, fold_j=j, state=k+1, state_i=int(s)+1, state_j=int(d)+1,
                             abs_correlation=similarity[s], bidirectional_margin=margin[s],
                             qualified=bool(passed[s]), cycle_state=bool(consistent[k]),
                             relabeling=int(s)!=int(d)))
        for s, target in enumerate(candidates):
            nearest_rows.append(dict(fold_i=i, fold_j=j, state_i=s+1, nearest_j=int(target)+1,
                                     nearest_similarity=raw[s, target],
                                     many_to_one=np.count_nonzero(candidates==target)>1,
                                     mutually_nearest=reverse[target]==s))
    return dict(order=order, stable=stable, composition=bool(stable.all()), rows=rows, nearest_rows=nearest_rows)


@njit(cache=True)
def _potts(cost, valid, penalty):
    n, k = cost.shape
    labels = np.full(n, -1, np.int64)
    path = np.empty((n,k), np.int64)
    scores = np.zeros(k)
    nxt = np.empty(k)
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        start = i
        scores[:] = cost[i]
        i += 1
        while i < n and valid[i]:
            cheapest = np.argmin(scores)
            for j in range(k):
                if scores[j] <= scores[cheapest] + penalty:
                    path[i,j] = j
                    nxt[j] = scores[j] + cost[i,j]
                else:
                    path[i,j] = cheapest
                    nxt[j] = scores[cheapest] + penalty + cost[i,j]
            scores[:] = nxt
            i += 1
        state = np.argmin(scores)
        for t in range(i-1,start-1,-1):
            labels[t] = state
            if t>start:
                state = path[t,state]
    return labels


def temporal_labels(scores, gfp, valid, penalty, *, weight="gfp", normalization=None):
    """Minimize weighted squared-correlation loss plus a cost for each transition."""
    if not np.isfinite(penalty) or penalty<0:
        raise ValueError("Transition penalty must be finite and nonnegative")
    if weight not in ("gfp", "uniform"):
        raise ValueError(weight)
    keep = np.asarray(valid, bool)
    if penalty == 0 or not keep.any():
        return np.where(keep, scores.argmax(axis=1), -1)
    denominator = np.mean(gfp[keep]**2) if normalization is None else normalization
    scale = gfp**2 / denominator if weight=="gfp" else np.ones(len(gfp))
    return _potts(scale[:, None]*(1-scores**2), keep, penalty)


def peak_labels(p, score):
    """Actual V3 historical B0, including its equal-distance similarity tie break."""
    labels, _ = interpolate_peaks(len(p.valid), p.peaks, score[p.peaks].argmax(axis=1), p.valid, score)
    return labels


def metrics(p, labels, corr, window):
    """Measure only labeled exposure; keep common window clips as observed boundary runs."""
    result = measure(p, labels, corr, edge_censor=window=="edge")
    if not result:
        return dict(duration_s=0., n_labeled_frames=0)
    n = int(round(result["duration_s"]/TR))
    runs = segments(labels)
    if window=="edge":
        bounded = labels.copy()
        for a, b in blocks(labels>=0):
            runs_in = segments(labels[a:b])
            bounded[a:a+runs_in[0][1]] = -1
            bounded[a+runs_in[-1][0]:b] = -1
        runs = segments(bounded)
    result.update(global_gev=sum(result[f"gev_{j+1}"] for j in range(K)), n_labeled_frames=n,
                  single_tr_fraction=np.mean([b-a==1 for a,b,_ in runs]),
                  min_state_coverage=min(result[f"coverage_{j+1}"] for j in range(K)))
    return result


def segment_scores(p, scores_dict, penalty, window="common", weight="gfp", admissible=None, *, summarize=True):
    old = {g: peak_labels(p, scores_dict[g]) for g in ("G0", "G1")}
    common = (old["G0"] >= 0) & (old["G1"] >= 0)
    if admissible is not None:
        common &= admissible
    if window not in ("common", "native", "fd_censor", "edge"):
        raise ValueError(window)
    out = {}
    norm = np.mean(p.amplitude[p.valid]**2) if p.valid.any() else 1.
    for method, (g, rule) in METHODS.items():
        corr = scores_dict[g]
        keep = p.valid if window=="native" else common
        if rule == "S0":
            labels = old[g] if window=="native" else np.where(keep, old[g], -1)
        else:
            labels = temporal_labels(corr, p.amplitude, keep, penalty, weight=weight, normalization=norm)
        entry = dict(labels=labels, corr=corr, gfp=p.amplitude)
        if summarize:
            entry["metrics"] = metrics(p, labels, corr, window)
        out[method] = entry
    return out


def segment_scan(prepared, maps_by_g, penalty, window="common", weight="gfp", admissible=None, *, summarize=True):
    score = {g: np.abs(prepared.units @ (v.maps if isinstance(v, Fit) else v).T)
             for g, v in maps_by_g.items()}
    return segment_scores(prepared, score, penalty, window, weight, admissible, summarize=summarize)


def fd_mask(displacement):
    """Exclude the threshold exceedance, one preceding and two following TR."""
    bad = np.flatnonzero(displacement>0.50)
    keep = np.ones(len(displacement), bool)
    for t in bad:
        keep[max(0,t-1):min(len(keep),t+3)] = False
    return keep
