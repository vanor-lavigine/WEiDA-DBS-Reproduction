"""Within-scan circular-shift tests and patient-level partial rank correlations."""

import numpy as np
from scipy.special import xlogy
from scipy.stats import rankdata
from sklearn.metrics.cluster import expected_mutual_information

from stats import METRICS, holm_adjust

K = 4
SEED = 20261122
SHIFT_DRAWS, PERMUTATIONS, BOOTSTRAPS = 99999, 19999, 5000


def joint_counts_by_shift(micro, phase):
    """Count (lag, micro, WEiDA) for two already aligned, valid state sequences."""
    n = len(micro)
    offsets = np.arange(n)
    rotated = phase[(offsets[None, :] - offsets[:, None]) % n]
    cells = (offsets[:, None] * K + micro[None, :]) * K + rotated
    return np.bincount(cells.ravel(), minlength=n*K*K).reshape(n, K, K)


def shift_profile(micro, phase):
    """AMI and joint counts for every circular displacement on the common interior."""
    micro, phase = np.asarray(micro), np.asarray(phase)
    if micro.shape != phase.shape or micro.ndim != 1:
        raise ValueError("Both state sequences must use the same scan and frame axis")
    positions = np.flatnonzero((micro >= 0) & (phase >= 0))
    if len(positions) < 2:
        return None
    if np.any(np.diff(positions) != 1):
        raise ValueError("Discontinuous overlap requires separate within-segment rotations")
    m, s = micro[positions], phase[positions]
    n = len(m)
    counts = joint_counts_by_shift(m, s)
    row, col = counts[0].sum(1), counts[0].sum(0)
    if np.count_nonzero(row) == np.count_nonzero(col) == 1:
        return None
    null_cells = np.outer(row, col) / n**2
    np.testing.assert_allclose(counts.mean(0)/n, null_cells, rtol=0, atol=1e-14)
    marginals = xlogy(row, row).sum() + xlogy(col, col).sum()
    mi = np.log(n) + (xlogy(counts, counts).sum(axis=(1, 2)) - marginals) / n
    occupied = counts[0][np.ix_(row > 0, col > 0)]
    expected = expected_mutual_information(occupied, n)
    normalizer = np.log(n) - marginals/(2*n) - expected
    ami = (mi - expected)/normalizer
    if min(np.count_nonzero(row), np.count_nonzero(col)) == 1:
        ami = np.zeros(n)
    return dict(ami=ami, observed_cells=counts[0]/n, null_cells=null_cells, n_frames=n)


def zero_motion_weights(delta_fd):
    """OLS intercept weights along the last (patient) axis, including bootstrap samples."""
    x = np.asarray(delta_fd, dtype=float)
    if x.ndim == 0 or x.shape[-1] < 3 or not np.isfinite(x).all():
        raise ValueError("Insufficient independent pairs or motion variation")
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    ss = np.sum(centered**2, axis=-1, keepdims=True)
    if np.any(np.ptp(x,axis=-1) == 0) or np.any(ss == 0):
        raise ValueError("Insufficient independent pairs or motion variation")
    return 1/x.shape[-1] - mean * centered / ss


def group_shifts(profiles, fd, rng, draws=SHIFT_DRAWS):
    people = sorted(p for p, visit in profiles if visit == "OFF" and
                    profiles[p, "OFF"] is not None and profiles[p, "ON"] is not None)
    x = np.array([fd[p, "ON"] - fd[p, "OFF"] for p in people])
    weights = zero_motion_weights(x)
    np.testing.assert_allclose([weights.sum(), weights @ x], [1, 0], atol=1e-12)
    profiles_by_visit = [profiles[p, visit] for p in people for visit in ("OFF","ON")]
    scan_weights = np.repeat(weights/2, 2)
    observed = sum(w*q["ami"][0] for w,q in zip(scan_weights, profiles_by_visit))
    expected = sum(w*q["ami"].mean() for w,q in zip(scan_weights, profiles_by_visit))
    random = np.zeros(draws)
    for w,q in zip(scan_weights, profiles_by_visit):
        a = q["ami"]
        random += w*a[rng.integers(len(a), size=draws)]
    extreme = np.count_nonzero(random >= observed - 1e-12)
    lower, upper = np.quantile(random, [.025,.975])
    result = dict(n_pairs=len(people), observed_ami=observed, null_ami=expected,
                  excess_ami=observed-expected, null_q025=lower, null_q975=upper,
                  n_draws=draws, extreme=int(extreme), p_shift=(extreme+1)/(draws+1))
    cells = {k:np.mean([q[k] for q in profiles_by_visit], axis=0)
             for k in ("observed_cells", "null_cells")}
    return result, cells


def residual_ranks(values, motion_ranks):
    ranks = rankdata(values, axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    return ranks - (ranks*motion_ranks).sum(axis=-1,keepdims=True)*motion_ranks


def unit_motion(motion):
    z = rankdata(motion, axis=-1)
    z -= z.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(z,axis=-1,keepdims=True)
    return np.divide(z,norm,out=np.zeros_like(z),where=norm>0)


def partial_rank(x, y, motion):
    z = unit_motion(motion)
    a, b = residual_ranks(x,z), residual_ranks(y,z)
    denom = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    return np.divide((a*b).sum(axis=-1),denom,out=np.full(denom.shape,np.nan),where=denom>1e-10)


def association(x, y, motion, seed, permutations=PERMUTATIONS, bootstraps=BOOTSTRAPS):
    complete = np.isfinite(x) & np.isfinite(y) & np.isfinite(motion)
    x, y, motion = (np.asarray(v)[complete] for v in (x,y,motion))
    n = len(x)
    result = dict(n_pairs=n, partial_r=np.nan, ci_low=np.nan, ci_high=np.nan,
                  p_raw=np.nan, n_extreme=np.nan, n_valid_boot=0, status="not_estimable")
    if n < 4:
        return result
    z = unit_motion(motion)
    a, b = residual_ranks(x,z), residual_ranks(y,z)
    r = float(partial_rank(x,y,motion))
    if not np.isfinite(r):
        return result

    # The reduced-model fitted rank scores vanish on projection onto residual space.
    # Each permuted residual has the same squared norm before this projection.
    perm_seed, boot_seed = np.random.SeedSequence(seed).spawn(2)
    permuted = np.random.default_rng(perm_seed).permuted(np.broadcast_to(b,(permutations,n)),axis=1)
    residuals = permuted - (permuted @ z)[:,None] * z
    denom = np.linalg.norm(residuals,axis=1)*np.linalg.norm(a)
    if np.any(denom <= 1e-10):
        raise ValueError("Degenerate reduced-model permutation")
    null = residuals @ a / denom
    extreme = np.count_nonzero(np.abs(null) >= abs(r)-1e-12)
    samples = np.random.default_rng(boot_seed).integers(n,size=(bootstraps,n))
    boot = partial_rank(x[samples],y[samples],motion[samples])
    valid = boot[np.isfinite(boot)]
    ci = np.percentile(valid,[2.5,97.5]) if len(valid) else [np.nan,np.nan]
    return dict(n_pairs=n, partial_r=r, ci_low=ci[0], ci_high=ci[1],
                p_raw=(extreme+1)/(permutations+1), n_extreme=int(extreme),
                n_valid_boot=len(valid), status="ok")


def association_family(measured, source_index, band_index):
    patients = measured[measured.condition.isin(["ON","OFF"])]
    grid = patients.pivot(index="subject_id", columns=["condition","state"],
                          values=["var_s2","fd",*METRICS]).sort_index()

    def difference(name,state):
        return (grid[name,"ON",state] - grid[name,"OFF",state]).to_numpy()

    x = difference("var_s2",1)
    motion = difference("fd",1)
    rows = []
    for state in range(1,K+1):
        for j, name in enumerate(METRICS):
            estimate = association(x,difference(name,state),motion,
                                   [SEED,source_index,band_index,state,j])
            rows.append(dict(state=state,metric=name,family_n=12,**estimate))
    for row, p in zip(rows,holm_adjust([r["p_raw"] for r in rows])):
        row["p_holm"] = p
    return rows
