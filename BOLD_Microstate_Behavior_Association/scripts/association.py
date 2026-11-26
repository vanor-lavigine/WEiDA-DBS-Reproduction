"""Participant-level two-sided Spearman tests and FD sensitivity."""

from math import factorial, prod
import sys

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

from io_sources import ENDPOINTS, PROTOCOL, REPO, require

sys.path.insert(0, str(REPO / "V4_WEiDA_Statistical_Framework_Microstate/scripts"))
from statistics_v4 import Resampling, hc3_t, tail_probability


SEED = PROTOCOL["seed"]
PERMS = PROTOCOL["permutations"]
BOOT = PROTOCOL["bootstrap"]
EPS = PROTOCOL["tolerance"]
MIN_PAIRS = PROTOCOL["minimum_pairs"]
MIN_FD_PAIRS = PROTOCOL["minimum_fd_pairs"]
RETEST = Resampling(SEED, PERMS, BOOT)


def correlation_rows(x, y):
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    a = (x * y).sum(axis=-1)
    b = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    return np.clip(np.divide(a, b, where=b > EPS, out=np.full_like(a, np.nan)), -1, 1)


def ranked_rows(x, y):
    return correlation_rows(stats.rankdata(x, axis=-1), stats.rankdata(y, axis=-1))


def step_permutations(counts, state, position=0):
    """Generate each multiset arrangement once."""
    if position == len(state):
        yield tuple(state)
        return
    for code, count in enumerate(counts):
        if count > 0:
            counts[code] -= 1
            state[position] = code
            yield from step_permutations(counts, state, position + 1)
            counts[code] += 1


def alternative_values(values, salt, budget=PERMS):
    """Enumerate or sample unique non-observed assignments of values to participants."""
    numbers, labels, counts = np.unique(values, return_inverse=True, return_counts=True)
    total = factorial(len(labels)) // prod(factorial(int(c)) for c in counts)
    observed = tuple(labels.tolist())
    if total - 1 <= budget:
        rows = [r for r in step_permutations(counts.tolist(), [0] * len(labels)) if r != observed]
        mode = "enumerated"
    else:
        random = np.random.default_rng([SEED, salt, len(values)])
        seen = {observed}
        rows = []
        while len(rows) < budget:
            candidate = tuple(random.permutation(labels).tolist())
            if candidate not in seen:
                rows.append(candidate)
                seen.add(candidate)
        mode = "monte_carlo"
    assignments = np.asarray(rows, dtype=np.int64).reshape(len(rows), len(values))
    return numbers[assignments], mode, total


def bootstrap_indices(n):
    # V6's UniqueResampling inherits this exact paired bootstrap RNG.
    return RETEST.resample(n, n, True)[0]


def bootstrap_summary(values, prefix=""):
    valid = values[np.isfinite(values)]
    low, high = np.quantile(valid, [.025, .975]) if len(valid) else (np.nan, np.nan)
    return {f"{prefix}ci95_low": low, f"{prefix}ci95_high": high,
            f"{prefix}bootstrap_requested": BOOT, f"{prefix}bootstrap_valid": len(valid),
            f"{prefix}bootstrap_failed": BOOT - len(valid)}


def residual_ranks(x, y, covariate):
    """Residualize ranked exposure and outcome on centered covariate ranks."""
    a, b, c = (stats.rankdata(v, axis=-1) for v in (x, y, covariate))
    a_mean, b_mean, c_mean = (v.mean(axis=-1, keepdims=True) for v in (a, b, c))
    ac, bc, cc = a - a_mean, b - b_mean, c - c_mean
    variance = (cc ** 2).sum(axis=-1, keepdims=True)
    alpha = np.divide((ac * cc).sum(axis=-1, keepdims=True), variance,
                      where=variance > EPS, out=np.full_like(variance, np.nan))
    beta = np.divide((bc * cc).sum(axis=-1, keepdims=True), variance,
                     where=variance > EPS, out=np.full_like(variance, np.nan))
    return ac - alpha * cc, bc - beta * cc


def fd_test(x, y, fd):
    good = np.isfinite(fd)
    x, y, fd = x[good], y[good], fd[good]
    n = len(x)
    d = dict(fd_n_pairs=n, fd_status="NOT_ESTIMABLE", fd_reason="Fewer than four complete FD pairs",
             fd_rho=np.nan, fd_p=np.nan, fd_ci95_low=np.nan, fd_ci95_high=np.nan, fd_bootstrap_valid=0,
             fd_bootstrap_requested=0, fd_bootstrap_failed=0, fd_bootstrap_constant_covariate=0,
             fd_bootstrap_constant_rank_residual=0, fd_permutation_draws=0, fd_permutation_valid=0,
             fd_permutation_mode="", fd_permutation_infinite=0, fd_permutation_space="")
    if n < MIN_FD_PAIRS:
        return d
    if min(np.ptp(x), np.ptp(y), np.ptp(fd)) <= 0:
        return dict(d, fd_reason="Constant feature, score improvement, or FD on the FD-complete subset")
    nuisance = np.c_[np.ones(n), stats.rankdata(fd)]
    a, b = stats.rankdata(x), stats.rankdata(y)
    design = np.c_[nuisance, a]
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return dict(d, fd_reason="FD rank-regression design is singular")
    residual_a, residual_b = residual_ranks(x, y, fd)
    rho = correlation_rows(residual_a, residual_b)
    if not np.isfinite(rho):
        return dict(d, fd_reason="A rank is constant after removing the FD rank")
    fitted = b - residual_b
    try:
        _, observed = hc3_t(design, b, 2)
    except ValueError:
        return dict(d, fd_reason="HC3 leverage reaches one on the FD-complete subset")
    if not np.isfinite(observed):
        return dict(d, fd_reason="Observed HC3 t is undefined")
    surrogate, mode, space = alternative_values(residual_b, salt=204)
    _, null = hc3_t(design, fitted + surrogate, 2)
    keep = ~np.isnan(null)  # A zero standard error with nonzero effect gives an extreme t.
    idx = bootstrap_indices(n)
    ax, by = residual_ranks(x[idx], y[idx], fd[idx])
    draws = correlation_rows(ax, by)
    boot_constant_fd = np.ptp(fd[idx], axis=1) == 0
    return dict(d, fd_status="OK", fd_reason="", fd_rho=float(rho),
                fd_p=tail_probability(null[keep], observed),
                fd_permutation_mode=mode, fd_permutation_space=str(space),
                fd_permutation_draws=len(surrogate), fd_permutation_valid=int(keep.sum()),
                fd_permutation_infinite=int(np.isinf(null).sum()),
                fd_bootstrap_constant_covariate=int(boot_constant_fd.sum()),
                fd_bootstrap_constant_rank_residual=int((~np.isfinite(draws) & ~boot_constant_fd).sum()),
                **bootstrap_summary(draws, prefix="fd_"))


def leave_one_out(x, y, rho):
    d = dict(loo_valid=0, loo_rho_low=np.nan, loo_rho_high=np.nan,
             loo_max_abs_shift=np.nan, loo_direction_changed=None)
    if len(x) <= MIN_PAIRS:
        return d
    keep = ~np.eye(len(x), dtype=bool)
    index = keep.nonzero()[1].reshape(len(x), -1)
    rows = ranked_rows(x[index], y[index])
    valid = rows[np.isfinite(rows)]
    if not len(valid):
        return d
    sign = lambda v: np.sign(np.where(np.abs(v) <= EPS, 0, v))
    return dict(loo_valid=len(valid), loo_rho_low=float(valid.min()), loo_rho_high=float(valid.max()),
                loo_max_abs_shift=float(np.abs(valid - rho).max()),
                loo_direction_changed=bool(np.any(sign(valid) != sign(rho))))


def association(x, y, fd):
    x, y, fd = (np.asarray(v, float) for v in (x, y, fd))
    needed = np.isfinite(x) & np.isfinite(y)
    x, y, fd = (v[needed] for v in (x, y, fd))
    n = len(x)
    ties_x, ties_y = n - len(np.unique(x)), n - len(np.unique(y))
    fd_summary = fd_test(x, y, fd)
    d = dict(n_pairs=n, feature_tie_excess=ties_x, improvement_tie_excess=ties_y,
             rho=np.nan, ci95_low=np.nan, ci95_high=np.nan, raw_p=np.nan,
             permutation_p=np.nan, permutation_mode="", permutation_space="",
             permutation_draws=0, permutation_valid=0, bootstrap_requested=0,
             bootstrap_valid=0, bootstrap_failed=0, bootstrap_constant_feature=0,
             bootstrap_constant_improvement=0, status="NOT_ESTIMABLE", reason="Fewer than three complete pairs",
             **leave_one_out(x[:0], y[:0], np.nan), **fd_summary)
    if n < MIN_PAIRS:
        return d
    if min(np.ptp(x), np.ptp(y)) <= 0:
        return dict(d, reason="Constant feature or score improvement")
    rho, raw_p = stats.spearmanr(x, y, alternative="two-sided")
    assigned, mode, space = alternative_values(stats.rankdata(y), salt=203)
    null = correlation_rows(stats.rankdata(x), assigned)
    idx = bootstrap_indices(n)
    x_boot, y_boot = x[idx], y[idx]
    boot = ranked_rows(x_boot, y_boot)
    return dict(d, rho=float(rho), raw_p=float(raw_p), permutation_p=tail_probability(null, rho),
                permutation_mode=mode, permutation_space=str(space), permutation_draws=len(assigned),
                permutation_valid=int(np.isfinite(null).sum()),
                bootstrap_constant_feature=int(np.count_nonzero(np.ptp(x_boot, axis=1) == 0)),
                bootstrap_constant_improvement=int(np.count_nonzero(np.ptp(y_boot, axis=1) == 0)),
                status="OK", reason="", **bootstrap_summary(boot), **leave_one_out(x, y, rho))


def correct(frame):
    """Correct a predeclared four-test family while leaving unavailable rows as NA."""
    require(frame.endpoint.tolist() == ["delta_" + item["field"] for item in ENDPOINTS],
            "An association family must contain the four locked endpoints in order")
    frame = frame.copy()
    frame["family_size"] = len(ENDPOINTS)
    for column, prefix in (("raw_p", "raw"), ("permutation_p", "permutation"), ("fd_p", "fd_permutation")):
        raw = frame[column].to_numpy(float)
        mask = np.isfinite(raw)
        for method, suffix in (("holm", "holm"), ("fdr_bh", "fdr")):
            corrected = multipletests(np.where(mask, raw, 1.), method=method)[1]
            frame[f"{prefix}_{suffix}_p"] = np.where(mask, corrected, np.nan)
    return frame
