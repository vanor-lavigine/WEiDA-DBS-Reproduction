"""Conditional patient permutations and joint-row bootstrap for a single image coefficient."""

from functools import lru_cache
from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution
from statsmodels.stats.multitest import multipletests

from identity_audit import VERIFIED
from pipeline_io import CFG, PRIVATE, RESULTS, TOL, require, write_csv


def registry():
    rows = []
    for o, outcome in enumerate([CFG["primary_outcome"], *CFG["domain_outcomes"]]):
        for x, predictor in enumerate(CFG["predictors"], start=1):
            rows.append(dict(endpoint=f"H{x}" if not o else f"E{(o-1)*3+x}",
                             family="primary" if not o else "exploratory",
                             outcome=outcome, predictor=predictor["key"], predictor_unit=predictor["unit"],
                             interpretation=predictor["status"]))
    return rows


@lru_cache(maxsize=32)
def alternatives(n):
    """Use a deterministic nonidentity set of participant orders or enumerate all."""
    target = CFG["permutations"]
    total = factorial(n) - 1
    observed = tuple(range(n))
    if total <= target:
        return np.array([r for r in permutations(range(n)) if r != observed], dtype=int).reshape(-1, n), "exhaustive"
    rng = np.random.default_rng([CFG["seed"], n, 841])
    orders, seen = [], {observed}
    while len(orders) < target:
        draw = rng.permutation(n)
        key = tuple(draw)
        if key not in seen:
            seen.add(key)
            orders.append(draw)
    return np.array(orders, dtype=int), "monte_carlo_unique"


@lru_cache(maxsize=32)
def bootstrap_rows(n):
    rng = np.random.default_rng([CFG["seed"], n, 842])
    return rng.integers(n, size=(CFG["bootstrap"], n))


def design_problem(x):
    n, p = x.shape
    rank = int(np.linalg.matrix_rank(x)) if n else 0
    if n <= p:
        return "Insufficient residual degrees of freedom", rank
    if rank != p:
        return "Rank-deficient full design", rank
    return "", rank


def hc3_stats(x, y):
    """Evaluate the last OLS coefficient and HC3 t for one response or a batch."""
    pinv = np.linalg.pinv(x)
    leverage = np.einsum("ij,ji->i", x, pinv)
    if np.max(leverage) >= 1 - TOL:
        n = np.shape(y)[:-1]
        return np.full(n, np.nan), np.full(n, np.nan), float(leverage.max())
    beta = y @ pinv.T
    scores = (y - beta @ x.T) / (1 - leverage)
    se = np.sqrt(scores ** 2 @ (pinv[-1] ** 2))
    estimate = beta[..., -1]
    t = np.divide(estimate, se, out=np.full_like(estimate, np.nan), where=np.isfinite(se) & (se > TOL))
    return estimate, t, float(leverage.max())


def residual_reference(design, y):
    reduced = design[:, :-1]
    predicted = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    return reduced, predicted, y - predicted


def bootstrap_coefficients(x, y):
    idx = bootstrap_rows(len(y))
    xb, yb = x[idx], y[idx]
    good = np.linalg.matrix_rank(xb) == x.shape[1]
    coeff = np.full(len(idx), np.nan)
    if good.any():
        coeff[good] = np.einsum("bn,bn->b", np.linalg.pinv(xb[good])[:, -1, :], yb[good])
    return coeff


def model_data(frame, endpoint, model):
    outcome, predictor = endpoint["outcome"], endpoint["predictor"]
    off = f"off_{outcome}"
    y = f"{outcome}_{'improvement' if model['outcome_scale'] == 'absolute' else 'percent_improvement'}"
    motion = [model["motion"]] if model["motion"] else []
    fields = [y, off, *motion, predictor]
    cohort = frame.loc[frame.visit_status.eq(VERIFIED)] if model["sample"] == "verified" else frame
    cohort = cohort.sort_values("participant_id")
    values = cohort[fields].apply(pd.to_numeric, errors="raise").to_numpy(float)
    keep = np.isfinite(values).all(axis=1)
    n = int(keep.sum())
    columns = ["intercept", off, *motion, predictor]
    design, target = np.c_[np.ones(n), values[keep, 1:]], values[keep, 0]
    diag = dict(n_in_sample=len(cohort), n_patients=n, n_excluded_missing=int((~keep).sum()),
                n_missing_outcome=int((~np.isfinite(values[:, 0])).sum()),
                n_missing_predictor=int((~np.isfinite(values[:, -1])).sum()),
                n_off_zero_in_sample=int(cohort[off].eq(0).sum()),
                n_outcome_unique=int(np.unique(target).size), n_predictor_unique=int(np.unique(design[:, -1]).size),
                n_zero_changes=int(np.count_nonzero(target == 0)))
    return design, target, cohort.loc[keep, "participant_id"].to_numpy(), columns, diag


def estimate(x, y):
    issue, rank = design_problem(x)
    blank = dict(beta=np.nan, ci95_low=np.nan, ci95_high=np.nan, hc3_t=np.nan, hc3_reference_p=np.nan,
                 permutation_p=np.nan, permutation_mode="", permutations_valid=0, permutations_failed=0,
                 permutation_fail_reason="", bootstrap_valid=0, bootstrap_failed=0, bootstrap_fail_reason="",
                 status="NOT_ESTIMABLE", reason=issue, design_columns=x.shape[1], design_rank=rank,
                 residual_df=len(y) - x.shape[1], max_leverage=np.nan)
    if issue:
        return blank
    beta, observed, leverage = hc3_stats(x, y)
    blank["max_leverage"] = leverage
    if not np.isfinite(observed):
        blank["reason"] = "HC3 statistic undefined at unit leverage or zero robust standard error"
        return blank
    _, fitted, residual = residual_reference(x, y)
    perm, mode = alternatives(len(y))
    batches = np.array_split(perm, max(1, (len(perm) + 999) // 1000))
    null = np.concatenate([np.asarray(hc3_stats(x, fitted + residual[indices])[1]) for indices in batches])
    usable = np.isfinite(null)
    blank.update(beta=float(beta), hc3_t=float(observed), permutation_mode=mode,
                 hc3_reference_p=2 * t_distribution.sf(abs(observed), len(y) - x.shape[1]),
                 permutations_valid=int(usable.sum()), permutations_failed=int((~usable).sum()),
                 permutation_fail_reason="Undefined HC3 statistic" if not usable.all() else "")
    if not usable.any():
        blank["reason"] = "All reduced-model residual permutations were undefined"
        return blank
    exceed = int((np.abs(null[usable]) >= abs(observed) - TOL).sum())
    p = (1 + exceed) / (1 + usable.sum())
    boot = bootstrap_coefficients(x, y)
    b_good = np.isfinite(boot)
    interval = np.quantile(boot[b_good], [.025, .975]) if b_good.any() else (np.nan, np.nan)
    return dict(blank, permutation_p=float(p), ci95_low=float(interval[0]), ci95_high=float(interval[1]),
                bootstrap_valid=int(b_good.sum()), bootstrap_failed=int((~b_good).sum()),
                bootstrap_fail_reason="Rank-deficient resampled full design" if not b_good.all() else "",
                status="OK", reason="")


def adjust(p_values, method, size):
    raw = np.asarray(p_values, float)
    require(len(raw) == size, "Prespecified correction family is incomplete")
    present = np.isfinite(raw)
    require(((raw[present] >= 0) & (raw[present] <= 1)).all(), "P values outside the unit interval")
    return np.where(present, multipletests(np.where(present, raw, 1.), method=method)[1], np.nan)


def adjust_model(table):
    groups = []
    for _, data in table.groupby("model", sort=False):
        data = data.copy()
        require(data.endpoint.is_unique, "Duplicate hypotheses in correction family")
        primary = data.family.eq("primary")
        data.loc[primary, "holm3_p"] = adjust(data.loc[primary, "permutation_p"], "holm", 3)
        data.loc[~primary, "bh9_q"] = adjust(data.loc[~primary, "permutation_p"], "fdr_bh", 9)
        data["holm12_p"] = adjust(data.permutation_p, "holm", 12)
        groups.append(data)
    return pd.concat(groups, ignore_index=True)


def leave_one(x, y, participants, full, endpoint):
    rows = []
    for k, patient in enumerate(participants):
        keep = np.arange(len(y)) != k
        problem, rank = design_problem(x[keep])
        beta, t, _ = (np.nan, np.nan, np.nan) if problem else hc3_stats(x[keep], y[keep])
        if not problem and not np.isfinite(t):
            problem = "HC3 statistic undefined after omission"
        rows.append(dict(participant_id=patient, endpoint=endpoint, leave_one_beta=beta, leave_one_hc3_t=t,
                         design_rank=rank, status=problem or "OK"))
    supported = np.array([row["leave_one_beta"] for row in rows if row["status"] == "OK"], float)
    observed = full["beta"]
    if len(supported):
        summary = dict(loo_n=len(supported), loo_beta_min=supported.min(), loo_beta_max=supported.max(),
                       loo_max_abs_beta_shift=np.abs(supported - observed).max() if np.isfinite(observed) else np.nan,
                       loo_direction_changes=int(np.count_nonzero(np.sign(supported) != np.sign(observed)))
                       if np.isfinite(observed) else np.nan)
    else:
        summary = dict(loo_n=0, loo_beta_min=np.nan, loo_beta_max=np.nan, loo_max_abs_beta_shift=np.nan,
                       loo_direction_changes=np.nan)
    return rows, summary


def run_models(analysis, results_dir=RESULTS, private_dir=PRIVATE):
    results, patient_diagnostics, patient_points = [], [], []
    for model in CFG["models"]:
        for endpoint in registry():
            x, y, people, columns, meta = model_data(analysis, endpoint, model)
            fit = estimate(x, y)
            result = dict(model=model["name"], outcome_scale=model["outcome_scale"],
                          motion_covariate=model["motion"] or "", **endpoint, **fit, **meta)
            result["effect_unit"] = f"{'points' if model['outcome_scale']=='absolute' else 'percentage points'} per {endpoint['predictor_unit']}"
            result["permutations_requested"] = CFG["permutations"]
            result["bootstrap_requested"] = CFG["bootstrap"]
            result["design"] = " + ".join(columns)
            if model["name"] == "Primary_FD":
                rows, loo = leave_one(x, y, people, fit, endpoint["endpoint"])
                patient_diagnostics.extend(rows)
                result.update(loo)
                if fit["status"] == "OK":
                    reduced, _, y_adjusted = residual_reference(x, y)
                    predictor = x[:, -1]
                    x_adjusted = predictor - reduced @ np.linalg.lstsq(reduced, predictor, rcond=None)[0]
                    patient_points.extend(dict(endpoint=endpoint["endpoint"], participant_id=person,
                                               residualized_image=u, residualized_improvement=v)
                                          for person, u, v in zip(people, x_adjusted, y_adjusted, strict=True))
            results.append(result)
    all_rows = adjust_model(pd.DataFrame(results))
    main = all_rows.loc[all_rows.model.eq("Primary_FD")]
    write_csv(main.loc[main.family.eq("primary")], results_dir / "Primary_Associations.csv")
    write_csv(main.loc[main.family.eq("exploratory")], results_dir / "Exploratory_Domain_Associations.csv")
    write_csv(all_rows.loc[all_rows.model.ne("Primary_FD")], results_dir / "Motion_and_Visit_Sensitivity.csv")
    write_csv(main[["endpoint", "family", "loo_n", "loo_beta_min", "loo_beta_max", "loo_max_abs_beta_shift",
                    "loo_direction_changes"]], results_dir / "Leave_One_Out_Summary.csv")
    write_csv(pd.DataFrame(patient_diagnostics), private_dir / "results/Leave_One_Out_by_Patient.csv")
    points = pd.DataFrame(patient_points)
    write_csv(points, private_dir / "results/Private_Association_Plot_Data.csv")
    return all_rows, points
