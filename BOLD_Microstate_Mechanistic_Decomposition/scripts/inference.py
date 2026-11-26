"""V7 paired inference using the previous V6 resampling and adjustment."""

import os
import sys

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from frozen_inputs import CFG, PRIVATE, RESULTS, V6, read_json, require, write_csv

SENSITIVITY = {"FD": "fd", "DVARS": "mean_std_dvars"}


def load_v6(workspace):
    os.environ["WEIDA_V6_DATA_ROOT"] = str(workspace)
    os.environ["NUMBA_CACHE_DIR"] = str(PRIVATE / "numba")
    sys.path.insert(0, str(V6 / "scripts"))
    import paired as v6

    old = read_json(V6 / "configs/protocol.json")
    for key in ("seed", "permutations", "bootstrap", "alpha"):
        require(old[key] == CFG[key], f"V6 resampling differs from the locked V7 setting ({key})")
    require(old["wald_ridge"] == CFG["wald_shrinkage"], "V6 composition shrinkage differs")
    return v6


def nothing(n, reason):
    return dict(n_patients=n, effect=np.nan, ci95_low=np.nan, ci95_high=np.nan, paired_dz=np.nan,
                test_statistic=np.nan, permutation_p=np.nan, permutations_valid=0, permutations_failed=0,
                bootstrap_valid=0, bootstrap_failed=0, status="NOT_ESTIMABLE", reason=reason)


def ci(v6, boot):
    low, high, valid = v6.interval(boot)
    return dict(ci95_low=low, ci95_high=high, bootstrap_valid=valid, bootstrap_failed=len(boot) - valid)


def unadjusted_test(v6, delta):
    delta = np.asarray(delta, float)
    vector = delta.ndim == 2
    finite = np.isfinite(delta).all(axis=1) if vector else np.isfinite(delta)
    y = delta[finite]
    n = len(y)
    if n < 3:
        return nothing(n, "Fewer than three complete patients"), np.array([])
    flip = v6.signs(n)
    sampled = y[v6.draws(n)].mean(axis=1)
    point = y.mean(axis=0)
    if vector:
        ridge = CFG["wald_shrinkage"]
        boot = np.linalg.norm(sampled, axis=1)
        estimate = np.linalg.norm(point)
        standardized = np.nan
        try:
            statistic = v6.paired_wald(y, ridge)
        except np.linalg.LinAlgError:
            return nothing(n, "Undefined ILR covariance"), boot
        null = np.concatenate([v6.paired_wald(s[..., None] * y, ridge) for s in np.array_split(flip, 10)])
    else:
        boot = sampled
        estimate = statistic = point.item()
        sd = y.std(ddof=1)
        standardized = estimate / sd if sd > 0 else np.nan
        null = (flip * y).mean(axis=1)
    supported = np.isfinite(null)
    if not supported.any():
        return nothing(n, "All sign-flip statistics are undefined"), boot
    p = v6.tail_probability(null[supported], statistic, two_sided=not vector)
    return dict(n_patients=n, effect=estimate, paired_dz=standardized, test_statistic=statistic,
                permutation_p=p, permutations_valid=int(supported.sum()),
                permutations_failed=int((~supported).sum()), status="OK", reason="", **ci(v6, boot)), boot


def adjusted_test(v6, delta, cov):
    vector = delta.ndim == 2
    keep = np.isfinite(cov) & (np.isfinite(delta).all(axis=1) if vector else np.isfinite(delta))
    n = int(keep.sum())
    if n < 3 or np.ptp(cov[keep]) == 0:
        return nothing(n, "Missing or constant scan quality or fewer than three complete patients")
    x = np.c_[np.ones(n), cov[keep]]
    if np.any((x * np.linalg.pinv(x).T).sum(axis=1) >= 1 - 1e-12):
        return nothing(n, "Leverage one prevents the V6 HC3 adjustment")
    y = delta[keep]
    summary, boot = v6.covariate_test(delta, cov, vector=vector)
    if boot is None or not np.isfinite(summary["p"]):
        return nothing(n, summary["status"])
    if vector:
        statistic = v6.regression_wald(x, y, 0, CFG["wald_shrinkage"])
    else:
        _, statistic = v6.hc3_t(x, y, 0)
    return dict(n_patients=n, effect=summary["effect"], paired_dz=np.nan, test_statistic=statistic,
                permutation_p=summary["p"], permutations_valid=summary["draws"], permutations_failed=0,
                status="OK", reason="", **ci(v6, boot))


def holm_complete(p, family_size):
    p = np.asarray(p, float)
    require(len(p) == family_size, "The prespecified Holm family is incomplete")
    ok = np.isfinite(p)
    require(np.all((p[ok] >= 0) & (p[ok] <= 1)), "Invalid raw P value")
    return np.where(ok, multipletests(np.where(ok, p, 1), method="holm")[1], np.nan)


def primary_tests(v6, paired):
    rows = []
    for spec in CFG["primary"]:
        delta = paired[spec["outcome"]].to_numpy(float)
        main, _ = unadjusted_test(v6, delta)
        provenance = "repeat of targeted follow-up H6" if spec.get("reuses_previous_H6") else "V7 decomposition"
        info = dict(endpoint=spec["id"], units=spec["unit"], provenance=provenance)
        rows.append(dict(model="Unadjusted", **info, **main))
        for model, column in SENSITIVITY.items():
            result = adjusted_test(v6, delta, paired[column].to_numpy(float))
            rows.append(dict(model=model, **info, **result))
    out = pd.DataFrame(rows)
    out["holm_family_size"] = len(CFG["primary"])
    for _, group in out.groupby("model", sort=False):
        require(group.endpoint.is_unique, "Repeated hypothesis within a model")
        out.loc[group.index, "holm6_p"] = holm_complete(group.permutation_p, len(CFG["primary"]))
    out["passes_holm6"] = out.holm6_p < CFG["alpha"]
    main = out.loc[out.model.eq("Unadjusted")].reset_index(drop=True)
    sensitivity = out.loc[out.model.ne("Unadjusted")].reset_index(drop=True)
    write_csv(main, RESULTS / "Mechanistic_Primary_Statistics.csv")
    write_csv(sensitivity, RESULTS / "Mechanistic_Motion_Sensitivity.csv")
    return main, sensitivity


def mean_bootstrap(v6, delta):
    x = np.asarray(delta, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < CFG["window_min_estimation_pairs"]:
        return dict(n_patients=n, effect=np.nan, ci95_low=np.nan, ci95_high=np.nan, bootstrap_valid=0,
                    bootstrap_failed=0), np.array([])
    boot = x[v6.draws(n)].mean(axis=1)
    return dict(n_patients=n, effect=float(x.mean()), **ci(v6, boot)), boot


def condition_summary(v6, paired, field, scale=1, keep=None):
    off = paired[f"off_{field}"].to_numpy(float) * scale
    on = paired[f"on_{field}"].to_numpy(float) * scale
    valid = np.isfinite(off) & np.isfinite(on)
    if keep is not None:
        valid &= keep
    off, on = off[valid], on[valid]
    result, _ = mean_bootstrap(v6, on - off)
    result.update(off_mean=off.mean() if len(off) else np.nan, on_mean=on.mean() if len(on) else np.nan)
    if result["bootstrap_valid"]:
        idx = v6.draws(len(off))
        for side, x in [("off", off), ("on", on)]:
            left, right, _ = v6.interval(x[idx].mean(axis=1))
            result.update({f"{side}_ci95_low": left, f"{side}_ci95_high": right})
    return result


def contributions(v6, paired, total_key, component_keys, threshold):
    wanted = [total_key, *component_keys]
    keep = paired[wanted].notna().all(axis=1)
    points, samples = {}, {}
    for key in wanted:
        points[key], samples[key] = mean_bootstrap(v6, paired.loc[keep, key])
    total = points[total_key]
    stable = (abs(total["effect"]) >= threshold and
              (total["ci95_low"] > 0 or total["ci95_high"] < 0))
    shares = {}
    for key in component_keys:
        if stable:
            denom = samples[total_key]
            valid = (np.abs(denom) >= threshold) & (np.sign(denom) == np.sign(total["effect"]))
            values = np.divide(samples[key] * 100., denom, out=np.full_like(denom, np.nan), where=valid)
            effect = points[key]["effect"] / total["effect"] * 100
            shares[key] = dict(effect=effect, n_patients=total["n_patients"], **ci(v6, values),
                               status="OK", reason="")
        else:
            shares[key] = dict(effect=np.nan, n_patients=total["n_patients"], ci95_low=np.nan, ci95_high=np.nan,
                               bootstrap_valid=0, status="PROPORTION_UNSTABLE", reason="Total below threshold or CI crosses zero")
    return points, shares
