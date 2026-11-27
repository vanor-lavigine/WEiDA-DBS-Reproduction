"""Fixed participant-level whole-vector location tests and bootstrap effects."""

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from round5_io import CFG, EDGES, NONDIAGONAL


def paired_cells(matrices, cells, column):
    views = {}
    for condition in ("ON", "OFF"):
        view = matrices.loc[matrices.condition.eq(condition)].pivot(
            index="participant_id", columns=EDGES, values=column,
        )
        views[condition] = view.loc[:, cells].sort_index()
    if not views["ON"].index.equals(views["OFF"].index) or len(views["ON"]) != CFG["cohort"]["pairs"]:
        raise ValueError("The frozen ON/OFF keys failed the paired-vector check")
    return views["ON"], views["OFF"]


def signs(n, seed, draws):
    space = (1 << n) - 1
    generator = np.random.default_rng(seed)
    masks = generator.choice(space, size=draws, replace=False).astype(np.uint64) + 1
    return 1 - 2 * ((masks[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)


def bootstrap_indices(n, seed, draws, stream):
    return np.random.default_rng([seed, stream, n]).integers(n, size=(draws, n))


def summarize_vectors(values):
    n = values.shape[-2]
    average = values.mean(axis=-2)
    deviations = values - average[..., None, :]
    covariance = (deviations.swapaxes(-1, -2) @ deviations) / (n - 1)
    return average, covariance


def shrink_covariance(cov, ridge):
    d = cov.shape[-1]
    return (1-ridge)*cov + (ridge*np.trace(cov, axis1=-2, axis2=-1)/d)[..., None, None]*np.eye(d)


def geometry(average, covariance, ridge):
    """Squared distance in the fixed shrunk covariance, with a fixed spectral cutoff."""
    ev, axes = np.linalg.eigh(shrink_covariance(covariance, ridge))
    keep = ev > np.maximum(ev[..., -1:] * CFG["covariance_relative_cutoff"], 0)
    oriented = (axes.swapaxes(-1, -2) @ average[..., None])[..., 0]
    result = np.divide(oriented**2, ev, out=np.zeros_like(oriented), where=keep).sum(axis=-1)
    return result, keep.sum(axis=-1)


def design(delta, stream):
    n, d = delta.shape
    if n != CFG["cohort"]["pairs"] or not np.isfinite(delta).all():
        raise ValueError("Whole-vector tests require all 25 defined paired profiles")
    observed = summarize_vectors(delta)
    null_signs = signs(n, CFG["seed"], CFG["permutations"])
    null_mean = null_signs @ delta / n
    crossproducts = delta.T @ delta
    null_cov = (crossproducts - n * null_mean[..., None] * null_mean[:, None, :]) / (n - 1)
    users = bootstrap_indices(n, CFG["seed"], CFG["bootstrap_draws"], stream)
    bootstrap = summarize_vectors(delta[users])
    return observed, (null_mean, null_cov), bootstrap


def omnibus_result(observed, randomized, bootstrap, ridge, n):
    average, cov = observed
    obs, effective_rank = geometry(average, cov, ridge)
    null, _ = geometry(*randomized, ridge)
    effects, bootstrap_rank = geometry(*bootstrap, ridge)
    chol = shrink_covariance(cov, ridge)
    raw_ev = np.linalg.eigvalsh(cov)
    shrunk_ev = np.linalg.eigvalsh(chol)
    raw_rank = int((raw_ev > raw_ev[-1] * CFG["covariance_relative_cutoff"]).sum())
    counts = int(np.count_nonzero(n*null >= n*obs - 1e-12))
    length_ci = np.quantile(np.linalg.norm(bootstrap[0], axis=1), (.025, .975))
    effect_ci = np.quantile(np.sqrt(effects), (.025, .975))
    return {
        "n_pairs": n, "dimensions": len(average), "ridge": ridge, "raw_covariance_rank": raw_rank,
        "test_covariance_rank": int(effective_rank),
        "raw_condition_number": raw_ev[-1]/raw_ev[0] if raw_rank == len(average) else np.nan,
        "test_condition_number": shrunk_ev[-1]/shrunk_ev[0] if effective_rank == len(average) else np.nan,
        "min_raw_eigenvalue": raw_ev[0], "max_raw_eigenvalue": raw_ev[-1],
        "covariance_policy": "spectral pseudoinverse" if effective_rank < len(average) else "full rank",
        "bootstrap_full_rank_draws": int(np.count_nonzero(bootstrap_rank == len(average))),
        "wald_statistic": n*obs, "raw_p": (counts + 1) / (len(null) + 1),
        "extreme_permutations": counts, "permutations": len(null), "bootstrap_draws": len(effects),
        "vector_norm": np.linalg.norm(average), "vector_norm_ci95_low": length_ci[0],
        "vector_norm_ci95_high": length_ci[1], "standardized_effect": np.sqrt(obs),
        "standardized_ci95_low": effect_ci[0], "standardized_ci95_high": effect_ci[1],
        "test": "two-sided location; upper tail of signed whole-profile Wald",
    }


def adjust(frame):
    p = frame.raw_p.fillna(1).to_numpy()
    keep = frame.raw_p.notna()
    order = np.argsort(p)
    holm = np.empty(len(p))
    holm[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return frame.assign(family_size=len(p), fdr_q=np.where(keep, false_discovery_control(p), np.nan),
                        holm_p=np.where(keep, holm, np.nan))


def p0(matrices, legacy):
    on, off = paired_cells(matrices, NONDIAGONAL, "p_weida")
    delta = (on-off).to_numpy(float)
    observed, randomized, bootstrap = design(delta, 105)
    n, _ = delta.shape
    all_ridges = []
    for ridge in CFG["ridge_sensitivity"]:
        all_ridges.append({"scope": "fixed_primary" if ridge == CFG["ridge"] else "fixed_sensitivity",
                           **omnibus_result(observed, randomized, bootstrap, ridge, n)})
    sensitivity = pd.DataFrame(all_ridges)
    primary = adjust(sensitivity.loc[sensitivity.scope.eq("fixed_primary")].copy())
    archive = legacy.loc[legacy.endpoint.eq("transition_omnibus")].iloc[0]
    np.testing.assert_allclose(primary.wald_statistic.item(), archive.statistic, rtol=0, atol=1e-10)
    np.testing.assert_allclose(primary.vector_norm.item(), archive.mean_difference_norm, rtol=0, atol=1e-12)
    primary = primary.assign(contrast="ON-OFF", method="Round5 A3 WEiDA non-diagonal",
                             legacy_v6_raw_p=archive.raw_p, legacy_v6_holm_p=archive.holm_p,
                             legacy_v6_wald=archive.statistic, legacy_v6_norm=archive.mean_difference_norm,
                             legacy_v6_family_size=archive.family_size)
    edge_names = [f"{source}->{target}" for source, target in NONDIAGONAL]
    average, covariance = observed
    low, high = np.quantile(bootstrap[0], (.025, .975), axis=0)
    vector = pd.DataFrame({
        "edge": edge_names, "from_state": [i for i, _ in NONDIAGONAL], "to_state": [j for _, j in NONDIAGONAL],
        "n_pairs": n, "mean_on": on.mean().to_numpy(), "mean_off": off.mean().to_numpy(),
        "mean_difference": average, "ci95_low": low, "ci95_high": high,
    })
    covariance_table = pd.DataFrame({
        "edge_i": np.repeat(edge_names, len(edge_names)), "edge_j": np.tile(edge_names, len(edge_names)),
        "n_pairs": n, "sample_covariance": covariance.ravel(),
        "shrunk_covariance": shrink_covariance(covariance, CFG["ridge"]).ravel(),
        "pearson_r": np.corrcoef(delta, rowvar=False).ravel(),
    })
    return {
        "Omnibus_Transition_Results": primary, "Omnibus_Transition_Vector": vector,
        "Omnibus_Transition_Covariance": covariance_table, "Omnibus_Transition_Sensitivity": sensitivity,
    }, on-off
