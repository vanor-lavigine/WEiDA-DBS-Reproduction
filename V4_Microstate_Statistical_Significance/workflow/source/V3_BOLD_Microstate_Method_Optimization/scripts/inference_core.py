"""Condition contrasts with scan FD and participant-level dependence."""
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import linalg, optimize, stats
from statsmodels.stats.multitest import multipletests

CONTRASTS = {"C1": (1, 0), "C2": (0, 1), "C3": (1, -1), "C4": (.5, .5)}
TR = 2.1


def model_data(df):
    """Drop unused condition columns for prespecified two-group analyses."""
    conditions = set(df.condition)
    if conditions == {"HC", "ON", "OFF"}:
        cols = ["ON", "OFF"]
        weights = {c: np.r_[0., w, 0.] for c, w in CONTRASTS.items()}
    elif conditions in ({"HC", "ON"}, {"HC", "OFF"}):
        cols = sorted(conditions - {"HC"})
        weights = {"C1" if cols == ["ON"] else "C2": np.array([0., 1., 0.])}
    elif conditions == {"ON", "OFF"}:
        cols = ["ON"]
        weights = {"C3": np.array([0., 1., 0.])}
    else:
        raise ValueError("unsupported observed conditions")
    x = np.column_stack([np.ones(len(df)), *((df.condition == c).astype(float) for c in cols),
                         df.fd - df.fd.mean()])
    groups = pd.factorize(df.subject, sort=True)[0]
    if np.linalg.matrix_rank(x) != x.shape[1] or df.groupby("condition").subject.nunique().min() < 3:
        raise ValueError("insufficient independent observations or singular design")
    return x, groups, weights


def score_cov(x, u, groups):
    n, k = x.shape
    g = max(groups) + 1
    scores = np.zeros((g, k))
    np.add.at(scores, groups, x * u[:, None])
    bread = np.linalg.inv(x.T @ x)
    return bread @ scores.T @ scores @ bread * g / (g - 1) * (n - 1) / (n - k)


def scalar_results(beta, cov, weights, df, divisor=np.nan):
    out = {}
    for c, w in weights.items():
        value = w @ beta
        variance = w @ cov @ w
        se = np.sqrt(variance) if variance > 0 else np.nan
        margin = stats.t.ppf(.975, df) * se
        out[c] = dict(Effect_Size=value, Standardized_Effect=value / divisor, SE=se,
                      CI_low=value - margin, CI_high=value + margin, inference_df=df,
                      P_raw=2 * stats.t.sf(abs(value / se), df))
    return out


def scalar(df, metric):
    x, groups, weights = model_data(df)
    y = df[metric].to_numpy(float)
    if np.ptp(y) < np.finfo(float).eps * max(1, np.abs(y).max()) * 8:
        raise ValueError("constant outcome")
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    cov = score_cov(x, residual, groups)
    sd = np.linalg.norm(residual) / np.sqrt(len(df) - x.shape[1])
    return scalar_results(beta, cov, weights, max(groups), sd)


def paired_change(df, metric):
    """Paired first difference regression conditional on a within-person FD change of zero."""
    visits = df[df.condition.isin(["ON", "OFF"])].pivot(index="subject", columns="condition",
                                                         values=[metric, "fd"]).dropna()
    change = visits[metric]["ON"] - visits[metric]["OFF"]
    fd_diff = visits["fd"]["ON"] - visits["fd"]["OFF"]
    design = np.column_stack([np.ones(len(visits)), fd_diff])
    if len(visits) < 4 or np.linalg.matrix_rank(design) != 2:
        raise ValueError("insufficient FD-adjusted complete pairs")
    model = sm.OLS(change, design).fit(cov_type="HC3", use_t=True)
    return scalar_results(model.params, model.cov_params(), {"C3":np.array([1., 0.])}, len(visits)-2)


def random_intercept(df, metric):
    """REML for a random subject intercept. Includes variance=0 boundary."""
    x, groups, weights = model_data(df)
    y = df[metric].to_numpy(float)
    n, k = x.shape
    ng = np.bincount(groups)
    g = len(ng)
    if (ng > 1).sum() < 2:
        raise ValueError("no repeated visits for variance component")
    z = np.column_stack((x, y))
    sums = np.zeros((g, k + 1))
    np.add.at(sums, groups, z)
    ztz = z.T @ z

    def profile(ratio, details=False):
        # Invert each I + ratio * J block using the rank-one identity.
        cross = ztz - sums.T @ ((ratio / (1 + ng * ratio))[:, None] * sums)
        bread = np.linalg.inv(cross[:k, :k])
        beta = bread @ cross[:k, k]
        sigma2 = (cross[k, k] - beta @ cross[:k, k]) / (n - k)
        if sigma2 <= 0:
            return np.inf
        objective = (n - k) * np.log(sigma2) + np.log1p(ratio * ng).sum() + np.linalg.slogdet(cross[:k, :k])[1]
        return (beta, sigma2 * bread) if details else objective

    opt = optimize.minimize_scalar(lambda v: profile(np.expm1(v)), bounds=(0, np.log1p(1e5)), method="bounded")
    ratio = np.expm1(opt.x) if opt.success and opt.fun < profile(0) else 0.
    if ratio > .999e5:
        raise ValueError("random intercept profile at upper numerical bound")
    beta, cov = profile(ratio, True)
    return {c: dict(row, variance_ratio=ratio) for c, row in scalar_results(beta, cov, weights, g-1).items()}


def count_results(df, counts, duration):
    x, groups, weights = model_data(df)
    y = np.asarray(counts, float)
    if not np.allclose(y, np.round(y)) or min(y) < 0 or y.sum() == 0:
        raise ValueError("invalid or all-zero episode counts")
    offset = np.log(np.asarray(duration))
    options = dict(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True})
    fit = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset).fit(**options)
    phi = fit.pearson_chi2 / fit.df_resid
    alpha = 0.
    if phi > 1.5:
        mu = fit.fittedvalues
        alpha = max(1e-8, (((y-mu)**2 - mu).sum()) / (mu**2).sum())
        fit = sm.GLM(y, x, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit(**options)
    if not fit.converged:
        raise ValueError("count regression did not converge")
    output = scalar_results(fit.params, fit.cov_params(), weights, max(groups))
    for row in output.values():
        row.update(pearson_dispersion=phi, nb2_alpha=alpha, rate_ratio=np.exp(row["Effect_Size"]),
                   rr_ci_low=np.exp(row["CI_low"]), rr_ci_high=np.exp(row["CI_high"]))
    return output


def ilr_coordinates(coverage, duration):
    coverage = np.asarray(coverage, float)
    if np.any(coverage < 0) or not np.isfinite(coverage).all() or not np.allclose(coverage.sum(axis=1), 1, atol=1e-8):
        raise ValueError("invalid coverage simplex")
    counts = coverage * np.asarray(duration)[:, None] / TR
    return np.log(counts + .5) @ linalg.helmert(coverage.shape[1]).T


def composition(df, coverage_cols, draws=9999, seed=20261125):
    """Restricted wild cluster bootstrap, multivariate CR1-studentized contrast."""
    x, groups, weights = model_data(df)
    y = ilr_coordinates(df[coverage_cols], df.duration_s)
    n, k = x.shape
    q = y.shape[1]
    subjects = np.eye(max(groups)+1)[groups]
    g = subjects.shape[1]
    bread = np.linalg.inv(x.T @ x)
    inv = bread @ x.T
    beta = inv @ y
    factor = g / (g-1) * (n-1) / (n-k)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1., 1.], size=(draws, g))
    output = {}
    for c, w in weights.items():
        scale = w @ bread @ w
        observed = w @ beta
        null_beta = beta - np.outer(bread @ w, observed) / scale
        residual = y - x @ beta
        infl = ((x @ bread @ w)[:, None] * subjects).T
        scores = infl @ residual
        cov = factor * scores.T @ scores
        if np.linalg.matrix_rank(cov) < q:
            raise ValueError("singular observed ILR covariance")
        statistic = observed @ np.linalg.solve(cov, observed)
        # For each cluster sign flip, refit the unrestricted model to y*=Xb0+U0*sign;
        # common signs preserve patient visit and cross-coordinate dependence.
        null_resid = y - x @ null_beta
        exceed = 0
        valid = 0
        for batch in np.array_split(signs, max(1, int(np.ceil(draws / 256)))):
            u = null_resid[None, :, :] * batch[:, groups, None]
            delta = inv @ u
            estimate = np.einsum("k,bkj->bj", w, delta)
            scores = infl @ (u - x @ delta)
            covstar = factor * (scores.swapaxes(1, 2) @ scores)
            good = np.linalg.matrix_rank(covstar) == q
            estimate = estimate[good]
            solved = np.linalg.solve(covstar[good], estimate[..., None])[..., 0]
            boot = np.einsum("bi,bi->b", estimate, solved)
            exceed += np.sum(boot >= statistic)
            valid += good.sum()
        if valid != draws:
            raise ValueError("singular ILR resample; test withheld")
        radius = stats.t.ppf(.975, g-1) * np.sqrt(cov.diagonal())
        output[c] = dict(Effect_Size=np.linalg.norm(observed), Wald_stat=statistic, ILR_dimension=q,
                         ILR_difference=json.dumps(observed.tolist()), draws=valid,
                         ILR_pointwise_CI=json.dumps(np.column_stack((observed-radius, observed+radius)).tolist()),
                         P_raw=(1+exceed)/(1+valid))
    return output


def adjust(s, method):
    """Keep undefined prespecified tests in the multiplicity denominator."""
    p = np.asarray(s, float)
    keep = np.isfinite(p)
    out = multipletests(np.where(keep, p, 1), method=method)[1]
    return np.where(keep, out, np.nan)


def mde():
    rows = []
    for name, df, scale in (("HC vs patient", 38, np.sqrt(1/15 + 1/25)), ("paired ON vs OFF", 24, 1/5)):
        for alpha in (.05, .0125):
            cutoff = stats.t.ppf(1-alpha/2, df)
            def power(d):
                delta = d / scale
                return stats.nct.sf(cutoff, df, delta) + stats.nct.cdf(-cutoff, df, delta)
            d = optimize.brentq(lambda d: power(d)-.8, .01, 4)
            rows.append(dict(comparison=name, alpha=alpha, power=.8, standardized_mde=d))
    return pd.DataFrame(rows)
