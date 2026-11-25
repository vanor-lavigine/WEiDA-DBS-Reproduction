"""Frozen participant-level tests for the three contrasts."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import special, stats

from features import EDGES, STATE


@dataclass(frozen=True)
class Metric:
    metric: str
    family: str
    unit: str
    scope: str = "primary"
    state: str = ""


def measures(method):
    state_prefix = "M" if method == "Microstate" else "W"
    fields = [("coverage", "Coverage", "proportion"), ("duration", "Duration", "s"),
              ("occurrence", "Occurrence", "episodes/s")]
    if method == "Microstate":
        fields += [("gev", "GEV", "proportion"), ("dispersion", "Spatial Dispersion", "unit-vector distance")]
    else:
        fields += [("var", "Local Metastability (WEiDA VAR)", "WEiDA squared distance / ROI")]
    metrics = [Metric(f"{key}_{i}", family, unit, state=f"{state_prefix}{i}")
               for key, family, unit in fields for i in STATE]
    metrics += [Metric(f"transition_{i}_{j}", "Transition Probability", "proportion",
                       state=f"{state_prefix}{i}->{state_prefix}{j}") for i, j in EDGES]
    globals_ = [("coverage_entropy", "normalized entropy"), ("global_pooled_duration", "s"),
                ("switching_rate", "switches/s"), ("global_occurrence", "episodes/s")]
    if method == "Microstate":
        globals_ += [("total_gev", "proportion"), ("global_dispersion", "unit-vector distance")]
    metrics += [Metric(key, "Overall metrics", unit, scope="overall") for key, unit in globals_]
    if method == "WEiDA":
        metrics += [Metric(f"native_duration_{i}", "WEiDA native Dwell Time", "s", "native", f"W{i}") for i in STATE]
        metrics += [Metric(f"native_transition_{i}_{j}", "WEiDA native Transition Probability", "proportion", "native",
                           f"W{i}->W{j}") for i, j in EDGES]
    return metrics


def safe_ratio(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, where=denominator > 0,
                     out=np.full(numerator.shape, np.nan))


def studentize(effect, se):
    effect, se = np.broadcast_arrays(effect, se)
    empty = np.where(effect == 0, 0., np.copysign(np.inf, effect))
    return np.divide(effect, se, where=se > 0, out=empty)


def welch_stat(x, y):
    return studentize(np.mean(x, axis=-1) - np.mean(y, axis=-1),
                      np.sqrt(np.var(x, ddof=1, axis=-1) / x.shape[-1] +
                              np.var(y, ddof=1, axis=-1) / y.shape[-1]))


def tail_probability(null, observed, two_sided=True):
    null, observed = (np.abs(null), abs(observed)) if two_sided else (null, observed)
    if not np.isfinite(observed):
        return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))
    tol = 5e-13 * max(1., abs(observed))
    return float((1 + np.count_nonzero(null >= observed - tol)) / (len(null) + 1))


@dataclass(frozen=True)
class Resampling:
    seed: int
    permutations: int
    bootstrap: int

    def rng(self, salt, nx, ny):
        return np.random.default_rng(np.random.SeedSequence([self.seed, salt, nx, ny]))

    @lru_cache(maxsize=32)
    def independent(self, nx, ny):
        orders = self.rng(1, nx, ny).random((self.permutations, nx + ny)).argsort(axis=1)
        return orders

    @lru_cache(maxsize=32)
    def signs(self, n):
        return self.rng(2, n, 0).choice([-1., 1.], size=(self.permutations, n))

    @lru_cache(maxsize=32)
    def resample(self, nx, ny, paired):
        rng = self.rng(3 + int(paired), nx, ny)
        first = rng.integers(0, nx, size=(self.bootstrap, nx))
        second = first if paired else rng.integers(0, ny, size=(self.bootstrap, ny))
        return first, second


def normality(x):
    return float(stats.shapiro(x).pvalue) if len(x) >= 3 and np.ptp(x) > 0 else np.nan


def j_correction(df):
    return np.exp(special.gammaln(df / 2) - .5 * np.log(df / 2) - special.gammaln((df - 1) / 2))


def standardized_effect(x, y, paired):
    nx, ny = x.shape[-1], y.shape[-1]
    difference = x.mean(axis=-1) - y.mean(axis=-1)
    if paired:
        return j_correction(nx - 1) * safe_ratio(difference, np.std(x - y, axis=-1, ddof=1))
    df = nx + ny - 2
    sd = np.sqrt(((nx - 1) * np.var(x, axis=-1, ddof=1) + (ny - 1) * np.var(y, axis=-1, ddof=1)) / df)
    return j_correction(df) * safe_ratio(difference, sd)


def rank_biserial(x, y, paired):
    if paired:
        d = (x - y).round(12)
        ranks = stats.rankdata(np.where(d != 0, abs(d), np.nan), axis=-1, nan_policy="omit")
        score = np.nansum(np.sign(d) * ranks, axis=-1)
        denom = np.nansum(ranks, axis=-1)
        return np.divide(score, denom, out=np.zeros_like(score), where=denom > 0)
    diffs = x.round(12)[..., :, None] - y.round(12)[..., None, :]
    return np.sign(diffs).mean(axis=(-1, -2))


def interval(values):
    finite = np.isfinite(values)
    if finite.sum() < len(values) * .95:
        return np.nan, np.nan
    return np.quantile(np.asarray(values)[finite], [.025, .975])


def outcome_test(x, y, paired, cfg, resampling):
    delta = x - y if paired else None
    diag = [normality(delta)] if paired else [normality(x), normality(y)]
    normal = bool(np.all(np.array(diag) >= cfg["normality_alpha"]))
    if paired:
        if normal:
            label = "paired t"
            statistic, p_raw = stats.ttest_rel(x, y)
        else:
            label = "Wilcoxon signed-rank"
            rd = delta.round(12)
            statistic, p_raw = stats.wilcoxon(rd, method="approx") if np.any(rd != 0) else (0., 1.)
        # A sign flip fixes the sum of squared differences, so |t| is monotone in |mean|.
        p_perm = tail_probability(resampling.signs(len(x)) @ delta / len(delta), delta.mean())
    else:
        if normal:
            label = "Welch t"
            statistic, p_raw = stats.ttest_ind(x, y, equal_var=False)
        else:
            label = "Mann-Whitney U"
            statistic, p_raw = stats.mannwhitneyu(x.round(12), y.round(12), method="asymptotic")
        pooled = np.r_[x, y]
        order = resampling.independent(len(x), len(y))
        p_perm = tail_probability(welch_stat(pooled[order[:, :len(x)]], pooled[order[:, len(x):]]), welch_stat(x, y))
    ix, iy = resampling.resample(len(x), len(y), paired)
    bx, by = x[ix], y[iy]
    low, high = interval(bx.mean(axis=1) - by.mean(axis=1))
    gs = standardized_effect(bx, by, paired)
    g_low, g_high = interval(gs)
    rb, rb_low, rb_high = np.nan, np.nan, np.nan
    if not normal:
        rb = float(rank_biserial(x, y, paired))
        rb_low, rb_high = interval(rank_biserial(bx, by, paired))
    return dict(test=label, statistic=statistic, normality_p_difference=diag[0] if paired else np.nan,
                normality_p_a=diag[0] if not paired else np.nan, normality_p_b=diag[1] if not paired else np.nan,
                mean_a=x.mean(), mean_b=y.mean(), sd_a=x.std(ddof=1), sd_b=y.std(ddof=1),
                median_a=np.median(x), median_b=np.median(y), effect_size=x.mean() - y.mean(), ci95_low=low, ci95_high=high,
                standardized_effect=float(standardized_effect(x, y, paired)),
                standardized_type="Hedges g_z" if paired else "Hedges g",
                standardized_ci95_low=g_low, standardized_ci95_high=g_high,
                standardized_bootstrap_valid=np.isfinite(gs).sum(), rank_biserial=rb,
                rank_biserial_ci95_low=rb_low, rank_biserial_ci95_high=rb_high,
                raw_p=p_raw, permutation_p=p_perm)


def hc3_t(design, outcomes, index):
    """HC3 t for a single contrast, with rows of outcomes representing resamples."""
    pinv = np.linalg.pinv(design)
    coef = np.asarray(outcomes) @ pinv.T
    residual = outcomes - coef @ design.T
    leverage = np.sum(design * pinv.T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("HC3 undefined at leverage one")
    weight = pinv[index] / (1 - leverage)
    se = np.linalg.norm(residual * weight, axis=-1)
    return coef[..., index], studentize(coef[..., index], se)


def freedman_lane(x, y, fd_x, fd_y, paired, resampling):
    if paired:
        outcome = x - y
        delta_fd = fd_x - fd_y
        reduced = delta_fd[:, None]
        design = np.column_stack((np.ones(len(x)), delta_fd))
        index = 0
    else:
        outcome = np.r_[x, y]
        fd = np.r_[fd_x, fd_y]
        reduced = np.column_stack((np.ones(len(fd)), fd - fd.mean()))
        group = np.r_[np.ones(len(x)), np.zeros(len(y))]
        design = np.column_stack((reduced, group))
        index = 2
    if np.linalg.matrix_rank(design) != design.shape[1]:
        return dict(fd_permutation_status="singular FD design")
    fitted = reduced @ np.linalg.lstsq(reduced, outcome, rcond=None)[0]
    residual = outcome - fitted
    if paired:
        surrogate = fitted + resampling.signs(len(x)) * residual
    else:
        surrogate = fitted + residual[resampling.independent(len(x), len(y))]
    try:
        effect, observed = hc3_t(design, outcome, index)
        _, null = hc3_t(design, surrogate, index)
    except ValueError as exc:
        return dict(fd_permutation_status=str(exc))
    return dict(fd_permutation_p=tail_probability(null, observed), fd_permutation_t=float(observed),
                fd_permutation_effect=float(effect), fd_permutation_status="ok")


def select_samples(data, key, a, b, paired):
    records = data.loc[data.condition.isin([a, b]), ["subject", "condition", key, "fd"]]
    if paired:
        view = records.pivot(index="subject", columns="condition", values=[key, "fd"]).dropna()
        return view[key][a].to_numpy(), view[key][b].to_numpy(), view.fd[a].to_numpy(), view.fd[b].to_numpy()
    fields = [records.loc[records.condition.eq(g)].dropna().sort_values("subject") for g in (a, b)]
    return fields[0][key].to_numpy(), fields[1][key].to_numpy(), fields[0].fd.to_numpy(), fields[1].fd.to_numpy()


def corrections(data, p_columns, keys):
    from inference_core import adjust

    for raw, stem in p_columns.items():
        if raw not in data:
            data[raw] = np.nan
        for name, how in (("fdr", "fdr_bh"), ("holm", "holm")):
            target = f"{stem}{name}_p"
            data[target] = np.nan
            for ids in data.groupby(keys, sort=False).groups.values():
                data.loc[ids, target] = adjust(data.loc[ids, raw], how)


def infer(data, method, cfg, resampling, v3_archive=None, available=True):
    from inference_core import scalar

    v3_lookup = {} if v3_archive is None else v3_archive.set_index(["experiment", "metric", "contrast"]).to_dict("index")
    table = []
    for metric in measures(method):
        key = metric.metric
        subset = data.loc[np.isfinite(data[key])] if available else data
        condition_set = set(subset.condition) if available else set()
        adjusted = {}
        if condition_set == {"HC", "ON", "OFF"}:
            try:
                adjusted = scalar(subset, key)
            except (ValueError, np.linalg.LinAlgError):
                pass
        for contrast, v3_code in zip(cfg["contrasts"], ("C2", "C1", "C3")):
            first, second = contrast.split("-")
            paired = contrast == "ON-OFF"
            if available:
                x, y, fd_x, fd_y = select_samples(data, key, first, second, paired)
            else:
                x, y = [], []
            na, nb = len(x), len(y)
            status = "ok" if min(na, nb) >= 3 else "insufficient defined observations"
            if not available:
                status = "WEiDA scan-level evidence unavailable"
            row = dict(method=method, **vars(metric), contrast=contrast, group_a=first, group_b=second,
                       n_a=na, n_b=nb, n_pairs=na if paired else np.nan, paired=paired, status=status)
            if status == "ok":
                row.update(outcome_test(x, y, paired, cfg, resampling))
                row.update(freedman_lane(x, y, fd_x, fd_y, paired, resampling))
            if v3_code in adjusted:
                new = adjusted[v3_code]
                row.update(fd_cr1_effect=new["Effect_Size"], fd_cr1_ci95_low=new["CI_low"],
                           fd_cr1_ci95_high=new["CI_high"], fd_cr1_p=new["P_raw"])
            archive_experiment = cfg["v3_state_archive"] if metric.scope == "primary" else cfg["v3_method"]
            archived = v3_lookup.get((archive_experiment, key, v3_code))
            if archived is not None:
                row.update(v3_fd_archive_status=archived["status"], v3_fd_archive_effect=archived["Effect_Size"],
                           v3_fd_archive_ci95_low=archived["CI_low"], v3_fd_archive_ci95_high=archived["CI_high"],
                           v3_fd_archive_raw_p=archived["P_raw"], v3_fd_archive_fdr_p=archived["P_FDR"],
                           v3_fd_archive_holm_p=archived["P_Holm"], v3_fd_archive_across_methods_p=archived["P_across_methods"])
                if archived["status"] == "ok" and v3_code in adjusted:
                    np.testing.assert_allclose([row["fd_cr1_effect"], row["fd_cr1_p"], row["fd_cr1_ci95_low"], row["fd_cr1_ci95_high"]],
                                               [archived[k] for k in ("Effect_Size", "P_raw", "CI_low", "CI_high")],
                                               atol=1e-11, rtol=1e-8)
            table.append(row)
    output = pd.DataFrame(table)
    p_columns = {"raw_p": "", "permutation_p": "permutation_", "fd_cr1_p": "fd_cr1_",
                 "fd_permutation_p": "fd_permutation_"}
    corrections(output, p_columns, ["family"])
    output["family_size"] = output.groupby("family").metric.transform("size")
    core = output.scope.eq("primary")
    overall = output[core].copy()
    corrections(overall, {"raw_p": "all_primary_", "permutation_p": "all_primary_permutation_"}, ["scope"])
    for col in overall.filter(regex=r"^all_primary_"):
        output.loc[core, col] = overall[col]
    alpha = cfg["alpha"]
    output["family_holm_and_permutation_holm"] = output.holm_p.lt(alpha) & output.permutation_holm_p.lt(alpha)
    return output


def toward_hc(data, results, cfg, resampling):
    indexed = results.set_index(["metric", "contrast"])
    records = []
    for metric in measures("Microstate"):
        key = metric.metric
        hc = data.loc[data.condition.eq("HC")].sort_values("subject")[key].dropna().to_numpy()
        pairs = data.loc[data.condition.isin(["OFF", "ON"])].pivot(index="subject", columns="condition", values=key).dropna()
        on, off = pairs.ON.to_numpy(), pairs.OFF.to_numpy()
        row = dict(**vars(metric), n_hc=len(hc), n_pairs=len(pairs), equivalence="not assessed: no prespecified equivalence margin")
        if len(hc) < 3 or len(pairs) < 3:
            records.append(row)
            continue
        c, a, b = hc.mean(), on.mean(), off.mean()
        ix, iy = resampling.resample(len(on), len(hc), False)
        h = hc[iy].mean(axis=1)
        gain = abs(b - c) - abs(a - c)
        lo, hi = interval(np.abs(off[ix].mean(axis=1) - h) - np.abs(on[ix].mean(axis=1) - h))
        mean_middle = (a + b) / 2
        half_change = (resampling.signs(len(on)) @ (on - off)) / (2 * len(on))
        null = np.abs(mean_middle - half_change - c) - np.abs(mean_middle + half_change - c)
        row.update(off_minus_hc=b - c, on_minus_hc=a - c, on_minus_off=a - b, toward_hc_gain=gain, ci95_low=lo,
                   ci95_high=hi, toward_p=tail_probability(null, gain, two_sided=False), closer_in_sample=gain > 0,
                   paired_direction_toward=(a - b) * (b - c) < 0,
                   off_hc_supported=indexed.loc[(key, "OFF-HC"), "family_holm_and_permutation_holm"],
                   on_hc_supported=indexed.loc[(key, "ON-HC"), "family_holm_and_permutation_holm"],
                   on_off_supported=indexed.loc[(key, "ON-OFF"), "family_holm_and_permutation_holm"],
                   on_hc_fdr_p=indexed.loc[(key, "ON-HC"), "fdr_p"],
                   on_hc_holm_p=indexed.loc[(key, "ON-HC"), "holm_p"])
        records.append(row)
    table = pd.DataFrame(records)
    corrections(table, {"toward_p": "toward_"}, ["scope"])
    table["off_detected_on_not_detected"] = table.off_hc_supported.eq(True) & table.on_hc_supported.eq(False)
    table["paired_change_toward_hc"] = (table.off_hc_supported.eq(True) & table.on_off_supported.eq(True)
                                       & table.paired_direction_toward.eq(True) & table.closer_in_sample.eq(True))
    return table
