"""Participant-level inference for the prespecified A3 transition families."""

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


V4 = Path(__file__).resolve().parents[4] / "V4_WEiDA_Statistical_Framework_Microstate" / "scripts"
sys.path.insert(0, str(V4))
try:
    from statistics_v4 import Resampling, interval, normality, rank_biserial, standardized_effect
finally:
    sys.path.pop(0)

STATES = ("M1", "M2", "M3", "M4")
CONDITIONS = ("HC", "OFF", "ON")
CONTRASTS = ("OFF-HC", "ON-OFF", "ON-HC")
METHODS = {"WEiDA": "p_weida", "ExitConditional": "p_exit"}
ENDPOINTS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
MINIMUM_N = 3
RANK_DECIMALS = 12


def _resampling(config):
    return Resampling(seed=config["seed"], permutations=0, bootstrap=config["bootstrap_draws"])


def _direction(value):
    if not np.isfinite(value):
        return "NA"
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _edge_frame(probabilities, source, target):
    edge = probabilities.loc[probabilities.from_state.eq(source) & probabilities.to_state.eq(target)]
    return edge.sort_values("participant_id")


def _description(values, resampling):
    n = len(values)
    q25, median, q75 = np.quantile(values, [.25, .5, .75]) if n else [np.nan] * 3
    low, high = np.nan, np.nan
    if n >= MINIMUM_N:
        ix, _ = resampling.resample(n, n, True)
        low, high = interval(values[ix].mean(axis=1))
    return dict(n=n, mean=values.mean() if n else np.nan, sd=values.std(ddof=1) if n > 1 else np.nan,
                median=median, q25=q25, q75=q75, mean_ci95_low=low, mean_ci95_high=high)


def _samples(edge, column, first, second, paired):
    grouped = [edge.loc[edge.condition.eq(group), ["participant_id", column]] for group in (first, second)]
    undefined = [int(frame[column].isna().sum()) for frame in grouped]
    if paired:
        view = edge.loc[edge.condition.isin((first, second))].pivot(
            index="participant_id", columns="condition", values=column)
        if len(view) != len(grouped[0]) or len(view) != len(grouped[1]):
            raise ValueError("ON/OFF participant cohorts differ")
        view = view.dropna(subset=[first, second])
        x, y = view[first].to_numpy(float), view[second].to_numpy(float)
    else:
        x, y = (frame[column].dropna().to_numpy(float) for frame in grouped)
    return x, y, dict(n_total_a=len(grouped[0]), n_total_b=len(grouped[1]),
                      n_undefined_a=undefined[0], n_undefined_b=undefined[1],
                      n_pairs_missing=len(grouped[0]) - len(x) if paired else np.nan)


def _raw_test(x, y, paired, nonconstant_normal_alpha):
    diagnostics = [normality(x-y)] if paired else [normality(x), normality(y)]
    normal = all(p >= nonconstant_normal_alpha for p in diagnostics)
    if paired and normal:
        name = "paired t"
        statistic, p = stats.ttest_rel(x, y, alternative="two-sided")
    elif paired:
        name = "Wilcoxon signed-rank"
        differences = (x-y).round(RANK_DECIMALS)
        statistic, p = (stats.wilcoxon(differences, alternative="two-sided", method="approx", zero_method="wilcox")
                        if np.any(differences != 0) else (0., 1.))
    elif normal:
        name = "Welch t"
        statistic, p = stats.ttest_ind(x, y, equal_var=False, alternative="two-sided")
    else:
        name = "Mann–Whitney U"
        statistic, p = stats.mannwhitneyu(x.round(RANK_DECIMALS), y.round(RANK_DECIMALS),
                                        method="asymptotic", alternative="two-sided", use_continuity=True)
    return normal, dict(test=name, statistic=statistic, raw_p=p,
                        normality_p_difference=diagnostics[0] if paired else np.nan,
                        normality_p_a=np.nan if paired else diagnostics[0],
                        normality_p_b=np.nan if paired else diagnostics[1])


def _comparison(x, y, paired, infer, config, resampling):
    n_a, n_b = len(x), len(y)
    mean_a, mean_b = (v.mean() if len(v) else np.nan for v in (x, y))
    mean_diff = (x-y).mean() if paired and n_a else mean_a-mean_b
    row = dict(n_a=n_a, n_b=n_b, n_pairs=n_a if paired else np.nan, paired=paired,
               mean_a=mean_a, mean_b=mean_b, mean_diff=mean_diff, direction=_direction(mean_diff),
               sd_a=x.std(ddof=1) if n_a > 1 else np.nan, sd_b=y.std(ddof=1) if n_b > 1 else np.nan,
               median_a=np.median(x) if n_a else np.nan, median_b=np.median(y) if n_b else np.nan,
               status="descriptive_only" if not infer else "ok", reason="", raw_p=np.nan,
               ci95_low=np.nan, ci95_high=np.nan)
    if paired:
        differences = (x-y).round(RANK_DECIMALS)
        row.update(paired_higher=np.count_nonzero(differences > 0), paired_lower=np.count_nonzero(differences < 0),
                   paired_same=np.count_nonzero(differences == 0))
    if min(n_a, n_b) < MINIMUM_N:
        row.update(status="unestimable" if infer else "descriptive_only", reason="insufficient_defined_observations")
        return row

    ix, iy = resampling.resample(n_a, n_b, paired)
    bx, by = x[ix], y[iy]
    if paired:
        low, high = interval((bx-by).mean(axis=1))
    else:
        low, high = interval(bx.mean(axis=1)-by.mean(axis=1))
    row.update(ci95_low=low, ci95_high=high)
    if not infer:
        return row

    normal, test = _raw_test(x, y, paired, config["normality_alpha"])
    g_draws = standardized_effect(bx, by, paired)
    g_low, g_high = interval(g_draws)
    row.update(test, standardized_effect=float(standardized_effect(x, y, paired)),
               standardized_type="Hedges g_z" if paired else "Hedges g",
               standardized_ci95_low=g_low, standardized_ci95_high=g_high,
               standardized_bootstrap_valid=np.isfinite(g_draws).sum())
    if not normal:
        rank_low, rank_high = interval(rank_biserial(bx, by, paired))
        row.update(rank_biserial=float(rank_biserial(x, y, paired)),
                   rank_biserial_ci95_low=rank_low, rank_biserial_ci95_high=rank_high)
    if not np.isfinite(row["raw_p"]):
        row.update(status="unestimable", reason="undefined_test_statistic")
    return row


def _correct(frame, family_size, alpha, holm=True):
    if len(frame) != family_size:
        raise ValueError("Multiplicity rows differ from the prespecified family size")
    known = frame.raw_p.notna().to_numpy()
    p = frame.raw_p.fillna(1.0).to_numpy(float)
    frame = frame.assign(family_size=family_size, fdr_q=np.where(known, multipletests(p, method="fdr_bh")[1], np.nan))
    frame["passes_fdr"] = frame.fdr_q.lt(alpha)
    if holm:
        frame["holm_p"] = np.where(known, multipletests(p, method="holm")[1], np.nan)
        frame["passes_holm"] = frame.holm_p.lt(alpha)
    return frame


def group_statistics(probabilities, config):
    """Aggregate all defined cells and correct the two separate 36-test families."""
    resampling = _resampling(config)
    descriptors, comparisons = [], []
    for method, column in METHODS.items():
        for source, target in product(STATES, repeat=2):
            infer = source != target
            if method == "ExitConditional" and not infer:
                continue
            edge = _edge_frame(probabilities, source, target)
            keys = dict(method=method, from_state=source, to_state=target, unit="proportion")
            n_defined = {}
            for condition in CONDITIONS:
                series = edge.loc[edge.condition.eq(condition), column]
                values = series.dropna().to_numpy(float)
                n_defined[f"n_defined_{condition}"] = len(values)
                descriptors.append(dict(**keys, condition=condition, n_total=len(series),
                                        n_undefined=len(series)-len(values), **_description(values, resampling)))
            for contrast in CONTRASTS:
                first, second = contrast.split("-")
                paired = contrast == "ON-OFF"
                x, y, eligibility = _samples(edge, column, first, second, paired)
                comparisons.append(dict(**keys, **n_defined, contrast=contrast, group_a=first, group_b=second,
                                        scope="prespecified" if infer else "descriptive_only",
                                        **eligibility, **_comparison(x, y, paired, infer, config, resampling)))

    tested = pd.DataFrame(comparisons)
    for method in METHODS:
        use = tested.method.eq(method) & tested.scope.eq("prespecified")
        corrected = _correct(tested.loc[use], 36, config["alpha"])
        columns = ["family_size", "fdr_q", "passes_fdr", "holm_p", "passes_holm"]
        for column in columns:
            if column not in tested:
                tested[column] = pd.Series(index=tested.index, dtype="boolean" if column.startswith("passes_") else float)
            tested.loc[use, column] = corrected[column]
    correction_cols = ["method", "from_state", "to_state", "contrast", "family_size", "status", "reason",
                       "raw_p", "fdr_q", "holm_p", "passes_fdr", "passes_holm"]
    corrections = tested.loc[tested.scope.eq("prespecified"), correction_cols].reset_index(drop=True)
    return pd.DataFrame(descriptors), tested, corrections


def _spearman(x, y, resampling):
    result = dict(rho=np.nan, raw_p=np.nan, ci95_low=np.nan, ci95_high=np.nan,
                  reason="", ci_reason="", bootstrap_valid=0)
    if len(x) < MINIMUM_N:
        return dict(result, reason="insufficient_defined_pairs")
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        reason = "both_constant" if np.ptp(x) == np.ptp(y) == 0 else (
            "constant_transition_change" if np.ptp(x) == 0 else "constant_clinical_improvement")
        return dict(result, reason=reason)
    rho, p = stats.spearmanr(x, y, alternative="two-sided")
    if not np.isfinite(rho) or not np.isfinite(p):
        return dict(result, reason="undefined_correlation")

    ix, _ = resampling.resample(len(x), len(x), True)
    rank_x, rank_y = (stats.rankdata(v[ix], axis=1) for v in (x, y))
    rank_x -= rank_x.mean(axis=1, keepdims=True)
    rank_y -= rank_y.mean(axis=1, keepdims=True)
    scale = np.linalg.norm(rank_x, axis=1) * np.linalg.norm(rank_y, axis=1)
    bootstrap = np.divide((rank_x*rank_y).sum(axis=1), scale, where=scale > 0,
                          out=np.full(len(ix), np.nan)).clip(-1, 1)
    low, high = interval(bootstrap)
    return dict(result, rho=rho, raw_p=p, ci95_low=low, ci95_high=high,
                bootstrap_valid=np.isfinite(bootstrap).sum(),
                ci_reason="" if np.isfinite(low) else "fewer_than_95_percent_of_resamples_defined")


def clinical_associations(probabilities, clinical, config):
    """Rank all 16 paired WEiDA changes against the five absolute score improvements."""
    scores = clinical.set_index("patient_id", verify_integrity=True)
    resampling = _resampling(config)
    rows = []
    for source, target in product(STATES, repeat=2):
        edge = _edge_frame(probabilities, source, target)
        patients = edge.loc[edge.condition.isin(["ON", "OFF"])].pivot(
            index="participant_id", columns="condition", values="p_weida")
        if set(patients.index) != set(scores.index):
            raise ValueError("Frozen ON/OFF and clinical participants differ")
        deltas = (patients.ON-patients.OFF).round(RANK_DECIMALS).to_numpy(float)
        for endpoint in ENDPOINTS:
            changes = scores.loc[patients.index, f"{endpoint}_improvement"].to_numpy(float)
            use = np.isfinite(deltas) & np.isfinite(changes)
            fit = _spearman(deltas[use], changes[use], resampling)
            rows.append(dict(method="WEiDA", from_state=source, to_state=target, clinical_endpoint=endpoint,
                             transition_change_definition="ON-OFF", score_improvement_definition="OFF-ON",
                             n_pairs_total=len(use), n_pairs=int(use.sum()), n_pairs_missing=int((~use).sum()),
                             n_missing_transition=int((~np.isfinite(deltas)).sum()),
                             n_missing_clinical=int((~np.isfinite(changes)).sum()),
                             direction=_direction(fit["rho"]), status="unestimable" if fit["reason"] else "ok", **fit))
    return _correct(pd.DataFrame(rows), 80, config["alpha"], holm=False)
