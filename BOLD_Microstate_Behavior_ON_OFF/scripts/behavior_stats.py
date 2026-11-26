"""Frozen ON–OFF comparisons and the twenty registered correlations."""

import numpy as np
import pandas as pd
from scipy import stats

from behavior_protocol import BEHAVIORS, KEYS, REGISTRY, STATES, STATS


def registry_check():
    registry = pd.read_csv(REGISTRY, dtype=str)
    expected = ([("A", "clinical", name) for name in BEHAVIORS]
                + [("B", "duration", state) for state in STATES]
                + [("C", "association", f"{state}:{name}") for state in STATES for name in BEHAVIORS])
    if list(registry.itertuples(index=False, name=None)) != expected:
        raise ValueError("The registered list does not match the locked protocol")


def matched_pairs(scores, scans):
    """Inner join on the person plus both audited visits."""
    for frame in (scores, scans):
        if frame[KEYS].isna().any().any() or any(not frame[name].is_unique for name in KEYS):
            raise ValueError("A clinical or imaging patient/visit key is absent or duplicated")
    joined = scores.merge(scans, on=KEYS, how="inner", validate="one_to_one").sort_values("patient_id", ignore_index=True)
    for index, _ in enumerate(STATES, 1):
        col = f"duration_{index}"
        joined[f"{col}_change"] = joined[f"{col}_on"] - joined[f"{col}_off"]
    for metric in BEHAVIORS:
        off, on = (joined[f"{metric}_{c}"].to_numpy(float) for c in ("off", "on"))
        improvement = off-on
        rate = np.divide(improvement, off, out=np.full(len(off), np.nan), where=np.isfinite(off) & (off > 0))
        joined[f"{metric}_improvement"] = improvement
        joined[f"{metric}_improvement_rate"] = rate
    return joined


def _average(x):
    return float(np.mean(x)) if len(x) else np.nan


def _median(x):
    return float(np.median(x)) if len(x) else np.nan


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) >= 2 else np.nan


def paired_test(delta):
    rounded = np.round(delta, STATS["wilcoxon_round_digits"])
    if len(delta) and np.all(delta == 0):
        return "all_zero", 0., 1., np.nan, ""
    if len(delta) < STATS["minimum_n"]:
        return "NA", np.nan, np.nan, np.nan, "fewer_than_three_complete_pairs"
    normality = float(stats.shapiro(delta).pvalue) if np.ptp(delta) > 0 else np.nan
    if normality >= STATS["normality_alpha"]:
        result = stats.ttest_1samp(delta, 0., alternative=STATS["alternative"])
        return "paired_t", float(result.statistic), float(result.pvalue), normality, ""
    if not rounded.any():
        return "Wilcoxon_signed_rank", 0., 1., normality, ""
    result = stats.wilcoxon(rounded, zero_method="wilcox", correction=False,
                            alternative=STATS["alternative"], method="approx")
    return "Wilcoxon_signed_rank", float(result.statistic), float(result.pvalue), normality, ""


def main_comparisons(frame, family):
    clinical = family == "clinical"
    endpoints = BEHAVIORS if clinical else STATES
    rows = []
    for idx, name in enumerate(endpoints):
        col = name if clinical else f"duration_{idx+1}"
        off, on = (frame[f"{col}_{c}"].to_numpy(float) for c in ("off", "on"))
        both = np.isfinite(off) & np.isfinite(on)
        delta = off[both]-on[both] if clinical else on[both]-off[both]
        n = len(delta)
        test, statistic, p, shapiro_p, reason = paired_test(delta)
        interval = (np.nan, np.nan)
        if n:
            rng = np.random.default_rng(np.random.SeedSequence([STATS["seed"], int(clinical), idx]))
            sample = delta[rng.integers(0, n, (STATS["bootstraps"], n))]
            interval = np.percentile(sample.mean(axis=1), [2.5, 97.5])
        results = dict(
            endpoint=name, family=family, direction="OFF minus ON" if clinical else "ON minus OFF",
            unit="score points" if clinical else "seconds", n=n, missing_pairs=len(frame)-n,
            on_mean=_average(on[both]), off_mean=_average(off[both]), on_median=_median(on[both]),
            off_median=_median(off[both]), on_sd=_sd(on[both]), off_sd=_sd(off[both]),
            mean_change=_average(delta), median_change=_median(delta), sd_change=_sd(delta),
            ci95_low=interval[0], ci95_high=interval[1], shapiro_p=shapiro_p,
            test=test, statistic=statistic, raw_p=p, reason=reason,
            n_increased=int(np.sum(delta > 0)), n_unchanged=int(np.sum(delta == 0)), n_decreased=int(np.sum(delta < 0)),
        )
        if clinical:
            rate = np.divide(delta, off[both], out=np.full(n, np.nan), where=off[both] > 0)
            observed = rate[np.isfinite(rate)]
            results.update(rate_fraction_mean=_average(observed), rate_fraction_median=_median(observed),
                           n_rate=len(observed), n_off_zero=int(np.sum(off[both] == 0)))
        rows.append(results)
    return adjust(pd.DataFrame(rows), family)


def descriptive_clinical(frame):
    rows = []
    for name in BEHAVIORS:
        a, b = (frame[f"{name}_{key}"].to_numpy(float) for key in ("off", "on"))
        keep = np.isfinite(a) & np.isfinite(b)
        off, on = a[keep], b[keep]
        improvement = off-on
        rate = np.divide(improvement, off, out=np.full(len(off), np.nan), where=off > 0)
        rows.append(dict(behavior=name, n=int(keep.sum()), off_mean=_average(off), on_mean=_average(on),
                         improvement_mean=_average(improvement), improvement_median=_median(improvement),
                         n_improved=int(np.sum(improvement > 0)), n_unchanged=int(np.sum(improvement == 0)),
                         n_worse=int(np.sum(improvement < 0)), n_rate=int(np.isfinite(rate).sum()),
                         mean_improvement_rate=_average(rate[np.isfinite(rate)])))
    return pd.DataFrame(rows)


def adjust(table, family):
    size = STATS["bh_slots"][family]
    if len(table) != size:
        raise ValueError(f"{family} requires {size} registered tests")
    raw = table.raw_p.to_numpy(float)
    valid = np.isfinite(raw)
    q = stats.false_discovery_control(np.where(valid, raw, STATS["missing_slot_p"]), method="bh")
    result = table.assign(fdr_q=np.where(valid, q, np.nan), bh_family_size=size)
    return result.assign(passes_fdr=result.fdr_q.lt(STATS["alpha"]))


def association(frame, sample):
    rows = []
    for idx, state in enumerate(STATES, 1):
        for name in BEHAVIORS:
            xx = frame[f"duration_{idx}_change"].to_numpy(float)
            yy = frame[f"{name}_improvement"].to_numpy(float)
            keep = np.isfinite(xx) & np.isfinite(yy)
            x, y = xx[keep], yy[keep]
            rho, p, reason = np.nan, np.nan, ""
            if len(x) < STATS["minimum_n"]:
                reason = "fewer_than_three_complete_pairs"
            elif np.ptp(x) == 0 or np.ptp(y) == 0:
                reason = "constant_duration" if np.ptp(x) == 0 else "constant_score_change"
                if np.ptp(x) == 0 and np.ptp(y) == 0:
                    reason = "both_changes_constant"
            else:
                result = stats.spearmanr(x, y, alternative=STATS["alternative"])
                rho, p = float(result.statistic), float(result.pvalue)
                if not (np.isfinite(rho) and np.isfinite(p)):
                    reason = "undefined_spearman"
            direction = "negative" if rho < 0 else "positive" if rho > 0 else "zero" if rho == 0 else "NA"
            rows.append(dict(sample=sample, state=state, behavior=name, n=len(x), missing_pairs=len(frame)-len(x),
                             rho=rho, raw_p=p, direction=direction, reason=reason))
    return adjust(pd.DataFrame(rows), "association")
