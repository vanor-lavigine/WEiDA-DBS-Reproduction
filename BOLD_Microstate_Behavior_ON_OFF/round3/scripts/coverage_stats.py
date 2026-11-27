"""Calculate fixed Coverage association families on identical case sets."""

import numpy as np
import pandas as pd
from scipy import stats

from round3_io import CFG, ENDPOINTS, STATES

TEST = CFG["statistics"]
DEFINITIONS = CFG["clinical_definitions"]


def summarize(vector, prefix):
    a = np.asarray(vector, dtype=float)
    q1, median, q3 = np.percentile(a, [25, 50, 75])
    return {f"{prefix}_mean": a.mean(), f"{prefix}_median": median, f"{prefix}_q25": q1, f"{prefix}_q75": q3}


def coverage_summary(pairs):
    rows = []
    tolerance = CFG["imaging"]["direction_tolerance"]
    for i, state in enumerate(STATES, 1):
        off, on, change = (pairs[name] for name in (f"coverage_{i}_off", f"coverage_{i}_on", f"delta_coverage_{i}"))
        rows.append(dict(state=state, n_pairs=len(pairs), **summarize(off, "off"), **summarize(on, "on"),
                         **summarize(change, "delta"), n_increased=int(change.gt(tolerance).sum()),
                         n_unchanged=int(change.abs().le(tolerance).sum()),
                         n_decreased=int(change.lt(-tolerance).sum())))
    return pd.DataFrame(rows)


def clinical_eligibility(pairs):
    rows = []
    for name in ENDPOINTS:
        off, on, absolute, rate = (pairs[f"{name}_{k}"].to_numpy(float) for k in
                                  ("off", "on", "improvement", "improvement_rate"))
        zero = off == 0
        missing = ~np.isfinite(rate) & ~zero
        rows.append(dict(clinical_endpoint=name, clinical_definition=DEFINITIONS[name], n_pairs=len(pairs),
                         n_valid_rate=int(np.isfinite(rate).sum()), n_off_zero=int(zero.sum()),
                         n_missing=int(missing.sum()), n_missing_off_score=int((missing & ~np.isfinite(off)).sum()),
                         n_missing_on_score=int((missing & np.isfinite(off) & ~np.isfinite(on)).sum()),
                         n_valid_absolute=int(np.isfinite(absolute).sum())))
    return pd.DataFrame(rows)


def spearman(x, y):
    if len(x) < TEST["minimum_n"]:
        return np.nan, np.nan, "insufficient_pairs"
    const_x, const_y = np.ptp(x) == 0, np.ptp(y) == 0
    if const_x or const_y:
        reason = ("both_constant" if const_x and const_y else
                  "constant_coverage_change" if const_x else "constant_clinical_improvement")
        return np.nan, np.nan, reason
    result = stats.spearmanr(x, y, alternative=TEST["alternative"])
    if not np.isfinite(result.statistic) or not np.isfinite(result.pvalue):
        return np.nan, np.nan, "undefined_spearman"
    return float(result.statistic), float(result.pvalue), ""


def adjust(frame):
    if len(frame) != TEST["bh_family_size"]:
        raise ValueError("Fixed BH family size differs from analysis slots")
    raw = frame.raw_p.to_numpy(float)
    finite = np.isfinite(raw)
    adjusted = stats.false_discovery_control(np.where(finite, raw, TEST["unestimable_placeholder"]), method="bh")
    frame = frame.assign(fdr_q=np.where(finite, adjusted, np.nan), bh_family_size=TEST["bh_family_size"])
    passed = frame.fdr_q.lt(TEST["alpha"])
    return frame.assign(passes_fdr=passed, evidence=np.select(
        [~finite, passed, frame.raw_p.lt(TEST["alpha"])],
        ["unestimable", "fdr_supported", "nominal_only"], default="no_evidence",
    ))


def associate(pairs):
    main, secondary, eligible = [], [], []
    for i, state in enumerate(STATES, 1):
        delta = pairs[f"delta_coverage_{i}"].to_numpy(float)
        for endpoint in ENDPOINTS:
            off, rate, absolute = (pairs[f"{endpoint}_{k}"].to_numpy(float)
                                   for k in ("off", "improvement_rate", "improvement"))
            off_zero = off == 0
            use = np.isfinite(delta) & np.isfinite(rate) & np.isfinite(absolute) & ~off_zero
            eligible.append(pd.DataFrame(dict(
                patient_id=pairs.loc[use, "patient_id"].to_numpy(), state=state, clinical_endpoint=endpoint,
                delta_coverage=delta[use], improvement_rate=rate[use], absolute_improvement=absolute[use],
            )))
            common = dict(state=state, clinical_endpoint=endpoint, clinical_definition=DEFINITIONS[endpoint],
                          n=int(use.sum()), n_off_zero=int(off_zero.sum()),
                          n_missing=int((~use & ~off_zero).sum()), n_excluded=int((~use).sum()))
            for matrix, values, analysis in ((main, rate, "rate"), (secondary, absolute, "absolute_same_cases")):
                rho, p, reason = spearman(delta[use], values[use])
                direction = ("positive" if rho > 0 else "negative" if rho < 0 else "zero" if rho == 0 else "NA")
                matrix.append(dict(**common, analysis=analysis, rho=rho, raw_p=p, direction=direction,
                                   reason=reason))
    return adjust(pd.DataFrame(main)), adjust(pd.DataFrame(secondary)), pd.concat(eligible, ignore_index=True)
