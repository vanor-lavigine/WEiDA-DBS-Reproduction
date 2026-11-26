"""Patient-level robustness inference and descriptive influence diagnostics."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from v10_inputs import CFG, TOL, V9, prior
from v10_frames import FORMAL, MASK_ONLY, ORIGINAL, SCAN_METRICS
import inference as v7
import v9_statistics as v9

V9_STATS = V9 / "results"
COVARIATES = {"Unadjusted": None, "FD": "delta_fd_mm", "DVARS": "delta_dvars"}


def standard_errors(v6, y, cov):
    if cov is None:
        boot = y[v6.draws(len(y))].mean(axis=1)
        se = y.std(ddof=1) / np.sqrt(len(y))
        kind = "Paired mean"
    else:
        x = np.c_[np.ones(len(y)), cov]
        pinv = np.linalg.pinv(x)
        residual = y - x @ (pinv @ y)
        leverage = np.sum(x * pinv.T, axis=1)
        se = np.linalg.norm((pinv[0] / (1 - leverage)) * residual)
        boot = v6.regression_bootstrap(y, cov, v6.draws(len(y)))
        kind = "HC3 intercept"
    return {"standard_error_pp": float(se), "standard_error_type": kind,
            "bootstrap_se_pp": float(boot[np.isfinite(boot)].std(ddof=1))}


def v9_occupancy_models(v6, original):
    tables = ["M1_Component_Statistics.csv", "M1_Component_Motion_Sensitivity.csv"]
    expected = pd.concat([prior.read_csv(V9_STATS / csv) for csv in tables])
    expected = expected.loc[expected.endpoint.eq("A1")].set_index("model", verify_integrity=True)
    old_family = prior.read_csv(V9_STATS / "V7_V9_Joint_Correction.csv")
    rows = []
    for model, predictor in COVARIATES.items():
        keep = original.K_C.notna()
        if predictor:
            keep &= original[predictor].notna()
        y = original.loc[keep, "K_C"].to_numpy(float)
        cov = original.loc[keep, predictor].to_numpy(float) if predictor else None
        if predictor:
            recomputed = v9.image_sensitivity(v6, y, cov)
        else:
            recomputed, _ = v7.unadjusted_test(v6, y)
        archived = expected.loc[model]
        exact = ("n_patients", "permutation_p", "permutations_valid", "permutations_failed",
                 "bootstrap_valid", "bootstrap_failed")
        approximate = ("effect", "ci95_low", "ci95_high", "test_statistic")
        for key in exact:
            prior.require(archived[key] == recomputed[key], f"V9 A1 replay differs at {model} {key}")
        np.testing.assert_allclose([float(archived[k]) for k in approximate],
                                   [recomputed[k] for k in approximate], atol=TOL, rtol=0)
        errors = standard_errors(v6, y, cov)
        if predictor and errors["standard_error_pp"] > 0:
            np.testing.assert_allclose(archived.effect / errors["standard_error_pp"], archived.test_statistic,
                                       atol=TOL, rtol=0)
        previous = old_family.loc[old_family.model.eq(model)]
        prior.require(len(previous) == 8 and previous.endpoint.is_unique
                      and previous.loc[previous.endpoint.eq("A1"), "holm8_p"].item() == archived.holm8_p,
                      "Original V9 Holm8 family is incomplete")
        rows.append(dict(model=model, endpoint="K_C", **{k: archived[k] for k in (*exact, *approximate)},
                         **errors, holm2_p=archived.holm2_p, holm8_p=archived.holm8_p,
                         original_prespecified_permutations=CFG["permutations"],
                         original_prespecified_bootstrap=CFG["bootstrap"],
                         rank_in_holm8=1+int(previous.permutation_p.lt(archived.permutation_p).sum()),
                         ci95_width_pp=archived.ci95_high-archived.ci95_low))
    frame = pd.DataFrame(rows)
    base = frame.loc[frame.model.eq("Unadjusted")].iloc[0]
    frame["effect_change_from_unadjusted_pp"] = frame.effect - base.effect
    frame["standard_error_change_from_unadjusted_pp"] = frame.standard_error_pp - base.standard_error_pp
    frame["bootstrap_se_change_from_unadjusted_pp"] = frame.bootstrap_se_pp - base.bootstrap_se_pp
    return frame


def formal_tests(v6, paired):
    new = []
    for endpoint in CFG["endpoints"]:
        r, _ = v7.unadjusted_test(v6, paired[endpoint].to_numpy(float))
        new.append(dict(endpoint=endpoint, mode=FORMAL, effect_unit="GEV percentage points",
                        permutations_requested=CFG["permutations"], bootstrap_requested=CFG["bootstrap"], **r))
    table = pd.DataFrame(new)
    table["holm2_p"] = v7.holm_complete(table.permutation_p, CFG["family"]["formal_quality_holm"])
    config = CFG["family"]
    earlier = prior.read_csv(V9_STATS / "V7_V9_Joint_Correction.csv")
    use = ["endpoint", "n_patients", "permutation_p", "status", "reason", "provenance"]
    old = earlier.loc[earlier.model.eq(config["prior_model"]), use]
    prior.require(len(old) == config["v7_v9_v10_holm"] - len(table) and old.endpoint.is_unique,
                  "V7 and V9 original Holm8 family is incomplete")
    sensitivity = table.assign(endpoint="censored_" + table.endpoint,
                               provenance="V10 fixed FD-censor sensitivity")[use]
    cumulative = pd.concat([old, sensitivity], ignore_index=True)
    prior.require(cumulative.endpoint.is_unique, "A cumulative endpoint is repeated")
    cumulative["family_size"] = config["v7_v9_v10_holm"]
    cumulative["holm10_p"] = v7.holm_complete(cumulative.permutation_p, config["v7_v9_v10_holm"])
    table["holm10_p"] = cumulative.loc[cumulative.provenance.eq(sensitivity.provenance.iloc[0]), "holm10_p"].to_numpy()
    return table, cumulative


def window_quality(scans):
    rows, missing = [], []
    for (mode, condition), d in scans.groupby(["mode", "condition"], sort=False):
        row = {"mode": mode, "condition": condition, "scans": len(d),
               "scans_with_any_fd_removal": int(d.removed_frames.gt(0).sum()),
               "scans_with_any_redecoded_label_change": int(d.redecoded_labels_changed.gt(0).sum()),
               "scans_missing_original_first_roi_tr": int(d.source_start_tr.sum()),
               "original_frames_total": int(d.original_frames.sum()), "removed_frames_total": int(d.removed_frames.sum()),
               "retained_frames_total": int(d.retained_frames.sum()), "retained_seconds_total": d.retained_seconds.sum(),
               "contiguous_blocks_total": int(d.valid_blocks.sum()),
               "labels_changed_by_redecode": int(d.redecoded_labels_changed.sum())}
        for field in ("removed_frames", "retained_seconds", "valid_blocks"):
            for summary in ("mean", "median", "min", "max"):
                row[f"{field}_{summary}"] = getattr(d[field], summary)()
        for field in SCAN_METRICS:
            row[f"{field}_mean"] = d[field].mean()
            row[f"{field}_n"] = d[field].count()
        rows.append(row)
        for reason, n in d.measurement_status.value_counts().items():
            missing.append(dict(mode=mode, condition=condition, status=reason, scans=n))
    return pd.DataFrame(rows), pd.DataFrame(missing)


def window_comparison(v6, pairs):
    measures = ("K_C", "K_R", "C_W", "C_F", "delta_gev_pp", "delta_coverage_pp", "delta_retained_seconds")
    formal_ids = set(pairs[FORMAL].loc[pairs[FORMAL].support, "participant_id"])
    rows = []
    for mode, d in pairs.items():
        for sample in ("All decomposable pairs", "Retained formal S1 pairs"):
            valid = d.support if sample == "All decomposable pairs" else d.support & d.participant_id.isin(formal_ids)
            for field in measures:
                effect, _ = v7.mean_bootstrap(v6, d.loc[valid, field])
                rows.append(dict(mode=mode, sample=sample, metric=field, **effect))
    complete = pairs[ORIGINAL].support & pairs[MASK_ONLY].support & pairs[FORMAL].support
    comparisons = [(MASK_ONLY, ORIGINAL), (FORMAL, MASK_ONLY), (FORMAL, ORIGINAL)]
    for to, before in comparisons:
        for endpoint in measures:
            values = pairs[to].loc[complete, endpoint] - pairs[before].loc[complete, endpoint]
            effect, _ = v7.mean_bootstrap(v6, values)
            rows.append(dict(mode=f"{to} minus {before}", sample="Common complete pairs across three modes",
                             metric=endpoint, **effect))
    return pd.DataFrame(rows)


def rank_correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def patient_diagnostics(pairs):
    old = pairs[ORIGINAL]
    formal = pairs[FORMAL]
    p = old[["participant_id", "K_C", "delta_coverage_pp", "delta_fd_mm", "delta_dvars"]].copy()
    p["K_C_after_censoring"] = formal.K_C
    p["change_in_K_C_due_to_processing"] = formal.K_C - old.K_C
    p["change_in_on_off_valid_seconds"] = formal.delta_retained_seconds - old.delta_retained_seconds
    p["off_removed_frames"] = formal.off_removed_frames
    p["on_removed_frames"] = formal.on_removed_frames

    distributions = []
    for metric in p.columns.drop("participant_id"):
        values = p[metric].dropna()
        distributions.append(dict(metric=metric, n=len(values), mean=values.mean(), sd=values.std(ddof=1),
                                  minimum=values.min(), q25=values.quantile(.25), median=values.median(),
                                  q75=values.quantile(.75), maximum=values.max()))

    relationships = [
        ("K_C", "delta_coverage_pp"), ("K_C", "delta_fd_mm"), ("delta_coverage_pp", "delta_fd_mm"),
        ("change_in_K_C_due_to_processing", "change_in_on_off_valid_seconds"),
        ("K_C_after_censoring", "change_in_on_off_valid_seconds"),
    ]
    rows, leave = [], []
    for x, y in relationships:
        two = p[["participant_id", x, y]].dropna()
        full = rank_correlation(two[x], two[y])
        loo = []
        for row in two.itertuples(index=False):
            rest = two.loc[two.participant_id.ne(row.participant_id)]
            coef = rank_correlation(rest[x], rest[y])
            loo.append(coef)
            leave.append(dict(metric=f"{x} with {y}", kind="Spearman rho", omitted_participant=row.participant_id,
                              estimate=coef))
        rows.append(dict(x=x, y=y, n=len(two), spearman_rho=full, **influence(full, loo)))
    return p, pd.DataFrame(distributions), pd.DataFrame(rows), leave


def influence(full, estimates):
    x = np.asarray(estimates, dtype=float)
    x = x[np.isfinite(x)]
    valid = len(x) > 0
    return dict(loo_n=len(x), loo_min=x.min() if valid else np.nan, loo_max=x.max() if valid else np.nan,
                loo_max_abs_shift=abs(x-full).max() if valid else np.nan,
                loo_sign_changes=int((np.sign(x) != np.sign(full)).sum()) if valid and np.isfinite(full) else np.nan)


def point_estimate(table, outcome, predictor):
    y = table[outcome].to_numpy(float)
    if predictor is None:
        return float(y.mean()) if len(y) else np.nan
    x = np.c_[np.ones(len(table)), table[predictor].to_numpy(float)]
    if len(y) < 3 or np.linalg.matrix_rank(x) != x.shape[1]:
        return np.nan
    return float(np.linalg.lstsq(x, y, rcond=None)[0][0])


def model_influence(pairs):
    rows, details = [], []
    models = [(ORIGINAL, mode, "K_C", x) for mode, x in COVARIATES.items()]
    models += [(mode, "Unadjusted", endpoint, None) for mode in (MASK_ONLY, FORMAL) for endpoint in CFG["endpoints"]]
    for mode, model, endpoint, predictor in models:
        keep = [endpoint, predictor] if predictor else [endpoint]
        d = pairs[mode].dropna(subset=keep)
        full = point_estimate(d, endpoint, predictor)
        loo = []
        for p in d.participant_id:
            estimate = point_estimate(d.loc[d.participant_id.ne(p)], endpoint, predictor)
            loo.append(estimate)
            details.append(dict(metric=endpoint, kind="Contribution pp", mode=mode, model=model,
                                omitted_participant=p, estimate=estimate))
        rows.append(dict(mode=mode, model=model, endpoint=endpoint, n=len(d), effect_pp=full, **influence(full, loo)))
    return pd.DataFrame(rows), details
