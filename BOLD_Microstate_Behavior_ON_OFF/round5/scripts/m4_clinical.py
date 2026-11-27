"""Audit the frozen BH80 and estimate the prespecified exploratory M4 models."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.api import OLS
from statsmodels.stats.multitest import multipletests


PATHS = {"delta_P44": ("M4", "M4"), "delta_P41": ("M4", "M1"), "delta_P34": ("M3", "M4")}
SCORE_KEYS = ["patient_id", "on_scan_id", "off_scan_id", "fold"]
EDGE_KEYS = ["from_state", "to_state", "clinical_endpoint"]
BASELINE_TERMS = [*PATHS]


def _round4_stats():
    path = Path(__file__).resolve().parents[2] / "round4/transition/scripts/transition_stats.py"
    spec = spec_from_file_location("round4_frozen_transition_statistics", path)
    source = module_from_spec(spec)
    spec.loader.exec_module(source)
    return source


def _paired_inputs(matrices, scores):
    """Verify each patient and both scan keys, then reconstruct exact count ratios."""
    patient_edges = matrices.loc[matrices.condition.isin(("ON", "OFF"))].copy()
    observed = patient_edges[["participant_id", "scan_id", "condition", "fold"]].drop_duplicates()
    if observed.duplicated(["participant_id", "condition"]).any() or len(observed) != 50:
        raise ValueError("P4 requires exactly one frozen ON and OFF scan for each of 25 participants")
    if not observed.groupby("participant_id").fold.nunique().eq(1).all():
        raise ValueError("P4 ON and OFF scans differ in fold")
    if observed.scan_id.duplicated().any() or patient_edges.duplicated(["scan_id", "from_state", "to_state"]).any():
        raise ValueError("P4 frozen scans or matrix edges are duplicated")
    wide = observed.pivot(index=["participant_id", "fold"], columns="condition", values="scan_id").reset_index()
    wide = wide.rename(columns={"participant_id": "patient_id", "ON": "on_scan_id", "OFF": "off_scan_id"})
    paired = wide[SCORE_KEYS].merge(scores, on=SCORE_KEYS, how="outer", validate="one_to_one", indicator=True)
    if len(paired) != 25 or not paired.pop("_merge").eq("both").all():
        raise ValueError("P4 imaging and clinical IDs, scan keys or folds do not match")
    if not paired.patient_id.is_unique:
        raise ValueError("P4 repeats a clinical participant")
    totals = patient_edges.groupby(["scan_id", "from_state"])["count"].transform("sum")
    p_exact = patient_edges["count"].div(totals.where(totals.gt(0)))
    np.testing.assert_allclose(patient_edges.p_weida, p_exact, rtol=0, atol=1e-13, equal_nan=True)
    patient_edges["p_weida"] = p_exact
    paired = paired.set_index("patient_id", verify_integrity=True).sort_index()
    return patient_edges, paired


def _recheck_round4(edges, scores, archived, cfg):
    stats = _round4_stats()
    replay = stats.clinical_associations(edges, scores.reset_index(), cfg)
    if (len(replay) != 80 or len(archived) != 80
            or replay.duplicated(EDGE_KEYS).any() or archived.duplicated(EDGE_KEYS).any()):
        raise ValueError("P4 clinical registry must contain all 80 unique Round4 endpoints")
    reference = archived.set_index(EDGE_KEYS).sort_index()
    audit = replay.set_index(EDGE_KEYS).sort_index()
    pd.testing.assert_index_equal(audit.index, reference.index)
    # Empty archived CSV fields are read as NaN. Compare all inferential and QC fields.
    for name in audit.columns:
        if pd.api.types.is_numeric_dtype(audit[name]) and not pd.api.types.is_bool_dtype(audit[name]):
            np.testing.assert_allclose(audit[name], reference[name], atol=1e-12, rtol=0, equal_nan=True,
                                       err_msg=f"Round4 clinical replay differs in {name}")
        elif not audit[name].fillna("").eq(reference[name].fillna("")).all():
            raise ValueError(f"Round4 clinical replay differs in {name}")
    replay["round4_replay_match"] = True
    replay["interpretation"] = "exploratory_historical_score_association"
    return replay


def _path_changes(edges, scores):
    columns = {}
    for term, (source, target) in PATHS.items():
        view = edges.loc[edges.from_state.eq(source) & edges.to_state.eq(target)]
        pairs = view.pivot(index="participant_id", columns="condition", values="p_weida").reindex(scores.index)
        columns[term] = pairs.ON - pairs.OFF
    delta = pd.DataFrame(columns)
    required = scores[["total_on", "total_off", "total_improvement"]].to_numpy(float)
    if not np.isfinite(delta.to_numpy()).all() or not np.isfinite(required).all():
        raise ValueError("P4 three-path and total score analyses require 25 complete frozen pairs")
    np.testing.assert_array_equal(required[:, 1] - required[:, 0], required[:, 2])
    return delta


def _vifs(values):
    centered = values - values.mean(axis=0)
    scale = np.linalg.norm(centered, axis=0)
    normalized = np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)
    condition = float(np.linalg.cond(normalized))
    vifs = []
    for i, total in enumerate(scale):
        target = normalized[:, i]
        others = np.delete(normalized, i, axis=1)
        weights, *_ = np.linalg.lstsq(others, target, rcond=None)
        unexplained = float(np.linalg.norm(target - others @ weights) ** 2)
        vifs.append(1 / unexplained if total > 0 and unexplained > 1e-12 else np.inf)
    return np.asarray(vifs), condition


def _bootstrap_r2(values, outcome, indices):
    """Fit an intercept and observe the same participant index on every column."""
    x = values[indices]
    y = outcome[indices]
    x -= x.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    threshold = singular[:, 0] * max(x.shape[1:]) * np.finfo(float).eps
    totals = np.square(y).sum(axis=1)
    usable = (singular[:, -1] > threshold) & (totals > 0)
    explained = np.square(np.einsum("bnk,bn->bk", u, y)).sum(axis=1)
    return np.divide(explained, totals, out=np.full(len(indices), np.nan), where=usable).clip(0, 1)


def _interval(samples):
    finite = samples[np.isfinite(samples)]
    if len(finite) < .95 * len(samples):
        return np.nan, np.nan, len(finite)
    low, high = np.quantile(finite, [.025, .975])
    return float(low), float(high), len(finite)


def _generator(config, salt):
    return np.random.default_rng(np.random.SeedSequence([config["seed"], salt]))


def _joint_response(changes, improvement, config):
    x = changes.to_numpy(float)
    y = improvement.to_numpy(float)
    n, p = x.shape
    vifs, condition = _vifs(x)
    centered = x - x.mean(axis=0)
    q, singular, _ = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.linalg.matrix_rank(centered))
    response = y - y.mean()
    tss = np.dot(response, response)
    estimates = dict(r_squared=np.nan, adjusted_r_squared=np.nan, raw_p=np.nan, fdr_q=np.nan, holm_p=np.nan,
                     ci95_low=np.nan, ci95_high=np.nan, bootstrap_valid=0)
    reason = ""
    if rank != p or tss == 0:
        reason = "rank_deficient_predictors" if rank != p else "constant_total_improvement"
    else:
        observed_r2 = float(np.clip(np.square(q.T @ response).sum() / tss, 0, 1))
        permutations = config["permutations"]
        orders = _generator(config, 50401).permuted(np.broadcast_to(np.arange(n), (permutations, n)), axis=1)
        null_r2 = np.square(response[orders] @ q).sum(axis=1) / tss
        permutation_p = float((1 + np.count_nonzero(null_r2 >= observed_r2 - 1e-12)) / (1 + permutations))
        indices = _generator(config, 50402).integers(n, size=(config["bootstrap_draws"], n))
        lower, upper, valid = _interval(_bootstrap_r2(x, y, indices))
        estimates = dict(r_squared=observed_r2, adjusted_r_squared=1 - (1 - observed_r2) * (n - 1) / (n - p - 1),
                         raw_p=permutation_p, fdr_q=permutation_p, holm_p=permutation_p,
                         ci95_low=lower, ci95_high=upper, bootstrap_valid=valid)
    results = pd.DataFrame([dict(
        model="total_improvement_on_three_M4_path_changes", predictors="delta_P44+delta_P41+delta_P34",
        outcome="total_score_OFF_minus_ON", design="exploratory_patient_level_unconstrained_direction",
        n_pairs=n, fraction_unit="0_to_1", rank=rank, residual_df=n-rank-1, min_singular_value=singular[-1],
        standardized_condition_number=condition, maximum_vif=vifs.max(), family_size=1, reason=reason,
        permutation_test="upper_tail_R2_permute_whole_patient_outcomes",
        permutations=config["permutations"], permutation_seed_salt=50401,
        bootstrap_draws=config["bootstrap_draws"], bootstrap_seed_salt=50402,
        passes_fdr=bool(estimates["fdr_q"] < config["alpha"]),
        passes_holm=bool(estimates["holm_p"] < config["alpha"]),
        **estimates,
    )])
    covariance, correlation = changes.cov(), changes.corr()
    diagnostics = pd.DataFrame([
        dict(first_path=first, second_path=second, n_pairs=n,
             sample_covariance=covariance.loc[first, second], pearson_r=correlation.loc[first, second],
             vif_first_path_in_three_predictor_model=vifs[i])
        for i, first in enumerate(PATHS) for second in PATHS
    ])
    return results, diagnostics


def _baseline_coefficients(changes, scores, config):
    """Four fixed OLS models with HC3 coefficient and full-block uncertainty."""
    outcome = scores.total_on.to_numpy(float)
    baseline = scores.total_off.to_numpy(float)
    # Scale the increments so coefficients are in score points per probability percentage point.
    path_pp = changes * 100
    baseline_x = np.column_stack([np.ones(len(outcome)), baseline])
    baseline_r2 = float(OLS(outcome, baseline_x).fit().rsquared)
    rows = []
    for terms in [(name,) for name in PATHS] + [tuple(PATHS)]:
        model = f"single_{terms[0]}" if len(terms) == 1 else "joint_three_M4_paths"
        predictors = np.column_stack([baseline, path_pp[list(terms)].to_numpy(float)])
        design = np.column_stack([np.ones(len(outcome)), predictors])
        names = ["intercept", "total_off", *terms]
        vifs, condition = _vifs(predictors)
        fit = OLS(outcome, design).fit(cov_type="HC3", use_t=True)
        model_rank = int(fit.model.rank)
        full_rank = model_rank == design.shape[1]
        confidence = fit.conf_int() if full_rank else np.full((len(names), 2), np.nan)
        common = dict(
            model=model, outcome="ON_total_score", n_pairs=len(outcome), model_rank=model_rank,
            residual_df=int(fit.df_resid), model_predictors=len(names)-1,
            standardized_condition_number=condition, max_model_vif=float(vifs.max()),
            r_squared=float(fit.rsquared), adjusted_r_squared=float(fit.rsquared_adj),
            baseline_only_r_squared=baseline_r2, incremental_r_squared=float(fit.rsquared-baseline_r2),
            interpretation="exploratory_historical_score_association",
        )
        for number, term in enumerate(names):
            path_term = term in terms
            units = ("score_points_per_percentage_point" if path_term else
                     "score_points_per_baseline_point" if term == "total_off" else "score_points")
            testable = full_rank and path_term
            rows.append(dict(
                **common, term=term, test_family="baseline_adjusted_BH_Holm7" if path_term else "descriptive_covariate",
                hypothesis=path_term, effect_unit=units, test="HC3_two_sided_t" if path_term else "none",
                estimate=float(fit.params[number]) if full_rank else np.nan,
                standard_error=float(fit.bse[number]) if full_rank else np.nan,
                ci95_low=float(confidence[number, 0]), ci95_high=float(confidence[number, 1]),
                statistic=float(fit.tvalues[number]) if testable else np.nan,
                raw_p=float(fit.pvalues[number]) if testable else np.nan,
                term_vif=float(vifs[number - 1]) if number else np.nan,
                reason="" if full_rank else "rank_deficient_predictors",
            ))
        if len(terms) == len(PATHS):
            null = np.column_stack([np.zeros((len(PATHS), 2)), np.eye(len(PATHS))])
            test = fit.f_test(null) if full_rank else None
            indices = _generator(config, 50403).integers(len(outcome), size=(config["bootstrap_draws"], len(outcome)))
            full_boot = _bootstrap_r2(predictors, outcome, indices)
            baseline_boot = _bootstrap_r2(baseline[:, None], outcome, indices)
            low, high, valid = _interval(np.maximum(0, full_boot - baseline_boot))
            rows.append(dict(
                **common, term="all_three_paths_block", test_family="baseline_adjusted_BH_Holm7", hypothesis=True,
                effect_unit="incremental_R_squared", test="HC3_joint_Wald_F",
                estimate=common["incremental_r_squared"] if full_rank else np.nan, standard_error=np.nan,
                ci95_low=low, ci95_high=high, statistic=float(test.fvalue) if test is not None else np.nan,
                raw_p=float(test.pvalue) if test is not None else np.nan, term_vif=np.nan,
                block_df_num=float(test.df_num) if test is not None else np.nan,
                block_bootstrap_valid=valid, block_bootstrap_draws=config["bootstrap_draws"],
                block_bootstrap_seed_salt=50403,
                reason="" if full_rank else "rank_deficient_predictors",
            ))
    table = pd.DataFrame(rows)
    correction = table.hypothesis
    if int(correction.sum()) != 7:
        raise ValueError("P4 baseline-adjusted test family must retain all seven fixed slots")
    pvals = table.loc[correction, "raw_p"]
    for column, method in (("fdr_q", "fdr_bh"), ("holm_p", "holm")):
        adjusted = multipletests(pvals.fillna(1.0), method=method)[1]
        table.loc[correction, column] = np.where(pvals.notna(), adjusted, np.nan)
    table["family_size"] = np.where(correction, 7, np.nan)
    table["passes_fdr"] = table.fdr_q.lt(config["alpha"])
    table["passes_holm"] = table.holm_p.lt(config["alpha"])
    return table


def _clinical_reference(matrices, scores):
    n_hc = matrices.loc[matrices.condition.eq("HC"), "participant_id"].nunique()
    rows = [dict(condition="HC", n=n_hc, mean_total_score=0., sd_total_score=0.,
                 score_basis="conventional_zero_descriptive_reference")]
    for condition in ("OFF", "ON"):
        values = scores[f"total_{condition.lower()}"]
        rows.append(dict(condition=condition, n=values.count(), mean_total_score=values.mean(),
                         sd_total_score=values.std(), score_basis="archived_historical_manual_total"))
    return pd.DataFrame(rows)


def run_clinical(matrices, scores, archived_associations, round4_config, config):
    """Reproduce all BH80 and fit the two frozen Round5 exploratory families."""
    edges, paired = _paired_inputs(matrices, scores)
    replay = _recheck_round4(edges, paired, archived_associations, round4_config)
    changes = _path_changes(edges, paired)
    joint, diagnostics = _joint_response(changes, paired.total_improvement, config)
    baseline = _baseline_coefficients(changes, paired, config)
    return {
        "M4_Clinical_Associations.csv": replay,
        "M4_Joint_Response_Analysis.csv": joint,
        "M4_Joint_Response_Predictor_Diagnostics.csv": diagnostics,
        "M4_Baseline_Adjusted_Models.csv": baseline,
        "M4_Clinical_Reference_Summary.csv": _clinical_reference(matrices, paired),
    }
