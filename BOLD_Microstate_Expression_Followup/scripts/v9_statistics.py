"""Prespecified patient-level V7 and V8 inference, with complete-family correction."""

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from v9_inputs import (
    CFG, COHORT, PAIR_KEYS, PRIVATE, V4, V4_FRAMEWORK, V6, V7_CFG, V7_TABLES,
    V8_CFG, V8_TABLES, digest, read_csv, require,
)
from v9_measurements import CONTRIBUTIONS, STATES
import inference as v7_inference
import regression as v8_regression


def load_resampling(workspace):
    os.environ["WEIDA_V6_DATA_ROOT"] = str(workspace)
    os.environ["NUMBA_CACHE_DIR"] = str(PRIVATE / "numba")
    sys.path.insert(0, str(V6 / "scripts"))
    import paired

    for name, module in [("micro_inference", V4), ("statistics_v4", V4_FRAMEWORK)]:
        live = Path(sys.modules[name].__file__)
        archived = module / "scripts" / live.name
        require(digest(live, normalise=True) == digest(archived, normalise=True),
                f"Loaded V6 inference dependency differs from the frozen code: {name}")
    return paired


def adjust_tables(new, prior, old_ids, family, prior_label):
    """Expose both correction scopes, retaining unavailable rows in each family."""
    subset = ["model", "endpoint", "n_patients", "permutation_p", "status", "reason"]
    scope = CFG["old_and_new_families"][family]
    new_ids = {h["id"] for h in CFG["hypotheses"] if h["family"] == family}
    require(len(new_ids) == scope["planned"] and len(old_ids) + len(new_ids) == scope["joint"],
            f"Prespecified family size differs for {family}")
    joint_col = f"holm{scope['joint']}_p"
    require(set(new.model) == set(prior.model), f"Prior and current models differ for {family}")
    new = new.copy()
    parts = []
    for model in new.model.drop_duplicates():
        current = new.loc[new.model.eq(model)]
        former = prior.loc[prior.model.eq(model), subset].copy()
        require(former.endpoint.is_unique and set(former.endpoint) == old_ids,
                f"Incorrect prior endpoint set for {family} {model}")
        require(current.endpoint.is_unique and set(current.endpoint) == new_ids,
                f"Incorrect new endpoint set for {family} {model}")
        new.loc[current.index, "holm2_p"] = v7_inference.holm_complete(current.permutation_p, scope["planned"])
        part = pd.concat([
            former.assign(provenance=prior_label),
            current[subset].assign(provenance="New V9 endpoint"),
        ], ignore_index=True)
        part["family_size"] = scope["joint"]
        part[joint_col] = v7_inference.holm_complete(part.permutation_p, scope["joint"])
        part[f"passes_holm{scope['joint']}"] = part[joint_col].lt(CFG["alpha"])
        current_p = part.set_index("endpoint")[joint_col]
        new.loc[current.index, joint_col] = current.endpoint.map(current_p)
        parts.append(part)
    new["passes_holm2"] = new.holm2_p.lt(CFG["alpha"])
    new[f"passes_holm{scope['joint']}"] = new[joint_col].lt(CFG["alpha"])
    return new, pd.concat(parts, ignore_index=True)


def image_sensitivity(v6, delta, cov):
    result = v7_inference.adjusted_test(v6, delta, cov)
    if result["status"] != "OK":
        return result
    # V7 uses the V6 HC3 statistic. Check the actual resamples so failures are counted correctly.
    keep = np.isfinite(cov) & np.isfinite(delta)
    y, x = delta[keep], cov[keep]
    design = np.c_[np.ones(len(x)), x]
    if not np.isfinite(result["test_statistic"]):
        return v7_inference.nothing(len(x), "Undefined observed HC3 statistic")
    reduced = x[:, None]
    fitted = reduced @ np.linalg.lstsq(reduced, y, rcond=None)[0]
    _, null = v6.hc3_t(design, fitted + v6.signs(len(x)) * (y - fitted), 0)
    supported = np.isfinite(null)
    if not supported.any():
        result = v7_inference.nothing(len(x), "Every residual sign-flip HC3 statistic is undefined")
        result["permutations_failed"] = len(null)
    elif not supported.all():
        result.update(permutations_valid=int(supported.sum()), permutations_failed=int((~supported).sum()),
                      permutation_p=v6.tail_probability(null[supported], result["test_statistic"]))
    return result


def imaging(v6, decomposition):
    rows = []
    for model, motion in CFG["imaging_models"].items():
        for h in CFG["hypotheses"]:
            if h["family"] != "imaging":
                continue
            delta = decomposition[h["field"]].to_numpy(float)
            if motion is None:
                result, _ = v7_inference.unadjusted_test(v6, delta)
            else:
                result = image_sensitivity(v6, delta, decomposition[motion].to_numpy(float))
            rows.append(dict(
                model=model, endpoint=h["id"], metric=h["name"], effect_unit=h["unit"], covariate=motion,
                effect_definition="Mean paired contribution" if motion is None else "Intercept at zero covariate change",
                n_planned=COHORT["image_pairs"], n_missing=COHORT["image_pairs"] - result["n_patients"],
                permutations_requested=CFG["permutations"], bootstrap_requested=CFG["bootstrap"],
                **result,
            ))
    prior = pd.concat([read_csv(table) for table in V7_TABLES], ignore_index=True)
    result, joint = adjust_tables(pd.DataFrame(rows), prior, {old["id"] for old in V7_CFG["primary"]},
                                  "imaging", "Archived V7 original")
    threshold = CFG["component_share_min_abs_gev_pp"]
    _, shares = v7_inference.contributions(v6, decomposition, "m1_shapley_w_pp", CONTRIBUTIONS, threshold)
    share_rows = []
    for h in (h for h in CFG["hypotheses"] if h["family"] == "imaging"):
        share_rows.append(dict(endpoint=h["id"], unit="percent of the original mean M1 weight contribution",
                               minimum_abs_mean_weight_pp=threshold, **shares[h["field"]]))
    return result, joint, pd.DataFrame(share_rows)


def expression_group_summary(v6, pairs):
    rows = []
    for k in STATES:
        group = v7_inference.condition_summary(v6, pairs, f"e{k}")
        group["mean_delta_on_minus_off"] = group.pop("effect")
        group["delta_ci95_low"] = group.pop("ci95_low")
        group["delta_ci95_high"] = group.pop("ci95_high")
        n = group["n_patients"]
        rows.append(dict(template=f"M{k}", unit="GFP squared weighted mean squared correlation",
                         n_eligible=COHORT["image_pairs"], n_complete_pairs=n,
                         n_missing_pair=COHORT["image_pairs"] - n,
                         n_off_observed=int(pairs[f"off_e{k}"].notna().sum()),
                         n_on_observed=int(pairs[f"on_e{k}"].notna().sum()),
                         off_ci95_low=np.nan, off_ci95_high=np.nan,
                         on_ci95_low=np.nan, on_ci95_high=np.nan) | group)
    return pd.DataFrame(rows)


def motor_associations(pairs, scores):
    require(not scores.duplicated(PAIR_KEYS).any() and not pairs.duplicated(PAIR_KEYS).any(),
            "Clinical and expression tables must each contain a single row per patient and pair")
    analysis = scores.merge(pairs, how="left", on=PAIR_KEYS, validate="one_to_one", indicator=True)
    require(analysis._merge.eq("both").all() and len(analysis) == len(pairs) == len(scores),
            "Clinical and expression pairs disagree")
    primary_models = {model["name"]: model for model in V8_CFG["models"]}
    results, records, points = [], [], []
    for name in CFG["clinical_models"]:
        model = primary_models[name]
        for h in CFG["hypotheses"]:
            if h["family"] != "clinical":
                continue
            endpoint = dict(outcome="motor_total", predictor=h["field"])
            x, y, participants, columns, sample = v8_regression.model_data(analysis, endpoint, model)
            fit = v8_regression.estimate(x, y)
            loo_rows, loo = v8_regression.leave_one(x, y, participants, fit, h["id"])
            records.extend(dict(model=name, **row) for row in loo_rows)
            if name == "Primary_FD" and fit["status"] == "OK":
                reduced, _, residual_y = v8_regression.residual_reference(x, y)
                residual_x = x[:, -1] - reduced @ np.linalg.lstsq(reduced, x[:, -1], rcond=None)[0]
                points.extend(dict(endpoint=h["id"], participant_id=p, adjusted_expression=a,
                                   adjusted_improvement=b)
                              for p, a, b in zip(participants, residual_x, residual_y, strict=True))
            results.append(dict(model=name, endpoint=h["id"], metric=h["name"], effect_unit=h["unit"],
                                motion_covariate=model["motion"], visit_rule=model["sample"],
                                design=" + ".join(columns), response="motor_total_off_minus_on",
                                permutations_requested=CFG["permutations"], bootstrap_requested=CFG["bootstrap"],
                                **fit, **sample, **loo))
    prior = pd.concat([read_csv(table) for table in V8_TABLES], ignore_index=True)
    prior = prior.loc[prior.model.isin(CFG["clinical_models"])]
    result, joint = adjust_tables(pd.DataFrame(results), prior, {row["endpoint"] for row in v8_regression.registry()},
                                  "clinical", "Archived V8 original")
    return result, joint, pd.DataFrame(records), pd.DataFrame(points)
