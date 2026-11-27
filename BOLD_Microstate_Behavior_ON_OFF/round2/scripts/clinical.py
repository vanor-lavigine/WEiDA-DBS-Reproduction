"""Describe frozen paired motor scores and reconcile their additive composition."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

ENDPOINTS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
COMPONENTS = ENDPOINTS[1:] + ("remaining_four",)
DOMAIN_COUNTS = dict(total=33, rigidity=5, bradykinesia=11, tremor=10, gait_balance=3, remaining_four=4)
REMAINDER = {"speech", "facial_expression", "arising", "posture"}
TOL = 1e-12


@dataclass(frozen=True)
class Clinical:
    profile: pd.DataFrame
    composition: pd.DataFrame | None
    private: pd.DataFrame
    audit: dict


def _summarize(x, prefix):
    if not len(x):
        return {f"{prefix}_{key}": np.nan for key in ("mean", "median", "q25", "q75")}
    q25, median, q75 = np.percentile(x, [25, 50, 75])
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_median": float(median),
            f"{prefix}_q25": float(q25), f"{prefix}_q75": float(q75)}


def _agree(current, locked, label):
    if not np.allclose(current, locked, rtol=0, atol=TOL, equal_nan=True):
        raise ValueError(f"Clinical data differ from frozen values for {label}")


def _response_profile(pairs, legacy):
    old = legacy.set_index("endpoint", verify_integrity=True)
    if (set(old.index) != set(ENDPOINTS) or not old.direction.eq("OFF minus ON").all()
            or not old.bh_family_size.eq(5).all()):
        raise ValueError("The original clinical table does not define the same five OFF-minus-ON endpoints")
    if not len(pairs) or pairs.patient_id.isna().any() or not pairs.patient_id.is_unique:
        raise ValueError("A complete and unique patient roster is required")
    private = pairs[["patient_id"]].copy()
    rows = []
    for endpoint in ENDPOINTS:
        a, b = (pairs[f"{endpoint}_{condition}"].to_numpy(float) for condition in ("off", "on"))
        x = np.r_[a, b]
        if not (np.isfinite(x).all() and (x >= 0).all() and np.equal(x, np.floor(x)).all()):
            raise ValueError(f"Invalid or incomplete frozen score for {endpoint}")
        positive_off = a > 0
        difference = a - b
        rate = np.divide(difference, a, out=np.full(len(a), np.nan), where=positive_off)
        _agree(difference, pairs[f"{endpoint}_improvement"].to_numpy(float), f"{endpoint} improvement")
        _agree(rate, pairs[f"{endpoint}_improvement_rate"].to_numpy(float), f"{endpoint} improvement rate")

        n = len(difference)
        improved, unchanged, worse = (int(indices.sum()) for indices in
                                      (difference > 0, difference == 0, difference < 0))
        positive_n = int(positive_off.sum())
        positive_improved = int((positive_off & (difference > 0)).sum())
        positive_unchanged = int((positive_off & (difference == 0)).sum())
        positive_worse = int((positive_off & (difference < 0)).sum())
        row = dict(
            endpoint=endpoint, n=n, **_summarize(a, "off"), **_summarize(b, "on"),
            **_summarize(difference, "improvement"), n_improved=improved, n_unchanged=unchanged, n_worse=worse,
            improved_fraction=improved/n, unchanged_fraction=unchanged/n, worse_fraction=worse/n,
            rate_n=positive_n, **_summarize(rate[positive_off], "rate"),
            off_zero_n=n-positive_n, off_positive_n=positive_n, off_positive_n_improved=positive_improved,
            off_positive_n_unchanged=positive_unchanged, off_positive_n_worse=positive_worse,
            off_positive_improved_fraction=positive_improved/positive_n if positive_n else np.nan,
            legacy_raw_p=float(old.at[endpoint, "raw_p"]), legacy_bh5_q=float(old.at[endpoint, "fdr_q"]),
        )
        comparisons = dict(
            n="n", off_mean="off_mean", on_mean="on_mean", off_median="off_median", on_median="on_median",
            improvement_mean="mean_change", improvement_median="median_change", n_improved="n_increased",
            n_unchanged="n_unchanged", n_worse="n_decreased", rate_mean="rate_fraction_mean",
            rate_median="rate_fraction_median", rate_n="n_rate", off_zero_n="n_off_zero",
        )
        _agree([row[key] for key in comparisons], old.loc[endpoint, list(comparisons.values())].to_numpy(float),
               f"{endpoint} original clinical table")
        for suffix, values in (("off", a), ("on", b), ("improvement", difference), ("improvement_rate", rate)):
            private[f"{endpoint}_{suffix}"] = values
        rows.append(row)
    return pd.DataFrame(rows), private


def _item_domains(dictionary):
    if (len(dictionary) != DOMAIN_COUNTS["total"] or not dictionary.item.is_unique
            or dictionary.item.isna().any() or not dictionary.column.is_unique):
        raise ValueError("The frozen dictionary must contain the 33 unique manual items")
    items = {domain: [] for domain in (*ENDPOINTS, "remaining_four")}
    for row in dictionary.itertuples(index=False):
        labels = [name.strip() for name in row.domains.split("|")]
        if ("total" not in labels or len(set(labels)) != len(labels) or len(labels) > 2
                or set(labels) - set(ENDPOINTS)):
            raise ValueError("The frozen manual items do not have the expected disjoint domains")
        items["total"].append(row.item)
        component = next((name for name in labels if name != "total"), "remaining_four")
        items[component].append(row.item)
    if ({key: len(value) for key, value in items.items()} != DOMAIN_COUNTS
            or set(items["remaining_four"]) != REMAINDER):
        raise ValueError("The dictionary does not partition the total into four predefined domains and four other items")
    return items


def _score_side(item_scores, pairs, items, side):
    keys = ["study_code", "patient_id", "scan_id"]
    candidate = item_scores.loc[item_scores.dbs_condition.eq(side.upper())
                                & item_scores.study_code.isin(pairs.study_code)].rename(
                                    columns={"participant_id": "patient_id"})
    if (len(candidate) != len(pairs) or candidate[keys].isna().any().any()
            or not candidate.identity_status.eq("confirmed_code_name_condition").all()
            or not candidate.target.eq("STN").all() or not candidate.demographic_target.eq("STN").all()
            or not candidate.n_items_valid.eq(len(items["total"])).all()):
        raise ValueError(f"The archived manual-score rows cannot be uniquely established for {side.upper()}")
    roster = pairs[["study_code", "patient_id", f"{side}_scan_id", f"{side}_assessment_date",
                    f"{side}_medication_state"]].rename(columns={f"{side}_scan_id": "scan_id"})
    visits = roster.merge(candidate, on=keys, how="left", validate="one_to_one", indicator=True)
    if not visits["_merge"].eq("both").all():
        raise ValueError(f"Clinical study, patient and scan keys do not reconcile for {side.upper()}")
    if (not visits.assessment_date.eq(visits[f"{side}_assessment_date"]).all()
            or not visits.medication_state.eq(visits[f"{side}_medication_state"]).all()):
        raise ValueError(f"Clinical assessment date or medication metadata do not reconcile for {side.upper()}")
    observed = visits[[f"item_{name}" for name in items["total"]]].to_numpy(float)
    if (not np.isfinite(observed).all() or not (observed >= 0).all() or not (observed <= 4).all()
            or not np.equal(observed, np.floor(observed)).all()):
        raise ValueError(f"Invalid archived manual item value for {side.upper()}")
    sums = pd.DataFrame({domain: visits[[f"item_{name}" for name in names]].sum(axis=1)
                         for domain, names in items.items()})
    if not np.array_equal(sums.loc[:, list(COMPONENTS)].sum(axis=1), sums["total"]):
        raise ValueError(f"Five disjoint manual-score sums do not reconcile to total for {side.upper()}")
    for endpoint in ENDPOINTS:
        _agree(sums[endpoint], pairs[f"{endpoint}_{side}"], f"{endpoint} manual values at {side.upper()}")
    return sums


def _composition(pairs, dictionary, item_scores):
    items = _item_domains(dictionary)
    off, on = (_score_side(item_scores, pairs, items, side) for side in ("off", "on"))
    delta = off - on
    _agree(delta.loc[:, list(COMPONENTS)].sum(axis=1), delta["total"], "additive improvement")
    total_mean = delta["total"].mean()
    data = [dict(component=domain, n=len(delta), item_count=len(items[domain]), off_mean=off[domain].mean(),
                 on_mean=on[domain].mean(), improvement_mean=delta[domain].mean(),
                 fraction_of_total_mean=delta[domain].mean()/total_mean if total_mean else np.nan)
            for domain in (*COMPONENTS, "total")]
    return pd.DataFrame(data), pd.DataFrame(
        {f"remaining_four_{side}": source["remaining_four"] for side, source in
         (("off", off), ("on", on), ("improvement", delta))})


def analyze_clinical(pairs, dictionary, legacy, item_scores=None):
    """Return publishable summaries and a separately identified private table."""
    pairs = pairs.reset_index(drop=True)
    profile, private = _response_profile(pairs, legacy)
    composition = None
    audit = {"n_clinical_pairs": len(pairs), "original_clinical_summary_checks": len(ENDPOINTS),
             "clinical_direction": "OFF minus ON", "rate_rule": "OFF minus ON divided by OFF; NA at OFF = 0",
             "additive_component_checks": 0, "composition_status": "archived_item_rows_unavailable"}
    if item_scores is not None:
        composition, other_items = _composition(pairs, dictionary, item_scores)
        private = pd.concat([private, other_items], axis=1)
        audit.update(additive_component_checks=2*len(pairs), composition_status="verified_from_archived_manual_items")
    return Clinical(profile, composition, private, audit)
