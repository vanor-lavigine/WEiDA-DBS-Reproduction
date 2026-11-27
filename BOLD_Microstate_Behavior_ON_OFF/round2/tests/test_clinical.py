"""Boundary and linkage tests for round-two clinical descriptions."""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROUND2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND2 / "scripts"))

from clinical import ENDPOINTS, analyze_clinical


@pytest.fixture
def frozen_example():
    dictionary = pd.read_csv(ROUND2.parent / "results/Score_Item_Dictionary.csv")
    pairs = []
    items = []
    for i in range(3):
        subject = f"example_{i}"
        study_code = f"example_study_{i}"
        pair = dict(patient_id=subject, study_code=study_code)
        for side in ("off", "on"):
            scan = f"{subject}_{side}"
            date = "2000-01-01"
            pair.update({f"{side}_scan_id": scan, f"{side}_assessment_date": date,
                         f"{side}_medication_state": "med_off"})
            row = dict(study_code=study_code, participant_id=subject, scan_id=scan, dbs_condition=side.upper(),
                       identity_status="confirmed_code_name_condition", target="STN", demographic_target="STN",
                       n_items_valid=33, assessment_date=date, medication_state="med_off")
            row.update({f"item_{name}": 0 for name in dictionary.item})
            if i == 0:
                row["item_speech"] = 2 if side == "off" else 1
                row["item_postural_tremor_right"] = 1 if side == "off" else 0
            elif i == 2:
                row["item_rigidity_neck"] = 1 if side == "off" else 2
            for name in ENDPOINTS:
                cols = [f"item_{entry.item}" for entry in dictionary.itertuples()
                        if name in {label.strip() for label in entry.domains.split("|")}]
                pair[f"{name}_{side}"] = sum(row[key] for key in cols)
            items.append(row)
        for name in ENDPOINTS:
            improvement = pair[f"{name}_off"] - pair[f"{name}_on"]
            pair[f"{name}_improvement"] = improvement
            pair[f"{name}_improvement_rate"] = improvement/pair[f"{name}_off"] if pair[f"{name}_off"] else np.nan
        pairs.append(pair)

    paired = pd.DataFrame(pairs)
    legacy = []
    for name in ENDPOINTS:
        a, b, d, rate = (paired[f"{name}_{suffix}"] for suffix in ("off", "on", "improvement", "improvement_rate"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_rate = rate.median()
        legacy.append(dict(endpoint=name, direction="OFF minus ON", n=len(paired), off_mean=a.mean(), on_mean=b.mean(),
                           off_median=a.median(), on_median=b.median(), mean_change=d.mean(), median_change=d.median(),
                           n_increased=d.gt(0).sum(), n_unchanged=d.eq(0).sum(), n_decreased=d.lt(0).sum(),
                           rate_fraction_mean=rate.mean(), rate_fraction_median=median_rate, n_rate=rate.notna().sum(),
                           n_off_zero=a.eq(0).sum(), raw_p=1., fdr_q=1., bh_family_size=5))
    return paired, dictionary, pd.DataFrame(legacy), pd.DataFrame(items)


def test_zero_off_retained_and_ratio_of_mean_contributions(frozen_example):
    pairs, dictionary, legacy, items = frozen_example
    values = analyze_clinical(pairs.iloc[::-1], dictionary, legacy, items.iloc[::-1])
    tremor = values.profile.set_index("endpoint").loc["tremor"]
    assert tremor.n == 3 and tremor.n_improved == 1 and tremor.n_unchanged == 2
    assert tremor.rate_n == 1 and tremor.off_zero_n == 2
    assert tremor.off_positive_improved_fraction == 1.0
    assert values.private.tremor_improvement_rate.isna().sum() == 2
    composition = values.composition.set_index("component")
    assert composition.at["remaining_four", "improvement_mean"] == pytest.approx(1/3)
    assert composition.at["rigidity", "fraction_of_total_mean"] == -1.0
    assert composition.drop(index="total").fraction_of_total_mean.sum() == pytest.approx(1.0)
    assert values.audit["additive_component_checks"] == 6


def test_partial_archive_or_misaligned_visits(frozen_example):
    pairs, dictionary, legacy, items = frozen_example
    profile_only = analyze_clinical(pairs, dictionary, legacy)
    assert profile_only.composition is None and len(profile_only.profile) == 5
    bad = items.copy()
    bad.loc[0, "scan_id"] = bad.loc[0, "scan_id"] + "_wrong"
    with pytest.raises(ValueError, match="scan keys"):
        analyze_clinical(pairs, dictionary, legacy, bad)
    bad = items.copy()
    bad.loc[0, "item_posture"] = 4
    with pytest.raises(ValueError, match="total manual values"):
        analyze_clinical(pairs, dictionary, legacy, bad)
