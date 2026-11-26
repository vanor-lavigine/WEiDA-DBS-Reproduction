"""Regression checks for pairing, units and the stage and cumulative families."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from severity import (HISTORICAL_STATUS, PRIOR_KEYS, STAGE_KEYS, add_corrections, check_units, cumulative,
                      merge_scores, one_row_per_scan, read_registry, run_stage)


@pytest.fixture
def example():
    registry = read_registry()
    features = registry.metric.unique()
    paired, measurements, scores = [], [], []
    for i in range(6):
        person = f"fixture-{i}"
        off, on = f"fixture-off-{i}", f"fixture-on-{i}"
        keys = dict(participant_id=person, on_scan_id=on, off_scan_id=off)
        paired.append(keys)
        scores.append(dict(keys, updrs_iii_on=24 - i, updrs_iii_off=35 + i, visit_evidence=HISTORICAL_STATUS))
        for condition, key in (("ON", on), ("OFF", off)):
            values = {name: i + .01 if condition == "ON" else i + .02 for name in features}
            measurements.append(dict(values, subject=person, scan=key, condition=condition, fd=.1 + i / 70,
                                     fold=i % 5, valid_frames=100, coverage_4=.15, occurrence_4=2 / 210))
    return registry, pd.DataFrame(paired).set_index("participant_id"), pd.DataFrame(measurements), pd.DataFrame(scores)


def align(example):
    registry, pairs, scans, scores = example
    scans = one_row_per_scan(pairs, scans, list(registry.metric.unique()))
    return registry, pairs, scans, scores


def test_exact_keys_are_used_even_when_score_rows_are_shuffled(example):
    registry, pairs, scans, scores = align(example)
    observed = merge_scores(pairs, scans.iloc[::-1], scores.sample(frac=1, random_state=17), HISTORICAL_STATUS)
    off = observed[observed.condition.eq("OFF")].sort_values("participant_id")
    on = observed[observed.condition.eq("ON")].sort_values("participant_id")
    assert off.score.tolist() == [35 + i for i in range(6)]
    assert on.score.tolist() == [24 - i for i in range(6)]
    assert registry.loc[registry.endpoint.eq("off_gev_1"), "display_multiplier"].item() == 100
    assert registry.loc[registry.endpoint.eq("on_occurrence_1"), "display_multiplier"].item() == 60


def test_wrong_scan_side_or_duplicate_score_fails(example):
    _, pairs, scans, scores = align(example)
    wrong = scores.copy()
    wrong.loc[0, ["on_scan_id", "off_scan_id"]] = wrong.loc[0, ["off_scan_id", "on_scan_id"]].to_numpy()
    with pytest.raises(ValueError, match="wrong patient"):
        merge_scores(pairs, scans, wrong, HISTORICAL_STATUS)
    with pytest.raises(ValueError, match="duplicated"):
        merge_scores(pairs, scans, pd.concat([scores, scores.iloc[:1]]), HISTORICAL_STATUS)


def test_missing_fd_changes_only_same_scan_sensitivity_and_severity_direction(example):
    registry, pairs, scans, scores = align(example)
    scans.loc[scans.condition.eq("OFF") & scans.participant_id.eq("fixture-2"), "fd"] = np.nan
    observed = merge_scores(pairs, scans, scores, HISTORICAL_STATUS)
    result = run_stage(observed, registry, "test archive", HISTORICAL_STATUS).set_index("endpoint")
    assert result.at["off_gev_1", "n_patients"] == 6
    assert result.at["off_gev_1", "fd_n_patients"] == 5
    assert result.at["on_gev_1", "fd_n_patients"] == 6
    assert result.at["off_gev_1", "rho"] == pytest.approx(+1.)
    assert result.at["on_gev_1", "rho"] == pytest.approx(-1.)


def test_frozen_units_and_double_conversion():
    data = pd.DataFrame({
        "valid_frames": [100], "occurrence_1": [4 / 210], "occurrence_3": [3 / 210],
        "switching_rate": [10 / 210], "gev_1": [.1], "gev_3": [.2], "dispersion_3": [.9],
        "duration_4": [15.75], "coverage_4": [.15], "occurrence_4": [2 / 210],
    })
    check_units(data)
    with pytest.raises(ValueError, match="per second"):
        check_units(data.assign(occurrence_1=60 * data.occurrence_1))
    with pytest.raises(ValueError, match="dwell time"):
        check_units(data.assign(duration_4=data.duration_4 * 2.1))


def test_eleven_tests_are_counted_with_unavailable_p_values():
    stage = pd.DataFrame({"endpoint": STAGE_KEYS, "permutation_p": [.002] + [np.nan] * 10})
    add_corrections(stage, "permutation_p", STAGE_KEYS, "11")
    assert stage.at[0, "holm11_p"] == pytest.approx(.022)
    assert stage.at[0, "bh11_p"] == pytest.approx(.022)
    assert stage.holm11_p.isna().sum() == 10
    with pytest.raises(ValueError, match="fully specified"):
        add_corrections(stage.iloc[:-1].copy(), "permutation_p", STAGE_KEYS, "11")


def test_cumulative_retains_old_p_and_stage_columns_with_15_tests(example):
    registry, pairs, scans, scores = align(example)
    stage = run_stage(merge_scores(pairs, scans, scores, HISTORICAL_STATUS), registry, "test archive", HISTORICAL_STATUS)
    stage["permutation_p"] = np.nan
    stage.loc[0, "permutation_p"] = .002
    add_corrections(stage, "permutation_p", STAGE_KEYS, "11")
    previous = pd.DataFrame({
        "endpoint": PRIOR_KEYS, "analysis_path": HISTORICAL_STATUS, "score_source": "WEiDA4-1_atlas1_group0",
        "family_size": 4, "status": "OK", "reason": "", "n_pairs": 25, "rho": .1,
        "ci95_low": -.2, "ci95_high": .3, "raw_p": .05, "permutation_p": .06,
        "permutation_holm_p": .24, "permutation_fdr_p": .06,
    })
    result = cumulative(stage, previous)
    assert len(result) == 15 and result.cumulative_family_size.eq(15).all()
    assert result.at[0, "holm15_p"] == pytest.approx(.03)
    assert result.at[0, "holm11_p"] == pytest.approx(.022)
    assert result.loc[11:, "permutation_p"].eq(.06).all()
    assert result.loc[11:, "holm4_p"].eq(.24).all()
    assert result.loc[1:10, "holm15_p"].isna().all()
