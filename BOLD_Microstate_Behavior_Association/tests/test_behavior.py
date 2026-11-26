"""Targeted checks for frozen pairs, visit evidence and the four-test family."""

from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from association import alternative_values, association, correct, fd_test
from io_sources import ENDPOINTS, FIELDS, ORIGINAL_FILES, archive_pairs, load_frozen, verified_scores
from run import family


def test_exact_randomization_counts_unique_tied_assignments():
    y = np.array([0., 1., 1., 2.])
    x = np.arange(4, dtype=float)
    orders, mode, size = alternative_values(y, salt=203)
    assert mode == "enumerated"
    assert size == 12
    assert {tuple(v) for v in orders} == set(permutations(y)) - {tuple(y)}
    observed = stats.spearmanr(x, y).statistic
    direct = [abs(stats.spearmanr(x, values).statistic) for values in set(permutations(y))]
    actual = association(x, y, np.full(len(y), np.nan))
    assert actual["permutation_valid"] == 11
    assert actual["permutation_p"] == pytest.approx(np.mean(np.array(direct) >= abs(observed) - 1e-12))
    assert actual["rho"] == pytest.approx(observed)
    assert actual["improvement_tie_excess"] == 1
    assert actual["bootstrap_valid"] + actual["bootstrap_failed"] == 2000
    assert actual["bootstrap_constant_improvement"] > 0


def test_monte_carlo_has_fixed_distinct_nonidentity_draws():
    values = np.arange(12, dtype=float)
    a, mode, total = alternative_values(values, salt=203)
    b, _, _ = alternative_values(values, salt=203)
    assert mode == "monte_carlo" and total > 10000
    assert len(a) == len({tuple(v) for v in a}) == 10000
    assert tuple(values) not in {tuple(v) for v in a}
    np.testing.assert_array_equal(a, b)


def test_constants_and_missing_data_are_explicit():
    result = association([2, 2, 2, np.nan], [5, 6, 7, 8], [0, 1, 3, 4])
    assert result["status"] == "NOT_ESTIMABLE" and "Constant" in result["reason"]
    assert result["n_pairs"] == 3
    assert np.isnan(result["raw_p"]) and np.isnan(result["permutation_p"])
    assert result["permutation_draws"] == result["bootstrap_valid"] == 0


def test_missing_fd_never_restricts_the_main_analysis():
    x = [1., 3, 0, 2, 4, 5]
    y = [0., 4, 2, 3, 3, 1]
    result = association(x, y, [np.nan, 1, 2, 3, 4, 5])
    reference = association(x, y, [0, 1, 2, 3, 4, 5])
    assert result["n_pairs"] == reference["n_pairs"] == 6
    assert result["raw_p"] == reference["raw_p"]
    assert result["permutation_p"] == reference["permutation_p"]
    assert result["ci95_low"] == reference["ci95_low"]
    assert result["fd_n_pairs"] == 5


def test_rank_freedman_lane_matches_separate_statsmodels_exact_result():
    x = np.array([0., 2, 1, 4, 3])
    y = np.array([4., 0, 1, 1, 3])
    fd = np.array([1., 3., 0., 4., 2.])
    xr, yr, fr = [stats.rankdata(v) for v in (x, y, fd)]
    reduced = sm.add_constant(fr)
    model = sm.OLS(yr, reduced).fit()
    full = np.c_[reduced, xr]
    observed = abs(sm.OLS(yr, full).fit(cov_type="HC3").tvalues[-1])
    ref_rho = np.corrcoef(sm.OLS(xr, reduced).fit().resid, model.resid)[0, 1]
    t_null = [abs(sm.OLS(model.fittedvalues + np.asarray(row), full).fit(cov_type="HC3").tvalues[-1])
              for row in set(permutations(model.resid))]
    result = fd_test(x, y, fd)
    assert result["fd_status"] == "OK"
    assert result["fd_rho"] == pytest.approx(ref_rho)
    assert result["fd_p"] == pytest.approx(np.mean(np.array(t_null) >= observed - 1e-12))
    assert result["fd_bootstrap_requested"] == result["fd_bootstrap_valid"] + result["fd_bootstrap_failed"]


def test_fixed_four_endpoint_holm_even_when_three_p_values_are_missing():
    source = pd.DataFrame({"endpoint": [f"delta_{item['field']}" for item in ENDPOINTS],
                           "raw_p": [.02, np.nan, np.nan, np.nan],
                           "permutation_p": [.02, np.nan, np.nan, np.nan],
                           "fd_p": [np.nan, .03, np.nan, np.nan]})
    adjusted = correct(source)
    assert adjusted.family_size.tolist() == [4] * 4
    assert adjusted.permutation_holm_p[0] == pytest.approx(.08)
    assert adjusted.permutation_fdr_p[0] == pytest.approx(.08)
    assert adjusted.fd_permutation_holm_p[1] == pytest.approx(.12)
    assert adjusted.permutation_holm_p.isna().sum() == 3


@pytest.fixture
def fake_frozen(tmp_path):
    tr, frame_count = 2.1, 100
    cohorts, manifests, deltas = [], [], []
    for i in range(25):
        pid = f"synthetic-patient-{i}"
        sides = {}
        for s, v in (("ON", 1), ("OFF", 0)):
            record = dict(scan=f"sub-STN{s.lower()}{i+101}", subject=pid, condition=s, fold=i % 5,
                          valid_frames=frame_count, fd=.2 + .01 * v, gev_1=.1 + .02 * v,
                          duration_4=12 - 2 * v, occurrence_1=(3 + v) / (tr * frame_count),
                          switching_rate=(11 + v) / (tr * frame_count))
            cohorts.append(record)
            sides[s] = record
        a, b = sides["ON"], sides["OFF"]
        manifests.append(dict(participant_id=pid, on_scan_id=a["scan"], off_scan_id=b["scan"],
                              on_mean_fd=a["fd"], off_mean_fd=b["fd"],
                              on_scan_time="2000-01-01 10:00:00", off_scan_time="2000-01-01 11:00:00",
                              scan_order="ON before OFF", on_medication_hours=np.nan, off_medication_hours=np.nan))
        deltas.append(dict(participant_id=pid, **{k: a[k] - b[k] for k in FIELDS + ["fd"]}))
    scans = pd.DataFrame(cohorts)
    hc = scans.iloc[:15].assign(scan=lambda d: "hc-synth-" + d.index.astype(str),
                                condition="HC", subject=lambda d: "hc-synth-" + d.index.astype(str))
    data = [pd.DataFrame(manifests), pd.concat([scans, hc]), scans, pd.DataFrame(deltas)]
    for rel, frame in zip(ORIGINAL_FILES.values(), data, strict=True):
        dest = tmp_path / rel
        dest.parent.mkdir(exist_ok=True, parents=True)
        frame.to_csv(dest, index=False)
    return tmp_path


def test_frozen_difference_direction_and_single_display_conversion(fake_frozen):
    paired, counts, _ = load_frozen(fake_frozen)
    assert counts["frozen_pd_pairs"] == 25 and counts["hc_scans"] == 15
    assert paired.delta_duration_4.iloc[0] == pytest.approx(-2)
    exp = {e["field"]: e["display_multiplier"] for e in ENDPOINTS}
    assert paired.delta_gev_1.iloc[0] * exp["gev_1"] == pytest.approx(2.)
    assert paired.delta_occurrence_1.iloc[0] * exp["occurrence_1"] == pytest.approx(60 / 210)
    assert paired.delta_switching_rate.iloc[0] * exp["switching_rate"] == pytest.approx(60 / 210)
    scores = pd.DataFrame({"participant_id": paired.index, "updrs_iii_off": 40, "updrs_iii_on": 30})
    results, joined = family(paired, scores, "test", "test fixture")
    assert joined.improvement.eq(10).all()
    assert results.status.eq("NOT_ESTIMABLE").all()


@pytest.mark.parametrize("kind", ["duplicate_scan", "swapped_conditions", "wrong_difference"])
def test_frozen_roster_detects_duplicates_wrong_condition_and_delta(fake_frozen, kind):
    rel = ORIGINAL_FILES["differences" if kind == "wrong_difference" else "manifest"]
    frame = pd.read_csv(fake_frozen / rel)
    if kind == "duplicate_scan":
        frame.loc[1, "on_scan_id"] = frame.loc[0, "on_scan_id"]
    elif kind == "swapped_conditions":
        cols = ["on_scan_id", "off_scan_id"]
        frame.loc[0, cols] = frame.loc[0, cols[::-1]].to_numpy()
    else:
        frame.loc[0, "duration_4"] *= -1
    frame.to_csv(fake_frozen / rel, index=False)
    with pytest.raises(ValueError):
        load_frozen(fake_frozen)


def test_archive_join_checks_exact_scan_condition_and_patient_keys(fake_frozen):
    pairs, _, _ = load_frozen(fake_frozen)
    # A study code may differ from the frozen image-side participant ID.
    sample = pairs.iloc[0]
    rows = pd.DataFrame({"scan": [sample.on_scan_id, sample.off_scan_id],
                         "condition": ["ON", "OFF"], "score": [22, 37],
                         "mapped_study_code": ["chart-lookup", "chart-lookup"]})
    raw, _ = archive_pairs(rows, pairs)
    assert raw.loc[0, "updrs_iii_off"] - raw.loc[0, "updrs_iii_on"] == 15
    with pytest.raises(ValueError, match="scan or DBS"):
        archive_pairs(rows.assign(condition=["OFF", "ON"]), pairs)
    with pytest.raises(ValueError, match="study codes"):
        archive_pairs(rows.assign(mapped_study_code=["A", "B"]), pairs)


def test_verified_dates_require_original_evidence_and_match_scan_day(fake_frozen, tmp_path):
    pairs, _, _ = load_frozen(fake_frozen)
    first = pairs.reset_index().iloc[0]
    person, on, off = first.participant_id, first.on_scan_id, first.off_scan_id
    side_path = tmp_path / "evidence.csv"
    score_path = tmp_path / "scores.csv"
    original = tmp_path / "synthetic_original.csv"
    original.write_text("synthetic fixture only\n", encoding="utf-8")
    pd.DataFrame([dict(participant_id=person, on_scan_id=on, off_scan_id=off,
                       updrs_iii_on=10, updrs_iii_off=20, visit_evidence="confirmed_assessment_date")]
                 ).to_csv(score_path, index=False)
    missing, _, reason = verified_scores(score_path, side_path, pairs)
    assert missing.empty and reason
    rows = pd.DataFrame([dict(participant_id=person, condition=c, scan_id=scan, source_path=original.name,
                              record_locator="fixture row 1", visit_basis="synthetic test",
                              assessment_date="2000-01-01") for c, scan in (("ON", on), ("OFF", off))])
    rows.to_csv(side_path, index=False)
    valid, manifests, reason = verified_scores(score_path, side_path, pairs)
    assert len(valid) == 1 and not reason and len(manifests) == 3
    rows.loc[1, "assessment_date"] = "2000-01-02"
    rows.to_csv(side_path, index=False)
    with pytest.raises(ValueError, match="Date-based evidence"):
        verified_scores(score_path, side_path, pairs)
