from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from identity_audit import CONFLICT, INSUFFICIENT, VERIFIED, score_visit_status, study_scan, target_from_record
from regression import adjust, estimate, hc3_stats, model_data


def test_visit_dates_are_compared_independently_of_name_and_condition():
    fake = "9001"
    assert study_scan(f"STN{fake}", "OFF") == f"sub-stnoff{fake}"
    assert study_scan(f"STN{fake}", "ON") == f"sub-stnon{fake}"
    assert study_scan(f"STN{fake}", "med_off") is None
    assert score_visit_status("2099-03-02", "2099-03-02", "2099-03-02", True) == VERIFIED
    assert score_visit_status("2099-03-02", "2099-03-02", None, True) == INSUFFICIENT
    assert score_visit_status("2099-03-02", "2099-03-02", "2099-03-09", True) == CONFLICT
    assert target_from_record("STN", "靶点不对称") == "mixed_or_asymmetric"
    assert target_from_record("STN", "左右电压不对称") == "STN"


def test_hc3_and_small_exact_freedman_lane_match_independent_ols():
    x = np.c_[np.ones(5), [0., 1., 1.5, 2., 4.], [-1., 2., 0., 1., -2.]]
    y = np.array([2., 4., 4., 5., 9.])
    baseline = sm.OLS(y, x[:, :-1]).fit()
    comparison = sm.OLS(y, x).fit(cov_type="HC3")
    beta, observed, _ = hc3_stats(x, y)
    assert beta == pytest.approx(comparison.params[-1])
    assert observed == pytest.approx(comparison.tvalues[-1])
    tail = 0
    for p in list(permutations(range(len(y))))[1:]:
        sample = baseline.fittedvalues + baseline.resid[list(p)]
        t = sm.OLS(sample, x).fit(cov_type="HC3").tvalues[-1]
        tail += abs(t) >= abs(comparison.tvalues[-1]) - 1e-12
    output = estimate(x, y)
    assert output["permutations_valid"] == 119
    assert output["permutation_p"] == pytest.approx((1 + tail) / 120)
    assert output["bootstrap_valid"] + output["bootstrap_failed"] == 2000


def test_only_percent_model_drops_off_zero():
    patients = pd.DataFrame({"participant_id": ["a", "b"], "visit_status": [VERIFIED, VERIFIED],
                             "off_tremor": [0, 2], "tremor_improvement": [0, 1],
                             "tremor_percent_improvement": [np.nan, 50], "fd": [0, .3], "test_x": [1., 2.]})
    endpoint = {"outcome": "tremor", "predictor": "test_x"}
    model = {"motion": "fd", "sample": "eligible", "outcome_scale": "absolute"}
    _, absolute, _, _, _ = model_data(patients, endpoint, model)
    _, percent, _, _, diag = model_data(patients, endpoint, dict(model, outcome_scale="percentage"))
    np.testing.assert_array_equal(absolute, [0, 1])
    np.testing.assert_array_equal(percent, [50])
    assert diag["n_off_zero_in_sample"] == diag["n_missing_outcome"] == 1


def test_holm_and_bh_keep_na_slots_in_denominators():
    np.testing.assert_allclose(adjust([.01, np.nan, .04], "holm", 3), [.03, np.nan, .08], equal_nan=True)
    raw = [.01, .02, np.nan, .2, .2, .2, .4, .8, 1.]
    np.testing.assert_allclose(adjust(raw, "fdr_bh", 9)[:3], [.09, .09, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="family is incomplete"):
        adjust(raw, "holm", 12)
