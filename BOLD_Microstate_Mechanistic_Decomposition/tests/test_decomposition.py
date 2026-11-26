from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inference import holm_complete
from measurements import log_ratio, readout, roi_acf, shapley, stable_percent, within_state_fit


def test_gev_uses_squared_weights_and_shapley_sums():
    power = np.array([1, 2, 3])
    fit = np.array([.5, .2, 1])
    w, f, gev = within_state_fit(np.array([True, False, True]), power, fit)
    assert np.isclose(w, 4 / 6)
    assert np.isclose(f, 3.25 / 4)
    assert np.isclose(gev, w * f)
    w_on, w_off = np.array([w, .8]), np.array([.4, .2])
    f_on, f_off = np.array([f, .2]), np.array([.9, .8])
    a, b = shapley(w_on, w_off, f_on, f_off)
    np.testing.assert_allclose(a + b, w_on * f_on - w_off * f_off, rtol=0, atol=1e-14)
    ratio = stable_percent(np.array([2., 2., -1.]), np.array([.001, 4., -1.]), .1)
    np.testing.assert_allclose(ratio, [np.nan, 50., 100.], equal_nan=True)


def test_gap_aware_episodes_and_exit_counts():
    labels = np.array([3, 3, -1, 0, 3, 3, 1, -1, 3, -1, 2, 3])
    output = readout(labels, np.ones(len(labels)), np.ones(len(labels)))
    assert output["m4_n_exits"] == 1
    assert output["m4_q42"] == 1.
    assert output["m4_q41"] == 0.
    assert output["episodes_4"] == 4
    assert output["right_boundary_episodes_4"] == 3
    assert output["complete_episodes_4"] == 1
    assert output["dwell_edge_mean_4"] == pytest.approx(4.2)
    assert output["duration_4"] == pytest.approx(output["coverage_4"] / output["occurrence_4"])
    assert output["duration_4"] != pytest.approx(1 / output["occurrence_4"])
    assert np.isfinite(output["m4_ilr_1"])


def test_no_exit_and_no_state_are_preserved_as_distinct_cases():
    no_exit = readout(np.array([3, 3, -1, 3]), np.ones(4), np.ones(4))
    assert no_exit["m4_exit_rate_min"] == 0
    assert np.isnan(no_exit["m4_q41"])
    assert np.isnan(no_exit["m4_ilr_1"])
    assert no_exit["occurrence_4"] > 0
    absent = readout(np.array([0, -1, 0]), np.ones(3), np.ones(3))
    assert absent["coverage_4"] == absent["occurrence_4"] == 0
    assert np.isnan(absent["duration_4"])
    np.testing.assert_allclose(log_ratio(np.array([.5, 0, 2]), np.array([1, .5, 1])),
                               [-np.log(2), np.nan, np.log(2)], equal_nan=True)


def test_acf_respects_valid_clock_blocks():
    data = np.array([1., 2., np.nan, 3., 4., 5.])[:, None]
    correlations, support = roi_acf(data, np.isfinite(data[:, 0]), 3)
    np.testing.assert_array_equal(support, [3, 1, 0])
    assert np.isnan(correlations[2, 0])


def test_holm_pads_unavailable_and_preserves_all_six_slots():
    raw = [.01, .02, np.nan, .03, .5, .8]
    np.testing.assert_allclose(holm_complete(raw, 6), [.06, .10, np.nan, .12, 1, 1], equal_nan=True)
    with pytest.raises(ValueError, match="family is incomplete"):
        holm_complete(raw[:-1], 6)
