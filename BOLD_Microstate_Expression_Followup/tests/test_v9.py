"""Targeted tests for unit conventions, frozen columns and complete Holm families."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v9_inputs import V8_CFG
from v9_measurements import CONTRIBUTIONS, decompose, frame_readout
from v9_statistics import adjust_tables, v8_regression


def test_raw_to_frozen_three_cycle_and_valid_mask():
    labels = np.array([0, 1, -1, 2])
    gfp = np.array([1.0, 2.0, np.nan, 3.0])
    corr = np.array([[.1, .2, .3, .4], [.5, .6, .7, .8], [np.nan] * 4, [.9, .2, .4, .6]])
    mapping = np.array([1, 2, 0, 3])
    assert not np.array_equal(mapping, np.argsort(mapping))
    e, gev, m1, count = frame_readout(labels, gfp, corr, mapping)
    expected_e = [(1 * .3**2 + 4 * .7**2 + 9 * .4**2) / 14,
                  (1 * .1**2 + 4 * .5**2 + 9 * .9**2) / 14,
                  (1 * .2**2 + 4 * .6**2 + 9 * .2**2) / 14,
                  (1 * .4**2 + 4 * .8**2 + 9 * .6**2) / 14]
    np.testing.assert_allclose(e, expected_e, rtol=0, atol=1e-15)
    np.testing.assert_allclose(gev, np.array([9 * .4**2, 1 * .1**2, 4 * .6**2, 0.0]) / 14, rtol=0, atol=1e-15)
    assert np.all(e >= gev)
    assert e.sum() != pytest.approx(1)
    assert count == 3
    assert m1["coverage_1"] == pytest.approx(1 / 3)
    assert m1["r_1"] == pytest.approx(9 / (14 / 3))


def test_rounded_correlation_is_clipped_within_the_frozen_tolerance():
    labels = np.array([0])
    power = np.array([1.0])
    mapping = np.arange(4)
    corr = np.array([[1.0 + 1e-13, -1e-13, 0.4, 0.0]])
    expression, _, _, _ = frame_readout(labels, power, corr, mapping)
    np.testing.assert_allclose(expression, [1, 0, .16, 0], atol=1e-15)
    corr[0, 0] = 1.01
    with pytest.raises(ValueError, match="outside the allowed range"):
        frame_readout(labels, power, corr, mapping)
    with pytest.raises(ValueError, match="no usable GFP"):
        frame_readout(labels, np.zeros(1), np.zeros((1, 4)), mapping)


def example_pair():
    return pd.DataFrame([dict(
        participant_id=1, off_coverage_1=.20, on_coverage_1=.30, off_m1_weight=.24, on_m1_weight=.24,
        off_m1_fit=.20, on_m1_fit=.30, off_m1_gev=.048, on_m1_gev=.072,
        m1_shapley_w_pp=0.0, m1_shapley_f_pp=2.4, m1_gev_delta_pp=2.4, fd=.01, mean_std_dvars=-.01,
    )])


def test_nested_contributions_are_per_patient_and_in_percentage_points():
    rows, audit = decompose(example_pair())
    assert rows.off_r.item() == pytest.approx(1.2)
    assert rows.on_r.item() == pytest.approx(.8)
    np.testing.assert_allclose(rows[CONTRIBUTIONS].iloc[0], [2.5, -2.5], rtol=0, atol=1e-12)
    assert audit["decomposable_pairs"] == 1
    assert audit["max_error_original_gev_change_pp"] < 1e-12


def test_no_pseudocount_when_m1_coverage_is_zero():
    pair = example_pair().assign(off_coverage_1=0., off_m1_weight=0., off_m1_fit=np.nan, off_m1_gev=0.)
    rows, audit = decompose(pair)
    assert rows[["off_r", *CONTRIBUTIONS]].isna().all(axis=None)
    assert audit["undecomposable_pairs"] == 1


def test_correct_using_full_precision_raw_p_and_keep_na_family_member():
    previous = pd.DataFrame([
        dict(endpoint=f"O{i}", permutation_p=.06 + i / 100, holm6_p=1.,
             model="Unadjusted", n_patients=25, status="OK", reason="")
        for i in range(6)
    ])
    raw = 0.0123456789
    added = pd.DataFrame([
        dict(endpoint="A1", permutation_p=raw, model="Unadjusted", n_patients=25, status="OK", reason=""),
        dict(endpoint="A2", permutation_p=np.nan, model="Unadjusted", n_patients=0,
             status="NOT_ESTIMABLE", reason="no complete pairs"),
    ])
    result, joint = adjust_tables(added, previous, set(previous.endpoint), "imaging", "Prior")
    assert result.loc[0, "holm2_p"] == pytest.approx(2 * raw)
    assert result.loc[0, "holm8_p"] == pytest.approx(8 * raw)
    assert result.loc[1, ["holm2_p", "holm8_p"]].isna().all()
    assert len(joint) == 8
    assert joint.loc[joint.endpoint.eq("O0"), "permutation_p"].item() == previous.permutation_p[0]


def test_clinical_reduced_model_has_off_score_and_motion_as_nuisance():
    model = next(row for row in V8_CFG["models"] if row["name"] == "Primary_FD")
    frame = pd.DataFrame(dict(participant_id=[2, 1], off_motor_total=[30, 40], motor_total_improvement=[5, -2],
                              fd=[-.01, .02], delta_e1_x100=[-1, 2]))
    x, y, patients, fields, _ = v8_regression.model_data(
        frame, {"outcome": "motor_total", "predictor": "delta_e1_x100"}, model,
    )
    reduced, _, _ = v8_regression.residual_reference(x, y)
    assert fields == ["intercept", "off_motor_total", "fd", "delta_e1_x100"]
    np.testing.assert_array_equal(patients, [1, 2])
    np.testing.assert_array_equal(y, [-2, 5])
    np.testing.assert_array_equal(x[:, -1], [2, -1])
    np.testing.assert_array_equal(reduced, [[1, 40, .02], [1, 30, -.01]])
