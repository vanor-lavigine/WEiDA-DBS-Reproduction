"""Boundary cases for the independent release verifier."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_round3 import ENDPOINTS, bh, rank_test, reconstruct_cells


def test_ranks_use_average_ties_and_unestimable_tests_remain_na():
    x = np.array([1., 1., 2., 3., 4.])
    y = np.array([4., 2., 2., 1., 1.])
    assert rank_test(x, y, 3) == pytest.approx(tuple(spearmanr(x, y)))
    assert np.isnan(rank_test(x, np.ones(5), 3)).all()
    assert np.isnan(rank_test(x[:2], y[:2], 3)).all()


def test_bh_keeps_unestimable_slots_and_stable_ties():
    p = np.array([.01, np.nan, .01, .04] + [np.nan] * 16)
    q = bh(p)
    assert q[[0, 2, 3]] == pytest.approx([.10, .10, 20*.04/3])
    assert np.isnan(q[~np.isfinite(p)]).all()


def test_off_zero_excludes_only_its_endpoint_and_keeps_nonimprovement():
    source = pd.DataFrame({
        "patient_id": ["mock_a", "mock_b", "mock_c", "mock_d"],
        **{f"delta_coverage_{i}": [-.2, 0., .1, .2] for i in range(1, 5)},
    })
    for ep in ENDPOINTS:
        source[f"{ep}_off"] = [0., 10., 10., 10.] if ep == "tremor" else [8., 10., 10., 10.]
        source[f"{ep}_improvement"] = [-1., 0., -3., 2.]
        off = source[f"{ep}_off"]
        source[f"{ep}_improvement_rate"] = source[f"{ep}_improvement"].div(off.where(off > 0))
    reconstructed, cases = reconstruct_cells(source, 3)
    assert reconstructed.at[("M1", "tremor"), "n"] == 3
    assert reconstructed.at[("M1", "total"), "n"] == 4
    assert reconstructed.at[("M1", "tremor"), "n_off_zero"] == 1
    assert reconstructed.at[("M1", "tremor"), "n_missing"] == 0
    tremor = cases.loc[cases.state.eq("M1") & cases.clinical_endpoint.eq("tremor")]
    assert set(tremor.patient_id) == {"mock_b", "mock_c", "mock_d"}
    assert (tremor.improvement_rate < 0).sum() == 1
    assert (tremor.improvement_rate == 0).sum() == 1
