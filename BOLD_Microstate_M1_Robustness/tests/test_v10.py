"""Software tests for the original clock, identities and correction scope."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from v10_frames import contributions, motion_mask, n_blocks, statistics_for_labels
from v10_statistics import formal_tests, standard_errors


def test_strict_fd_threshold_in_physical_clock():
    fd = np.array([np.nan, 0.1, 0.50, 0.51, 0.1, 0.1, 0.1, 0.1, 0.6, 0.1])
    mask = motion_mask(fd)
    np.testing.assert_array_equal(mask, [True, True, False, False, False, False, True, False, False, False])
    assert n_blocks(mask) == 2
    np.testing.assert_array_equal(motion_mask(np.array([np.nan, .51, .17])), [False, False, False])


def test_censoring_denominator_and_raw_state_mapping():
    labels = np.array([-1, 2, 0, 2])
    gfp = np.array([99., 1., 2., 3.])
    raw_to_frozen = np.array([1, 2, 0, 3])
    c, w, r, f, gev, reason = statistics_for_labels(labels, gfp, np.full((4, 4), .5), raw_to_frozen)
    assert not reason
    assert c[0] == pytest.approx(2/3)
    assert w[0] == pytest.approx(10/14)
    assert r[0] == pytest.approx(5/(14/3))
    assert f[0] == pytest.approx(.25)
    assert gev[0] == pytest.approx(10/(14*4))


def test_unavailable_m1_and_empty_window_remain_undefined():
    gfp = np.ones(3)
    corr = np.full((3, 4), .5)
    c, _, r, f, _, reason = statistics_for_labels(np.array([1, 1, 2]), gfp, corr, np.arange(4))
    assert c[0] == 0
    assert reason == "No retained M1 frames"
    assert np.isnan(r[0]) and np.isnan(f[0])
    *values, reason = statistics_for_labels(np.full(3, -1), gfp, corr, np.arange(4))
    assert reason == "No retained frames" and np.isnan(values).all()


def test_hierarchical_decomposition_in_own_window():
    row = dict(off_measurement_status="OK", on_measurement_status="OK",
               off_coverage=.2, on_coverage=.3, off_weight=.22, on_weight=.24,
               off_fit=.2, on_fit=.3, off_gev=.044, on_gev=.072,
               off_relative_amplitude=1.1, on_relative_amplitude=.8,
               off_mean_fd_mm=.3, on_mean_fd_mm=.1, off_mean_std_dvars=1., on_mean_std_dvars=.8,
               off_retained_seconds=360., on_retained_seconds=320.)
    unavailable = dict(row, on_measurement_status="No retained M1 frames", on_fit=np.nan)
    values, audit = contributions(pd.DataFrame([row, unavailable]))
    assert values.loc[0, "K_C"] == pytest.approx(2.375)
    assert values.loc[0, "K_R"] == pytest.approx(-1.875)
    assert values.loc[0, "C_W"] == pytest.approx(.5)
    assert values.loc[0, "delta_gev_pp"] == pytest.approx(2.8)
    assert values.loc[0, "C_F"] == pytest.approx(2.3)
    assert values.loc[1, ["K_C", "K_R"]].isna().all()
    assert audit["complete_pairs"] == audit["unavailable_pairs"] == 1


def test_hc3_standard_error_matches_independent_statsmodels():
    x = np.array([.05, .18, -.1, -.17, .25, .34, -.34])
    y = np.array([.3, -.2, 1.7, .2, .1, -.5, .13])
    fake = SimpleNamespace(
        draws=lambda n: np.random.default_rng(123).integers(0, n, (2000, n)),
        regression_bootstrap=lambda y, x, draws: np.asarray([
            np.linalg.lstsq(np.c_[np.ones(len(i)), x[i]], y[i], rcond=None)[0][0] for i in draws
        ]),
    )
    hc3 = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HC3").bse[0]
    assert standard_errors(fake, y, x)["standard_error_pp"] == pytest.approx(hc3)
    assert standard_errors(fake, y, None)["standard_error_pp"] == pytest.approx(y.std(ddof=1)/np.sqrt(len(y)))


def test_cumulative_holm_counts_unavailable_endpoint(monkeypatch):
    import v10_statistics as stats

    def fake(_, delta):
        n = int(np.count_nonzero(np.isfinite(delta)))
        return dict(permutation_p=.001 if n else np.nan, n_patients=n, reason="",
                    status="OK" if n else "NOT_ESTIMABLE"), np.array([])

    monkeypatch.setattr(stats.v7, "unadjusted_test", fake)
    table, combined = formal_tests(None, pd.DataFrame(dict(K_C=[1., 2., 3.], K_R=[np.nan]*3)))
    assert len(combined) == 10 and combined.endpoint.is_unique
    assert table.loc[0, "holm2_p"] == pytest.approx(.002)
    assert table.loc[0, "holm10_p"] == pytest.approx(.010)
    assert table.loc[1, ["holm2_p", "holm10_p"]].isna().all()
