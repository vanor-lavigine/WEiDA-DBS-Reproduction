"""Synthetic checks for global synchrony dispersion and baseline adjustment."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slow5_phase_clinical import BAND, WEIDA_BAND, partial_correlation, scan_metric


def test_synchronous_regions_have_zero_temporal_dispersion():
    t = np.arange(209) * 2.1
    wave = np.sin(2*np.pi*0.055*t)
    metrics = scan_metric(np.tile(wave[:, None], (1, 78)))
    assert metrics["metric"] < 1e-12
    assert metrics["trimmed"] < 1e-12


def test_two_frequencies_create_phase_drift_independent_of_roi_order():
    t = np.arange(210) * 2.1
    x = np.sin(2*np.pi*t[:, None]*np.repeat([0.047, 0.063], 39))
    m = scan_metric(x)
    assert m["metric"] > 0.15
    np.testing.assert_allclose(m["metric"], scan_metric(x[:, ::-1])["metric"], atol=1e-12)


def test_partial_change_reduces_to_sign_reversed_on_after_matching_baselines():
    x0,y0,x1,y1 = np.random.default_rng(205).normal(size=(4,25))
    result = partial_correlation(x1-x0, y0-y1, x0, y0)
    on = partial_correlation(x1, y1, x0, y0)
    np.testing.assert_allclose(result["r"], -on["r"], atol=1e-14)
    assert result["df"] == 21


def test_primary_passband_uses_the_requested_frequencies():
    t = np.arange(210)*2.1
    x = np.sin(2*np.pi*.055*t[:, None] + np.arange(78)[None,:]/10)
    assert scan_metric(x, WEIDA_BAND)["median_roi_band_power_fraction"] > .95
    assert scan_metric(x, BAND)["median_roi_band_power_fraction"] < .01


def test_default_filter_is_the_primary_frequency_band():
    x = np.random.default_rng(23).normal(size=(209,78))
    assert scan_metric(x) == scan_metric(x, WEIDA_BAND)
