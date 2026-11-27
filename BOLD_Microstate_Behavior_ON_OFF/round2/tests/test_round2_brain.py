"""Focused tests for the new duration and health-reference estimands."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from brain import bootstrap_distance, distance_change


def test_absolute_distance_handles_lower_crossing_and_equal_distance():
    off = np.array([3., 8., 1., 3., 4., 5.])
    on = np.array([4., 2., 8., 7., 8., 5.])
    before, after, change = distance_change(off, on, 5.)
    np.testing.assert_array_equal(before, [2., 3., 4., 2., 1., 0.])
    np.testing.assert_array_equal(after, [1., 3., 3., 2., 3., 0.])
    np.testing.assert_array_equal(change, [1., 0., 1., 0., -2., 0.])


def test_bootstrap_matches_independent_scalar_reconstruction():
    hc = np.array([4., 6., 8.])
    off, on = np.array([3., 11.]), np.array([9., 7.])
    draws, seed = 20, 20261127
    rng = np.random.default_rng(seed)
    hc_samples = [rng.integers(0, len(hc), len(hc)) for _ in range(draws)]
    pair_samples = [rng.integers(0, len(off), len(off)) for _ in range(draws)]
    expected = []
    for h, patients in zip(hc_samples, pair_samples, strict=True):
        center = sum(hc[i] for i in h)/len(h)
        expected.append(sum(abs(off[i]-center)-abs(on[i]-center) for i in patients)/len(patients))
    np.testing.assert_allclose(bootstrap_distance(hc, off, on, draws=draws, seed=seed), expected)


def test_bootstrap_preserves_pairs_and_resamples_the_center():
    hc = np.array([4., 6., 8.])
    fixed_pair = np.array([3., 11.])
    unchanged = bootstrap_distance(hc, fixed_pair, fixed_pair, draws=20, seed=5)
    np.testing.assert_array_equal(unchanged, np.zeros(20))
    reestimated = bootstrap_distance(hc, np.array([3., 3.]), np.array([9., 9.]), draws=20, seed=5)
    assert np.ptp(reestimated) > 0
