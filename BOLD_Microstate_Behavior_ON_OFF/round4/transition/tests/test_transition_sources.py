"""Exercise the gap and denominator rules even though A3 has no interior gaps."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from transition_sources import measure_scan


class TransitionBoundaryTests(unittest.TestCase):
    def test_interrupted_series_preserves_each_block(self):
        result = measure_scan([0, 0, -1, 1, 3, 3, -1, 3, 0, -1, 2])
        expected_counts = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 1],
        ])
        np.testing.assert_array_equal(result.counts, expected_counts)
        np.testing.assert_array_equal(result.frames, [3, 1, 1, 3])
        np.testing.assert_array_equal(result.block_starts, [0, 3, 7, 10])
        np.testing.assert_array_equal(result.block_ends, [1, 5, 8, 10])
        np.testing.assert_array_equal(result.starts, [1, 1, 1, 1])
        np.testing.assert_array_equal(result.ends, [2, 0, 1, 1])
        np.testing.assert_array_equal(result.episodes, [2, 1, 1, 2])
        np.testing.assert_array_equal(result.enters, [1, 0, 0, 1])
        np.testing.assert_array_equal(result.exits, [0, 1, 0, 1])
        np.testing.assert_allclose(result.durations, [3.15, 2.1, 2.1, 3.15])
        np.testing.assert_allclose(result.p_weida[0], [1, 0, 0, 0])
        np.testing.assert_allclose(result.p_weida[3], [.5, 0, 0, .5])
        np.testing.assert_allclose(result.p_exit[3], [1, 0, 0, np.nan], equal_nan=True)
        self.assertTrue(np.isnan(result.p_weida[2]).all())
        self.assertTrue(np.isnan(result.p_exit[[0, 2]]).all())

    def test_leading_trailing_and_singleton_gaps(self):
        result = measure_scan([-1, 2, -1, -1, 3, -1])
        self.assertEqual(result.counts.sum(), 0)
        self.assertTrue(np.isnan(result.p_weida).all())
        self.assertTrue(np.isnan(result.p_exit).all())
        np.testing.assert_array_equal(result.episodes, [0, 0, 1, 1])
        np.testing.assert_array_equal(result.block_starts, result.block_ends)

    def test_unknown_label_is_rejected(self):
        for labels in ([0, 4], [0, -2], [0., 1.], [[0, 1]]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                measure_scan(labels)


if __name__ == "__main__":
    unittest.main()
