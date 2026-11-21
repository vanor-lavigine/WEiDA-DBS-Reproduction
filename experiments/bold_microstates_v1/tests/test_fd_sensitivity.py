"""Synthetic checks for the paired motion-covariate test."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fd_sensitivity import adjusted_endpoint


class MotionAdjustmentTests(unittest.TestCase):
    x = np.array([-0.14, -0.09, -0.02, 0.03, 0.07, 0.11, 0.18, 0.31])
    y = np.array([0.2, 0.5, -0.4, 0.8, -0.2, 0.1, -0.7, -0.3])

    def fit(self, x, y):
        return adjusted_endpoint(x, y, 199, 250,
                                 np.random.default_rng(19), np.random.default_rng(42))

    def test_adjustment_targets_equal_motion_and_retains_the_null_slope(self):
        original = self.fit(self.x, self.y)
        shifted = self.fit(self.x, self.y + 4 * self.x)
        self.assertNotAlmostEqual(self.y.mean(), (self.y + 4 * self.x).mean())
        for key in ("adjusted_delta", "intercept_hc3_se", "adjusted_p_raw",
                    "adjusted_ci_low", "adjusted_ci_high"):
            self.assertAlmostEqual(original[key], shifted[key], places=12)
        self.assertAlmostEqual(shifted["fd_slope_per_0_1_mm"] - original["fd_slope_per_0_1_mm"], 0.4)

    def test_against_explicit_subject_level_refits(self):
        result = self.fit(self.x, self.y)
        design = np.column_stack([np.ones(len(self.x)), self.x])
        gram_inv = np.linalg.inv(design.T @ design)
        diagonal = np.diag(design @ gram_inv @ design.T)

        def intercept_t(outcomes):
            coef = np.linalg.lstsq(design, outcomes, rcond=None)[0]
            adjusted_residuals = (outcomes - design @ coef) / (1 - diagonal)
            covariance = gram_inv @ design.T @ np.diag(adjusted_residuals**2) @ design @ gram_inv
            return coef[0], coef[0] / np.sqrt(covariance[0, 0])

        estimate, observed = intercept_t(self.y)
        self.assertAlmostEqual(result["adjusted_delta"], estimate)
        self.assertAlmostEqual(result["intercept_t"], observed)
        restricted = np.linalg.lstsq(self.x[:, None], self.y, rcond=None)[0] * self.x
        signs = np.random.default_rng(19).integers(0, 2, (199, len(self.x)), dtype=np.int8) * 2 - 1
        null_t = [intercept_t(restricted + sign * (self.y - restricted))[1] for sign in signs]
        extreme = np.count_nonzero(np.abs(null_t) >= abs(observed) - 1e-12)
        self.assertEqual(result["adjusted_p_raw"], (extreme + 1) / 200)
        indices = np.random.default_rng(42).integers(0, len(self.x), (250, len(self.x)))
        intercepts = [np.linalg.lstsq(design[ix], self.y[ix], rcond=None)[0][0]
                      for ix in indices if np.linalg.matrix_rank(design[ix]) == 2]
        np.testing.assert_allclose([result["adjusted_ci_low"], result["adjusted_ci_high"]],
                                   np.percentile(intercepts, [2.5, 97.5]), rtol=0, atol=1e-12)

    def test_constant_nonzero_fd_cannot_identify_an_equal_motion_intercept(self):
        result = self.fit(np.full(8, 0.1), self.y)
        self.assertEqual(result["status"], "not_estimable")


if __name__ == "__main__":
    unittest.main()
