"""Hand-check WEiDA row orientation, enrichment tails and paired rank ties."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched24_cells import cell_tests,winner_weights
from same_band_stats import joint_counts_by_shift,zero_motion_weights


class MatchedCellTests(unittest.TestCase):
    def test_orientation_enrichment_and_margins(self):
        m = np.array([0,0,0,0,0,1,1,2,2,2,3,1])
        s = np.array([1,3,0,2])[m]
        shifts = joint_counts_by_shift(m,s).transpose(0,2,1)
        fd = np.array([-.1,.1,.2])
        rows,patient_s2 = cell_tests([shifts]*6,fd,seed=9,draws=799)
        lookup = {(r["weida"],r["micro"]):r for r in rows}
        enriched = lookup[2,1]
        self.assertEqual(enriched["raw_frames"],30)
        self.assertAlmostEqual(enriched["observed_pct"],100*5/12)
        self.assertAlmostEqual(enriched["shift_expected_pct"],100*(5/12)**2)
        self.assertTrue(enriched["enriched_holm"])
        self.assertEqual(lookup[1,2]["raw_frames"],0)
        self.assertEqual(lookup[1,2]["p_upper"],1.)
        np.testing.assert_allclose(patient_s2[:,0],enriched["excess_pp"])
        self.assertEqual(sum(r["raw_frames"] for r in rows),72)

    def test_ties_and_motion_intercept(self):
        np.testing.assert_array_equal(winner_weights(np.array([[1,1,0,-1],[1,2,0,3]])),
                                      [[.5,.5,0,0],[0,0,0,1]])
        x = np.array([[-.1,.2,.3],[.2,-.1,.2]])
        w = zero_motion_weights(x)
        np.testing.assert_allclose(w.sum(1),1)
        np.testing.assert_allclose((w*x).sum(1),0,atol=1e-14)
        with self.assertRaises(ValueError):
            zero_motion_weights(np.full(24,.1))
