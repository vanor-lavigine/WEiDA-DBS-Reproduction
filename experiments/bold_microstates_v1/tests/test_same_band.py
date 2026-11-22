"""Hand checks for zero margins, rotations, and confound adjustment."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from same_band_stats import association, group_shifts, partial_rank, shift_profile


class SameBandTests(unittest.TestCase):
    def test_ami_preserves_margins_and_boundary_frames(self):
        m = np.array([-1,0,0,2,2,1,1,0,0,0,-1])
        s = np.array([3,1,1,3,3,0,0,1,1,1,3])
        profile = shift_profile(m,s)
        m, s = m[1:-1],s[1:-1]
        expected = [adjusted_mutual_info_score(m,np.roll(s,i)) for i in range(len(m))]
        np.testing.assert_allclose(profile["ami"],expected,atol=1e-12)
        self.assertAlmostEqual(profile["ami"][0],1.0)
        np.testing.assert_allclose(profile["null_cells"],np.outer(np.bincount(m,minlength=4),
                                   np.bincount(s,minlength=4))/len(m)**2)

    def test_constant_and_incomplete_state_series(self):
        self.assertIsNone(shift_profile(np.array([-1,1,1]),np.array([1,2,2])))
        p = shift_profile(np.array([0,0,0,0]),np.array([1,1,2,2]))
        np.testing.assert_array_equal(p["ami"],np.zeros(4))
        with self.assertRaisesRegex(ValueError,"Discontinuous"):
            shift_profile(np.array([1,1,-1,0,0]),np.zeros(5,dtype=int))

    def test_patient_weights_and_zero_motion_estimand(self):
        profiles, fds = {}, {}
        for i in range(5):
            for c in ("ON","OFF"):
                profiles[i,c] = {"ami":np.array([.2+.1*i]*3), "observed_cells":np.eye(4)/4,
                                 "null_cells":np.eye(4)/4}
                fds[i,c] = float(i) if c=="ON" else 0
        got, _ = group_shifts(profiles,fds,np.random.default_rng(3),draws=19)
        self.assertAlmostEqual(got["observed_ami"],.2)
        self.assertAlmostEqual(got["excess_ami"],0)
        self.assertEqual(got["p_shift"],1)

    def test_partial_rank_ties_and_missing_visit(self):
        x = np.array([6,3,4,2,1,5,5,np.nan])
        y = np.array([9,1,8,4,6,3,1,7])
        z = np.array([2,3,3,1,6,2,1,3])
        a,b,c = [rankdata(v[:-1]) for v in (x,y,z)]
        design = np.column_stack([np.ones(len(c)),c])
        residual = np.eye(len(c))-design @ np.linalg.pinv(design)
        expected = np.corrcoef(residual@a,residual@b)[0,1]
        reported = association(x,y,z,[7],permutations=199,bootstraps=150)
        self.assertEqual(reported["n_pairs"],7)
        self.assertAlmostEqual(expected,reported["partial_r"])
        self.assertTrue(np.isnan(partial_rank(z,y,z)))
