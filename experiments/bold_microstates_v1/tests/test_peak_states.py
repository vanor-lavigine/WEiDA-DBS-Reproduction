"""Numerical references and synthetic regression checks for the EEG-style BOLD adaptation."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peak_states import aahc, backfit, group_maps, interpolate_peaks, match_to_signed, prepare_scan


def slow_aahc(maps, k):
    groups = [[i] for i in range(len(maps))]
    while True:
        axes, utility = [], []
        for g in groups:
            members = maps[g]
            axis = np.linalg.eigh(members.T @ members)[1][:, -1]
            axes.append(axis)
            utility.append(np.sum(np.abs(members @ axis)))
        if len(groups) == k:
            return np.asarray(axes)
        weakest = np.argmin(utility)
        orphaned = groups.pop(weakest)
        axes.pop(weakest)
        for member in orphaned:
            groups[np.argmax(np.abs(axes @ maps[member]))].append(member)


class PeakTests(unittest.TestCase):
    def test_aahc_matches_recomputed_scatter_and_ignores_each_map_sign(self):
        rng = np.random.default_rng(147)
        y = rng.normal(size=(19, 10))
        y -= y.mean(axis=1, keepdims=True)
        a = aahc(y)
        for alternative in [slow_aahc(y, 4), aahc(y * rng.choice([-1, 1], size=(len(y), 1)))]:
            similarity = np.abs(a @ alternative.T)
            i, j = linear_sum_assignment(-similarity)
            np.testing.assert_allclose(similarity[i, j], 1, atol=1e-12)

    def test_gfp_is_spatial_sd_before_unit_normalization(self):
        rng = np.random.default_rng(17)
        x = rng.normal(size=(70, 12))
        scan = prepare_scan(x, .15)
        z = (x-x.mean(axis=0))/x.std(axis=0, ddof=1)
        g = z.std(axis=1)
        p = [j for j in range(1,len(g)-1) if g[j-1] < g[j] > g[j+1]]
        np.testing.assert_allclose(scan.gfp, g)
        np.testing.assert_array_equal(scan.peaks, p)
        a, b, _ = backfit(scan, scan.individual)
        c, d, _ = backfit(scan, scan.individual * np.array([1,-1,-1,1])[:,None])
        np.testing.assert_array_equal(a,c)
        np.testing.assert_array_equal(b,d)

    def test_midpoint_ties_edges_and_gaps_against_linear_indicator_interpolation(self):
        n = 29
        valid = np.ones(n, dtype=bool)
        valid[15] = False
        peaks = np.array([1,4,7,11,13,17,20,23,25,27])
        labels = np.array([3,2,0,1,0,2,1,3,0,2])
        actual, ties = interpolate_peaks(n, peaks, labels, valid)
        expected = np.full(n,-1)
        for p,v in [(peaks[:5],labels[:5]),(peaks[5:],labels[5:])]:
            t = np.arange(p[0],p[-1]+1)
            s = np.vstack([np.interp(t,p,(v==j).astype(float)) for j in range(4)]).argmax(0)
            changes = np.where(np.diff(s)!=0)[0]+1
            take = slice(changes[0],changes[-1])
            expected[t[take]] = s[take]
        np.testing.assert_array_equal(actual,expected)
        self.assertGreater(ties,0)
        self.assertEqual(actual[15],-1)
        blank, _ = interpolate_peaks(5,np.array([1,3]),np.array([2,2]),np.ones(5,dtype=bool))
        self.assertTrue((blank==-1).all())

    def test_group_fit_invariant_to_member_permutations_and_repeated_weight(self):
        rng = np.random.default_rng(444)
        target = rng.normal(size=(4,16))
        target -= target.mean(1,keepdims=True)
        target /= np.linalg.norm(target,axis=1,keepdims=True)
        bank = np.stack([target[rng.permutation(4)] * rng.choice([-1,1],size=(4,1)) for _ in range(5)])
        axes, info = group_maps(bank,[1,.5,.5,1,1])
        _, corr, _ = match_to_signed(target,axes)
        np.testing.assert_allclose(corr.diagonal(),1,atol=1e-12)
        self.assertAlmostEqual(info["best_mean_squared_r"],1)


if __name__ == "__main__":
    unittest.main()
