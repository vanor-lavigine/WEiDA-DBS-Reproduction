"""Constructed checks for matching, boundary clipping, and paired resampling."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched24_analysis import complete_m1_duration, endpoint, paired_gap, shared_interior


class MatchedTests(unittest.TestCase):
    def test_common_frames_and_cut_episode(self):
        labels = np.array([-1,0,0,0,1,0,0,2,-1])
        other = np.array([-1,-1,1,2,1,0,2,-1,-1])
        keep = shared_interior((labels,other))
        np.testing.assert_array_equal(np.flatnonzero(keep),[2,3,4,5,6])
        self.assertAlmostEqual(complete_m1_duration(labels,keep),4.2)
        with self.assertRaisesRegex(ValueError,"continuous"):
            shared_interior((labels,np.array([-1,0,1,-1,0,2,-1,-1,-1])))

    def test_drop_patient_from_both_pipelines(self):
        d = np.arange(2*5*6,dtype=float).reshape(2,5,6)
        d[1,1,4] = np.nan
        d[0,0,0] = np.nan
        samples = endpoint(d,[4,1,2,0],4)
        np.testing.assert_array_equal(samples[0],d[0,[4,2]][:,[1,4,0]].T)
        np.testing.assert_array_equal(samples[1],d[1,[4,2]][:,[1,4,0]].T)

    def test_identical_sources_have_zero_paired_gap(self):
        rng = np.random.default_rng(8)
        src = rng.integers(1,20,size=(3,22))
        result = paired_gap(np.stack([src,src]),[30,1])
        self.assertEqual(result["n_pairs"],22)
        for key in ("gap_syn9_minus_105","paired_ci_low","paired_ci_high"):
            self.assertEqual(result[key],0.0)
