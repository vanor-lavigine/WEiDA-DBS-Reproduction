"""Synthetic checks for incomplete state support and score-independent map alignment."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_replacement_105 import fit_band, scan_features


def test_unobserved_class_has_zero_occurrence_and_undefined_duration():
    labels = np.repeat([0,1,2,1,0,1,2], 4)
    scan = SimpleNamespace(units=np.eye(4)[labels], gfp=np.ones(len(labels)), peaks=np.arange(len(labels)))
    values, qc, transitions = scan_features(scan,np.eye(4))
    assert values["M4","occurrence_s"] == values["M4","coverage"] == 0
    assert np.isnan(values["M4","duration_s"]) and np.isnan(values["all","duration_s"])
    assert qc["retained"] == 20 and values["all","pooled_duration_s"] == 8.4
    assert np.isnan(transitions[3]).all()


def test_identical_conditions_align_to_the_same_four_maps_without_scores():
    rng=np.random.default_rng(710)
    x,y = rng.normal(size=(2,210,12))
    names=["a_off","b_off","a_on","b_on"]
    roster=pd.DataFrame({"condition":["OFF","OFF","ON","ON"]},index=names)
    scans=dict(zip(names,[x,y,x,y]))
    _, maps, qc=fit_band(scans,dict.fromkeys(names,0.),roster,(.04,.07),None)
    np.testing.assert_allclose(maps["OFF"],maps["ON"],atol=1e-12)
    np.testing.assert_allclose(qc["ON"]["matched_r"],1,atol=1e-12)
