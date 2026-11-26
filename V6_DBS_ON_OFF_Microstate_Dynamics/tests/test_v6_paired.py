"""Regression coverage for the new paired V6 entry points."""

from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / "scripts"))
from paired import adjust, association, classify, covariate_test, draws, influence, omnibus, scalar, signs
from prepare import summarize_labels


def test_archived_numbering_and_gaps():
    labels = np.array([-1,0,0,1,0,-1,2,2,3,3,-1])
    corr = np.full((len(labels),4),.5)
    gfp = np.ones(len(labels))
    x = summarize_labels(labels,corr,gfp,np.array([3,0,1,2]))
    assert x["valid_frames"]==8 and x["interior_unlabeled"]==1
    assert x["count_4_1"]==1 and x["count_4_2"]==0 and x["count_4_4"]==1
    np.testing.assert_allclose(x["duration_4"],3.15)
    assert x["dwell_window_p75_4"]>x["duration_4"]
    assert np.isnan(x["dwell_edge_mean_4"])
    np.testing.assert_allclose(x["gev_4"],.25*3/8)


def test_monte_carlo_flip_and_bootstrap_schedule():
    flip = signs(25)
    assert flip.shape==(10000,25) and np.unique(flip,axis=0).shape==flip.shape
    assert not np.any(np.all(flip==1,axis=1))
    assert draws(25).shape==(2000,25)
    assert draws(25).min()==0 and draws(25).max()==24


def test_scalar_against_exact_flip_and_paired_t():
    x = np.linspace(.1,.7,12)
    y = 4.+np.sin(np.arange(len(x)))
    fd = np.cos(np.arange(len(x)))
    r,d = scalar(y+x,y,fd,fd[::-1],np.ones(len(x),dtype=bool))
    perm = np.array(list(product([-1,1],repeat=len(x))))
    expected = np.mean(np.abs(perm@x/len(x)) >= abs(x.mean())-1e-12)
    np.testing.assert_allclose(r["permutation_p"],expected)
    np.testing.assert_allclose(r["paired_t_p"],ttest_rel(y+x,y).pvalue)
    assert d.shape==(2000,5) and r["bootstrap_failed"]==0


def test_freedman_lane_intercept_targets_zero_delta_fd():
    cov = np.linspace(-.1,.6,25)
    noise = np.sin(np.arange(25))*.1
    change = 3.7+2.9*cov+noise
    expected = np.linalg.lstsq(np.c_[np.ones(len(cov)),cov],change,rcond=None)[0][0]
    r,b = covariate_test(change,cov)
    np.testing.assert_allclose(r["effect"],expected)
    assert len(b)==2000 and r["draws"]==10000 and r["p"]<.01


def test_omnibus_keeps_direction_and_untested_coordinates():
    rng = np.random.default_rng(501)
    off = rng.normal(size=(25,12))
    delta = rng.normal(0,.04,off.shape)+.02
    fd = rng.normal(size=25)
    r,boot,components = omnibus(off+delta,off,fd,fd[::-1],np.ones(len(fd),dtype=bool))
    np.testing.assert_allclose(r["mean_difference_norm"],np.linalg.norm(delta.mean(axis=0)))
    assert r["dimensions"]==12 and components.shape==(2000,12) and boot.shape==(2000,3)
    assert 0<=r["fd_bootstrap_direction_consistency"]<=1


def test_gates_and_family_scope():
    t = pd.DataFrame([dict(endpoint="gev_1",family="Primary",raw_p=.02,permutation_p=.02,fd_p=.02),
                      dict(endpoint="duration_4",family="Primary",raw_p=.20,permutation_p=.20,fd_p=.20),
                      dict(endpoint="transition_4_1",family="Transition localization",raw_p=.001,permutation_p=.001,fd_p=.001)])
    m = pd.DataFrame([dict(endpoint="transition_omnibus",family="Exploratory omnibus",raw_p=.004,permutation_p=.20,fd_p=.005)])
    out = classify(adjust(t),adjust(m)).set_index("endpoint")
    assert out.loc["gev_1","permutation_holm_p"]==.04 and out.loc["gev_1","permutation_evidence_class"]=="A"
    assert out.loc["transition_4_1","permutation_evidence_class"]=="D"
    assert out.loc["transition_4_1","fd_evidence_class"]=="B"


def test_influence_retains_every_pair():
    d = np.arange(1.,9.)
    result,loo = influence(d)
    expected = (d.sum()-d)/(len(d)-1)
    np.testing.assert_allclose(loo[:,0],expected)
    assert result["leave_one_out_direction_consistency"]==1


def test_spearman_resampling_and_outlier_check():
    x = np.arange(1.,7.)
    y = x[::-1]
    result = association(x,y,np.array([1.,0.,1.,2.,.5,.3]))
    np.testing.assert_allclose([result["rho"],result["loo_rho_low"],result["loo_rho_high"]],-1.)
    np.testing.assert_allclose(result["permutation_p"],2/720.)
