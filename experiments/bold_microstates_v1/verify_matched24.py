"""Independently reconstruct matched windows, state measures, and full/deletion inference."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

from matched24_data import load_study,subjects
from run import write_json
from same_band_analysis import SOURCES,spatial_correlation
from stats import METRICS
from verify_peak_analysis import holm,summarize_labels
from verify_same_band import amplitude_labels,partial_from_correlations


def rebuild(study,cache,fold):
    native = loadmat(cache/f"phase_{fold:02d}.mat")["centers"]
    corr = np.corrcoef(study.paper,native)[:4,4:]
    order = max(permutations(range(4)),key=lambda q:sum(corr[i,j] for i,j in enumerate(q)))
    phase = native[list(order)]
    center = np.array([float(f"{v:.5g}") for v in phase[1]])
    amp = np.load(cache/f"amp_{fold:02d}.npz")["centers"]
    active = subjects(fold)
    visits = np.full((2,24,2,14),np.nan)
    for p in active:
        for v in (0,1):
            pair = [study.visits[si,p,v] for si in range(2)]
            micro = [amplitude_labels(item.wave,amp) for item in pair]
            frame = np.flatnonzero(np.min(micro,axis=0)>=0)
            np.testing.assert_array_equal(frame,np.arange(frame[0],frame[-1]+1))
            for si,item in enumerate(pair):
                features = item.features[frame]
                dist = np.stack([np.sum((features-c)**2,axis=1) for c in phase])
                s2 = features[np.argmin(dist,axis=0)==1]
                n = len(s2)
                var = (s2.var(0,ddof=1).mean()+n/(n-1)*np.mean((s2.mean(0)-center)**2)) if n>=2 else np.nan
                visits[si,p,v] = [item.peaks.fd,var,*summarize_labels(micro[si][frame]).ravel()]
    return visits,phase,amp


def independent_family(deltas,people,table,full):
    for si,source in enumerate(SOURCES):
        group = table[table.source==source]
        assert len(group)==12
        np.testing.assert_allclose(holm(group.p_raw.to_numpy()),group.p_holm,atol=1e-12)
        for row in group.itertuples():
            j = METRICS.index(row.metric)
            column = 2+3*(row.state-1)+j
            block = deltas[:,people,:][:,:,[1,column,0]]
            complete = np.isfinite(block).all(axis=(0,2))
            x,y,fd = block[si,complete].T
            n = len(x)
            assert n==row.n_pairs
            np.testing.assert_allclose(partial_from_correlations(x,y,fd),row.partial_r,atol=1e-11)
            # OLS t on reconstructed outcomes, independent of the residual-correlation implementation.
            cov = np.column_stack([np.ones(n),rankdata(fd)])
            design = np.column_stack([cov,rankdata(x)])
            rank_y = rankdata(y)
            fitted = cov @ np.linalg.lstsq(cov,rank_y,rcond=None)[0]
            perm,boot = np.random.SeedSequence([20261122,si,0,row.state,j]).spawn(2)
            ix = np.random.default_rng(perm).permuted(np.tile(np.arange(n),(19999,1)),axis=1)
            yy = np.column_stack([rank_y,(fitted+(rank_y-fitted)[ix]).T])
            inv = np.linalg.pinv(design)
            coef = inv @ yy
            se = np.linalg.norm(yy-design@coef,axis=0)/np.sqrt(n-3)*np.linalg.norm(inv[-1])
            t = abs(coef[-1]/se)
            tail = np.count_nonzero(t[1:]>=t[0]-1e-11)
            np.testing.assert_allclose((tail+1)/20000,row.p_raw,atol=1e-12)
            if full:
                draws = np.random.default_rng(boot).integers(n,size=(5000,n))
                rs = partial_from_correlations(x[draws],y[draws],fd[draws])
                np.testing.assert_allclose(np.nanquantile(rs,[.025,.975]),[row.ci_low,row.ci_high],atol=1e-10)


def verify(study,cache):
    visits,phase,amp = rebuild(study,cache,0)
    deltas = visits[:,:,1]-visits[:,:,0]
    independent_family(deltas,subjects(0),pd.read_csv(cache/"associations.csv"),True)
    fixed = pd.read_csv(cache/"fixed.csv")
    stability = []
    for fold in range(1,25):
        name = study.fold_names[fold]
        people = subjects(fold)
        independent_family(deltas,people,fixed[fixed.fold==name],False)
        vv,ff,aa = rebuild(study,cache,fold)
        independent_family(vv[:,:,1]-vv[:,:,0],people,pd.read_csv(cache/f"refit_{name}.csv"),False)
        for label,c in (("WEiDA",spatial_correlation(phase,ff)),("micro",abs(spatial_correlation(amp,aa)))):
            for i,r in enumerate(c.diagonal(),1):
                stability.append(dict(fold=name,method=label,state=i,r_to_full=r))
    pd.DataFrame(stability).sort_values(["fold","method","state"]).to_csv(cache/"template_stability.csv",index=False)
    missing = []
    for si,src in enumerate(SOURCES):
        for state in range(4):
            idx = 2+3*state+2
            absent = ~np.isfinite(visits[si,:,:,idx])
            missing.append(dict(source=src,state=state+1,off_absent=int(absent[:,0].sum()),
                                on_absent=int(absent[:,1].sum()),patients_without_paired_duration=int(absent.any(axis=1).sum())))
    write_json(cache/"verification.json",dict(data_kind="real",full_endpoints_checked=24,
               deletion_endpoints_checked=24*2*2*12,full_template_and_refits=25,
               matched_participant_visits_per_source=48,independent_test="rank OLS t, identical Freedman–Lane draws",
               full_duration_missing=missing))
    print("Verified 24 primary estimates, intervals and Holm p-values and 1,152 deletion estimates and raw/Holm p-values.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        verify(load_study(args.data_root),args.cache)
