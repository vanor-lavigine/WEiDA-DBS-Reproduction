"""Independent numerical checks from band-limited frames and published group templates."""

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist
from scipy.special import entr, rel_entr
from scipy.stats import hypergeom, rankdata

from compare_legacy_105 import REPORTS, read_inputs
from verify_peak_analysis import holm, summarize_labels

METRICS = ("coverage","occurrence_per_min","duration_sec")
SOURCES = ("105","SyN9")
BANDS = ("primary","slow5")


def amplitude_labels(wave,templates):
    z = (wave-wave.mean(0))/wave.std(0,ddof=1)
    peaks = argrelmax(z.std(1))[0]
    on_peaks = np.abs(1-cdist(z[peaks],templates,"correlation")).argmax(1)
    frame = np.arange(peaks[0],peaks[-1]+1)
    indicators = np.stack([np.interp(frame,peaks,on_peaks==s) for s in range(4)],axis=1)
    active = (indicators >= indicators.max(1,keepdims=True)-1e-13).argmax(1)
    transitions = np.flatnonzero(np.diff(active))+1
    labels = np.full(len(z),-1)
    if len(transitions)>=2:
        use = slice(transitions[0],transitions[-1])
        labels[frame[use]] = active[use]
    return labels


def rotations(m,s):
    n = len(m)
    joints = np.array([np.bincount(4*m+np.roll(s,i),minlength=16).reshape(4,4)/n for i in range(n)])
    r,c = np.bincount(m,minlength=4),np.bincount(s,minlength=4)
    mean_cell = r[:,None]*c[None,:]/n**2
    observed = rel_entr(joints,mean_cell).sum((1,2))
    h = (entr(r/n).sum()+entr(c/n).sum())/2
    emi = 0.0
    for a in r[r>0]:
        for b in c[c>0]:
            possible = np.arange(max(1,a+b-n),min(a,b)+1)
            contribution = possible/n*np.log(possible*n/(a*b))
            emi += hypergeom.pmf(possible,n,a,b) @ contribution
    return (observed-emi)/(h-emi), joints[0], joints.mean(0)


def partial_from_correlations(x, y, fd):
    scores = []
    for v in (x,y,fd):
        z = rankdata(v,axis=-1)
        z -= z.mean(axis=-1,keepdims=True)
        norm = np.linalg.norm(z,axis=-1,keepdims=True)
        scores.append(np.divide(z,norm,out=np.zeros_like(z),where=norm>0))
    a,b,c = scores
    ab,ac,bc = (np.sum(v*w,axis=-1) for v,w in ((a,b),(a,c),(b,c)))
    return (ab-ac*bc)/np.sqrt((1-ac**2)*(1-bc**2))


def verify_associations(records,report,si,bi):
    people = sorted(p for p,c in records if c=="OFF")
    motion = np.array([records[p,"ON"][0]-records[p,"OFF"][0] for p in people])
    var = np.array([records[p,"ON"][1]-records[p,"OFF"][1] for p in people])
    dy = np.stack([records[p,"ON"][2]-records[p,"OFF"][2] for p in people])
    for row in report.itertuples():
        j = METRICS.index(row.metric)
        y0 = dy[:,row.state-1,j]
        good = np.isfinite(var)&np.isfinite(y0)&np.isfinite(motion)
        x,y,fd = var[good],y0[good],motion[good]
        n = len(x)
        assert n == row.n_pairs
        estimate = partial_from_correlations(x,y,fd)
        np.testing.assert_allclose(estimate,row.partial_r,atol=1e-11)
        cov = np.stack([np.ones(n),rankdata(fd)],axis=1)
        full = np.column_stack([cov,rankdata(x)])
        rank_y = rankdata(y)
        fixed = cov @ np.linalg.lstsq(cov,rank_y,rcond=None)[0]
        r0,r1 = np.random.SeedSequence([20261122,si,bi,row.state,j]).spawn(2)
        index = np.random.default_rng(r0).permuted(np.tile(np.arange(n),(19999,1)),axis=1)
        all_y = np.column_stack([rank_y,(fixed+(rank_y-fixed)[index]).T])
        coeff = np.linalg.pinv(full) @ all_y
        se = np.linalg.norm(all_y-full@coeff,axis=0)/np.sqrt(n-3) * np.linalg.norm(np.linalg.pinv(full)[-1])
        t = abs(coeff[-1]/se)
        tail = np.count_nonzero(t[1:] >= t[0]-1e-11)
        assert tail == row.n_extreme and abs(row.p_raw-(tail+1)/20000)<1e-12
        draw = np.random.default_rng(r1).integers(n,size=(5000,n))
        boot = partial_from_correlations(x[draw],y[draw],fd[draw])
        np.testing.assert_allclose(np.nanpercentile(boot,[2.5,97.5]),[row.ci_low,row.ci_high],atol=1e-10)
    np.testing.assert_allclose(holm(report.p_raw.to_numpy()),report.p_holm,atol=1e-12)


def verify(root,cache,report):
    maps = pd.read_csv(report/"templates.tsv",sep="\t",index_col=[0,1,2,3])
    archive = read_inputs(root,list(maps.columns))
    phase_reference = pd.read_csv(root/"V100脑深部刺激WEiDA复现_20261118/代码与辅助/论文图3A质心读数.csv",index_col=0)
    effects = pd.read_csv(report/"associations.csv")
    coupling = pd.read_csv(report/"cooccurrence.csv").set_index(["source","band"])
    grid = pd.read_csv(report/"state_pairs.csv").set_index(["source","band","micro","weida"])
    visits = pd.read_csv(report/"cooccurrence_by_visit.csv").set_index(["source","band","condition"])
    for si,src in enumerate(SOURCES):
        for bi,band in enumerate(BANDS):
            f = loadmat(cache/f"{src}_{band}.mat",squeeze_me=True)
            micro = maps.loc[(src,band,"micro"),:].to_numpy()
            weida = maps.loc[(src,band,"WEiDA"),:].to_numpy()
            # Check that the chosen whole-configuration correspondence maximizes the signed score.
            c = 1-cdist(phase_reference.to_numpy().T,weida,"correlation")
            np.testing.assert_allclose(np.trace(c),max(c[np.arange(4),p].sum() for p in permutations(range(4))),atol=1e-11)
            original_index = cdist(weida,f["centers"]).argmin(1)
            vcenter = f["var_centers"][original_index[1]]
            records,profile = {},{}
            start = 0
            for k,scan in archive[si].items():
                sl = slice(start,start+len(scan.values))
                start = sl.stop
                m = amplitude_labels(f["wave"][sl],micro)
                features = f["features"][sl]
                s = cdist(features,weida).argmin(1)
                within = features[s==1]
                n = len(within)
                v = (within.var(0,ddof=1).mean()+n/(n-1)*np.mean((within.mean(0)-vcenter)**2)) if n>1 else np.nan
                records[k] = (scan.mean_fd,v,summarize_labels(m))
                use = m>=0
                assert np.all(np.diff(np.flatnonzero(use))==1)
                profile[k] = rotations(m[use],s[use])
            row = coupling.loc[(src,band)]
            people = sorted(p for p,c in records if c=="OFF")
            fd = np.array([records[p,"ON"][0]-records[p,"OFF"][0] for p in people])
            rng = np.random.default_rng([20261122,si,bi,99])
            simulations = []
            joint = []
            for p in people:
                per_visit = []
                for visit in ("OFF","ON"):
                    a,obs,expected = profile[p,visit]
                    per_visit.append(np.r_[a[0],a.mean(),a[rng.integers(len(a),size=99999)]])
                    joint.append((obs,expected))
                simulations.append(np.mean(per_visit,axis=0))
            fitted = np.linalg.lstsq(np.column_stack([np.ones(len(fd)),fd]),simulations,rcond=None)[0][0]
            q = np.quantile(fitted[2:],[.025,.975])
            np.testing.assert_allclose([fitted[0],fitted[1],*q],
                                       row[["observed_ami","null_ami","null_q025","null_q975"]],rtol=0,atol=1e-10)
            tail = np.sum(fitted[2:] >= fitted[0]-1e-12)
            assert row.extreme==tail and abs(row.p_shift-(tail+1)/100000)<1e-12
            obs,expected = np.mean(joint,axis=0)
            tab = grid.loc[(src,band),:].sort_index()
            np.testing.assert_allclose(tab.observed_fraction,obs.ravel(),atol=1e-12)
            np.testing.assert_allclose(tab.shift_expected_fraction,expected.ravel(),atol=1e-12)
            for visit in ("ON","OFF","HC"):
                av = [a for (_,v),(a,_,_) in profile.items() if v==visit]
                visit_row = visits.loc[(src,band,visit)]
                assert len(av)==visit_row.n_scans
                np.testing.assert_allclose([np.mean([a[0] for a in av]),np.mean([a.mean() for a in av])],
                                           visit_row[["observed_ami","null_ami"]],atol=1e-11)
            verify_associations(records,effects[(effects.source==src)&(effects.band==band)],si,bi)
            print(f"PASS {src} {band}: shift distribution, 16 cells, visit summaries, 12 correlations, resampling tails and intervals.")
    np.testing.assert_allclose(holm(coupling.p_shift.to_numpy()),coupling.p_holm_four,atol=1e-12)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ("data-root","cache","results"):
        cli.add_argument("--"+flag,type=Path,required=True)
    args = cli.parse_args()
    verify(args.data_root,args.cache,args.results)
