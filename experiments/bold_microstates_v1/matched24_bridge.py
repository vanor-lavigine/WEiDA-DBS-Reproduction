"""Describe the focal effect across the historical and matched scoring definitions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from compare_legacy_105 import REPORTS,read_inputs
from matched24_data import load_study,subjects
from same_band_analysis import SOURCES
from verify_same_band import amplitude_labels,partial_from_correlations
from verify_peak_analysis import summarize_labels


def focal_value(features,phase,labels,keep=None):
    idx = cdist(features,phase,"sqeuclidean").argmin(1)==1
    if keep is not None:
        idx &= keep
        labels = np.where(keep,labels,-1)
    members = features[idx]
    n = len(members)
    c = np.array([float(f"{x:.5g}") for x in phase[1]])
    # Variance about the fixed prototype = within-group variance + mean offset.
    var = ((members.var(axis=0,ddof=1).mean() + n/(n-1)*((members.mean(0)-c)**2).mean())
           if n>=2 else np.nan)
    return var,summarize_labels(labels)[0,2]


def correlations(a,mask):
    return [float(partial_from_correlations(*row[mask].T)) for row in a]


def run(root,cache):
    study = load_study(root)
    aux = root/"V100脑深部刺激WEiDA复现_20261118/代码与辅助"
    previous = REPORTS/"same_band/results"
    prior = pd.read_csv(previous/"associations.csv")
    old = pd.read_csv(previous/"templates.tsv",sep="\t",index_col=[0,1,2,3])
    shared = pd.read_csv(cache/"shared_templates.tsv",sep="\t",index_col=[0,1])
    raw = read_inputs(root,study.axis)
    records = np.full((4,2,24,2,3),np.nan)
    b,a = signal.butter(2,[.04,.07],btype="bandpass",fs=1/2.1)
    for si,(dataset,src) in enumerate(zip(raw,SOURCES)):
        archive = (cache/"legacy_native_features.mat",aux/"MATLAB四状态分析结果.mat")[si]
        features = loadmat(archive,squeeze_me=True,variable_names=["Time_all","WEiDA_Eig"])
        lengths = [len(scan.values) for scan in dataset.values()]
        np.testing.assert_array_equal(features["Time_all"][0],np.repeat(np.arange(1,len(lengths)+1),lengths))
        phase,amp = [old.loc[(src,"primary",m)].sort_index().to_numpy() for m in ("WEiDA","micro")]
        offset,native = 0,{}
        for (person,cond),scan in dataset.items():
            window = slice(offset,offset+len(scan.values))
            offset = window.stop
            if cond=="HC":
                continue
            trend = signal.detrend(scan.values,axis=0)
            wave = signal.filtfilt(b,a,trend-trend.mean(0),axis=0,padlen=12)
            v = [*focal_value(features["WEiDA_Eig"][window],phase,amplitude_labels(wave,amp)),scan.mean_fd]
            native[person,cond] = np.asarray(v)
            if person in study.person_ids:
                p = study.person_ids.index(person)
                records[0,si,p,("OFF","ON").index(cond)] = v
        # Check the historical definition against the exact prior real-data estimate before restricting.
        d = np.stack([native[p,"ON"]-native[p,"OFF"] for p,c in native if c=="OFF"])
        valid = np.isfinite(d).all(axis=1)
        ref = prior[(prior.source==src)&(prior.band=="primary")&(prior.state==1)&(prior.metric=="duration_sec")].iloc[0]
        assert valid.sum()==ref.n_pairs
        np.testing.assert_allclose(partial_from_correlations(*d[valid].T),ref.partial_r,atol=1e-11)
        for p in subjects(0):
            for v in range(2):
                visit = study.visits[si,p,v]
                fd = visit.peaks.fd
                records[1,si,p,v] = [*focal_value(visit.features,phase,amplitude_labels(visit.wave,amp)),fd]
                labels = [amplitude_labels(study.visits[j,p,v].wave,shared.loc["micro"].to_numpy()) for j in range(2)]
                keep = (labels[0]>=0)&(labels[1]>=0)
                for stage,mask in ((2,None),(3,keep)):
                    records[stage,si,p,v] = [*focal_value(visit.features,shared.loc["WEiDA"].to_numpy(),labels[si],mask),fd]
    d = records[:,:,:,1,:]-records[:,:,:,0,:]
    masks = np.isfinite(d).all(axis=(1,3))
    every = masks.all(axis=0)
    rows = []
    for i,stage in enumerate(("previous_native","previous_templates_aligned_frames","shared_templates_native_windows","primary_common_window")):
        rho = correlations(d[i],masks[i])
        same = correlations(d[i],every)
        rows.append(dict(stage=stage,n_both=int(masks[i].sum()),r_105=rho[0],r_syn9=rho[1],gap=rho[1]-rho[0],
                         n_all_stages=int(every.sum()),same_cases_r_105=same[0],same_cases_r_syn9=same[1],
                         same_cases_gap=same[1]-same[0]))
    baseline = pd.read_csv(cache/"focal_gap.csv").iloc[0]
    np.testing.assert_allclose([*correlations(d[-1],masks[-1]),masks[-1].sum()],
                               [baseline.r_105,baseline.r_syn9,baseline.n_pairs],atol=1e-10)
    pd.DataFrame(rows).to_csv(cache/"descriptive_bridge.csv",index=False)
    loo = []
    for k in range(1,25):
        keep = masks[0] & (np.arange(24)!=k-1)
        rho = correlations(d[0],keep)
        loo.append(dict(fold=study.fold_names[k],n_pairs=int(keep.sum()),r_105=rho[0],r_syn9=rho[1],gap=rho[1]-rho[0]))
    pd.DataFrame(loo).sort_values("fold").to_csv(cache/"previous_definition_loo.csv",index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root,args.cache)
