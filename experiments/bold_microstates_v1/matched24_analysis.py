"""Score shared-frame, shared-template changes and their patient-deletion influence."""

import argparse
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from core import sequence_metrics
from matched24_data import load_study, subjects
from peak_states import backfit
from run import digest, write_json
from same_band_analysis import SOURCES, spatial_correlation
from same_band_stats import BOOTSTRAPS, PERMUTATIONS, SEED, SHIFT_DRAWS, association, group_shifts, partial_rank, shift_profile
from stats import METRICS, holm_adjust
from verify_peak_analysis import holm, summarize_labels
from verify_same_band import amplitude_labels, partial_from_correlations


def shared_interior(labels):
    keep = (labels[0]>=0) & (labels[1]>=0)
    frames = np.flatnonzero(keep)
    if len(frames)<2 or np.any(np.diff(frames)!=1):
        raise ValueError("Matched visit lacks one continuous common exposure")
    return keep


def complete_m1_duration(labels,keep):
    """Mean duration of M1 episodes that fit entirely inside the common interior."""
    changes = np.flatnonzero(np.r_[True, np.diff(labels)!=0, True])
    start,end = changes[:-1],changes[1:]
    complete = (labels[start]==0) & keep[start] & keep[end-1]
    return 2.1*np.mean((end-start)[complete]) if complete.any() else np.nan


def score(study,cache,fold):
    phase = loadmat(cache/f"phase_{fold:02d}.mat",squeeze_me=True)
    amp = np.load(cache/f"amp_{fold:02d}.npz")
    people = subjects(fold)
    assert phase["fold"] == fold and phase["train_people"] == len(people) and amp["starts"] == 4*len(people)
    corr = spatial_correlation(study.paper,phase["centers"])
    _,order = linear_sum_assignment(-corr)
    centers, vcenter = phase["centers"][order],phase["var_centers"][order][1]
    np.testing.assert_allclose(phase["var_centers"],np.array([[float(f"{v:.5g}") for v in row] for row in phase["centers"]]),rtol=0,atol=0)
    full = fold == 0
    measures = np.full((2,24,2,15),np.nan)  # FD, S2 VAR, 4x3 metrics, complete M1 duration
    profiles, fds = [{},{}], [{},{}]
    windows, counts, clipped = [], [[],[]], [0,0]
    train_counts, train_ssd = np.zeros(4,dtype=int),np.zeros(4)
    for p in people:
        for visit in range(2):
            pair = [study.visits[si,p,visit] for si in range(2)]
            labels = [backfit(v.peaks,amp["centers"])[0] for v in pair]
            keep = shared_interior(labels)
            windows.append(int(keep.sum()))
            for si,(v,label) in enumerate(zip(pair,labels)):
                np.testing.assert_array_equal(label,amplitude_labels(v.wave,amp["centers"]))
                dist = cdist(v.features,centers,"sqeuclidean")
                state = dist.argmin(axis=1)
                train_counts += np.bincount(state,minlength=4)
                train_ssd += np.bincount(state,weights=dist.min(axis=1),minlength=4)
                s2 = keep & (state==1)
                n = int(s2.sum())
                counts[si].append(n)
                var = np.square(v.features[s2]-vcenter).sum()/(78*(n-1)) if n>=2 else np.nan
                common = np.where(keep,label,-1)
                rows = sequence_metrics(common,2.1,4)["states"]
                values = np.array([[r[m] for m in METRICS] for r in rows])
                np.testing.assert_allclose(values,summarize_labels(common),rtol=1e-12,equal_nan=True)
                complete = complete_m1_duration(label,keep)
                edges = np.flatnonzero(keep)[[0,-1]]
                for e,step in zip(edges,(-1,1)):
                    if 0<=e+step<len(label) and label[e]==label[e+step]==0:
                        clipped[si] += 1
                measures[si,p,visit] = [v.peaks.fd,var,*values.ravel(),complete]
                if full:
                    key = (p,("OFF","ON")[visit])
                    profiles[si][key] = shift_profile(common,state)
                    fds[si][key] = v.peaks.fd
    np.testing.assert_array_equal(train_counts,phase["counts"][order])
    np.testing.assert_allclose(train_ssd,phase["ssd"][order],rtol=1e-10)
    assert train_counts.sum() == phase["n_train"]
    matches = [dict(reference=name,state=i+1,spatial_r=float(val)) for name,mat in
               zip(("WEiDA_paper","micro_105","micro_SyN9"),[corr[:,order],*amp["matches"]])
               for i,val in enumerate(np.diag(mat))]
    qc = dict(people=len(people),train_frames_both=int(train_counts.sum()),common_frames_per_source=sum(windows),
              common_min=min(windows),common_median=float(np.median(windows)),common_max=max(windows),
              matlab_version=str(phase["matlab_version"]),amplitude_fit_starts=int(amp["starts"]),
              amplitude_objective=float(amp["objective"]),s2_min_105=min(counts[0]),s2_min_syn9=min(counts[1]),
              s2_insufficient_visits_105=sum(n<2 for n in counts[0]),s2_insufficient_visits_syn9=sum(n<2 for n in counts[1]),
              clipped_m1_edges_105=clipped[0],clipped_m1_edges_syn9=clipped[1])
    deltas = measures[:,:,1,:]-measures[:,:,0,:]
    shifts = []
    if full:
        for si,source in enumerate(SOURCES):
            out,_ = group_shifts(profiles[si],fds[si],np.random.default_rng([SEED,si,0,99]))
            shifts.append(dict(source=source,**out))
        for row,p in zip(shifts,holm_adjust([r["p_shift"] for r in shifts])):
            row["p_holm_two"] = p
    return deltas,qc,matches,shifts,{"micro":amp["centers"],"WEiDA":centers}


def endpoint(deltas,people,column):
    # (pipeline, variable, patient); one common complete-case set for both pipelines.
    a = deltas[:,people,:][:,:,[1,column,0]].transpose(0,2,1)
    return a[:,:,np.isfinite(a).all(axis=(0,1))]


def family(deltas,people,bootstrap):
    rows = [[],[]]
    for s in range(4):
        for j,metric in enumerate(METRICS):
            a = endpoint(deltas,people,2+3*s+j)
            for si,source in enumerate(SOURCES):
                res = association(*a[si],seed=[SEED,si,0,s+1,j],bootstraps=BOOTSTRAPS if bootstrap else 0)
                if res["status"]=="ok":
                    np.testing.assert_allclose(res["partial_r"],partial_from_correlations(*a[si]),atol=1e-11)
                rows[si].append(dict(source=source,state=s+1,metric=metric,family_n=12,**res))
    for group in rows:
        raw = np.array([r["p_raw"] for r in group])
        adj = holm_adjust(raw)
        np.testing.assert_allclose(adj,holm(raw),atol=1e-12,equal_nan=True)
        for r,p in zip(group,adj):
            r["p_holm"] = p
    return pd.DataFrame(rows[0]+rows[1])


def paired_gap(a,seed):
    est = [float(partial_rank(*src)) for src in a]
    n = a.shape[-1]
    draws = np.random.default_rng(seed).integers(n,size=(19999,n))
    r = [partial_rank(*(v[draws] for v in src)) for src in a]
    gaps = r[1]-r[0]
    ci = np.nanquantile(gaps,[.025,.975])
    other = [partial_from_correlations(*(v[draws] for v in src)) for src in a]
    np.testing.assert_allclose(gaps,other[1]-other[0],atol=1e-10,equal_nan=True)
    return dict(n_pairs=n,r_105=est[0],r_syn9=est[1],gap_syn9_minus_105=est[1]-est[0],
                paired_ci_low=ci[0],paired_ci_high=ci[1],draws=len(gaps),valid=int(np.isfinite(gaps).sum()))


def save_loo(table,fold_name,mode):
    table = table[["source","state","metric","family_n","n_pairs","partial_r","p_raw","p_holm"]].copy()
    table.insert(0,"mode",mode)
    table.insert(0,"fold",fold_name)
    return table


def run(study,root,cache,folds):
    for fold in folds:
        name = "all" if fold==0 else study.fold_names[fold]
        deltas,qc,matches,shifts,templates = score(study,cache,fold)
        write_json(cache/f"qc_{name}.json",dict(fold=name,**qc,matches=matches))
        table = family(deltas,subjects(fold),bootstrap=fold==0)
        if fold:
            save_loo(table,name,"refit").to_csv(cache/f"refit_{name}.csv",index=False)
        else:
            table.to_csv(cache/"associations.csv",index=False)
            focus = 2+METRICS.index("duration_sec")
            full = subjects(0)
            gaps = [dict(endpoint=label,**paired_gap(endpoint(deltas,full,column),[SEED,24,offset]))
                    for offset,(label,column) in enumerate((("M1_duration",focus),("M1_complete_episodes",14)))]
            pd.DataFrame(gaps).to_csv(cache/"focal_gap.csv",index=False)
            pd.DataFrame(shifts).to_csv(cache/"cooccurrence.csv",index=False)
            pd.DataFrame([dict(method=method,state=i+1,**dict(zip(study.axis,v))) for method,values in templates.items()
                          for i,v in enumerate(values)]).to_csv(cache/"shared_templates.tsv",index=False,sep="\t")
            aux = root/"V100脑深部刺激WEiDA复现_20261118/代码与辅助"
            parents = ["legacy105","v100_syn9"]
            hashes = {f"{p}/{f}":digest(aux/"同人群精确度对照"/p/f) for p in parents for f in ("inputs.mat","author_matched_k4.mat")}
            hashes["author_feature_code"] = digest(root/"WEiDA-DBS-author-matlab/StateSpace_modelling.m")
            hashes["paper_template_file"] = digest(aux/"论文图3A质心读数.csv")
            hashes["prior_state_templates"] = digest(Path(__file__).resolve().parents[2]/"reports/bold_microstates_v1/same_band/results/templates.tsv")
            write_json(cache/"audit.json",dict(data_kind="real",**study.audit,source_sha256=hashes,
                       tr_s=2.1,frequency_hz=[.04,.07],python=platform.python_version(),platform=platform.platform(),
                       numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                       seed=SEED,permutations=PERMUTATIONS,bootstrap_marginal=BOOTSTRAPS,bootstrap_gap=19999,
                       circular_shifts=SHIFT_DRAWS))
            fixed = [save_loo(family(deltas,subjects(k),bootstrap=False),study.fold_names[k],"fixed")
                     for k in range(1,25)]
            pd.concat(fixed).to_csv(cache/"fixed.csv",index=False)
        print(f"Completed matched changes for {name}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--folds",type=int,nargs="+",default=list(range(25)),choices=range(25))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(load_study(args.data_root),args.data_root,args.cache,args.folds)
