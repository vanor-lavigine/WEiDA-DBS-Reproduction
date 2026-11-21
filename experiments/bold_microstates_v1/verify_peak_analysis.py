"""Independent label, clustering-objective, and inferential checks on both real datasets."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist

from compare_legacy_105 import REPORTS, read_inputs


def axis(members):
    if len(members) == 1:
        v = members[0]
    else:
        v = members.T @ np.linalg.eigh(members @ members.T)[1][:,-1]
    return v / np.linalg.norm(v)


def reference_aahc(maps):
    # Recompute all cluster contributions from their memberships at each level.
    clusters = [[i] for i in range(len(maps))]
    while True:
        centers = [axis(maps[g]) for g in clusters]
        if len(centers) == 4:
            return np.stack(centers)
        loss = [sum(abs(maps[g] @ v)) for g,v in zip(clusters, centers)]
        weakest = int(np.argmin(loss))
        removed = clusters.pop(weakest)
        centers.pop(weakest)
        target = np.abs(maps[removed] @ np.asarray(centers).T).argmax(1)
        for j, i in zip(target, removed):
            clusters[j].append(i)


def summarize_labels(labels):
    change = np.flatnonzero(np.r_[True, labels[1:]!=labels[:-1], True])
    runs = labels[change[:-1]]
    valid = runs >= 0
    frames = np.bincount(runs[valid], weights=np.diff(change)[valid], minlength=4)
    episodes = np.bincount(runs[valid], minlength=4)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.column_stack([frames/frames.sum(), episodes*60/(2.1*frames.sum()), 2.1*frames/episodes])


def evaluate(values, signed, centers):
    z = (values-values.mean(0))/values.std(0, ddof=1)
    p = argrelmax(z.std(axis=1))[0]
    spatial = np.abs(1-cdist(z, centers, "correlation"))
    full = spatial.argmax(1)
    times = np.arange(p[0], p[-1]+1)
    indicators = np.array([np.interp(times,p,full[p]==k) for k in range(4)]).T
    candidates = np.isclose(indicators, indicators.max(1,keepdims=True), rtol=0, atol=1e-13)
    nearest = candidates.argmax(1)
    midpoint = np.where(candidates,spatial[times],-np.inf).argmax(1)
    result = {"old_signed": summarize_labels((1-cdist(z,signed,"cosine")).argmax(1)),
              "peak_direct": summarize_labels(full)}
    for name, restricted in [("peak_interp",nearest),("midpoint_spatial",midpoint)]:
        # Keep only interior episodes; the two boundary runs are incomplete.
        changes = np.flatnonzero(np.diff(restricted)!=0)+1
        labels = np.full(len(z),-1)
        if len(changes) >= 2:
            labels[times[changes[0]:changes[-1]]] = restricted[changes[0]:changes[-1]]
        result[name] = summarize_labels(labels)
    result["paired_method_change"] = result["peak_interp"]-result["old_signed"]
    peak_maps = z[p]-z[p].mean(1,keepdims=True)
    return result, reference_aahc(peak_maps)


def verify_group(sets, centers, expected):
    people = Counter(key for key,maps in sets)
    w = np.array([1/people[key] for key,maps in sets])
    aligned = []
    for _,maps in sets:
        i,j = linear_sum_assignment(-(centers @ maps.T)**2)
        aligned.append(maps[j[np.argsort(i)]])
    aligned = np.asarray(aligned)
    quality = np.average(np.sum(aligned*centers,axis=2)**2, weights=w, axis=0).mean()
    np.testing.assert_allclose(quality,expected,rtol=0,atol=1e-10)
    for k in range(4):
        principal = axis(aligned[:,k]*np.sqrt(w)[:,None])
        np.testing.assert_allclose(abs(principal@centers[k]),1,rtol=0,atol=1e-10)


def holm(p):
    adjusted = np.empty(len(p))
    floor = 0
    for remaining,i in zip(range(len(p),0,-1),np.argsort(p)):
        floor = max(floor,min(1,remaining*p[i]))
        adjusted[i] = floor
    return adjusted


def verify_family(records, table):
    people = sorted({person for person,c in records if c=="OFF"})
    x_all = np.array([records[s,"ON"][0]-records[s,"OFF"][0] for s in people])
    delta = np.array([records[s,"ON"][1]-records[s,"OFF"][1] for s in people])
    raw, adj = [], []
    for row in table.itertuples():
        j = ("coverage","occurrence_per_min","duration_sec").index(row.metric)
        y0 = delta[:,row.state-1,j]
        keep = np.isfinite(y0)
        x,y = x_all[keep],y0[keep]
        n = len(x)
        assert n == row.n_pairs and row.status == "ok"
        xc = x-x.mean()
        v = xc/(xc@xc)
        w = 1/n-x.mean()*v
        leverage = 1/n+xc*v
        def tstat(a):
            b0,b1 = a@w,a@v
            residual = a-b0[...,None]-b1[...,None]*x
            return b0/np.linalg.norm(residual*(w/(1-leverage)),axis=-1)
        def rng(purpose):
            return np.random.default_rng(np.random.SeedSequence([20261121,1,row.state,j,purpose]))
        signs = rng(0).integers(0,2,(10000,n),dtype=np.int8)*2-1
        yy = y/np.max(np.abs(y)) if np.any(y) else y
        tol = 16*np.finfo(float).eps
        count_raw = np.sum(abs((signs*yy).mean(1)) >= abs(yy.mean())-tol)
        fitted = x*((x@y)/(x@x))
        observed = tstat(y)
        count = np.sum(abs(tstat(fitted+signs*(y-fitted))) >= abs(observed)-2*tol*max(1,abs(observed)))
        raw.append((count_raw+1)/10001)
        adj.append((count+1)/10001)
        assert count == row.n_extreme
        sample = rng(1).integers(n,size=(2000,n))
        bx,by = x[sample],y[sample]
        bx0 = bx-bx.mean(1)[:,None]
        slope = (bx0*(by-by.mean(1)[:,None])).sum(1)/(bx0*bx0).sum(1)
        ci = np.percentile(by.mean(1)-slope*bx.mean(1),[2.5,97.5])
        got = [y.mean(),y@w,observed,raw[-1],adj[-1],*ci,*np.percentile(by.mean(1),[2.5,97.5])]
        want = [row.original_delta,row.adjusted_delta,row.intercept_t,row.original_p_raw,row.adjusted_p_raw,
                row.adjusted_ci_low,row.adjusted_ci_high,row.original_ci_low,row.original_ci_high]
        np.testing.assert_allclose(got,want,rtol=0,atol=1e-10)
    np.testing.assert_allclose(holm(raw),table.original_p_holm,atol=1e-12)
    np.testing.assert_allclose(holm(adj),table.adjusted_p_holm,atol=1e-12)


def main(root, report):
    newmaps = pd.read_csv(report/"results/templates.tsv",sep="\t",index_col=[0,1])
    raw = dict(zip(("105","SyN9"),read_inputs(root,list(newmaps.columns))))
    shared = sorted(raw["105"].keys() & raw["SyN9"].keys())
    summary = json.loads((report/"results/summary.json").read_text(encoding="utf-8"))
    old_paths = {"105": REPORTS/"legacy_105/results/templates_105.tsv", "SyN9": REPORTS/"templates.tsv"}
    stats = pd.read_csv(report/"results/paired_stats.csv")
    mid = pd.read_csv(report/"midpoint/paired_stats.csv")
    tables = pd.concat([stats,mid.assign(variant="midpoint_spatial")],ignore_index=True)
    pooled_sets = []
    for src, scans in raw.items():
        signed = pd.read_csv(old_paths[src],sep="\t",index_col=0).to_numpy()
        sets, records = [], {}
        group = newmaps.loc[src].to_numpy()
        for key,s in scans.items():
            result, maps = evaluate(s.values,signed,group)
            sets.append((key[0],maps))
            for variant,states in result.items():
                records.setdefault(variant,{})[key] = (s.mean_fd,states)
            if key in shared:
                offset = raw["105"][key].first_frame-s.first_frame
                common, maps = evaluate(s.values[offset:],signed,newmaps.loc["shared"].to_numpy())
                records.setdefault("common24_peak",{})[key] = (s.mean_fd,common["peak_interp"])
                pooled_sets.append((key[0],maps))
        verify_group(sets,group,summary["fit"][src]["best_mean_squared_r"])
        for variant,data in records.items():
            table = tables[(tables.source==src)&(tables.variant==variant)]
            assert len(table)==12
            verify_family(data,table)
    verify_group(pooled_sets,newmaps.loc["shared"].to_numpy(),summary["fit"]["shared"]["best_mean_squared_r"])
    print(f"PASS: 3 group fits, {len(tables)} endpoint rows, raw and adjusted Holm, effects, intervals and sign flips.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",required=True,type=Path)
    parser.add_argument("--report-dir",required=True,type=Path)
    args = parser.parse_args()
    main(args.data_root,args.report_dir)
