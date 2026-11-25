"""Cross-fold correspondence of E04 templates, determined without group outcomes."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def correspondence(maps, minimum=.60, margin=.05):
    banks = {}
    for f, group in maps[maps.fold>=0].groupby("fold"):
        group = group.sort_values("state")
        if group.state.tolist() != [1,2,3,4]:
            raise ValueError("Expected one map per E04 axis per held-out fold")
        a=group[[f"ROI_{i+1}" for i in range(78)]].to_numpy(float)
        a -= a.mean(axis=1,keepdims=True)
        a /= np.linalg.norm(a,axis=1,keepdims=True)
        banks[int(f)] = a
    if set(banks) != set(range(5)):
        raise ValueError("Expected five E04 training map sets")
    edges, rows = {}, []
    ids = np.arange(4)
    for i,j in itertools.combinations(sorted(banks),2):
        corr=np.abs(banks[i]@banks[j].T)
        _,perm=linear_sum_assignment(-corr)
        edges[i,j]=perm
        edges[j,i]=np.argsort(perm)
        matched=corr[ids,perm]
        other=corr.copy()
        other[ids,perm]=-np.inf
        gap=matched-np.maximum(other.max(axis=1),other.max(axis=0)[perm])
        for s in ids:
            rows.append(dict(fold_i=i,fold_j=j,state_i=s+1,state_j=perm[s]+1,
                             abs_correlation=matched[s],bidirectional_margin=gap[s],
                             qualified=bool(matched[s]>=minimum and gap[s]>=margin)))
    cycle_ok=all(np.array_equal(edges[j,k][edges[i,j]],edges[i,k])
                 for i,j,k in itertools.permutations(banks,3))
    passed=all(r["qualified"] for r in rows) and cycle_ok
    align={0:ids, **{f: edges[0,f] for f in range(1,5)}}
    return pd.DataFrame(rows).assign(cycles_consistent=cycle_ok,stable_composition=passed), align, passed


def audit_table(template_file, experiment, output):
    maps=pd.read_csv(template_file)
    subset=maps[maps.experiment==experiment]
    rows, align, passed=correspondence(subset)
    if passed:
        pool = subset[subset.fold.isin([-1,0])].copy()
        cols=[f"ROI_{i+1}" for i in range(78)]
        mat=[]
        for f in [0,-1]:
            x=pool[pool.fold==f].sort_values("state")[cols].to_numpy()
            x-=x.mean(1,keepdims=True)
            mat.append(x/np.linalg.norm(x,axis=1,keepdims=True))
        corr=np.abs(mat[0]@mat[1].T)
        idx,perm=linear_sum_assignment(-corr)
        sim=corr[idx,perm].copy()
        corr[idx,perm]=-np.inf
        gap=sim-np.maximum(corr.max(1),corr.max(0)[perm])
        if min(sim)>=.60 and min(gap)>=.05:
            align[-1]=perm
    output.mkdir(parents=True,exist_ok=True)
    rows.assign(experiment=experiment).to_csv(output/"Composition_Correspondence.csv",index=False)
    print(experiment,"all-fold composition gate",passed,"min matched |r|",rows.abs_correlation.min(),
          "min bidirectional margin",rows.bidirectional_margin.min(),"cycles",rows.cycles_consistent.all())
    return align if passed else {}


def reindex_coverage(d, align):
    out=d.copy()
    cols=[f"coverage_{j+1}" for j in range(4)]
    for f, order in align.items():
        use=d.fold==f
        out.loc[use,cols]=d.loc[use,[cols[j] for j in order]].to_numpy()
    return out


def state_family(data, experiment, alignment):
    """Reindex all scan features into arbitrary R1–R4 axes from fold zero."""
    frames=[]
    for name in [experiment,experiment+"_edge_censored"]:
        original = data[(data.experiment==name) & data.fold.isin(alignment)]
        if original.empty:
            continue
        reindexed = original.copy()
        reindexed["experiment"] = experiment+"_R"+name[len(experiment):]
        for f, order in alignment.items():
            sel=original.fold==f
            old=original.loc[sel]
            for i,j in enumerate(order):
                for m in ["coverage","duration","occurrence","count","gev","duration_sd","recurrence"]:
                    reindexed.loc[sel,f"{m}_{i+1}"] = old[f"{m}_{j+1}"].to_numpy()
                reindexed.loc[sel,f"stable_{i+1}"] = True
                for u,v in enumerate(order):
                    reindexed.loc[sel,f"transition_{i+1}_{u+1}"] = old[f"transition_{j+1}_{v+1}"].to_numpy()
        frames.append(reindexed)
    return pd.concat(frames,ignore_index=True) if frames else data.iloc[:0].copy()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("templates",type=Path)
    parser.add_argument("experiment")
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    audit_table(a.templates,a.experiment,a.output)
