"""Run the frozen feature matrix. No group inference is imported here."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, silhouette_score
from threadpoolctl import threadpool_limits

from features_v3 import TR, backfill, measure, segments
from models_v3 import Fitted, GROUPS, SEED, align, individual, prepare, train

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "private"
RESULTS = ROOT / "results"


def folds(scans, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = {}
    for g in ("ON", "HC"):
        people = sorted({r["subject"] for r in scans if r["condition"] == g})
        for i, person in enumerate(rng.permutation(people)):
            ids[person] = i%5
    return np.array([ids[s["subject"]] for s in scans])


def historical_anchor():
    p = ROOT.parent / "reports/bold_microstates_v1/eeg_replacement_105/results.json"
    return np.array(json.loads(p.read_text(encoding="utf-8"))["bands"]["weida_band"]["maps"]["ON"])


def k_evaluation(scans, prepared, banks, folds_array, indices):
    """Inner participant-block CV, evaluate two disjoint training pools on each held-out split."""
    rows = []
    inner_folds = np.unique(folds_array[indices])
    for k in range(2,11):
        records = []
        for f in inner_folds:
            test = indices[folds_array[indices] == f]
            train_idx = indices[folds_array[indices] != f]
            levels = [v for v in inner_folds if v != f]
            train_a = train_idx[np.isin(folds_array[train_idx], levels[::2])]
            train_b = train_idx[np.isin(folds_array[train_idx], levels[1::2])]
            def fit(sub):
                return train([scans[i] for i in sub], [prepared[i] for i in sub],
                             banks[k][sub], k=k)
            main, aa, bb = fit(train_idx), fit(train_a), fit(train_b)
            _, similarity, _ = align(aa.maps, bb.maps)
            for i in test:
                p = prepared[i]
                units = p.units[p.valid]
                scores = np.abs(units @ main.maps.T)
                y = scores.argmax(axis=1)
                prevalence = np.bincount(y,minlength=k) / len(y)
                # Compute silhouette on GFP maxima, avoiding dominant lag-1 near-duplicates.
                yp = np.abs(p.units[p.peaks] @ main.maps.T).argmax(axis=1)
                distance = np.maximum(1-np.abs(p.units[p.peaks] @ p.units[p.peaks].T), 0)
                np.fill_diagonal(distance,0)
                sil = silhouette_score(distance,yp,metric="precomputed") if 1<len(set(yp))<len(yp) else np.nan
                stability = adjusted_rand_score(np.abs(units @ aa.maps.T).argmax(axis=1),
                                               np.abs(units @ bb.maps.T).argmax(axis=1))
                records.append(dict(subject=scans[i]["subject"],
                    heldout_gev=np.average(scores.max(axis=1)**2,weights=p.amplitude[p.valid]**2),
                    silhouette=sil, cluster_stability=stability, reproducibility=similarity.mean(),
                    minimum_prevalence=prevalence.min()))
        row = pd.DataFrame(records).groupby("subject").mean().mean().to_dict()
        rows.append(dict(k=k, degrees_freedom=k*76, **row))
    # Equal rank weights for predeclared fit, partition and map criteria.
    table = pd.DataFrame(rows)
    table["rank_sum"] = sum(rankdata(-table[x]) for x in ("heldout_gev","cluster_stability","reproducibility"))
    chosen = int(table.sort_values(["rank_sum","k"]).iloc[0].k)
    return chosen, table


def run(scans, scope, transform, only=None):
    RESULTS.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    scans = sorted(scans,key=lambda r:r["scan"])
    split = folds(scans)
    allidx = np.arange(len(scans))
    anchor = historical_anchor()
    p_cache, banks, fits = {}, {}, {}
    out_rows, match_rows, k_rows, maps_list, lineage = [],[],[],[],[]
    registry = [
        ("E00","F0","P0","B0","T0",4,"N0"), ("E01","F0","P0","B0","T1",4,"N0"),
        ("E02","F1","P0","B0","T1",4,"N0"), ("E03","F0","P0","B1","T1",4,"N0"),
        ("E04","F1","P0","B1","T1",4,"N0"), ("E05","F1","P1","B1","T1",4,"N0"),
        ("E06","F1","P2","B1","T1",4,"N0"), ("E07","F1","P2","B1","T1",8,"N0"),
        ("E08","F1","P0","B1","T2",4,"N0"), ("E09","F1","P0","B1","T3",4,"N0"),
        ("E10","F1","P0","B1","T1",0,"N0"), ("E11","F0","P0","B1","T2",4,"N0"),
        ("F2","F2","P0","B1","T1",4,"N0"), ("F3","F3","P0","B1","T1",4,"N0"),
        ("N1","F1","P0","B1","T1",4,"N1"), ("B2","F1","P0","B2","T1",4,"N0"),
        ("T2_HC_ON","F1","P0","B1","T2_HC_ON",4,"N0"),
        ("T2_HC_OFF","F1","P0","B1","T2_HC_OFF",4,"N0"),
        ("E03_common_time","F0","P0","B1_common","T1",4,"N0"),
        ("E04_common_time","F1","P0","B1_common","T1",4,"N0"),
    ]
    gate_path = ROOT / "manifests/provenance_gate.json"
    include_raw = gate_path.is_file() and json.loads(gate_path.read_text(encoding="utf-8-sig"))["n2_scales_available"]
    roi_scale = {}
    if include_raw and (not only or "N2" in only):
        from input_data import amplitude_scales
        roi_scale = amplitude_scales()
        registry.append(("N2","F1","P0","B1","T1",4,"N2"))
    elif only and "N2" in only:
        raise ValueError("N2 requires verified raw-relative ROI amplitude scales")
    if scope == "PD":
        registry = [r for r in registry if not r[4].startswith("T2_HC")]
        registry += [("LO_ON_F0","F0","P0","B0","ON",4,"N0"),
                     ("LO_ON_F1","F1","P0","B1","ON",4,"N0")]
    prefix = "PD_" if scope == "PD" else ""
    if only:
        registry = [r for r in registry if r[0] in only]
        if len(registry)!=len(set(only)):
            raise ValueError("Unknown or ineligible selected experiment")
        replacing = {prefix+r[0]+s for r in registry for s in ("","_edge_censored")}
        def preserve(path, ktable=False):
            if not path.is_file() or (ktable and prefix+"E10" in replacing):
                return []
            df=pd.read_csv(path,low_memory=False)
            if "experiment" in df:
                df=df[~df.experiment.isin(replacing)]
            return df.to_dict("records")
        out_rows=preserve(PRIVATE/(prefix+"Subject_Level_Features.csv"))
        match_rows=preserve(RESULTS/(prefix+"Template_Matching.csv"))
        k_rows=preserve(RESULTS/(prefix+"K_Evaluation.csv"),ktable=True)
        maps_list=preserve(RESULTS/(prefix+"Group_Templates.csv"))
        lineage=preserve(RESULTS/(prefix+"State_Lineage.csv"))
    for exp, band, pol, rule, scheme, fixed_k, norm in registry:
        exp = prefix+exp
        print("Starting", exp, flush=True)
        if (band,norm) not in p_cache:
            data = [(s["x"]*roi_scale[s["scan"]] if norm=="N2" else s["x"]) for s in scans]
            p_cache[band,norm] = [prepare(transform(x,band),norm) for x in data]
        pp = p_cache[band,norm]
        sel = allidx
        if scheme.startswith("T2_HC_"):
            sel = allidx[[s["condition"] in ("HC",scheme.split("_")[-1]) for s in scans]]
        k_options = list(range(2,11)) if not fixed_k else [fixed_k]
        for k in k_options:
            if pol!="P2" and (band,norm,k) not in banks:
                banks[band,norm,k] = np.stack([individual(p,k) for p in pp])
        if band=="F1" and norm=="N0" and fixed_k==4 and pol=="P0":
            np.savez_compressed(PRIVATE/(prefix+"individual_maps.npz"),
                                scans=np.array([s["scan"] for s in scans]),maps=banks[band,norm,4])
        for fold in ([-1] if scheme=="T0" else [-1,0,1,2,3,4]):
            tr = sel if fold<0 else sel[split[sel]!=fold]
            te = sel if fold<0 else sel[split[sel]==fold]
            if scheme=="ON":
                tr=tr[[scans[i]["condition"]=="ON" for i in tr]]
            if not fixed_k:
                k, table = k_evaluation(scans,pp,{v:banks[band,norm,v] for v in k_options},split,tr)
                k_rows.extend(table.assign(fold=fold,chosen=table.k==k).to_dict("records"))
            else:
                k = fixed_k
            # For fixed K4 unsigned names use the historical ON maps only for ordering.
            # K8 signed names use both directions of all four frozen axes. For data-driven
            # unequal K only global outcomes can be pooled when folds select different K.
            ref = anchor if k==4 else None
            if pol=="P2" and k==8:
                ref=np.stack([anchor,-anchor],axis=1).reshape(8,-1)
            if scheme=="T0":
                fitted=Fitted(anchor,np.arange(4),np.ones(4,bool),[],"P0",4)
            else:
                key = (band,norm,k,pol,scheme,fold)
                if key not in fits:
                    fits[key] = train([scans[i] for i in tr],[pp[i] for i in tr],
                        banks[band,norm,k][tr] if pol!="P2" else None,
                        scheme,k,pol,reference=ref)
                fitted=fits[key]
            if not fixed_k and k!=4:
                fitted.stable[:] = False
            match_rows.extend(dict(experiment=exp,fold=fold,**row) for row in fitted.match_rows)
            for j,(m,fam) in enumerate(zip(fitted.maps, fitted.family)):
                maps_list.append(dict(experiment=exp,fold=fold,variant=j+1,state=fam+1,
                                      **{f"ROI_{n+1}":v for n,v in enumerate(m)}))
                if k!=4:
                    dot=anchor @ m
                    lineage.extend(dict(experiment=exp,fold=fold,state=fam+1,historical_axis=h+1,
                               spatial_dot=v,nearest=bool(h==np.abs(dot).argmax()))
                                   for h,v in enumerate(dot))
            for i in te:
                s,p=scans[i],pp[i]
                labels,scores,dots=backfill(p,fitted,"B1" if rule=="B1_common" else rule)
                if rule=="B1_common":
                    labels[backfill(p,fitted,"B0")[0]<0]=-1
                row=dict(experiment=exp, fit="full" if fold<0 else "crossfit",fold=fold,
                         **{k:s[k] for k in ("subject","condition","scan","fd")})
                metrics=measure(p,labels,scores)
                row.update(metrics)
                if pol=="P2":
                    ok=labels>=0
                    selected=scores[ok,labels[ok]]
                    row["negative_assignment_fraction"]=np.mean(selected<0)
                    row["positive_projection_gev"]=np.average(np.maximum(selected,0)**2,
                                                               weights=p.amplitude[ok]**2)
                if not metrics:
                    raise ValueError(f"No usable episodes, experiment={exp}, fold={fold}")
                if pol=="P1":
                    good = (labels[:-1]>=0)&(labels[1:]>=0)
                    before,after=labels[:-1][good],labels[1:][good]
                    row["axis_sign_switch_rate"]=np.sum((before//2==after//2)&(before!=after))/row["duration_s"]
                    row["between_axis_switch_rate"]=np.sum(before//2!=after//2)/row["duration_s"]
                    for ax in range(k):
                        pos,neg=(row["coverage_"+str(j)] for j in (2*ax+1,2*ax+2))
                        row[f"sign_imbalance_{ax+1}"]=(pos-neg)/(pos+neg) if pos+neg else np.nan
                row["stable_states"]=",".join(str(j+1) for j in np.flatnonzero(fitted.stable))
                row.update({f"stable_{j+1}":bool(ok) for j,ok in enumerate(fitted.stable)})
                out_rows.append(row)
                edge=measure(p,labels,scores,edge_censor=True)
                if edge:
                    out_rows.append({**row, **edge,"experiment":exp+"_edge_censored",
                        **{key:np.nan for key in row if key.startswith(("axis_sign","between_axis","sign_imbalance","negative_assignment","positive_projection"))}})
                directory=PRIVATE/"backfits"/exp/row["fit"]
                directory.mkdir(exist_ok=True,parents=True)
                top=np.sort(scores,axis=1)[:,-2:]
                top[~p.valid] = np.nan
                np.savez_compressed(directory/(s["scan"]+".npz"),
                    signed_similarity=np.where(p.valid[:,None],dots,np.nan).astype("float32"),
                    family_similarity=np.where(p.valid[:,None],scores,np.nan).astype("float32"),
                    top=top[:,1],second=top[:,0],margin=top[:,1]-top[:,0],
                    low_margin=top[:,1]-top[:,0]<.05,labels=labels,valid=p.valid,peaks=p.peaks,
                    segments=np.array(segments(labels)),family=fitted.family)
        save_tables(prefix,out_rows,match_rows,k_rows,maps_list,lineage)
        print("Completed",exp,flush=True)


def save_tables(prefix, rows, matches, k_rows, maps_list, lineage):
    output = PRIVATE/(prefix+"Subject_Level_Features.csv")
    temporary = output.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(temporary,index=False)
    temporary.replace(output)
    metrics = ["switching_rate","global_occurrence","global_pooled_duration","global_equal_duration",
               "mean_similarity","mean_margin","transition_entropy"]
    table = pd.DataFrame(rows)
    metrics += [x for x in table if x.startswith(("coverage_","duration_","occurrence_","gev_",
                "axis_sign","between_axis","sign_imbalance","recurrence_","transition_","negative_assignment","positive_projection","low_margin")) and x not in metrics]
    table.groupby(["experiment","fit","condition"])[metrics].agg(["count","mean","std"]).to_csv(
        RESULTS/(prefix+"Method_Feature_Summary.csv"))
    for name,records in [("Template_Matching",matches),("K_Evaluation",k_rows),("State_Lineage",lineage),
                         ("Group_Templates",maps_list)]:
        if records:
            pd.DataFrame(records).to_csv(RESULTS/(prefix+name+".csv"),index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["PD","all"],required=True)
    parser.add_argument("--only",nargs="+",help="Refresh named V3 outputs while preserving completed experiments")
    args=parser.parse_args()
    gate = json.loads((ROOT / "manifests/provenance_gate.json").read_text(encoding="utf-8-sig"))
    if not gate["pd_comparable"] or (args.scope == "all" and not gate["hc_comparable"]):
        raise ValueError("Requested scope has not passed provenance verification")
    from input_data import load_scans, transform
    with threadpool_limits(limits=1):
        input_scans=load_scans()
        if args.scope=="PD":
            input_scans=[s for s in input_scans if s["condition"]!="HC"]
        run(input_scans,args.scope,transform,args.only)
