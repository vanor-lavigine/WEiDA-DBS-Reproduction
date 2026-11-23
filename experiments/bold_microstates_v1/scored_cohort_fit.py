"""Fit two score-blind four-state models on complete, roster-selected DBS visits."""
import argparse
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from core import assign_states, prepare_bold, sequence_metrics, spherical_kmeans
from peak_states import group_maps, prepare_scan
from run import digest, load_roi


def fit(roster, input_root, output):
    output.mkdir(exist_ok=False)
    if set(roster)!= {"run_id", "person", "condition", "roi_digest"}:
        raise ValueError("Fit roster must contain identity and provenance only")
    roster = roster.sort_values("run_id").set_index("run_id", verify_integrity=True)
    pairs = roster.groupby("person").condition.agg(list)
    if len(pairs)!=24 or not all(sorted(c)==["OFF","ON"] for c in pairs):
        raise ValueError("Expected exactly 24 paired patients")
    runs = pd.read_csv(input_root/"input/runs.tsv", sep="\t").set_index("run_id",verify_integrity=True)
    ids = pd.read_csv(input_root/"input/metadata.verified.tsv",sep="\t",
                      usecols=["run_id","subject_id","cohort","condition","mapping_source"]).set_index("run_id",verify_integrity=True)
    pack = pd.read_csv(input_root/"packing-manifest.csv")
    regions = json.loads((input_root/"aal78-order.json").read_text())
    if len(set(regions))!=78 or len(regions)!=78:
        raise ValueError("Invalid AAL78 region order")
    people = {}
    signed_scans, peak_scans, numerical_hashes = [], [], set()
    from hashlib import sha256
    for run_id, visit in roster.iterrows():
        run, identity = runs.loc[run_id], ids.loc[run_id]
        entry = pack.iloc[int(run.packing_row)-1]
        path = Path(run.roi_time_series)
        if (identity.cohort!="DBS" or identity.condition!=visit.condition or not pd.notna(identity.mapping_source)
                or entry["alias"]!=visit.person or entry["condition"]!=visit.condition
                or not (visit.roi_digest==entry.source_sha256==run.source_sha256==digest(path))):
            raise ValueError("A selected scan failed its identity or digest check")
        people.setdefault(visit.person,set()).add(identity.subject_id)
        if float(run.tr_sec)!=2.1 or int(run.n_frames_expected)!=210 or int(run.n_regions_expected)!=78:
            raise ValueError("Unexpected TR or dimensions in frozen input")
        raw, names = load_roi(path)
        if raw.shape!=(210,78) or set(names)!=set(regions):
            raise ValueError("Unexpected input shape or anatomical labels")
        data = raw[:,[names.index(name) for name in regions]]
        fingerprint = sha256(np.ascontiguousarray(data,dtype="<f8").tobytes()).hexdigest()
        if fingerprint in numerical_hashes:
            raise ValueError("Duplicate numerical scan")
        numerical_hashes.add(fingerprint)
        full = prepare_bold(data)
        peak = prepare_scan(data, fd=np.nan)
        if not full["valid"].all() or not (peak.gfp>0).all():
            raise ValueError("Selected input has unassignable frames")
        signed_scans.append(full["unit_vectors"])
        peak_scans.append(peak)
    if any(len(v)!=1 for v in people.values()) or len(set.union(*people.values()))!=24:
        raise ValueError("Patient identifiers conflict with the verified input")

    peak_centers, peak_fit = group_maps([s.individual for s in peak_scans],np.ones(len(peak_scans)))
    frames = np.concatenate(signed_scans)
    # Each patient has two 210-frame visits, giving every patient equal weight.
    signed = spherical_kmeans(frames,k=4,n_init=100,max_iter=300,seed=20261121,
                              weights=np.full(len(frames),1/(2*210)),polarity="signed")
    if signed["n_occupied_clusters"]!=4:
        raise ValueError("Signed fit returned an empty group state")
    centers = {"AAHC":peak_centers, "signed":signed["centers"]}
    templates = []
    metrics, labels, assigned = [], [], Counter()
    peak_counts = [len(s.peaks) for s in peak_scans]
    peak_num = peak_den = 0.0
    for i, run_id in enumerate(roster.index):
        s = peak_scans[i]
        similarities = np.abs(s.units @ peak_centers.T)
        peak_labels = similarities.argmax(axis=1)
        weights = s.gfp[s.peaks]**2
        peak_num += np.dot(weights,similarities[s.peaks,peak_labels[s.peaks]]**2)
        peak_den += weights.sum()
        signed_labels = assign_states(signed_scans[i],centers["signed"],polarity="signed")["labels"]
        for method, seq in (("AAHC",peak_labels),("signed",signed_labels)):
            measured = sequence_metrics(seq,2.1,4)
            if measured["n_valid_frames"]!=210:
                raise ValueError("Frame backfitting is incomplete")
            assigned[method] += measured["n_valid_frames"]
            labels.extend((method,run_id,t,int(state)+1) for t,state in enumerate(seq))
            for row in measured["states"]:
                metrics.append(dict(method=method,run_id=run_id,**row))
    for method,maps in centers.items():
        np.testing.assert_allclose(np.linalg.norm(maps,axis=1),1,atol=1e-12)
        templates.extend(dict(method=method,state=i+1,**dict(zip(regions,row))) for i,row in enumerate(maps))
    summary = {"data_kind":"real","patients":len(pairs),"runs":len(roster),"frames_per_run":210,
               "n_roi":len(regions),"tr_sec":2.1,"k":4,"n_hc":0,"assigned_frames":dict(assigned),
               "peaks_total":sum(peak_counts),"peaks_per_scan_min":min(peak_counts),
               "peaks_per_scan_max":max(peak_counts),"pooled_peak_gev":float(peak_num/peak_den),
               "peak_group_fit":peak_fit,"signed_n_init":100,"signed_max_iter":300,"seed":20261121,
               "signed_best_objective":signed["objective"],"signed_winning_iterations":signed["n_iter"],
               "environment":{"python":platform.python_version(),**{p:version(p) for p in ("numpy","scipy","pandas")}}}
    pd.DataFrame(labels,columns=["method","run_id","frame","state"]).to_csv(output/"sequences.tsv",sep="\t",index=False)
    pd.DataFrame(metrics).to_csv(output/"metrics.tsv",sep="\t",index=False,na_rep="n/a")
    pd.DataFrame(templates).to_csv(output/"templates.tsv",sep="\t",index=False)
    (output/"fit_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster",type=Path,required=True)
    parser.add_argument("--input-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    fit(pd.read_csv(args.roster,sep="\t"),args.input_root,args.output)
