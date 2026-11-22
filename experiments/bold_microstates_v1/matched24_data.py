"""Audit aligned inputs and fit shared unsigned amplitude templates for matched patients."""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from adapt_syn9 import CONDITION_CODE
from compare_legacy_105 import REPORTS, read_inputs
from peak_states import PeakScan, group_maps, prepare_scan
from same_band_analysis import SOURCES, spatial_correlation
from run import digest


@dataclass
class Visit:
    wave: np.ndarray
    features: np.ndarray
    peaks: PeakScan


@dataclass
class Study:
    visits: dict
    person_ids: list
    fold_names: dict
    axis: list
    reference: np.ndarray
    paper: np.ndarray
    audit: dict


def subjects(fold):
    """The public leave-one-out label is assigned only at the report boundary."""
    return [p for p in range(24) if p + 1 != fold]


def load_study(root):
    aux = root / "V100脑深部刺激WEiDA复现_20261118/代码与辅助"
    matched = aux / "同人群精确度对照"
    refs = pd.read_csv(REPORTS / "same_band/results/templates.tsv", sep="\t", index_col=[0,1,2,3])
    axis = refs.columns.tolist()
    reference = np.stack([refs.loc[(src,"primary","micro")].sort_index().to_numpy() for src in SOURCES])
    paper = pd.read_csv(aux / "论文图3A质心读数.csv", index_col="roi")
    np.testing.assert_array_equal(axis,paper.index)
    assert list(paper.columns) == ["LMB-SM","V-VA/SM","FPC-V","SM-DM"]
    source_data = read_inputs(root,axis)
    roster = pd.read_csv(matched / "同人群清单.csv",keep_default_na=False)
    syn_pack = pd.read_csv(aux / "复现输入清单.csv").set_index(["alias","condition"],verify_integrity=True)
    audit_root = root / "WEiDA-raw-audit-private-20261116"
    legacy_dir = root / "STN-DBS_105_AAL78_20261117/PD_DBS_zcc/derivatives/fmripost"
    ident = pd.read_csv(root / "WEiDA-raw-audit-private-20261116/own_pipeline_20261116/stn_replication/cohort.tsv",
                        sep="\t").set_index("bids_label",verify_integrity=True)
    off = roster[roster.condition=="OFF"]
    on = roster[roster.condition=="ON"]
    np.testing.assert_array_equal(off.alias,on.alias)
    people = ident.loc[off.scan,"person_id"].tolist()
    assert len(people) == len(set(people)) == 24 and people == ident.loc[on.scan,"person_id"].tolist()
    common = {(p,c) for p,c in set(source_data[0]) & set(source_data[1]) if c in ("OFF","ON")}
    assert common == {(p,c) for p in people for c in ("OFF","ON")}
    # Opaque, reproducible ordering derived from private OFF-file fingerprints.
    digests = [hashlib.sha256(("matched24:" + a + b).encode()).hexdigest() for a,b in
               off[["105_series_sha256","v100_series_sha256_before_alignment"]].itertuples(index=False,name=None)]
    fold_names = {int(p)+1:f"J{i:02d}" for i,p in enumerate(np.argsort(digests),1)}

    lengths = roster.frames.to_numpy()
    b,a = signal.butter(2,[.04,.07],btype="bandpass",fs=1/2.1)
    visits, errors, read_errors = {}, [], []
    for si,folder in enumerate(("legacy105","v100_syn9")):
        src = matched / folder
        cells = loadmat(src/"inputs.mat")["ts_cell"]
        data = loadmat(src/"author_matched_k4.mat",squeeze_me=True,
                       variable_names=["WEiDA_Eig","Time_all","frames","TR"])
        assert data["TR"] == 2.1 and cells.shape == (63,2)
        np.testing.assert_array_equal(data["frames"], lengths)
        np.testing.assert_array_equal(data["Time_all"][0],np.repeat(np.arange(1,64),lengths))
        np.testing.assert_array_equal(data["Time_all"][1],np.repeat(roster.condition_code,lengths))
        edges = np.r_[0,lengths.cumsum()]
        for i,row in enumerate(roster.itertuples(index=False)):
            if row.condition == "HC":
                continue
            who = ident.loc[row.scan]
            key = (who.person_id,row.condition)
            p,visit = people.index(who.person_id), ("OFF","ON").index(row.condition)
            old, new = (dataset[key] for dataset in source_data)
            assert who.condition == row.condition and row.condition_code == CONDITION_CODE[who.condition]
            assert new.first_frame == 0 and row.v100_removed_frame_zero_based == ("0" if old.first_frame else "")
            scan = (old,new)[si]
            aligned = scan.values[old.first_frame-scan.first_frame:]
            assert cells[i,1].item() == row.condition_code and aligned.shape == (row.frames,78)
            if si == 0:
                path = legacy_dir / row.scan / f"{row.scan}_BOLD_AAL1ctx_MNI_basic36_sm6.csv"
                checksum = roster.iloc[i]["105_series_sha256"]
            else:
                path = audit_root/"syn9_stn_hc_batch/review"/syn_pack.loc[(row.alias,row.condition),"source"]
                checksum = row.v100_series_sha256_before_alignment
            assert digest(path) == checksum
            packed = cells[i,0].T
            # The independent TSV loader and the MATLAB pack writer parse decimal
            # text with different readers. Verify each coordinate and use the pack
            # that actually generated the archived author phase features.
            np.testing.assert_allclose(packed,aligned,rtol=2e-15,atol=1e-15)
            read_errors.append(np.max(abs(packed-aligned)))
            trendless = signal.detrend(packed,axis=0)
            wave = signal.filtfilt(b,a,trendless-trendless.mean(0),axis=0,padlen=3*(max(len(a),len(b))-1))
            features = data["WEiDA_Eig"][edges[i]:edges[i+1]]
            phi = np.angle(signal.hilbert(wave,axis=0))
            norm2 = (1 + abs(np.exp(2j*phi).mean(1))**2)/2
            error = np.max(abs(norm2-(features**2).sum(1)))
            assert error < 1e-9 and np.isfinite(features).all()
            errors.append(error)
            visits[si,p,visit] = Visit(wave,features,prepare_scan(wave,scan.mean_fd))
    total = sum(len(v.wave) for (si,p,vis),v in visits.items() if si==0)
    assert len(visits) == 96 and total == 10077
    audit = dict(pairs=len(people),visits_per_source=48,aligned_input_frames_per_source=total,
                 frames_off=int(off.frames.sum()),frames_on=int(on.frames.sum()),
                 scans_with_frame0_removed_off=int((off.frames==209).sum()),
                 scans_with_frame0_removed_on=int((on.frames==209).sum()),
                 archived_phase_norm_max_abs_error=max(errors),pack_vs_source_max_abs_error=max(read_errors))
    return Study(visits,people,fold_names,axis,reference,paper.to_numpy().T,audit)


def fit_amplitudes(study,cache,folds):
    for fold in folds:
        destination = cache / f"amp_{fold:02d}.npz"
        if destination.exists():
            raise FileExistsError(destination)
        maps = [study.visits[s,p,v].peaks.individual for s in range(2) for p in subjects(fold) for v in range(2)]
        raw,info = group_maps(maps,np.full(len(maps),.25))
        corr = np.stack([spatial_correlation(ref,raw) for ref in study.reference])
        score = abs(corr).mean(axis=0)
        rows,order = linear_sum_assignment(-score)
        np.testing.assert_array_equal(rows,np.arange(4))
        # Sign used only for plotting. All amplitude fitting uses absolute projections.
        centers = raw[order] * np.sign(corr[1,rows,order])[:,None]
        np.savez(destination,centers=centers,matches=np.array([spatial_correlation(r,centers) for r in study.reference]),
                 starts=info["starts"],objective=info["best_mean_squared_r"])
        print(f"Completed shared amplitude fit {fold:02d} ({len(maps)} scans)",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--folds",type=int,nargs="+",default=list(range(25)),choices=range(25))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        fit_amplitudes(load_study(args.data_root),args.cache,args.folds)
