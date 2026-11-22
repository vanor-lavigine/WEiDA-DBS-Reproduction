"""Compare same-band WEiDA states and polarity-invariant peak amplitude states on real scans."""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import signal
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_mutual_info_score

from adapt_syn9 import CONDITION_CODE
from compare_legacy_105 import REPORTS, read_inputs
from eeg_peak_analysis import metrics, train
from peak_states import K, backfit, prepare_scan
from run import new_output, write_json
from same_band_stats import BOOTSTRAPS, PERMUTATIONS, SEED, SHIFT_DRAWS, association_family, group_shifts, shift_profile
from stats import holm_adjust

BANDS = {"primary": (.04,.07), "slow5": (.01,.027)}
SOURCES = ("105","SyN9")


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spatial_correlation(a, b):
    return 1 - cdist(a,b,metric="correlation")


def spatial_rows(source, band, method, correlations):
    return [dict(source=source, band=band, method=method, reference=i+1, state=j+1,
                 spatial_r=correlations[i,j], assigned=(i==j))
            for i in range(K) for j in range(K)]


def audit_scans(raw, inputs):
    cells = loadmat(inputs)["ts_cell"]
    if cells.shape != (len(raw),2):
        raise ValueError("Mismatch between MATLAB pack and verified roster")
    for ((_,condition),scan), (matrix,code) in zip(raw.items(),cells):
        np.testing.assert_array_equal(scan.values,matrix.T)
        assert int(code.item()) == CONDITION_CODE[condition]


def fit_combination(raw, cached, broad, paper, band, source, aux):
    values = loadmat(cached,squeeze_me=True)
    wave, features = values["wave"],values["features"]
    centers = values["centers"]
    var_centers = values["var_centers"]
    within = values["time_index"]
    lengths = [len(s.values) for s in raw.values()]
    np.testing.assert_array_equal(within[0],np.repeat(np.arange(1,len(raw)+1),lengths))
    np.testing.assert_array_equal(within[1],np.repeat([CONDITION_CODE[c] for _,c in raw],lengths))
    native = values["labels"].astype(int)-1
    assert native.shape == (sum(lengths),) and wave.shape == features.shape == (len(native),78)
    np.testing.assert_array_equal(cdist(features,centers).argmin(1), native)
    center_check = np.stack([features[native==j].mean(0) for j in range(K)])
    np.testing.assert_allclose(center_check,centers,rtol=0,atol=1e-10)
    # The author computes local VAR with centroids as reloaded from CSV.
    np.testing.assert_allclose(var_centers,centers,rtol=6e-5,atol=1e-9)
    corr = spatial_correlation(paper,centers)
    row, order = linear_sum_assignment(-corr)
    np.testing.assert_array_equal(row,np.arange(K))
    phase_labels = np.argsort(order)[native]
    phase_match = corr[:,order]
    matched_centers = centers[order]
    var_s2_by_key, s2_counts = {}, []
    preps, phases, errors = {}, {}, []
    b,a = signal.butter(2,BANDS[band],btype="bandpass",fs=1/2.1)
    padlen = 3*(max(len(a),len(b))-1)
    offset = 0
    for key, scan in raw.items():
        window = slice(offset,offset+len(scan.values))
        filtered = wave[window]
        trendless = signal.detrend(scan.values,axis=0)
        check = signal.filtfilt(b,a,trendless-trendless.mean(0),axis=0,padlen=padlen)
        error = np.max(np.abs(check-filtered))/np.max(np.abs(filtered))
        assert error < 1e-8
        errors.append(error)
        phi = np.angle(signal.hilbert(filtered,axis=0))
        expected_norm2 = (1 + np.abs(np.exp(2j*phi).mean(1))**2)/2
        np.testing.assert_allclose(np.sum(features[window]**2,axis=1),expected_norm2,rtol=0,atol=1e-9)
        phases[key] = phase_labels[window]
        s2 = phases[key] == 1
        n_s2 = int(s2.sum())
        s2_counts.append(n_s2)
        var_s2_by_key[key] = (np.sum((features[window][s2]-var_centers[order[1]])**2)/(78*(n_s2-1))
                              if n_s2>=2 else np.nan)
        preps[key] = prepare_scan(filtered,scan.mean_fd)
        offset += len(scan.values)

    verification = {}
    if source=="SyN9" and band=="primary":
        saved = loadmat(aux/"MATLAB四状态分析结果.mat",variable_names=["WEiDA_Eig"])["WEiDA_Eig"]
        np.testing.assert_allclose(features,saved,rtol=0,atol=1e-10)
        old_var = loadmat(aux/"MATLAB局部亚稳态结果.mat")["OPs"][:,order[1]]
        new_var = np.array(list(var_s2_by_key.values()))
        np.testing.assert_allclose(old_var,new_var,rtol=0,atol=1e-12)
        verification["archive_s2_var_max_error"] = float(np.max(abs(old_var-new_var)))

    fitted, micro_match, fit = train([(k[0],s) for k,s in preps.items()],broad)
    measures, profiles, n_peaks, n_retained = [], {}, [], []
    for key, prep in preps.items():
        micro, _, diagnostic = backfit(prep,fitted)
        for record in metrics(key,prep.fd,micro):
            measures.append(dict(**record,var_s2=var_s2_by_key[key]))
        profile = shift_profile(micro,phases[key])
        if profile is not None:
            keep = micro>=0
            m, s = micro[keep],phases[key][keep]
            for lag in (0,len(m)//3):
                check = adjusted_mutual_info_score(m,np.roll(s,lag))
                np.testing.assert_allclose(profile["ami"][lag],check,rtol=0,atol=1e-10)
        profiles[key] = profile
        n_peaks.append(diagnostic["peaks"])
        n_retained.append(diagnostic["n_retained"])
    summary = dict(scans=len(raw),frames=sum(lengths),matlab=values["matlab_version"],
                   verified_nearest_labels=len(native),filter_max_relative_error=max(errors),
                   weida_native_id_in_paper_order=(order+1).tolist(),
                   weida_match_r=np.diag(phase_match),micro_match_abs_r=np.diag(micro_match),
                   s2_frames_min=min(s2_counts),s2_frames_median=float(np.median(s2_counts)),
                   s2_frames_max=max(s2_counts),peaks_median=float(np.median(n_peaks)),
                   common_frames_min=min(n_retained),common_frames_median=float(np.median(n_retained)),
                   common_frames_max=max(n_retained),**verification,**fit)
    matches = spatial_rows(source,band,"WEiDA_vs_paper",phase_match) + spatial_rows(source,band,"micro_vs_broad",micro_match)
    return pd.DataFrame(measures), profiles, matches, {"WEiDA":matched_centers,"micro":fitted}, summary


def run(root, cache, destination):
    refs = pd.read_csv(REPORTS/"eeg_peaks/results/templates.tsv",sep="\t",index_col=[0,1])
    axis = refs.columns.tolist()
    aux = root/"V100脑深部刺激WEiDA复现_20261118/代码与辅助"
    legacy = root/"WEiDA_STN25_HC15_K4_R2020b_20261117"
    paper_frame = pd.read_csv(aux/"论文图3A质心读数.csv",index_col="roi")
    np.testing.assert_array_equal(paper_frame.index,axis)
    assert paper_frame.columns.tolist() == ["LMB-SM","V-VA/SM","FPC-V","SM-DM"]
    paper = paper_frame.to_numpy().T
    source_data = dict(zip(SOURCES,read_inputs(root,axis)))
    metadata = dict(data_kind="real",tr_seconds=2.1,seed=SEED,shift_draws=SHIFT_DRAWS,
                    association_permutations=PERMUTATIONS,bootstrap_draws=BOOTSTRAPS,
                    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
                    scipy=scipy.__version__,sklearn=sklearn.__version__,combinations={})
    metadata["author_feature_code_sha256"] = file_digest(root/"WEiDA-DBS-author-matlab/StateSpace_modelling.m")
    metadata["paper_figure_weights_sha256"] = file_digest(aux/"论文图3A质心读数.csv")
    measurements, coupling, by_visit, joint, matches, template_rows = {}, [], [], [], [], []
    maps = {}
    for si,src in enumerate(SOURCES):
        raw = source_data[src]
        audit_scans(raw, legacy/"inputs.mat" if src=="105" else aux/"MATLAB分析输入.mat")
        fd = {k:s.mean_fd for k,s in raw.items()}
        for bi,band in enumerate(BANDS):
            print(f"Fitting amplitude states and measuring shift coupling: {src} {band}",flush=True)
            measured,profiles,spatial,templates,quality = fit_combination(raw,cache/f"{src}_{band}.mat",
                                                          refs.loc[src].to_numpy(),paper,band,src,aux)
            key = dict(source=src,band=band)
            measurements[src,band] = measured
            metadata["combinations"][f"{src}_{band}"] = quality
            matches.extend(spatial)
            for method, weights in templates.items():
                maps[src,band,method] = weights
                for state, v in enumerate(weights,1):
                    template_rows.append(dict(**key,method=method,state=state,**dict(zip(axis,v))))
            for visit in ("HC","OFF","ON"):
                aligned = [p for (_,c),p in profiles.items() if c==visit and p is not None]
                by_visit.append(dict(**key,condition=visit,n_scans=len(aligned),
                                     observed_ami=np.mean([p["ami"][0] for p in aligned]),
                                     null_ami=np.mean([p["ami"].mean() for p in aligned])))
            result, cell = group_shifts(profiles,fd,np.random.default_rng([SEED,si,bi,99]))
            coupling.append(dict(**key,**result))
            for i in range(K):
                for j in range(K):
                    obs, exp = cell["observed_cells"][i,j],cell["null_cells"][i,j]
                    joint.append(dict(**key,micro=i+1,weida=j+1,observed_fraction=obs,
                                      shift_expected_fraction=exp,excess_pp=100*(obs-exp)))
    for row,p in zip(coupling,holm_adjust([r["p_shift"] for r in coupling])):
        row["p_holm_four"] = p
    for band in BANDS:
        for method in ("micro","WEiDA"):
            corr = spatial_correlation(maps["105",band,method],maps["SyN9",band,method])
            matches.extend(spatial_rows("105_to_SyN9",band,method,corr))
    for src in SOURCES:
        for method in ("micro","WEiDA"):
            corr = spatial_correlation(maps[src,"primary",method],maps[src,"slow5",method])
            matches.extend(spatial_rows(src,"primary_to_slow5",method,corr))

    associations = []
    for si,src in enumerate(SOURCES):
        for bi,band in enumerate(BANDS):
            print(f"Testing 12 motion-adjusted change associations: {src} {band}",flush=True)
            family = association_family(measurements[src,band],si,bi)
            associations.extend(dict(source=src,band=band,**row) for row in family)
    out = new_output(destination)
    for name, rows in [("cooccurrence",coupling),("cooccurrence_by_visit",by_visit),
                       ("state_pairs",joint),("associations",associations),("state_matching",matches)]:
        pd.DataFrame(rows).to_csv(out/(name+".csv"),index=False,na_rep="NA")
    pd.DataFrame(template_rows).to_csv(out/"templates.tsv",sep="\t",index=False)
    write_json(out/"summary.json",metadata)
    print("Completed four shift tests and 48 associations on real data.",flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    run(args.data_root,args.cache,args.out)
