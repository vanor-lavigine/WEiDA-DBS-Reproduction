"""Replace WEiDA phase features with condition-specific EEG-style amplitude maps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import re
import secrets

import numpy as np
import pandas as pd
import scipy
from scipy import signal, stats

from clinical_archive_rerun import ENGLISH_DOMAINS, VERSIONS, clinical_visits, scan_key
from core import sequence_metrics
from legacy_clinical25 import read_archive
from li_clinical_stats import correlation
from peak_states import group_maps, interpolate_peaks, match_to_signed, prepare_scan
from stats import holm_adjust


BANDS = {"weida_band": (0.04, 0.07)}
TR = 2.1
DURATION = "duration_s"
RATE = "occurrence_s"
COVERAGE = "coverage"


def fit_band(scans, motion, roster, band, reference, conditions=("OFF", "ON")):
    b, a = signal.butter(2, band, btype="bandpass", fs=1/TR)
    prepared = {}
    for name, values in scans.items():
        centered = signal.detrend(values, axis=0)
        centered -= centered.mean(axis=0)
        filtered = signal.filtfilt(b, a, centered, axis=0, padlen=12)
        prepared[name] = prepare_scan(filtered, motion[name])
    maps, qc = {}, {}
    for condition in conditions:
        individual = [prepared[n].individual for n in roster.index[roster.condition == condition]]
        centers, fit = group_maps(individual, np.ones(len(individual)))
        anchor = maps.get("OFF", reference)
        if anchor is not None:
            centers, corr, gap = match_to_signed(anchor, centers)
            fit.update(matched_r=np.diag(corr).tolist(), assignment_margin=gap,
                       spatial_correlations=corr.tolist())
        maps[condition] = centers
        qc[condition] = fit
    return prepared, maps, qc


def scan_features(scan, centers):
    similarity = np.abs(scan.units @ centers.T)
    peaks = scan.peaks
    best = similarity[peaks].argmax(axis=1)
    labels, _ = interpolate_peaks(len(scan.gfp), peaks, best, scan.gfp > 0, similarity)
    metrics = sequence_metrics(labels, TR, 4)
    states = metrics["states"]
    retained = metrics["n_valid_frames"]
    if retained <= 0:
        raise ValueError("No complete state episode")
    durations = np.array([row["duration_sec"] for row in states])
    rate = sum(row["n_episodes"] for row in states)/(TR*retained)
    features = {("all", DURATION): durations.mean(), ("all", RATE): rate,
                ("all", "pooled_duration_s"): 1/rate}
    for row in states:
        state = "M"+str(row["state"])
        features.update({(state, DURATION): row["duration_sec"],
                         (state, RATE): row["occurrence_per_min"]/60,
                         (state, COVERAGE): row["coverage"]})
    np.testing.assert_allclose(sum(row["coverage"] for row in states), 1)
    gev = np.average(similarity[peaks,best]**2, weights=scan.gfp[peaks]**2)
    diagnostics = dict(peaks=len(peaks), retained=retained, frames=len(labels), peak_gev=gev)
    return features, diagnostics, metrics["frame_transition_probabilities"]


def add_adjustment(rows, method, key):
    raw = np.array([1.0 if row["p_raw"] is None else row["p_raw"] for row in rows])
    corrected = stats.false_discovery_control(raw) if method == "bh" else holm_adjust(raw)
    for row, p in zip(rows, corrected):
        row[key] = float(p) if row["p_raw"] is not None else None


def band_stats(observations, roster, scores):
    pairs = roster.reset_index().pivot(index="person", columns="condition", values="run")
    yoff = scores.loc[pairs.OFF].to_numpy()
    yon = scores.loc[pairs.ON].to_numpy()
    if (yoff <= 0).any():
        raise ValueError("Invalid baseline for proportional improvement")
    benefit = 100*(yoff-yon)/yoff
    paired, clinical, sensitivity = [], [], []
    for (state, metric), values in observations.items():
        xoff, xon = values.loc[pairs.OFF].to_numpy(), values.loc[pairs.ON].to_numpy()
        tests = (("off_score", xoff, yoff), ("on_score", xon, yon), ("on_benefit", xon, benefit))
        for outcome, x, y in tests:
            row = dict(state=state, metric=metric, contrast=outcome, **correlation(x,y))
            (sensitivity if metric == "pooled_duration_s" else clinical).append(row)
        if metric == "pooled_duration_s":
            continue
        ok = np.isfinite(xoff) & np.isfinite(xon)
        delta = xon[ok]-xoff[ok]
        n = len(delta)
        mean = delta.mean()
        se = stats.sem(delta)
        if n < 3 or se <= 0:
            raise ValueError("Undefined paired t statistic")
        width = stats.t.ppf(.975, n-1)*se
        paired.append(dict(state=state, metric=metric, n=n, off=xoff[ok].mean(), on=xon[ok].mean(),
                           delta=mean, ci_low=mean-width, ci_high=mean+width,
                           p_raw=float(stats.ttest_rel(xon[ok],xoff[ok]).pvalue)))
    if len(clinical)!=42 or len(paired)!=14:
        raise ValueError("Incomplete test family")
    add_adjustment(clinical, "bh", "q_bh42")
    add_adjustment(paired, "holm", "p_holm14")
    return paired, clinical, sensitivity


def analyze_archive(bids_root, project):
    scans, motion, axis, _ = read_archive(bids_root, project)
    names = sorted(scans)
    ids = [re.fullmatch(r"sub-STN(off|on)(\d+)", name, re.I) for name in names]
    roster = pd.DataFrame({"run":names,"person":[m[2] for m in ids],
                           "condition":[m[1].upper() for m in ids]}).set_index("run")
    if roster.shape[0]!=50 or roster.person.nunique()!=25 or len({x.tobytes() for x in scans.values()})!=50:
        raise ValueError("Expected the 25 distinct audited pairs")
    prepared, maps, alignment = fit_band(scans, motion, roster, BANDS["weida_band"], None)
    observations, diagnostics, matrices = {}, [], {"OFF":[], "ON":[]}
    for name, record in roster.iterrows():
        condition = record.condition
        features, qc, matrix = scan_features(prepared[name], maps[condition])
        observations[name] = features
        diagnostics.append(dict(condition=condition, **qc))
        matrices[condition].append(matrix)
    summaries, transitions = {}, []
    for condition, group in pd.DataFrame(diagnostics).groupby("condition"):
        summaries[condition] = dict(peaks=int(group.peaks.sum()), frames=int(group.frames.sum()),
                                    retained=int(group.retained.sum()), mean_peak_gev=group.peak_gev.mean())
        array = np.stack(matrices[condition])
        for i,j in np.ndindex(4,4):
            col = array[:,i,j]
            transitions.append(dict(condition=condition, source=i+1, target=j+1,
                                    n=int(np.isfinite(col).sum()), mean=float(np.nanmean(col))))
    salt = secrets.token_hex(32)
    main, duplicate = [clinical_visits(project, folder, series, salt) for folder,series in VERSIONS[:2]]
    pd.testing.assert_series_equal(main.score, duplicate.score)
    np.testing.assert_array_equal(main.score, main[list(ENGLISH_DOMAINS)].sum(axis=1))
    scores = pd.Series([main.score.at[scan_key(name,salt)] for name in names], index=names)
    paired, clinical, sensitivity = band_stats(pd.DataFrame(observations).T, roster, scores)
    fitted = {"weida_band":dict(maps={k:v.tolist() for k,v in maps.items()}, alignment=alignment,
                               qc=summaries, transitions=transitions, paired=paired, clinical=clinical,
                               pooled_duration_sensitivity=sensitivity)}
    return dict(data_kind="real", patients=25, visits=50, tr=TR, lengths=dict(Counter(map(len,scans.values()))),
                axis=axis, bands=fitted, environment=dict(python=platform.python_version(),numpy=np.__version__,
                                                         pandas=pd.__version__,scipy=scipy.__version__))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", required=True, type=Path)
    parser.add_argument("--clinical-root", required=True, type=Path)
    args=parser.parse_args()
    print(json.dumps(analyze_archive(args.bids_root, args.clinical_root), allow_nan=False))
