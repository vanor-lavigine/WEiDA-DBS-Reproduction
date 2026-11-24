"""Patient-level motor correlations of global BOLD phase metastability."""
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
from legacy_clinical25 import read_archive
from li_clinical_stats import correlation
from stats import holm_adjust


TR = 2.1
BAND = (0.010, 0.027)
WEIDA_BAND = (0.04, 0.07)
EDGE = 24


def scan_metric(x, band=WEIDA_BAND):
    centered = signal.detrend(x, axis=0)
    f, spectrum = signal.periodogram(centered, fs=1/TR, axis=0, window="hann", detrend=False)
    selected = (f >= band[0]) & (f <= band[1])
    fraction = spectrum[selected].sum(axis=0) / spectrum[1:].sum(axis=0)
    if not np.all((fraction > 0) & (fraction <= 1)):
        raise ValueError("Invalid input band spectral power")
    b, a = signal.butter(2, band, btype="bandpass", fs=1/TR)
    filtered = signal.filtfilt(b, a, centered, axis=0, padlen=12)
    analytic = signal.hilbert(filtered, axis=0)
    if not np.isfinite(analytic).all() or (abs(analytic) == 0).any():
        raise ValueError("Undefined analytic phase")
    phase = np.angle(analytic)
    order = abs(np.exp(1j*phase).mean(axis=1))
    for i in (0, len(x)//2, len(x)-1):
        expected = np.cos(phase[i, :, None] - phase[i, None, :]).mean()
        np.testing.assert_allclose(order[i]**2, expected, atol=1e-12, rtol=0)
    if not np.all((order >= 0) & (order <= 1+1e-12)):
        raise ValueError("Out-of-range Kuramoto order parameter")
    return dict(metric=order.std(ddof=1), trimmed=order[EDGE:-EDGE].std(ddof=1),
                median_roi_band_power_fraction=float(np.median(fraction)))


def partial_correlation(x, y, *covariates):
    design = np.column_stack([np.ones(len(x)), *covariates])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("Rank-deficient adjustment")
    observations = np.column_stack([x, y])
    residual = observations - design @ np.linalg.lstsq(design, observations, rcond=None)[0]
    r = stats.pearsonr(residual[:, 0], residual[:, 1]).statistic
    df = len(x) - design.shape[1] - 1
    return dict(r=r, p=2*stats.t.sf(abs(r)*np.sqrt(df/(1-r**2)), df), df=df)


def check_correlation(x, y, fd, trimmed, permutations):
    result = correlation(x, y)
    partial = partial_correlation(x, y, fd)
    a, b = x-x.mean(), y-y.mean()
    null = np.abs(b[permutations] @ a)
    permutation_p = (1 + np.count_nonzero(null >= abs(b @ a)-1e-14))/(len(null)+1)
    return dict(**result, spearman=stats.spearmanr(x,y).statistic, partial_fd_r=partial["r"],
                partial_fd_p=partial["p"],
                trimmed=correlation(trimmed, y), permutation_p=permutation_p)


def analyze_archive(bids_root, project, band=WEIDA_BAND):
    scans, motion, axis, _ = read_archive(bids_root, project)
    if len(scans) != 50 or len({x.tobytes() for x in scans.values()}) != 50:
        raise ValueError("Expected 50 unique scans")
    rows = []
    for run, x in sorted(scans.items()):
        match = re.fullmatch(r"sub-STN(off|on)(\d+)", run, re.I)
        rows.append(dict(run=run, condition=match[1].upper(), person=match[2],
                         fd=motion[run], **scan_metric(x, band)))
    frame = pd.DataFrame(rows)
    salt = secrets.token_hex(32)
    main, duplicate = [clinical_visits(project, folder, series, salt) for folder,series in VERSIONS[:2]]
    pd.testing.assert_series_equal(main.score, duplicate.score)
    np.testing.assert_array_equal(main.score, main[list(ENGLISH_DOMAINS)].sum(axis=1))
    frame["score"] = [main.score.at[scan_key(label, salt)] for label in frame.run]
    wide = frame.pivot(index="person", columns="condition", values=["metric","trimmed","fd","score"])
    if wide.shape != (25, 8) or not np.isfinite(wide.to_numpy()).all():
        raise ValueError("Missing paired measurements")
    rng = np.random.default_rng(20261124)
    permutations = np.array([rng.permutation(len(wide)) for _ in range(19999)])
    effects = []
    for name in ("OFF", "ON", "change"):
        if name == "change":
            contrast = wide.xs("ON", axis=1, level=1)-wide.xs("OFF", axis=1, level=1)
            y = -contrast.score
        else:
            contrast = wide.xs(name, axis=1, level=1)
            y = contrast.score
        effects.append(dict(contrast=name, **check_correlation(
            contrast.metric.to_numpy(), y.to_numpy(), contrast.fd.to_numpy(),
            contrast.trimmed.to_numpy(), permutations)))
    for effect, adjusted in zip(effects, holm_adjust([row["p_raw"] for row in effects])):
        effect["p_holm3"] = float(adjusted)
    off = wide.xs("OFF", axis=1, level=1)
    on = wide.xs("ON", axis=1, level=1)
    dx, benefit = on.metric-off.metric, off.score-on.score
    influence = {}
    for name, x, y in [("OFF", off.metric, off.score), ("ON", on.metric, on.score),
                       ("change", dx, benefit)]:
        estimates = [stats.pearsonr(x.drop(i), y.drop(i)).statistic for i in wide.index]
        influence[name] = [min(estimates), max(estimates)]
    posthoc = dict(leave_one_patient_out_r_range=influence,
                   change_adjusted_off_metric_and_score=partial_correlation(dx, benefit, off.metric, off.score))
    return dict(data_kind="real", patients=len(wide), scans=len(scans), roi_count=len(axis),
                frames=sum(map(len,scans.values())), lengths=dict(Counter(map(len,scans.values()))),
                TR=TR, band_hz=band, global_sd_mean=frame.groupby("condition").metric.mean().to_dict(),
                spectral_qc=dict(n_scan_roi=len(axis)*len(scans),
                                 median_of_scan_median_roi_power_fraction=frame.median_roi_band_power_fraction.median()),
                effects=effects, posthoc=posthoc, environment=dict(python=platform.python_version(), scipy=scipy.__version__,
                                                 numpy=np.__version__, pandas=pd.__version__))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--bids-root", type=Path, required=True)
    args.add_argument("--clinical-root", type=Path, required=True)
    args.add_argument("--band", choices=("slow5", "weida"), default="weida")
    cfg = args.parse_args()
    selected = BAND if cfg.band == "slow5" else WEIDA_BAND
    print(json.dumps(analyze_archive(cfg.bids_root, cfg.clinical_root, selected), indent=2, allow_nan=False))
