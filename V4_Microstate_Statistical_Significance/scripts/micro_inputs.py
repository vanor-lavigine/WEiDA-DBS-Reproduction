"""Frozen scan identity checks, template matching and A3 feature readouts."""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.linalg import helmert


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
V3 = ROOT / "WEiDA-DBS-V3" / "V3_BOLD_Microstate_Method_Optimization"
RELIABILITY = HERE.parent / "V4_BOLD_Microstate_Reliability"
PREVIOUS = HERE.parent / "V4_WEiDA_Statistical_Framework_Microstate" / "scripts"
sys.path[:0] = [str(RELIABILITY / "scripts"), str(V3 / "scripts")]
os.environ["WEIDA_105_ROOT"] = str(ROOT)

from v4_models import cohort, match_banks, peak_labels, prepare_scans, temporal_labels, TR


SETTINGS = json.loads((HERE / "configs/protocol.json").read_text(encoding="utf-8"))
STATE = range(1, 5)
EDGE = [(i, j) for i in STATE for j in STATE if i != j]
COORD = helmert(4)
STATE_FAMILIES = ("coverage", "duration", "occurrence", "gev", "dispersion")
FEATURES = tuple(f"{kind}_{i}" for kind in STATE_FAMILIES for i in STATE) + tuple(f"transition_{i}_{j}" for i in STATE for j in STATE)
FEATURES += ("switching_rate",) + tuple(f"ilr_{j}" for j in range(1, 4)) + tuple(f"sorted_ilr_{j}" for j in range(1, 4))
AXIS_COLUMNS = [f"ROI_{i}" for i in range(1, 79)]


@dataclass
class Inputs:
    scans: list
    split: np.ndarray
    prepared: list
    banks: dict
    matches: dict
    features: pd.DataFrame


def scheme(readout, order):
    """Normalize matched states to the order of the first frozen G1 fold."""
    labels, corr, amplitude = readout
    good = labels >= 0
    frames = int(good.sum())
    if not frames:
        return np.full(len(FEATURES), np.nan)
    starts = good & np.r_[True, labels[1:] != labels[:-1]]
    adjacent = good[:-1] & good[1:]
    trans = np.bincount(4 * labels[:-1][adjacent] + labels[1:][adjacent], minlength=16).reshape(4, 4)
    ordered = trans[np.ix_(order, order)]
    source = ordered.sum(axis=1, keepdims=True)
    transition = np.divide(ordered, source, out=np.full((4, 4), np.nan), where=source > 0)
    state = labels[good]
    total = np.bincount(state, minlength=4)[order]
    segments = np.bincount(labels[starts], minlength=4)[order]
    duration = np.divide(TR * total, segments, out=np.full(4, np.nan), where=segments > 0)
    similarity = np.clip(corr[good, state], 0, 1)
    power = amplitude[good] ** 2
    gev = np.bincount(state, weights=power * similarity ** 2, minlength=4)[order] / power.sum()
    distances = np.bincount(state, weights=np.sqrt(2 * (1 - similarity)), minlength=4)[order]
    dispersion = np.divide(distances, total, out=np.full(4, np.nan), where=total > 0)
    return np.r_[total / frames, duration, segments / (TR * frames), gev, dispersion, transition.ravel(),
                 (ordered.sum() - ordered.trace()) / (TR * frames), COORD @ np.log(total + .5),
                 COORD @ np.log(np.sort(total) + .5)]


def segment_a3(prepared, g0, g1, penalty):
    corr0 = np.abs(prepared.units @ g0.T)
    corr1 = np.abs(prepared.units @ g1.T)
    window = (peak_labels(prepared, corr0) >= 0) & (peak_labels(prepared, corr1) >= 0)
    norm = np.mean(prepared.amplitude[prepared.valid] ** 2)
    labels = temporal_labels(corr1, prepared.amplitude, window, penalty, normalization=norm)
    return labels, corr1, prepared.amplitude


def feature_mask(stable, match_composition=None):
    edge = np.outer(stable, stable).ravel()
    global_match = stable.all() if match_composition is None else match_composition
    return np.r_[np.tile(stable, len(STATE_FAMILIES)), edge, True, np.repeat(global_match, 3), np.ones(3, bool)]


def load_inputs():
    cfg = SETTINGS
    if cfg["k"] != 4 or cfg["tr_seconds"] != TR or cfg["main_method"] != "A3":
        raise ValueError("Unexpected fixed state model")
    scans, split = cohort()
    check = pd.DataFrame([{k: v for k, v in scan.items() if k != "x"} for scan in scans]).assign(fold=split)
    assert len(check) == check.scan.nunique() == 65 and check.subject.nunique() == 40
    assert check.condition.value_counts().to_dict() == {"OFF": 25, "ON": 25, "HC": 15}
    assert check.groupby("subject").fold.nunique().eq(1).all()
    assert set(check.loc[check.condition.eq("ON"), "subject"]) == set(check.loc[check.condition.eq("OFF"), "subject"])
    prepared = prepare_scans(scans)
    archive = RELIABILITY / "private"
    with np.load(archive / "V4_Maps.npz") as saved:
        banks = {k: saved[k] for k in ("G0", "G1")}
    match = match_banks(banks["G1"])
    old = pd.read_csv(archive / "V4_Scan_Metrics.csv", low_memory=False)
    old = old.loc[old.method.eq("A3") & old.window.eq("common")].set_index("scan", verify_integrity=True)
    rows = []
    for i, scan in enumerate(scans):
        h = int(split[i])
        name = scan["scan"]
        oldrow = old.loc[name]
        assert oldrow.fold == h and oldrow.subject == scan["subject"] and oldrow.condition == scan["condition"]
        assert np.isclose(oldrow.fd, scan["fd"], rtol=0, atol=1e-14) and oldrow.penalty == cfg["lambda"]
        current = segment_a3(prepared[i], banks["G0"][h], banks["G1"][h], cfg["lambda"])
        with np.load(archive / "frames" / "A3" / f"{name}.npz") as saved:
            np.testing.assert_array_equal(current[0], saved["labels"])
            np.testing.assert_allclose(current[1], saved["corr"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(current[2], saved["gfp"], atol=1e-12, rtol=0)
        data = scheme(current, match["order"][h])
        original = [oldrow[f"{kind}_{j+1}"] for kind in STATE_FAMILIES[:4] for j in match["order"][h]]
        np.testing.assert_allclose(data[:16], original, atol=1e-12, rtol=0, equal_nan=True)
        np.testing.assert_allclose(data[FEATURES.index("switching_rate")], oldrow.switching_rate, atol=1e-12)
        n = int(np.sum(current[0] >= 0))
        assert n == oldrow.n_labeled_frames
        data[~feature_mask(match["stable"])] = np.nan
        rows.append(dict(scan=name, subject=scan["subject"], condition=scan["condition"], fold=h,
                         valid_frames=n, fd=scan["fd"], **dict(zip(FEATURES, data))))
    return Inputs(scans, split, prepared, banks, match, pd.DataFrame(rows))


def write_templates(data, results):
    atlas = pd.read_csv(V3 / "manifests" / "ROI_Axis.csv")
    bank = data.banks["G1"]
    fold_rows = []
    for fold, maps in enumerate(bank):
        for i, j in enumerate(data.matches["order"][fold]):
            aligned = maps[j] * (1 if maps[j] @ bank[0, i] >= 0 else -1)
            fold_rows.append(dict(fold=fold, state=i+1, original_state=j+1, stable=data.matches["stable"][i],
                                  **dict(zip(atlas.AAL78, aligned))))
    frame = pd.DataFrame(fold_rows)
    frame.to_csv(results / "Fold_Templates.csv", index=False)
    frame.loc[frame.fold.eq(0)].to_csv(results / "Reference_Templates.csv", index=False)
    pd.DataFrame(data.matches["rows"]).to_csv(results / "State_Matching.csv", index=False)
