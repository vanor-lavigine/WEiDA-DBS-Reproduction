"""Participant-level HC versus STN comparisons on the original fMRIpost series."""
import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from eeg_replacement_105 import BANDS, COVERAGE, DURATION, RATE, TR, fit_band, scan_features
from legacy_clinical25 import read_archive
from peak_states import group_maps, match_to_signed
from stats import holm_adjust

CONDITIONS = ("OFF", "ON", "HC")
SCALE = {DURATION: 1, RATE: 60, COVERAGE: 100}
UNITS = {DURATION: "s", RATE: "episodes/min", COVERAGE: "%"}


def control_hashes(manifest):
    table = pd.read_csv(manifest, dtype=str)
    hc = table.loc[table.condition == "1"].set_index("scan", verify_integrity=True)
    if len(hc) != 15:
        raise ValueError("Expected the 15 verified control scans")
    return hc.sha256.to_dict()


def read_controls(bids_root, project, controls, axis):
    scans, motion = {}, {}
    roster = project / "data/ts_fmripost_nodbs_AAL78_all"
    if len(controls) != 15:
        raise ValueError("Expected 15 controls")
    for name, digest in sorted(controls.items()):
        if not name.startswith("sub-HC") or Path(name).name != name:
            raise ValueError("Unexpected control label")
        if not (roster / (name + "_BOLD_AAL1+PD25_MNI_basic36_sm6.csv")).is_file():
            raise ValueError("Control absent from author archive")
        path = bids_root / "derivatives/fmripost" / name / (name + "_BOLD_AAL1ctx_MNI_basic36_sm6.csv")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Control source fingerprint differs from frozen manifest")
        frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
        raw, = (bids_root / name / "func").glob("*_bold.json")
        conf, = (bids_root / "derivatives/fmriprep" / name / "func").glob("*desc-confounds_timeseries.tsv")
        fd = pd.read_csv(conf, sep="\t").framewise_displacement.to_numpy()
        x = frame.to_numpy()
        if list(frame.columns) != axis or json.loads(raw.read_text())["RepetitionTime"] != TR:
            raise ValueError("Control axis or TR mismatch")
        np.testing.assert_array_equal(frame.index, np.arange(len(frame)))
        if len(x) != len(fd) or not np.isfinite(x).all() or (x.std(0) == 0).any():
            raise ValueError("Invalid control series")
        if not np.isnan(fd[0]) or not np.isfinite(fd[1:]).all() or (fd[1:] < 0).any():
            raise ValueError("Invalid control motion summary")
        scans[name], motion[name] = x, float(fd[1:].mean())
    return scans, motion


def contrast(group, hc, state, metric, condition):
    patient = np.asarray(group, dtype=float)
    control = np.asarray(hc, dtype=float)
    patient = patient[np.isfinite(patient)] * SCALE[metric]
    control = control[np.isfinite(control)] * SCALE[metric]
    n, m = len(patient), len(control)
    if min(n, m) < 2:
        raise ValueError("Insufficient observations in a comparison")
    difference = patient.mean() - control.mean()
    dof = n + m - 2
    var1, var2 = patient.var(ddof=1), control.var(ddof=1)
    pooled_var = ((n-1)*var1 + (m-1)*var2) / dof
    se = np.sqrt(pooled_var * (1/n + 1/m))
    model = stats.ttest_ind(patient, control, equal_var=True)
    np.testing.assert_allclose(model.statistic, difference/se, atol=1e-12)
    width = stats.t.ppf(.975, dof) * se
    return dict(condition=condition, state=state, metric=metric, unit=UNITS[metric],
                n_hc=m, n_patient=n, mean_hc=float(control.mean()), mean_patient=float(patient.mean()),
                sd_hc=float(np.sqrt(var2)), sd_patient=float(np.sqrt(var1)), delta=float(difference),
                ci_low=float(difference-width), ci_high=float(difference+width),
                t=float(model.statistic), df=dof, p_raw=float(model.pvalue),
                p_welch=float(stats.ttest_ind(patient, control, equal_var=False).pvalue))


def measure(prepared, roster, maps):
    observations = {condition: [] for condition in CONDITIONS}
    diagnostics = {condition: [] for condition in CONDITIONS}
    for name, condition in roster.condition.items():
        features, qc, _ = scan_features(prepared[name], maps[condition])
        observations[condition].append(features)
        diagnostics[condition].append(qc)
    result = []
    for condition in ("OFF", "ON"):
        for state in ("all", "M1", "M2", "M3", "M4"):
            for metric in ((DURATION, RATE) if state == "all" else (DURATION, RATE, COVERAGE)):
                groups = [[record[state, metric] for record in observations[c]] for c in (condition, "HC")]
                result.append(contrast(*groups, state, metric, condition))
    corrected = holm_adjust(np.array([r["p_raw"] for r in result]))
    for row, p in zip(result, corrected):
        row["p_holm28"] = float(p)
    qc_summary = {}
    for group, rows in diagnostics.items():
        q = pd.DataFrame(rows)
        qc_summary[group] = dict(peaks=int(q.peaks.sum()), frames=int(q.frames.sum()),
                                 retained=int(q.retained.sum()), mean_peak_gev=float(q.peak_gev.mean()))
    return result, qc_summary


def analyze(bids_root, project, controls):
    scans, motion, axis, _ = read_archive(bids_root, project)
    hc, fd = read_controls(bids_root, project, controls, axis)
    scans.update(hc)
    motion.update(fd)
    if len({x.tobytes() for x in scans.values()}) != 65:
        raise ValueError("Expected 65 distinct scan signals")
    groups = {n: ("HC" if n in hc else "OFF" if "off" in n.lower() else "ON") for n in sorted(scans)}
    roster = pd.Series(groups, name="condition").to_frame()
    counts = dict(Counter(groups.values()))
    assert counts == {"HC":15, "OFF":25, "ON":25}
    prepared, maps, alignment = fit_band(scans, motion, roster, BANDS["weida_band"], None, CONDITIONS)
    primary, qc = measure(prepared, roster, maps)
    weights = [1/counts[c] for c in roster.condition]
    common, _ = group_maps([prepared[n].individual for n in roster.index], weights)
    common, _, _ = match_to_signed(maps["OFF"], common)
    sensitivity, common_qc = measure(prepared, roster, dict.fromkeys(CONDITIONS, common))
    comparisons = [dict(analysis="group_templates", **r) for r in primary]
    comparisons.extend(dict(analysis="common_template", **r) for r in sensitivity)
    pairs = {}
    for a,b in (("OFF","HC"),("ON","HC"),("OFF","ON")):
        pairs[a+"_"+b] = np.abs(np.diag(maps[a] @ maps[b].T)).tolist()
    return dict(data_kind="real", counts=counts, n_people=40, n_scans=65, tr=TR, band_hz=list(BANDS["weida_band"]),
                roi_count=len(axis), axis=axis, comparisons=comparisons, qc=qc, common_qc=common_qc,
                lengths={c:dict(Counter(len(scans[n]) for n in roster.index[roster.condition==c])) for c in CONDITIONS},
                mean_fd_mm={c:float(np.mean([motion[n] for n in roster.index[roster.condition==c]])) for c in CONDITIONS},
                maps={k:v.tolist() for k,v in maps.items()}, common_map=common.tolist(),
                alignment=alignment, matched_abs_r=pairs, hc_matched_sha256=len(controls),
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, pandas=pd.__version__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("bids-root", "author-root", "control-manifest"):
        parser.add_argument("--"+option, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.bids_root, args.author_root, control_hashes(args.control_manifest)), allow_nan=False))
