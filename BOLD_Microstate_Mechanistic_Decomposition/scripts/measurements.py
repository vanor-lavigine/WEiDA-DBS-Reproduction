"""Gap-aware readouts and exact decompositions from frozen frame assignments."""

import numpy as np
import pandas as pd
from scipy.linalg import helmert

from frozen_inputs import (CFG, COUNT_COLUMNS, FEATURES, PRIVATE, STATES, TOL, TR, load_module, require, write_csv)

M4_DESTINATIONS = (1, 2, 3)
COORDINATES = helmert(3)


def divide(numerator, denominator):
    return numerator / denominator if denominator > 0 else np.nan


def within_state_fit(assigned, weights, fit):
    """Use the stored V6 score and the original whole-window GFP squared denominator."""
    total = weights.sum()
    weight = weights[assigned].sum()
    fit_weighted = np.dot(weights[assigned], fit[assigned] ** 2)
    return divide(weight, total), divide(fit_weighted, weight), divide(fit_weighted, total)


def readout(labels, power, fit):
    valid = labels >= 0
    n_valid = int(valid.sum())
    seconds = n_valid * TR
    adjacent = valid[:-1] & valid[1:]
    counts = np.bincount(labels[:-1][adjacent] * 4 + labels[1:][adjacent], minlength=16).reshape(4, 4)
    starts = np.flatnonzero(valid & np.r_[True, labels[1:] != labels[:-1]])
    ends = np.flatnonzero(valid & np.r_[labels[:-1] != labels[1:], True]) + 1
    left = ~np.r_[False, valid][starts]
    right = ~np.r_[valid, False][ends]
    complete = ~(left | right)
    roi_labels = labels[valid]
    valid_blocks = int(np.count_nonzero(valid & ~np.r_[False, valid[:-1]]))
    out = {"valid_frames": n_valid, "valid_seconds": seconds, "valid_blocks": valid_blocks}
    for state in STATES:
        assigned = roi_labels == state - 1
        segment = labels[starts] == state - 1
        n_frames, n_segments = int(assigned.sum()), int(segment.sum())
        w, f, gev = within_state_fit(assigned, power[valid], fit[valid])
        n_complete = int((segment & complete).sum())
        out.update({
            f"frames_{state}": n_frames, f"episodes_{state}": n_segments, f"complete_episodes_{state}": n_complete,
            f"left_boundary_episodes_{state}": int((segment & left).sum()),
            f"right_boundary_episodes_{state}": int((segment & right).sum()),
            f"coverage_{state}": divide(n_frames, n_valid), f"occurrence_{state}": divide(n_segments, seconds),
            f"duration_{state}": divide(n_frames * TR, n_segments), f"gev_{state}": gev,
        })
        if state == 1:
            out.update(m1_weight=w, m1_fit=f, m1_gev=gev)
        if state == 4:
            complete_frames = int((ends - starts)[segment & complete].sum())
            out.update(m4_complete_frames=complete_frames,
                       m4_complete_coverage=divide(complete_frames, n_valid),
                       m4_complete_occurrence=divide(n_complete, seconds),
                       dwell_edge_mean_4=divide(complete_frames * TR, n_complete))
    if not np.isfinite(out["m1_fit"]):
        out["m1_fit_status"] = "Undefined M1 GFP squared total"
    else:
        out["m1_fit_status"] = "OK"
    out.update(zip(COUNT_COLUMNS, counts.ravel(), strict=True))
    exits = counts[3, :3]
    n_exit = int(exits.sum())
    require(n_exit == out["episodes_4"] - out["right_boundary_episodes_4"], "M4 exit identity failed")
    out.update(m4_n_exits=n_exit, m4_exit_rate_min=divide(n_exit * 60, seconds),
               m4_exit_intensity_min=divide(n_exit * 60, out["frames_4"] * TR),
               m4_exit_rate_status="OK" if n_valid else "No valid time",
               m4_exit_destination_status="OK" if n_exit else "No observed M4 exit")
    for j, numerator in zip(M4_DESTINATIONS, exits, strict=True):
        out[f"m4_q4{j}"] = divide(numerator, n_exit)
    if n_exit:
        smooth = exits + CFG["composition_pseudocount"]
        ilr = COORDINATES @ np.log(smooth / smooth.sum())
    else:
        ilr = np.full(2, np.nan)
    out["m4_ilr_1"], out["m4_ilr_2"] = ilr
    return out


def archived_frames(path, mapping):
    with np.load(path, allow_pickle=False) as cached:
        raw = cached["labels"]
        corr = cached["corr"]
        power = cached["gfp"] ** 2
    n = CFG["cohort"]["acquired_frames"]
    require(raw.shape == (n,) and corr.shape == (n, 4) and power.shape == (n,), "Unexpected frame archive shape")
    require(np.issubdtype(raw.dtype, np.integer) and np.isin(raw, (-1, 0, 1, 2, 3)).all(), "Invalid frozen label")
    valid = raw >= 0
    require(np.isfinite(power[valid]).all() and np.isfinite(corr[valid]).all(), "Nonfinite frozen GFP or correlation")
    fit, labels = np.zeros(n), np.full(n, -1, dtype=int)
    fit[valid] = np.clip(corr[valid, raw[valid]], 0, 1)
    labels[valid] = mapping[raw[valid]]
    return labels, power, fit, corr


def roi_acf(filtered, valid, max_lag):
    """Pool lagged products over pairs in the same uninterrupted valid block."""
    roi_count = filtered.shape[1]
    acf = np.full((max_lag, roi_count), np.nan)
    support = np.zeros(max_lag, dtype=int)
    bounds = np.flatnonzero(np.diff(np.r_[False, valid, False].astype(int))).reshape(-1, 2)
    if not len(bounds):
        return acf, support
    central = filtered - filtered[valid].mean(axis=0)
    for lag in range(1, max_lag + 1):
        blocks = [(central[a:b-lag], central[a+lag:b]) for a, b in bounds if b-a > lag]
        if not blocks:
            break
        earlier = np.vstack([left for left, _ in blocks])
        later = np.vstack([right for _, right in blocks])
        support[lag - 1] = len(earlier)
        a, b = np.sum(earlier ** 2, axis=0), np.sum(later ** 2, axis=0)
        denom = np.sqrt(a * b)
        acf[lag - 1] = np.divide(np.sum(earlier * later, axis=0), denom,
                                 out=np.full(roi_count, np.nan), where=denom > 0)
    return acf, support


def load_roi_diagnostics(inputs):
    """Use the frozen V3 loader for existing ROI derivatives and their clock."""
    module = load_module("_v7_v3_input_data", inputs.v3_loader)
    require(module.TR == TR and module.ROOT.resolve() == inputs.workspace, "Original ROI reader root or TR mismatch")
    scans = {row["scan"]: row for row in module.load_scans()}
    require(set(scans) == set(inputs.frame_files), "ROI series roster disagrees with frozen frames")
    return module, scans


def diagnose_roi(scan, data, filtered, labels, power, corr, inputs):
    valid = labels >= 0
    require(data["subject"] == scan.subject and data["condition"] == scan.condition,
            "ROI diagnostic series does not have the frozen scan identity")
    # Cross-check ROI order and acquisition clock against the saved per-frame scores.
    roi_valid = np.isfinite(filtered).all(axis=1)
    require(not np.any(valid & ~roi_valid), "A labeled frame has no original ROI series")
    scaled = filtered[roi_valid] - filtered[roi_valid].mean(axis=0)
    scaled /= scaled.std(axis=0, ddof=1)
    scaled -= scaled.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(scaled, axis=1)
    aligned = np.zeros_like(filtered)
    aligned[roi_valid] = scaled / lengths[:, None]
    amplitude = np.zeros(len(filtered))
    amplitude[roi_valid] = lengths / np.sqrt(filtered.shape[1])
    np.testing.assert_allclose(amplitude[valid] ** 2, power[valid], atol=TOL, rtol=0)
    np.testing.assert_allclose(np.abs(aligned[valid] @ inputs.bank[scan.fold].T),
                               corr[valid], atol=TOL, rtol=0)
    coefficients, support = roi_acf(filtered, valid, CFG["acf_max_lag_frames"])
    row_ids = ("scan", "subject", "condition")
    records = []
    for j, region in enumerate(inputs.roi_names):
        positive = np.flatnonzero(coefficients[:, j] < CFG["acf_efold"])
        records.append({
            **{key: getattr(scan, key) for key in row_ids}, "roi": region, "acf_lag1": coefficients[0, j],
            "first_efold_seconds": (positive[0] + 1) * TR if len(positive) else np.nan,
            "last_supported_lag_seconds": np.count_nonzero(support) * TR,
            "efold_observed": bool(len(positive)), "adjacent_roi_pairs": support[0],
        })
    return records


def measure_inputs(inputs):
    module, series = load_roi_diagnostics(inputs)
    records, acf_records = [], []
    for idx, scan in enumerate(inputs.scans.itertuples(index=False), start=1):
        labels, power, fit, corr = archived_frames(inputs.frame_files[scan.scan], inputs.frame_order[scan.fold])
        filtered = module.transform(series[scan.scan]["x"], CFG["baseline"]["band"])
        acf_records.extend(diagnose_roi(scan, series[scan.scan], filtered, labels, power, corr, inputs))
        for window in CFG["windows"]:
            start, stop = window["start"], window["stop"]
            result = readout(labels[start:stop], power[start:stop], fit[start:stop])
            records.append(dict(scan=scan.scan, subject=scan.subject, condition=scan.condition, fold=scan.fold,
                                window=window["window"], start=start, stop=stop, **result))
        if idx % 13 == 0:
            print(f"Frozen frame and ROI readouts: {idx}/{len(inputs.scans)} scans", flush=True)
    windows = pd.DataFrame(records)
    roi = pd.DataFrame(acf_records)
    full = windows.loc[windows.window.eq("Full")].set_index("scan", verify_integrity=True).sort_index()
    frozen = inputs.scans.set_index("scan").loc[full.index]
    for key in [*FEATURES, *COUNT_COLUMNS, "valid_frames", "valid_blocks", "dwell_edge_mean_4"]:
        np.testing.assert_allclose(full[key], frozen[key], atol=TOL, rtol=0, equal_nan=True)
    m1_geometrical = (full.m1_weight * full.m1_fit).mask(full.m1_weight.eq(0), 0)
    np.testing.assert_allclose(full.m1_gev, m1_geometrical, atol=TOL, rtol=0, equal_nan=True)
    for prefix, duration in [("", "duration_4"), ("m4_complete_", "dwell_edge_mean_4")]:
        coverage = full[f"{prefix}coverage{'_4' if not prefix else ''}"]
        occurrence = full[f"{prefix}occurrence{'_4' if not prefix else ''}"]
        np.testing.assert_allclose(full[duration], coverage / occurrence.where(occurrence > 0),
                                   atol=TOL, rtol=0, equal_nan=True)
    write_csv(windows, PRIVATE / "results/Scan_Window_Readouts.csv")
    write_csv(roi, PRIVATE / "results/ROI_Temporal_Autocorrelation.csv")
    return full.reset_index(), windows, roi


def stable_percent(contribution, total, threshold):
    return np.divide(contribution * 100, total, out=np.full_like(total, np.nan, dtype=float),
                     where=np.isfinite(total) & (np.abs(total) >= threshold))


def shapley(w_on, w_off, f_on, f_off):
    return (w_on - w_off) * (f_on + f_off) / 2, (f_on - f_off) * (w_on + w_off) / 2


def log_ratio(on, off):
    out = np.full_like(on, np.nan, dtype=float)
    keep = (on > 0) & (off > 0) & np.isfinite(on) & np.isfinite(off)
    out[keep] = np.log(on[keep]) - np.log(off[keep])
    return out


def paired_readouts(full, inputs):
    paired = pd.DataFrame(index=inputs.pairs.index)
    metrics = ["m1_gev", "m1_weight", "m1_fit", "coverage_1", "frames_1", "duration_4", "coverage_4", "occurrence_4",
               "m4_complete_coverage", "m4_complete_occurrence", "dwell_edge_mean_4", "m4_complete_frames",
               "m4_n_exits", "m4_exit_rate_min", "m4_exit_intensity_min", "m4_q41", "m4_q42", "m4_q43",
               "m4_ilr_1", "m4_ilr_2", "frames_4", "episodes_4", "complete_episodes_4", "left_boundary_episodes_4",
               "right_boundary_episodes_4", "valid_frames", "valid_seconds", "valid_blocks"]
    for side in ("off", "on"):
        frame = full.loc[full.condition.eq(side.upper())].set_index("subject").loc[paired.index]
        paired = paired.join(frame[metrics].add_prefix(f"{side}_"))
    for key in metrics:
        paired[f"delta_{key}"] = paired[f"on_{key}"] - paired[f"off_{key}"]
    off, on = paired.loc[:, "off_m1_weight"].to_numpy(), paired.loc[:, "on_m1_weight"].to_numpy()
    fit_off, fit_on = paired.loc[:, "off_m1_fit"].to_numpy(), paired.loc[:, "on_m1_fit"].to_numpy()
    a, b = shapley(on, off, fit_on, fit_off)
    paired["m1_shapley_w_pp"], paired["m1_shapley_f_pp"] = a * 100, b * 100
    paired["m1_gev_delta_pp"] = paired.delta_m1_gev * 100
    calculable = np.isfinite(a + b)
    np.testing.assert_allclose((a + b)[calculable], paired.delta_m1_gev.to_numpy()[calculable], atol=TOL, rtol=0)
    for k in ("w", "f"):
        paired[f"m1_{k}_share_pct"] = stable_percent(paired[f"m1_shapley_{k}_pp"].to_numpy(),
                                                     paired.m1_gev_delta_pp.to_numpy(), CFG["share_min_abs_gev_pp"])
    for suffix, columns in {"": ("coverage_4", "occurrence_4", "duration_4"),
                            "_edge": ("m4_complete_coverage", "m4_complete_occurrence", "dwell_edge_mean_4")}.items():
        a, b, d = [log_ratio(paired[f"on_{k}"].to_numpy(), paired[f"off_{k}"].to_numpy()) for k in columns]
        paired[f"m4_log_coverage{suffix}"] = a
        paired[f"m4_negative_log_occurrence{suffix}"] = -b
        paired[f"m4_log_duration{suffix}"] = d
        np.testing.assert_allclose(a - b, d, atol=TOL, rtol=0, equal_nan=True)
    paired["fd"] = inputs.pairs.on_mean_fd - inputs.pairs.off_mean_fd
    paired["mean_std_dvars"] = inputs.pairs.on_std_dvars - inputs.pairs.off_std_dvars
    paired["m4_exit_rate_min"] = paired.delta_m4_exit_rate_min
    paired["m4_ilr_1"], paired["m4_ilr_2"] = paired.delta_m4_ilr_1, paired.delta_m4_ilr_2
    paired["m1_decomposition_status"] = np.where(calculable, "OK", "M1 fit undefined in one or both scans")
    paired["m4_log_decomposition_status"] = np.where(paired.m4_log_duration.notna(), "OK", "M4 absent in a scan")
    paired["m4_boundary_status"] = np.where(paired.m4_log_duration_edge.notna(), "OK", "No complete M4 episode in a scan")
    paired["m4_exit_composition_status"] = np.where(paired.m4_ilr_1.notna(), "OK", "No actual M4 exit in a scan")
    write_csv(paired.reset_index(), PRIVATE / "results/Paired_Decompositions.csv")
    return paired
