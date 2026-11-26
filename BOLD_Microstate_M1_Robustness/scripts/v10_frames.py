"""Keep the original TR clock and A3 cost scale while censoring motion."""

import numpy as np
import pandas as pd

from v10_inputs import CFG, PARAM, TOL, prior
from v9_measurements import decompose as v9_decompose

ORIGINAL = "original_window"
FORMAL = CFG["formal_quality_mode"]
MASK_ONLY = CFG["descriptive_quality_mode"]
MODES = (ORIGINAL, MASK_ONLY, FORMAL)
SIDES = ("OFF", "ON")
SCAN_METRICS = ("coverage", "weight", "relative_amplitude", "fit", "gev")


def motion_mask(fd):
    rule = CFG["fd"]
    valid = np.ones(len(fd), dtype=bool)
    for i in np.flatnonzero(fd > rule["strictly_greater_than"]):
        start = max(0, i - rule["preceding_frames"])
        stop = min(len(fd), i + rule["following_frames"] + 1)
        valid[start:stop] = False
    return valid


def n_blocks(mask):
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def statistics_for_labels(labels, gfp, scores, mapping):
    """Use the retained window as the denominator of descriptive state metrics."""
    used = labels >= 0
    n = int(used.sum())
    c, w, r, f, gev = np.full((5, PARAM["k"]), np.nan)
    if not n:
        return c, w, r, f, gev, "No retained frames"
    power = gfp[used] ** 2
    total = power.sum()
    if not np.isfinite(total) or total <= 0:
        return c, w, r, f, gev, "No valid whole-window GFP squared denominator"
    raw = labels[used]
    aligned = mapping[raw]
    fit_power = power * np.clip(scores[used, raw], 0, 1) ** 2
    counts = np.bincount(aligned, minlength=PARAM["k"])
    weight = np.bincount(aligned, weights=power, minlength=PARAM["k"])
    fit = np.bincount(aligned, weights=fit_power, minlength=PARAM["k"])
    c, w, gev = counts/n, weight/total, fit/total
    defined = (counts > 0) & (weight > 0)
    r[defined] = (weight[defined]/counts[defined]) / (total/n)
    f[defined] = fit[defined]/weight[defined]
    np.testing.assert_allclose((c*r)[defined], w[defined], atol=TOL, rtol=0)
    np.testing.assert_allclose((w*f)[defined], gev[defined], atol=TOL, rtol=0)
    if counts[0] == 0:
        reason = "No retained M1 frames"
    elif not defined[0]:
        reason = "No valid M1 GFP squared denominator"
    else:
        reason = ""
    return c, w, r, f, gev, reason


def replay(inps):
    rows = []
    evidence = {
        "archived_s1_label_replays_exact": 0,
        "v4_censored_window_replays_exact": 0,
        "censored_blocks_replayed_independently": 0,
        "max_archived_gfp_abs_error": 0.0,
        "max_g1_fit_correlation_abs_error": 0.0,
        "max_censored_v4_gev_abs_error": 0.0,
    }
    decoder = inps.decoder
    for scan in inps.cohort.itertuples(index=False):
        with np.load(inps.frames[scan.scan], allow_pickle=False) as frozen:
            labels, gfp, scores = (frozen[key] for key in ("labels", "gfp", "corr"))
        n, k = PARAM["original_clock_frames"], PARAM["k"]
        prior.require(labels.shape == gfp.shape == (n,) and scores.shape == (n, k)
                      and np.issubdtype(labels.dtype, np.integer) and np.isin(labels, np.arange(-1, k)).all(),
                      "Invalid frozen labels, GFP or similarity arrays")
        common = labels >= 0
        roi = inps.source_scans[scan.scan]["x"]
        native = decoder.prepare(decoder.transform(roi, PARAM["band"]), PARAM["normalization"])
        prior.require(not common[:scan.source_start].any() and not native.valid[:scan.source_start].any()
                      and np.all(~common | native.valid) and int(common.sum()) == scan.valid_frames,
                      "Original ROI and label windows are misaligned")
        np.testing.assert_allclose(native.amplitude, gfp, atol=TOL, rtol=0, equal_nan=True)
        reference_scores = np.abs(native.units[native.valid] @ inps.banks[scan.fold].T)
        np.testing.assert_allclose(reference_scores, scores[native.valid], atol=TOL, rtol=0)
        evidence["max_archived_gfp_abs_error"] = max(evidence["max_archived_gfp_abs_error"],
                                                     float(np.max(abs(native.amplitude[native.valid] - gfp[native.valid]))))
        evidence["max_g1_fit_correlation_abs_error"] = max(evidence["max_g1_fit_correlation_abs_error"],
                                                           float(np.max(abs(reference_scores - scores[native.valid]))))
        scale = float(np.mean(gfp[native.valid] ** 2))
        prior.require(np.isfinite(scale) and scale > 0, "Invalid original A3 normalization constant")
        refit = decoder.temporal_labels(scores, gfp, common, PARAM["lambda"], normalization=scale)
        np.testing.assert_array_equal(refit, labels)
        evidence["archived_s1_label_replays_exact"] += 1

        fd = inps.fd[scan.scan]
        allowed = motion_mask(fd)
        np.testing.assert_array_equal(allowed, decoder.fd_mask(fd))
        retain = common & allowed
        baseline_labels = np.where(retain, labels, -1)
        redone = decoder.temporal_labels(scores, gfp, retain, PARAM["lambda"], normalization=scale)
        np.testing.assert_array_equal(redone >= 0, retain)
        edges = np.flatnonzero(np.diff(np.r_[False, retain, False]))
        for lo, hi in edges.reshape(-1, 2):
            separate = decoder.temporal_labels(scores[lo:hi], gfp[lo:hi], np.ones(hi - lo, dtype=bool),
                                               PARAM["lambda"], normalization=scale)
            np.testing.assert_array_equal(redone[lo:hi], separate)
            evidence["censored_blocks_replayed_independently"] += 1
        label_changes = int(np.count_nonzero(redone[retain] != baseline_labels[retain]))
        mapping = inps.mapping[scan.fold]
        for mode, output in zip(MODES, (labels, baseline_labels, redone), strict=True):
            valid = output >= 0
            frames = int(valid.sum())
            c, w, r, f, gev, reason = statistics_for_labels(output, gfp, scores, mapping)
            if mode == ORIGINAL:
                np.testing.assert_allclose(c, [getattr(scan, f"coverage_{j+1}") for j in range(k)],
                                           atol=TOL, rtol=0, equal_nan=True)
                np.testing.assert_allclose(gev, [getattr(scan, f"gev_{j+1}") for j in range(k)],
                                           atol=TOL, rtol=0, equal_nan=True)
            if mode == FORMAL:
                old = inps.old_qc.loc[scan.scan]
                prior.require(frames == old.n_labeled_frames, "V4 FD-censored retained TR count differs")
                raw_gev = np.array([old[f"gev_{j+1}"] for j in range(k)])
                np.testing.assert_allclose(c[mapping], [old[f"coverage_{j+1}"] for j in range(k)],
                                           atol=TOL, rtol=0, equal_nan=True)
                np.testing.assert_allclose(gev[mapping], raw_gev, atol=TOL, rtol=0, equal_nan=True)
                if frames:
                    evidence["max_censored_v4_gev_abs_error"] = max(evidence["max_censored_v4_gev_abs_error"],
                                                                   float(np.max(abs(gev[mapping] - raw_gev))))
                evidence["v4_censored_window_replays_exact"] += 1
            rows.append(dict(
                mode=mode, scan=scan.scan, subject=scan.subject, condition=scan.condition, fold=scan.fold,
                mean_fd_mm=scan.fd, mean_std_dvars=scan.mean_std_dvars,
                source_start_tr=scan.source_start, normalization_gfp_squared=scale,
                original_frames=int(common.sum()), retained_frames=frames,
                removed_frames=int(common.sum())-frames, retained_seconds=frames*PARAM["tr_s"],
                valid_blocks=n_blocks(valid), original_fd_spikes=int(np.sum(fd > CFG["fd"]["strictly_greater_than"])),
                redecoded_labels_changed=label_changes if mode == FORMAL else 0,
                measurement_status="OK" if not reason else reason,
                **dict(zip(SCAN_METRICS, (c[0], w[0], r[0], f[0], gev[0]), strict=True)),
            ))
    return pd.DataFrame(rows), evidence


def contributions(paired):
    """Compute hierarchical Shapley components within one retained window."""
    paired = paired.copy()
    support = paired.off_measurement_status.eq("OK") & paired.on_measurement_status.eq("OK")
    f = (paired.off_fit + paired.on_fit)/2
    w = (paired.off_weight + paired.on_weight)/2
    c = (paired.off_coverage + paired.on_coverage)/2
    r = (paired.off_relative_amplitude + paired.on_relative_amplitude)/2
    paired["K_C"] = (100 * (paired.on_coverage-paired.off_coverage) * r * f).where(support)
    paired["K_R"] = (100 * (paired.on_relative_amplitude-paired.off_relative_amplitude) * c * f).where(support)
    paired["C_W"] = (100 * (paired.on_weight-paired.off_weight) * f).where(support)
    paired["C_F"] = (100 * (paired.on_fit-paired.off_fit) * w).where(support)
    paired["delta_gev_pp"] = 100 * (paired.on_gev - paired.off_gev)
    paired["delta_coverage_pp"] = 100 * (paired.on_coverage - paired.off_coverage)
    paired["delta_fd_mm"] = paired.on_mean_fd_mm - paired.off_mean_fd_mm
    paired["delta_dvars"] = paired.on_mean_std_dvars - paired.off_mean_std_dvars
    paired["delta_retained_seconds"] = paired.on_retained_seconds - paired.off_retained_seconds
    a = (paired.K_C + paired.K_R - paired.C_W)[support]
    b = (paired.K_C + paired.K_R + paired.C_F - paired.delta_gev_pp)[support]
    np.testing.assert_allclose(a, 0, atol=TOL, rtol=0)
    np.testing.assert_allclose(b, 0, atol=TOL, rtol=0)
    paired["support"] = support
    return paired, {
        "complete_pairs": int(support.sum()), "unavailable_pairs": int((~support).sum()),
        "max_error_weight_identity_pp": float(a.abs().max()) if len(a) else None,
        "max_error_gev_identity_pp": float(b.abs().max()) if len(b) else None,
    }


def paired_tables(scan_table, inputs):
    fields = [*SCAN_METRICS, "measurement_status", "removed_frames", "retained_seconds", "valid_blocks",
              "mean_fd_mm", "mean_std_dvars", "redecoded_labels_changed"]
    tables, audit = {}, {}
    for mode in MODES:
        table = scan_table.loc[scan_table["mode"].eq(mode)]
        joined = inputs.roster[prior.PAIR_KEYS].copy()
        for side in SIDES:
            key = f"{side.lower()}_scan_id"
            selected = table.loc[table.condition.eq(side)].set_index("scan", verify_integrity=True)
            selected = selected.loc[joined[key]]
            np.testing.assert_array_equal(selected.subject, joined.participant_id)
            for field in fields:
                joined[f"{side.lower()}_{field}"] = selected[field].to_numpy()
        tables[mode], audit[mode] = contributions(joined)
    original, _ = v9_decompose(inputs.decomposition)
    values = tables[ORIGINAL]
    for computed, expected in [("K_C", "k_coverage_pp"), ("K_R", "k_relative_amplitude_pp"),
                               ("C_W", "m1_shapley_w_pp"), ("C_F", "m1_shapley_f_pp"),
                               ("delta_gev_pp", "m1_gev_delta_pp"), ("delta_fd_mm", "fd"),
                               ("delta_dvars", "mean_std_dvars")]:
        np.testing.assert_allclose(values[computed], original[expected], atol=TOL, rtol=0, equal_nan=True)
    return tables, audit
