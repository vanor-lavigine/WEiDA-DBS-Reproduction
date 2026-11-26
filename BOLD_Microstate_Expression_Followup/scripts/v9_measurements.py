"""V7 weight decomposition and whole-window expression from archived backfits."""

import numpy as np
import pandas as pd

from v9_inputs import CFG, PAIR_KEYS, TOL, require


SIDES = ("off", "on")
STATES = tuple(CFG["reported_expression_states"])
GEV_FIELDS = [f"gev_{k}" for k in range(1, 5)]
CONTRIBUTIONS = ["k_coverage_pp", "k_relative_amplitude_pp"]
ORIGINAL = ["m1_shapley_w_pp", "m1_shapley_f_pp", "m1_gev_delta_pp"]


def max_error(errors):
    x = np.asarray(errors, dtype=float)
    finite = x[np.isfinite(x)]
    return float(np.abs(finite).max()) if len(finite) else None


def decompose(paired):
    fields = ["coverage_1", "m1_weight", "m1_fit", "m1_gev"]
    paired = paired[["participant_id", *[f"{s}_{f}" for s in SIDES for f in fields],
                     *ORIGINAL, "fd", "mean_std_dvars"]].copy()
    for side in SIDES:
        c, w, f, gev = (paired[f"{side}_{field}"].to_numpy(float) for field in fields)
        status = np.select(
            [~np.isfinite(c) | (c <= 0), ~np.isfinite(w) | (w <= 0), ~np.isfinite(f), ~np.isfinite(gev)],
            ["No usable M1 coverage", "No usable M1 GFP squared weight",
             "Undefined weighted M1 fit", "Undefined M1 GEV"],
            default="OK",
        )
        valid = status == "OK"
        require(np.all(c[valid] <= 1 + TOL) and np.all(w[valid] <= 1 + TOL)
                and np.all((f[valid] >= -TOL) & (f[valid] <= 1 + TOL)), "Invalid V7 M1 coverage, weight or fit scale")
        r = np.divide(w, c, out=np.full_like(w, np.nan), where=valid)
        np.testing.assert_allclose((c * r)[valid], w[valid], atol=TOL, rtol=0)
        np.testing.assert_allclose((c * r * f)[valid], gev[valid], atol=TOL, rtol=0)
        paired[f"{side}_r"] = r
        paired[f"{side}_decomposition_status"] = status
    valid = paired[[f"{side}_decomposition_status" for side in SIDES]].eq("OK").all(axis=1).to_numpy()
    f_mean = (paired.on_m1_fit + paired.off_m1_fit) / 2
    c_mean = (paired.on_coverage_1 + paired.off_coverage_1) / 2
    r_mean = (paired.on_r + paired.off_r) / 2
    paired[CONTRIBUTIONS[0]] = (100 * (paired.on_coverage_1 - paired.off_coverage_1) * r_mean * f_mean).where(valid)
    paired[CONTRIBUTIONS[1]] = (100 * (paired.on_r - paired.off_r) * c_mean * f_mean).where(valid)
    weighted = paired[CONTRIBUTIONS[0]] + paired[CONTRIBUTIONS[1]]
    whole = weighted + paired.m1_shapley_f_pp
    np.testing.assert_allclose(weighted[valid], paired.loc[valid, "m1_shapley_w_pp"], atol=TOL, rtol=0)
    np.testing.assert_allclose(whole[valid], paired.loc[valid, "m1_gev_delta_pp"], atol=TOL, rtol=0)
    np.testing.assert_allclose(
        100 * (paired.on_m1_gev - paired.off_m1_gev)[valid], whole[valid], atol=TOL, rtol=0,
    )
    paired["pair_decomposition_status"] = np.where(valid, "OK", "One or both conditions undefined")
    audit = {
        "decomposable_pairs": int(valid.sum()),
        "undecomposable_pairs": int((~valid).sum()),
        "decomposition_status_counts": {
            side: {label: int(n) for label, n in paired[f"{side}_decomposition_status"].value_counts().items()}
            for side in SIDES
        },
        "max_error_original_weight_contribution_pp": max_error(weighted - paired.m1_shapley_w_pp),
        "max_error_original_gev_change_pp": max_error(whole - paired.m1_gev_delta_pp),
    }
    return paired, audit


def frame_readout(labels, gfp, corr, mapping):
    """Return E, hard-label GEV and M1 diagnostics in the original frozen numbering."""
    n = len(labels)
    require(labels.shape == (n,) and gfp.shape == (n,) and corr.shape == (n, 4), "Invalid frame archive shape")
    require(np.issubdtype(labels.dtype, np.integer) and np.isin(labels, (-1, 0, 1, 2, 3)).all(),
            "Invalid frozen frame label")
    require(np.issubdtype(mapping.dtype, np.integer) and np.array_equal(np.sort(mapping), np.arange(4)),
            "Invalid raw-to-frozen state permutation")
    valid = labels >= 0
    observed = corr[valid]
    require(np.isfinite(gfp[valid]).all() and np.all(gfp[valid] >= 0), "Invalid archived GFP")
    require(np.isfinite(observed).all() and np.all((observed >= -TOL) & (observed <= 1 + TOL)),
            "Archived absolute correlations outside the allowed range")
    power = np.square(gfp[valid])
    total = power.sum()
    require(np.isfinite(total) and total > 0, "The frozen valid window has no usable GFP squared total")
    aligned = np.empty_like(observed)
    aligned[:, mapping] = np.clip(observed, 0, 1)
    weighted = power[:, None] * np.square(aligned)
    assigned = mapping[labels[valid]][:, None] == np.arange(4)
    expression = weighted.sum(axis=0) / total
    hard_gev = np.where(assigned, weighted, 0).sum(axis=0) / total
    require(np.all(expression >= hard_gev - TOL), "Whole-window expression is below hard-label GEV")
    m1 = assigned[:, 0]
    weight = power[m1].sum()
    m1_values = {
        "coverage_1": float(m1.mean()),
        "weight_1": weight / total,
        "fit_1": weighted[m1, 0].sum() / weight if weight > 0 else np.nan,
        "r_1": power[m1].mean() / power.mean() if m1.any() and weight > 0 else np.nan,
    }
    return expression, hard_gev, m1_values, int(valid.sum())


def expression_tables(inputs, decomposition):
    records = []
    errors = []
    for scan in inputs.cohort.itertuples(index=False):
        path = inputs.frames.get(scan.scan)
        row = dict(scan=scan.scan, subject=scan.subject, condition=scan.condition, fold=scan.fold,
                   archived_frame_status="MISSING" if path is None else "OK",
                   expected_valid_frames=scan.valid_frames)
        if path is None:
            records.append(dict(row, **{f"e{k}": np.nan for k in STATES}))
            continue
        with np.load(path, allow_pickle=False) as archived:
            labels = archived["labels"]
            require(labels.shape == (inputs.acquired_frames,), "Unexpected nominal archived window length")
            require(np.all(labels[scan.source_frames:] < 0), "Frames beyond the acquired scan were marked valid")
            expression, gev, m1, n = frame_readout(labels, archived["gfp"], archived["corr"], inputs.mapping[scan.fold])
        reference = np.asarray([getattr(scan, f) for f in GEV_FIELDS])
        np.testing.assert_allclose(gev, reference, atol=TOL, rtol=0)
        require(n == scan.valid_frames, "Recomputed valid-frame count differs from the V6 common window")
        errors.extend(gev - reference)
        records.append(dict(row, **{f"e{k}": expression[k - 1] for k in STATES}, **m1,
                            valid_frames=n, **{f"recomputed_{f}": v for f, v in zip(GEV_FIELDS, gev, strict=True)}))

    scans = pd.DataFrame(records)
    pairs = inputs.roster[PAIR_KEYS].copy()
    paired_matches = 0
    for side in SIDES:
        scan_key = f"{side}_scan_id"
        available = scans.loc[scans.condition.eq(side.upper())].set_index("scan", verify_integrity=True)
        aligned = available.loc[pairs[scan_key]]
        np.testing.assert_array_equal(aligned.subject, pairs.participant_id)
        for k in STATES:
            pairs[f"{side}_e{k}"] = aligned[f"e{k}"].to_numpy()
        pairs[f"{side}_frame_status"] = aligned.archived_frame_status.to_numpy()
        ready = aligned.archived_frame_status.eq("OK").to_numpy()
        if ready.any():
            # Independently recover R from sums of the original per-frame GFP².
            for name, original in (("coverage_1", "coverage_1"), ("weight_1", "m1_weight"),
                                   ("fit_1", "m1_fit"), ("r_1", "r")):
                np.testing.assert_allclose(aligned[name].to_numpy()[ready], decomposition[f"{side}_{original}"][ready],
                                           atol=TOL, rtol=0, equal_nan=True)
            paired_matches += int(ready.sum())
    for k in STATES:
        pairs[f"delta_e{k}"] = pairs[f"on_e{k}"] - pairs[f"off_e{k}"]
        pairs[f"delta_e{k}_x100"] = 100 * pairs[f"delta_e{k}"]
    audit = {
        "scans_with_verified_hard_label_gev": len(errors) // 4,
        "missing_frame_archives_by_condition": {
            str(condition): int(n) for condition, n in scans.loc[scans.archived_frame_status.eq("MISSING")]
            .condition.value_counts().items()
        },
        "source_scans_shorter_than_nominal_window": int((inputs.cohort.source_frames < inputs.acquired_frames).sum()),
        "max_error_hard_label_gev_proportion": max_error(errors),
        "patient_scans_with_independent_m1_frame_check": paired_matches,
        "e_ge_hard_label_gev_verified_for_available_scans": bool(errors),
        "verified_folds": len(inputs.mapping),
    }
    return scans, pairs, audit
