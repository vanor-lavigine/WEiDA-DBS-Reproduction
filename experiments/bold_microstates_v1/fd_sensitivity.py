#!/usr/bin/env python3
"""Post hoc paired ON−OFF differences adjusted for the uncentered change in mean FD.

For each of the original twelve endpoints test the intercept at delta FD = 0.
Use patient-level Freedman–Lane residual sign flips with an HC3-studentized
intercept. Bootstrap entire (delta FD, delta outcome) pairs for marginal CIs.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adapt_syn9 import read_json, require
from run import new_output, verified_metadata, write_json
from stats import FAMILY_ON_OFF, METRICS, _bootstrap_mean, _rng, holm_adjust, signflip_pvalue


FD_COLUMN = "framewise_displacement_mean"


def adjusted_endpoint(x, y, n_flips, n_bootstrap, flip_rng, bootstrap_rng):
    """Estimate and test the intercept in y = intercept + slope * x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Expected one finite FD and outcome difference per eligible patient")
    if n_flips < 1 or n_bootstrap < 1:
        raise ValueError("Both resampling counts must be positive")
    design = np.column_stack([np.ones(len(x)), x])
    if len(x) < 3 or np.linalg.matrix_rank(design) < 2:
        return {"status": "not_estimable", "reason": "insufficient_pairs_or_rank"}

    weights = np.linalg.pinv(design)
    intercept, slope = weights @ y
    leverage = np.sum(design * weights.T, axis=1)
    if np.any(1 - leverage <= 32 * np.finfo(float).eps):
        return {"status": "not_estimable", "reason": "unit_leverage_in_full_model"}
    # The first row of the pseudoinverse maps observations to the intercept.
    hc3_weights = weights[0] / (1 - leverage)
    residuals = y - (intercept + slope * x)
    hc3_se = np.linalg.norm(hc3_weights * residuals)
    eps = np.finfo(float).eps
    if hc3_se <= 32 * eps * np.linalg.norm(y) * np.linalg.norm(hc3_weights):
        return {"status": "not_estimable", "reason": "undefined_HC3_t_for_perfect_fit"}
    observed_t = intercept / hc3_se

    # H0 retains the FD slope and fixes the intercept to zero. Hold x fixed
    # and flip one reduced-model residual per patient, then refit both terms.
    null_fit = x * (np.dot(x, y) / np.dot(x, x))
    null_residuals = y - null_fit
    extreme = 0
    for start in range(0, n_flips, 4096):
        size = min(4096, n_flips - start)
        signs = flip_rng.integers(0, 2, size=(size, len(x)), dtype=np.int8) * 2 - 1
        outcomes = null_fit + signs * null_residuals
        coefficients = outcomes @ weights.T
        errors = outcomes - coefficients @ design.T
        se = np.linalg.norm(errors * hc3_weights, axis=1)
        if (se == 0).any():
            return {"status": "not_estimable", "reason": "undefined_resampled_HC3_t"}
        t_values = coefficients[:, 0] / se
        tolerance = 32 * eps * max(1.0, abs(observed_t))
        extreme += int(np.count_nonzero(np.abs(t_values) >= abs(observed_t) - tolerance))

    indices = bootstrap_rng.integers(0, len(x), size=(n_bootstrap, len(x)))
    bx, by = x[indices], y[indices]
    valid = np.ptp(bx, axis=1) > 0
    bx, by = bx[valid], by[valid]
    if valid.any():
        x_mean, y_mean = bx.mean(axis=1), by.mean(axis=1)
        xc = bx - x_mean[:, None]
        slopes = (xc * (by - y_mean[:, None])).sum(axis=1) / (xc * xc).sum(axis=1)
        ci = np.percentile(y_mean - x_mean * slopes, [2.5, 97.5])
    else:
        ci = (np.nan, np.nan)
    return {
        "status": "ok", "reason": "", "adjusted_delta": intercept,
        "fd_slope_per_0_1_mm": slope * 0.1, "adjusted_ci_low": ci[0], "adjusted_ci_high": ci[1],
        "intercept_hc3_se": hc3_se, "intercept_t": observed_t, "n_extreme": extreme,
        "adjusted_p_raw": (extreme + 1) / (n_flips + 1),
        "n_bootstrap_valid": int(valid.sum()),
    }


def read_pairs(run_dir):
    fit_dir = run_dir / "analysis_k4"
    fit = read_json(fit_dir / "summary.json")
    require((fit["status"], fit["data_kind"], fit["mode"], fit["polarity"], fit["k"],
             fit["atlas"], fit["n_rois"], fit["n_complete_dbs_pairs"], fit["n_hc"]) ==
            ("completed", "real", "analysis", "signed", 4, "AAL78", 78, 26, 15),
            "Expected the completed real SyN9 K=4 common-template fit")
    meta = verified_metadata(run_dir / "input/metadata.verified.tsv").sort_index()
    fitted = verified_metadata(fit_dir / "identity_used.tsv").sort_index()
    identity = ["subject_id", "cohort", "condition", "mapping_source"]
    require(meta[identity].equals(fitted[identity]), "Cohort or pairing differs from model fit")
    dbs = meta.loc[meta["cohort"].eq("DBS")].reset_index(drop=True)
    require(len(dbs) == 52 and dbs["motion_status"].eq("available").all(),
            "Expected motion records for 52 paired DBS scans")
    dbs[FD_COLUMN] = pd.to_numeric(dbs[FD_COLUMN], errors="raise")
    for row in dbs.itertuples(index=False):
        source = pd.read_csv(row.motion_source, sep="\t", usecols=["framewise_displacement"])
        fd = source["framewise_displacement"].dropna()
        require(len(source) == 210 and len(fd) == int(row.framewise_displacement_n)
                and len(fd) > 0 and np.isfinite(fd).all() and (fd >= 0).all()
                and np.isclose(fd.mean(), row.framewise_displacement_mean, rtol=0, atol=1e-12),
                "Mean FD does not match the source confounds")
    fd = dbs.pivot(index="subject_id", columns="condition", values=FD_COLUMN).sort_index()
    require(fd.shape == (26, 2) and set(fd.columns) == {"ON", "OFF"}
            and np.isfinite(fd.to_numpy()).all(), "Expected 26 complete ON/OFF motion pairs")
    metrics = pd.read_csv(fit_dir / "metrics.tsv", sep="\t", na_values=["n/a"])
    scored = metrics.merge(dbs[["run_id", "subject_id", "condition"]], on="run_id",
                           validate="many_to_one")
    require(len(scored) == 52 * 4 and set(scored["state"]) == {1, 2, 3, 4},
            "Missing fitted state metrics")
    paired = scored.pivot(index="subject_id", columns=["condition", "state"],
                          values=list(METRICS)).sort_index()
    require(paired.index.equals(fd.index), "Motion and outcome pairs are not aligned")
    return fd["ON"] - fd["OFF"], paired


def analyze(run_dir, output_dir):
    delta_fd, paired = read_pairs(run_dir)
    source_dir = run_dir / "inference_k4"
    settings = read_json(source_dir / "analysis_summary.json")
    flips, boots, seed = settings["n_permutations"], settings["n_bootstrap"], settings["seed"]
    require((flips, boots, seed) == (10000, 2000, 20261121), "Original resampling settings differ")
    baseline = pd.read_csv(source_dir / "stats.csv")
    baseline = baseline.loc[baseline["family"].eq(FAMILY_ON_OFF)].set_index(["state", "metric"])
    require(len(baseline) == 12 and baseline.index.is_unique, "Expected all original ON/OFF tests")
    rows = []
    for state in range(1, 5):
        for index, metric in enumerate(METRICS):
            delta = paired[metric, "ON", state] - paired[metric, "OFF", state]
            eligible = np.isfinite(delta)
            x, y = delta_fd[eligible].to_numpy(), delta[eligible].to_numpy()
            ref = baseline.loc[(state, metric)]
            require(len(y) == ref.n_subjects and ref.status == "ok",
                    "Paired outcome availability differs from the original analysis")
            # Verify the original comparison using the original ordering and RNG streams.
            raw = signflip_pvalue(y, flips, _rng(seed, 1, state, index, 0))
            low, high, _ = _bootstrap_mean(y, boots, _rng(seed, 1, state, index, 1))
            np.testing.assert_allclose([y.mean(), low, high, raw],
                                       [ref.estimate, ref.ci_low, ref.ci_high, ref.p_raw],
                                       rtol=0, atol=1e-12)
            adjusted = adjusted_endpoint(x, y, flips, boots, _rng(seed, 1, state, index, 0),
                                         _rng(seed, 1, state, index, 1))
            rows.append({
                "state": state, "metric": metric, "n_pairs": len(y), "family_size": 12,
                "original_delta": ref.estimate, "original_ci_low": ref.ci_low,
                "original_ci_high": ref.ci_high, "original_p_raw": ref.p_raw,
                "original_p_holm": ref.p_holm, "adjusted_p_raw": np.nan, **adjusted,
            })
    table = pd.DataFrame(rows)
    np.testing.assert_allclose(holm_adjust(table["original_p_raw"]), table["original_p_holm"],
                               rtol=0, atol=1e-12)
    table["adjusted_p_holm"] = holm_adjust(table["adjusted_p_raw"])
    output = new_output(output_dir)
    table.to_csv(output / "fd_adjusted_stats.csv", index=False, na_rep="NA")
    design = np.column_stack([np.ones(len(delta_fd)), delta_fd])
    summary = {
        "data_kind": "real", "analysis": "post_hoc_delta_FD_sensitivity",
        "covariate": "mean fMRIPrep FD ON minus OFF, uncentered, mm",
        "target": "ON minus OFF intercept at delta FD = 0 mm",
        "test": "two-sided Freedman-Lane sign flips on reduced-model residuals, HC3 t",
        "ci": "95% marginal patient-pairs percentile bootstrap with refitting",
        "n_dbs_pairs": len(delta_fd), "n_verified_scan_FD_means": 2 * len(delta_fd),
        "mean_delta_fd_mm": delta_fd.mean(), "median_delta_fd_mm": delta_fd.median(),
        "fd_on_higher": (delta_fd > 0).sum(), "fd_on_lower": (delta_fd < 0).sum(),
        "fd_equal": (delta_fd == 0).sum(),
        "max_full_model_leverage": np.max(np.sum(design * np.linalg.pinv(design).T, axis=1)),
        "n_sign_flips": flips, "n_bootstrap": boots, "seed": seed,
        "family_size": len(table), "original_rows_reproduced": len(table),
        "estimable": table["adjusted_p_raw"].notna().sum(),
        "holm_below_0_05": table["adjusted_p_holm"].lt(0.05).sum(),
    }
    write_json(output / "fd_adjusted_summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Original private K=4 run")
    parser.add_argument("--out", type=Path, required=True, help="New group-level output directory")
    args = parser.parse_args()
    analyze(args.run_dir.resolve(), args.out)
