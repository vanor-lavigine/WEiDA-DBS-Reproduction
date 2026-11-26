"""Audit and execute V7 on the frozen BOLD microstate analysis chain."""

import argparse
from importlib import metadata
from pathlib import Path
import subprocess
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from frozen_inputs import (CFG, LOGS, PRIVATE, REPO, RESULTS, TOL, digest, load_inputs, require, write_json)
from inference import load_v6, primary_tests
from measurements import measure_inputs, paired_readouts
from summaries import duration_summary, exit_summary, gev_summary, window_summary


def check_release_diff():
    changed = subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), "diff", "--name-only",
         CFG["report_base"]], encoding="utf-8")
    require(all(path.startswith("BOLD_Microstate_Mechanistic_Decomposition/") for path in changed.splitlines()),
            "Tracked content outside the V7 module differs from the report base")


def verification(inputs, full, paired, windows, roi, primary, motion):
    check_release_diff()
    require(all(digest(Path(path)) == previous for path, previous in inputs.original_files.items()),
            "A frozen source changed during execution")
    require(len(primary) == 6 and set(motion.groupby("model").size()) == {6},
            "The primary or sensitivity family size is wrong")
    require(len(windows) == len(full) * len(CFG["windows"]) and len(roi) == len(full) * 78,
            "A frozen scan, ROI or window is missing")
    gev_error = paired.m1_gev_delta_pp - paired.m1_shapley_w_pp - paired.m1_shapley_f_pp
    log_error = paired.m4_log_duration - paired.m4_log_coverage - paired.m4_negative_log_occurrence
    complete_error = (paired.m4_log_duration_edge - paired.m4_log_coverage_edge -
                      paired.m4_negative_log_occurrence_edge)
    np.testing.assert_allclose(gev_error.dropna(), 0, atol=TOL, rtol=0)
    np.testing.assert_allclose(log_error.dropna(), 0, atol=TOL, rtol=0)
    np.testing.assert_allclose(complete_error.dropna(), 0, atol=TOL, rtol=0)
    show_max = lambda delta: float(delta.abs().max()) if delta.notna().any() else None
    packages = ("numpy", "pandas", "scipy", "statsmodels", "Matplotlib", "threadpoolctl")
    write_json({
        "status": "verified", "no_tracked_legacy_changes": True, "frozen_inputs_unchanged_after_analysis": True,
        "frozen_files_rehashed": len(inputs.original_files), "scans_reproducing_all_frozen_primary_features": len(full),
        "scans_with_105_roi_projection_and_gfp_reconciled": len(full),
        "n_independent_patients": len(paired), "m1_pair_identity_count": int(gev_error.notna().sum()),
        "m1_max_shapley_residual_gev_percentage_points": show_max(gev_error),
        "m4_main_pair_identity_count": int(log_error.notna().sum()), "m4_max_log_residual": show_max(log_error),
        "m4_complete_pair_identity_count": int(complete_error.notna().sum()),
        "m4_complete_max_log_residual": show_max(complete_error),
        "bootstrap": CFG["bootstrap"], "permutations": CFG["permutations"], "seed": CFG["seed"],
        "holm_family_per_model": 6, "models": ["Unadjusted", "FD", "DVARS"],
        "python": sys.version.split()[0], "packages": {p: metadata.version(p) for p in packages},
        "command": "python -B -X utf8 BOLD_Microstate_Mechanistic_Decomposition/scripts/run_v7.py --prior-input-index <local targeted follow-up input index>",
    }, LOGS / "Run_Verification.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent, help="Root matching the V4 freeze receipt")
    parser.add_argument("--prior-input-index", type=Path, required=True, help="Local targeted follow-up Input_Hashes.json")
    parser.add_argument("--audit-only", action="store_true", help="Verify and inventory source assets before analysis")
    args = parser.parse_args()
    workspace, index = args.workspace.resolve(strict=True), args.prior_input_index.resolve(strict=True)
    check_release_diff()
    with threadpool_limits(limits=1):
        inputs = load_inputs(workspace, index)
        print("Verified V4 feature inputs, V6 paired tables and the archived five-fold label collection.", flush=True)
        if args.audit_only:
            return
        v6 = load_v6(workspace)
        full, windows, roi = measure_inputs(inputs)
        paired = paired_readouts(full, inputs)
        primary, motion = primary_tests(v6, paired)
        print("Completed all locked sign-flip and head-motion models.", flush=True)
        gev_summary(v6, paired)
        duration_summary(v6, paired)
        exit_summary(v6, paired, full)
        window_summary(v6, windows, roi, inputs.pairs.index)
        verification(inputs, full, paired, windows, roi, primary, motion)
    write_json({"executable": sys.executable, "arguments": sys.argv, "cwd": str(Path.cwd()),
                "workspace": str(workspace), "index": str(index)}, PRIVATE / "Actual_Run_Command.json")
    print(f"V7 statistics verified and saved to {RESULTS.name}.", flush=True)


if __name__ == "__main__":
    main()
