"""Run the locked V10 original-window and motion-censored robustness review."""

import argparse
from importlib.metadata import version
from pathlib import Path
import platform

import pandas as pd
from threadpoolctl import threadpool_limits

from v10_frames import FORMAL, paired_tables, replay
from v10_inputs import CFG, MODULE, PRIVATE, PROTOCOL, REPO, RESULTS, load, prior, verify_inputs_unchanged
from v10_statistics import (
    formal_tests, model_influence, patient_diagnostics, v9_occupancy_models,
    window_comparison, window_quality,
)
from v9_statistics import load_resampling


def pair_availability(pairs):
    rows = []
    for mode, values in pairs.items():
        for (off, on), group in values.groupby(["off_measurement_status", "on_measurement_status"], sort=False):
            rows.append(dict(mode=mode, off_status=off, on_status=on,
                             decomposable=bool(group.support.all()), patients=len(group)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent, help="Root of the verified historical archives")
    parser.add_argument("--v9-index", type=Path, help="Location of the original private V9 input index")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    v9_index = args.v9_index or workspace / "WEiDA-DBS-V9/BOLD_Microstate_Expression_Followup/private/manifests/Input_Hashes.json"
    with threadpool_limits(limits=1):
        inputs = load(workspace, v9_index.resolve())
        scans, replay_audit = replay(inputs)
        pairs, algebra = paired_tables(scans, inputs)
        v6 = load_resampling(workspace)
        legacy = v9_occupancy_models(v6, pairs["original_window"])
        sensitivity, family = formal_tests(v6, pairs[FORMAL])
        exposures, scan_reasons = window_quality(scans)
        mode_effects = window_comparison(v6, pairs)
        individuals, distributions, correlations, correlation_loo = patient_diagnostics(pairs)
        effects_loo, effect_records = model_influence(pairs)
        checked_files = verify_inputs_unchanged()

    public = {
        "V9_Occupancy_Motion_Replay.csv": legacy,
        "Censored_Contribution_Tests.csv": sensitivity,
        "V7_V9_V10_Holm10.csv": family,
        "Window_Comparison.csv": mode_effects,
        "Scan_Window_Summary.csv": exposures,
        "Scan_Measurement_Status.csv": scan_reasons,
        "Pair_Availability.csv": pair_availability(pairs),
        "Patient_Diagnostic_Distributions.csv": distributions,
        "Patient_Diagnostic_Associations.csv": correlations,
        "Leave_One_Out_Summary.csv": effects_loo,
    }
    for name, frame in public.items():
        prior.write_csv(frame, RESULTS / name)
    private = {
        "Scans.csv": scans,
        "Paired_Decomposition.csv": pd.concat([table.assign(mode=mode) for mode, table in pairs.items()], ignore_index=True),
        "Patient_Diagnostics.csv": individuals,
        "Patient_Leave_One_Out.csv": pd.DataFrame(correlation_loo + effect_records),
    }
    for name, frame in private.items():
        prior.write_csv(frame, PRIVATE / "results" / name)

    prior.write_json(inputs.fingerprints, RESULTS / "Input_Fingerprints.json")
    command = f"python -B -X utf8 {MODULE.name}/scripts/run_v10.py"
    audit = dict(
        command=command, protocol_commit=prior.git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(REPO))),
        python=platform.python_version(), packages={name: version(name) for name in
                                                     ("numpy", "pandas", "scipy", "statsmodels", "threadpoolctl", "numba")},
        threads=1, report_base=CFG["report_base"], anchor_v9=CFG["v9_results_source"],
        seed=CFG["seed"], permutations=CFG["permutations"], bootstrap=CFG["bootstrap"],
        locked_files_unchanged=checked_files, roi_scans=len(inputs.cohort), patient_pairs=len(inputs.roster),
        original_210_TR_FD_verified_scans=len(inputs.fd),
        short_roi_series_missing_first_tr=int(inputs.cohort.source_start.sum()),
        fd_definition=CFG["fd"]["definition"], fd_units=CFG["fd"]["units"],
        replay=replay_audit, algebra=algebra,
    )
    prior.write_json(audit, RESULTS / "Run_Manifest.json")
    print(f"Verified {len(inputs.fd)} original FD clocks and {checked_files} input hashes.")
    print(legacy[["model", "effect", "standard_error_pp", "bootstrap_se_pp", "permutation_p", "holm2_p", "holm8_p"]]
          .to_string(index=False))
    print(sensitivity[["endpoint", "n_patients", "effect", "ci95_low", "ci95_high", "permutation_p", "holm2_p", "holm10_p"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
