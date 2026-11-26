"""Lock, run and record four prespecified V9 microstate tests."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import platform
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from v9_inputs import (
    CFG, MODULE, PRIVATE, PROTOCOL, REPO, check_lock_unchanged, git, lock_inputs, write_csv, write_json,
)
from v9_measurements import CONTRIBUTIONS, decompose, expression_tables
from v9_statistics import expression_group_summary, imaging, load_resampling, motor_associations


def private_diagnostics(decomposition, residuals, associations):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = PRIVATE / "figures"
    target.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    for index, column in enumerate(CONTRIBUTIONS):
        values = decomposition[column].dropna().to_numpy()
        ax.scatter(np.full(len(values), index), values, alpha=0.5)
        if len(values):
            ax.scatter(index, values.mean(), color="black", marker="_", s=250)
    ax.set_xticks([0, 1], ["Occupancy", "Relative amplitude"])
    ax.axhline(0, color="0.6", lw=1)
    ax.set_ylabel("ON−OFF contribution (GEV percentage points)")
    fig.savefig(target / "M1_Contribution_Diagnostics.png", dpi=160)
    plt.close(fig)
    if residuals.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), layout="constrained", sharey=True)
    for ax, endpoint, state in zip(axes, ("B1", "B2"), ("M1", "M4"), strict=True):
        points = residuals.loc[residuals.endpoint.eq(endpoint)]
        x, y = points.adjusted_expression, points.adjusted_improvement
        ax.scatter(x, y)
        if len(x):
            slope = associations.loc[associations.endpoint.eq(endpoint) & associations.model.eq("Primary_FD"),
                                     "beta"].item()
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, slope * xx, color="0.25")
        ax.set_title(f"{state} | n={len(x)}")
        ax.set_xlabel("Adjusted expression change (0.01 units)")
    axes[0].set_ylabel("Adjusted improvement (motor score points)")
    fig.savefig(target / "Continuous_Expression_Residual_Diagnostics.png", dpi=160)
    plt.close(fig)


def write_run_audit(inputs, decomposition_audit, expression_audit, a, b):
    project_path = MODULE.relative_to(REPO).as_posix()
    packages = ("numpy", "pandas", "scipy", "statsmodels", "threadpoolctl", "matplotlib", "numba")
    records = {
        "source_commits": {key: CFG[key] for key in
                           ("report_base", "v8_source_commit", "v8_archive_commit", "v7_imaging_commit")},
        "branch": git("branch", "--show-current"),
        "protocol_commit": git("log", "-1", "--format=%H", "--", str(PROTOCOL.relative_to(REPO))),
        "working_directory": "Repository root",
        "command_with_default_source_layout": f"python -B -X utf8 {project_path}/scripts/run_v9.py",
        "archive_verification_command": f"python -B -X utf8 {project_path}/scripts/verify_v8_archive.py",
        "input_index": "private/manifests/Input_Hashes.json",
        "indexed_available_files": len(inputs.manifest["file_sha256"]),
        "inputs_reverified_after_analysis": True,
        "seed": CFG["seed"],
        "requested_resampling_per_estimate": {"permutation": CFG["permutations"], "bootstrap": CFG["bootstrap"]},
        "python_version": platform.python_version(),
        "software_versions": {package: version(package) for package in packages},
        "threadpool_limit": 1,
        "imaging_cohort": {"paired_patients": len(inputs.roster), "frozen_scans": len(inputs.cohort)},
        "clinical_visit_counts": CFG["clinical_visit_counts"],
        "decomposition_checks": decomposition_audit,
        "expression_checks": expression_audit,
        "main_estimates_available": {
            "imaging": int(a.loc[a.model.eq("Unadjusted"), "permutation_p"].notna().sum()),
            "clinical": int(b.loc[b.model.eq("Primary_FD"), "permutation_p"].notna().sum()),
        },
    }
    write_json(records, MODULE / "results/Run_Manifest.json")
    write_json(dict(records, started_from=str(REPO), argv=[sys.executable, "-B", "-X", "utf8", *sys.argv],
                    analysis_finished_utc=datetime.now(timezone.utc).isoformat()), PRIVATE / "manifests/Run.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent,
                        help="Root containing the verified V7 and V8 source worktrees")
    parser.add_argument("--v7-index", type=Path, help="Override the path to the V7 private input index")
    parser.add_argument("--v8-source", type=Path, help="Override the path to the complete V8 source module")
    parser.add_argument("--lock-only", action="store_true", help="Verify and lock the input files without inference")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    v7 = args.v7_index or workspace / "WEiDA-DBS-V7/BOLD_Microstate_Mechanistic_Decomposition/private/Input_Hashes.json"
    v8 = args.v8_source or workspace / "WEiDA-DBS-V8/BOLD_Microstate_STN_Original_Scores"
    inputs = lock_inputs(workspace, v7.resolve(), v8.resolve())
    if args.lock_only:
        print(f"Locked {len(inputs.manifest['file_sha256'])} files, including {len(inputs.frames)} frame archives.")
        return

    v6 = load_resampling(workspace)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")
    with threadpool_limits(limits=1):
        decomposition, decomposition_audit = decompose(inputs.decomposition)
        scans, pairs, expression_audit = expression_tables(inputs, decomposition)
        a, joint_a, shares = imaging(v6, decomposition)
        descriptive = expression_group_summary(v6, pairs)
        b, joint_b, loo, residuals = motor_associations(pairs, inputs.clinical)
        private_diagnostics(decomposition, residuals, b)

    check_lock_unchanged(inputs.manifest)
    public_results = {
        "M1_Component_Statistics": a.loc[a.model.eq("Unadjusted")],
        "M1_Component_Motion_Sensitivity": a.loc[a.model.ne("Unadjusted")],
        "M1_Component_Shares": shares,
        "V7_V9_Joint_Correction": joint_a,
        "Continuous_Expression_Group_Summary": descriptive,
        "Continuous_Expression_Motor_Associations": b,
        "V8_V9_Clinical_Joint_Correction": joint_b,
    }
    private_results = {
        "M1_Paired_Decomposition": decomposition,
        "Continuous_Expression_by_Scan": scans,
        "Continuous_Expression_by_Pair": pairs,
        "Association_Leave_One_by_Patient": loo,
        "Association_Adjusted_Points": residuals,
    }
    for name, frame in public_results.items():
        write_csv(frame, MODULE / "results" / f"{name}.csv")
    for name, frame in private_results.items():
        write_csv(frame, PRIVATE / "results" / f"{name}.csv")
    write_run_audit(inputs, decomposition_audit, expression_audit, a, b)
    print(f"V9 verified {len(inputs.frames)} frozen frame archives before patient-level inference.")
    print("A1/A2")
    print(a.loc[a.model.eq("Unadjusted"), ["endpoint", "effect", "permutation_p", "holm2_p", "holm8_p"]]
          .to_string(index=False))
    print("B1/B2")
    print(b.loc[b.model.eq("Primary_FD"), ["endpoint", "beta", "permutation_p", "holm2_p", "holm14_p"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
