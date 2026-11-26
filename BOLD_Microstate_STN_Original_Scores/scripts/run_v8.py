"""Audit original STN ratings, freeze the input ledger and run the requested V8 tests."""

import argparse
from importlib import metadata
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

from audit_stage import run_audit
from identity_audit import VERIFIED
from pipeline_io import (CFG, PRIVATE, PROTOCOL, REPO, RESULTS, discover_sources, digest, fingerprint, git, load_imaging,
                         read_json, require, verify_frozen_worktree, write_json)
from regression import registry, run_models

LOCK = PRIVATE / "manifests/Prespecified_Inference_Lock.json"


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=REPO.parent)
    parser.add_argument("--score-dir", type=Path, help="Directory containing the three original workbooks")
    parser.add_argument("--v7-module", type=Path, help="Authorized V7 analysis directory with its private paired readouts")
    parser.add_argument("--prior-input-index", type=Path, help="Targeted follow-up private input index")
    parser.add_argument("--audit-only", action="store_true", help="Verify subjects and lock inputs before any association")
    return parser.parse_args()


def write_private_figures(points, all_rows):
    if points.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outcomes = registry()
    main = all_rows.loc[all_rows.model.eq("Primary_FD")].set_index("endpoint")
    fig, axes = plt.subplots(4, 3, figsize=(13, 13), layout="constrained")
    fig.suptitle("Private STN association diagnostics, FD and OFF score residualized", fontsize=15)
    for ax, test in zip(axes.flat, outcomes, strict=True):
        tag = test["endpoint"]
        data = points.loc[points.endpoint.eq(tag)]
        row = main.loc[tag]
        ax.set_title(f"{tag} · {test['outcome']} · n={row.n_patients}", fontsize=10)
        ax.set_xlabel(f"Residualized image predictor ({test['predictor_unit']})", fontsize=8)
        ax.set_ylabel("Residualized improvement (points)", fontsize=8)
        ax.axhline(0, lw=.45, color="#c6ced4")
        ax.axvline(0, lw=.45, color="#c6ced4")
        if not data.empty:
            x, y = data.residualized_image, data.residualized_improvement
            ax.scatter(x, y, s=20, alpha=.8, color="#075E65")
            limit = np.array([x.min(), x.max()])
            ax.plot(limit, row.beta * limit, color="#A8472A", lw=1)
    fig.savefig(PRIVATE / "figures/STN_Patient_Association_Diagnostics.png", dpi=420)
    fig.savefig(PRIVATE / "figures/STN_Patient_Association_Diagnostics.pdf")
    plt.close(fig)


def main():
    opts = options()
    workspace = opts.workspace.resolve(strict=True)
    score_dir = (opts.score_dir or workspace).resolve(strict=True)
    v7 = (opts.v7_module or workspace / "WEiDA-DBS-V7/BOLD_Microstate_Mechanistic_Decomposition").resolve(strict=True)
    prior = (opts.prior_input_index or workspace / "WEiDA-DBS-Targeted/BOLD_Microstate_Targeted_Followup"
             / "private/manifests/Input_Hashes.json").resolve(strict=True)
    verify_frozen_worktree()
    paths = discover_sources(workspace, score_dir, v7, prior)
    signatures = fingerprint(paths)
    with threadpool_limits(limits=1):
        imaging = load_imaging(paths)
        paired, analysis, old = run_audit(paths, imaging)
        lock = dict(baseline=CFG["v7_commit"], protocol_sha256=digest(PROTOCOL), seed=CFG["seed"], endpoints=registry(),
                    exclusion_rules=CFG["visit_rules"], input_files=signatures, main_n=len(analysis),
                    independently_verified_n=int(analysis.visit_status.eq(VERIFIED).sum()),
                    private_table_sha256=digest(PRIVATE / "results/STN_Analysis_Pairs.csv"))
        if opts.audit_only:
            require(fingerprint(paths) == signatures, "One of the input sources changed during discovery")
            if LOCK.is_file():
                require(read_json(LOCK) == lock, "Inputs or protocol changed from the previous inference lock")
            write_json(lock, LOCK)
            print(f"Locked {len(paired)} STN scores, {len(analysis)} eligible pairs and "
                  f"{lock['independently_verified_n']} independently verified date pairs.", flush=True)
            return
        require(LOCK.is_file() and read_json(LOCK) == lock, "Run --audit-only to create a matching pre-inference lock first")
        protocol_path = PROTOCOL.relative_to(REPO).as_posix()
        require(git("ls-files", "--", protocol_path).strip() == protocol_path and
                not git("diff", "HEAD", "--", protocol_path), "Commit the verified protocol before computing association P values")
        from legacy_replay import replay_legacy
        old_stats = replay_legacy(workspace, imaging, old, analysis)
        print("Reproduced the historical four-endpoint model and replaced scores on the same matched patients.", flush=True)
        all_rows, points = run_models(analysis)
    write_private_figures(points, all_rows)
    require(fingerprint(paths) == signatures, "One of the input sources changed during inference")
    verify_frozen_worktree()
    require(all_rows.groupby("model").size().eq(12).all() and len(old_stats) == 12, "A prespecified table is incomplete")
    write_json({"command": sys.argv, "python_executable": sys.executable, "working_directory": str(Path.cwd()),
                "input_lock": str(LOCK), "input_files": signatures}, PRIVATE / "manifests/Run_Command.json")
    versions = ("numpy", "pandas", "scipy", "statsmodels", "openpyxl", "matplotlib")
    write_json({"imaging_v7": CFG["v7_commit"], "imaging_v6": CFG["v6_commit"], "seed": CFG["seed"],
                "permutations_requested_per_endpoint": CFG["permutations"], "bootstrap_requested_per_endpoint": CFG["bootstrap"],
                "n_original_stn_pairs": len(paired), "n_main_eligible_pairs": len(analysis),
                "n_independently_verified_pairs": lock["independently_verified_n"], "primary_family": 3,
                "exploratory_family": 9, "global_family": 12, "models": list(all_rows.model.unique()),
                "old_full_analysis_reproduced": True, "input_files_unchanged": True, "historical_files_unchanged": True,
                "python": sys.version.split()[0], "packages": {v: metadata.version(v) for v in versions}},
               RESULTS / "Run_Verification.json")
    print(f"Completed all {len(all_rows)} prespecified model rows.", flush=True)


if __name__ == "__main__":
    main()
