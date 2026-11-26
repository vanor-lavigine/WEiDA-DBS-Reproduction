"""Run the date-unrestricted STN paired behavior experiment and fixed sensitivities."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import platform

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd

from behavior_protocol import CFG, PRIVATE, PROTOCOL, REGISTRY, REPO, RESULTS, ROOT, STATS, digest, git, write_csv, write_json
from behavior_stats import association, descriptive_clinical, main_comparisons, matched_pairs, registry_check


def options():
    run = argparse.ArgumentParser(description=__doc__)
    run.add_argument("--workspace", type=Path, default=REPO.parent)
    run.add_argument("--eeg-review", type=Path)
    return run.parse_args()


def merge_hashes(*groups):
    merged = {}
    for group in groups:
        for path, checksum in group.items():
            name = str(Path(path).resolve())
            if name in merged and checksum != merged[name]:
                raise ValueError("A shared input was audited against two different digests")
            merged[name] = checksum
    return merged


def main():
    args = options()
    started = datetime.now(timezone.utc)
    workspace = args.workspace.resolve()
    pdf = (args.eeg_review or workspace.parent / "微状态/脑电微状态.pdf").resolve()
    git("merge-base", "--is-ancestor", CFG["source_commit"], "HEAD")
    git("diff", "--exit-code", "HEAD", "--", f"{ROOT.name}/configs", f"{ROOT.name}/ANALYSIS_PLAN.md")
    registry_check()

    for directory in (RESULTS, PRIVATE / "clinical", PRIVATE / "imaging", PRIVATE / "matplotlib", ROOT / "reports"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")
    from clinical import load_clinical
    from imaging_inputs import load_imaging
    from behavior_report import audit_report, plot_association, plot_clinical, report

    scores, full, clinical_audit, dictionary, score_hash = load_clinical(workspace, PRIVATE / "clinical")
    imaging = load_imaging(workspace, PRIVATE / "imaging")
    main_pairs = matched_pairs(scores, imaging.original)
    quality_pairs = matched_pairs(scores, imaging.quality)
    date_pairs = main_pairs.loc[main_pairs.date_verified.eq(True)].copy()
    flow = dict(clinical_reference=len(full), imaging=len(imaging.original), eligible_clinical_linkage=len(scores),
                main=len(main_pairs), date_verified=len(date_pairs), quality=len(quality_pairs),
                primary_pairs_without_exact_image=len(scores)-len(main_pairs))
    for frame in (main_pairs, quality_pairs):
        if not len(frame):
            raise ValueError("There are no exact clinical and imaging visit pairs")
    if set(dictionary["domains"]) != set(CFG["behaviors"]):
        raise ValueError("The verified score dictionary differs from the analysis protocol")
    print(f"Verified score and scan inputs. Main n={len(main_pairs)}, date n={len(date_pairs)}, quality n={len(quality_pairs)}.",
          flush=True)
    a = main_comparisons(main_pairs, "clinical")
    b = main_comparisons(main_pairs, "duration")
    c = association(main_pairs, "main")
    date = association(date_pairs, "date_verified")
    quality = association(quality_pairs, "fd_censor_s1")
    sensitivity = pd.concat([date, quality], ignore_index=True).merge(
        c[["state", "behavior", "rho", "fdr_q"]].rename(columns={"rho":"main_rho", "fdr_q":"main_fdr_q"}),
        on=["state", "behavior"], validate="many_to_one",
    )

    files = {
        "Clinical_Paired_Results.csv": a, "Duration_Paired_Results.csv": b,
        "Duration_Motor_Associations.csv": c, "Association_Sensitivities.csv": sensitivity,
        "Clinical_Reference_Descriptives.csv": descriptive_clinical(full),
        "Score_Item_Dictionary.csv": pd.DataFrame(dictionary["items"]).assign(
            domains=lambda x: x.domains.map(" | ".join),
            observed_header=lambda x: x.observed_header.map(lambda headers: " | ".join(str(h) for h in headers)),
        ), "Cohort_Flow.csv": pd.DataFrame([flow]),
    }
    for name, table in files.items():
        write_csv(table, RESULTS / name)
    write_json(dictionary, RESULTS / "Score_Item_Dictionary.json")
    for name, table in (("Main_Pairs.csv", main_pairs), ("Date_Verified_Pairs.csv", date_pairs),
                        ("Quality_Pairs.csv", quality_pairs)):
        write_csv(table, PRIVATE / "results" / name)

    protocol_sources = (PROTOCOL, ROOT / "ANALYSIS_PLAN.md", REGISTRY, pdf)
    source_lock = merge_hashes(score_hash, imaging.input_hashes, {path: digest(path) for path in protocol_sources})
    for name, expected in source_lock.items():
        if digest(name) != expected:
            raise RuntimeError("A locked input changed during the run")
    write_json(source_lock, PRIVATE / "Input_Hashes.json")
    audit_report(clinical_audit, imaging.audit, flow)
    report(a, b, c, date, quality, flow)
    for private in (False, True):
        plot_clinical(main_pairs, a, private)
        plot_association(main_pairs, c, private)

    packages = ("numpy", "pandas", "scipy", "matplotlib", "numba", "openpyxl")
    public = {path.relative_to(ROOT).as_posix(): digest(path, text=path.suffix in {".md", ".csv"})
              for folder in ("results", "figures", "reports") for path in sorted((ROOT / folder).glob("*"))
              if path.is_file() and path.name != "Run_Manifest.json"}
    manifest = dict(
        source_commit=CFG["source_commit"], analysis_plan_commit=git("log", "-1", "--format=%H", "--",
                                                                     f"{ROOT.name}/ANALYSIS_PLAN.md"),
        execution_head=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
        command=f"python -B -X utf8 {ROOT.name}/scripts/run_behavior_onoff.py",
        verify_command=f"python -B -X utf8 {ROOT.name}/scripts/verify_behavior_onoff.py",
        protocol=CFG, protocol_sha256=digest(PROTOCOL, text=True), registry_sha256=digest(REGISTRY, text=True),
        seed=STATS["seed"], source_file_count=len(source_lock),
        input_collection_sha256=sha256("".join(x+"\n" for x in sorted(source_lock.values())).encode("ascii")).hexdigest(),
        original_workbook_sha256={Path(k).name: v for k, v in source_lock.items() if Path(k).suffix.lower() == ".xlsx"},
        eeg_review_sha256=digest(pdf), cohort_flow=flow, imaging_fingerprints=imaging.audit["source_sha256"],
        start_utc=started.isoformat(), end_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        packages={name: importlib.metadata.version(name) for name in packages},
        scripts_sha256={path.name: digest(path, text=True) for path in sorted((ROOT / "scripts").glob("*.py"))},
        public_outputs_sha256=public,
    )
    write_json(manifest, RESULTS / "Run_Manifest.json")
    print(f"Completed A={len(a)}, B={len(b)}, C={len(c)} and {len(sensitivity)} sensitivity tests.", flush=True)


if __name__ == "__main__":
    main()
