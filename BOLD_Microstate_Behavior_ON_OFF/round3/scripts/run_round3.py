"""Run only the two Coverage-to-improvement Spearman families."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import platform

from round3_io import (BASE, CFG, MODULE, PRIVATE, PROTOCOL, REGISTRY, REPO, RESULTS, ROOT,
                       digest, git, verify_prior, verify_registry, write_csv, write_json)


def main():
    started = datetime.now(timezone.utc)
    preserved = verify_prior()
    verify_registry()
    for directory in (RESULTS, ROOT / "reports", PRIVATE / "results", PRIVATE / "matplotlib"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")

    from coverage_figures import render_figures
    from coverage_inputs import load_inputs
    from coverage_report import write_report
    from coverage_stats import associate, clinical_eligibility, coverage_summary

    sources = load_inputs(REPO.parent, MODULE, coverage_tolerance=CFG["imaging"]["closure_tolerance"],
                          expected_patients=CFG["cohort"]["n_pairs"])
    main_table, secondary, eligible = associate(sources.pairs)
    descriptive = coverage_summary(sources.pairs)
    eligibility = clinical_eligibility(sources.pairs)
    support = {}
    for row in eligibility.itertuples(index=False):
        support[row.clinical_endpoint] = {"n_rate":row.n_valid_rate, "n_off_zero":row.n_off_zero,
                                          "n_other_missing":row.n_missing}
    print(f"Verified {len(sources.pairs)} pairs using frozen ON/OFF scan and fold keys. Clinical rate Ns: "
          + ", ".join(f"{e}={v['n_rate']}" for e, v in support.items()), flush=True)

    stats_files = {
        "Coverage_Clinical_Rate_Associations.csv": main_table,
        "Coverage_Clinical_Absolute_Associations.csv": secondary,
        "Coverage_Change_Summary.csv": descriptive,
        "Clinical_Rate_Eligibility.csv": eligibility,
    }
    created = []
    for filename, table in stats_files.items():
        path = RESULTS / filename
        write_csv(table, path)
        created.append(path)
    write_csv(sources.pairs, PRIVATE / "results/Coverage_Clinical_Pairs.csv")
    write_csv(eligible, PRIVATE / "results/Association_Eligible_Pairs.csv")
    figures = render_figures(main_table, sources.pairs, ROOT / "figures", PRIVATE / "figures")
    created.extend(ROOT / "figures" / name for name in figures["public"])
    report = ROOT / "reports/ON_OFF_Coverage_Clinical_Rate_Report.md"
    write_report(report, main_table, secondary, descriptive, eligibility)
    created.append(report)

    private_lock = {str(Path(name).resolve()): expected for name, expected in sources.input_hashes.items()}
    for name, expected in private_lock.items():
        if digest(name) != expected:
            raise ValueError("A private source changed during the third-round run")
    write_json({"file_sha256": private_lock}, PRIVATE / "Round3_Input_Hashes.json")
    command = "python -B -X utf8 " + ROOT.relative_to(REPO).as_posix() + "/scripts/"
    manifest = dict(
        reference_commit=BASE, reference_tree=CFG["reference_tree"], execution_head=git("rev-parse", "HEAD"),
        branch=git("branch", "--show-current"), plan_commit=git("log", "-1", "--format=%H", "--",
                                                              (ROOT / "ANALYSIS_PLAN.md").relative_to(REPO).as_posix()),
        command=command+"run_round3.py", verify_command=command+"verify_round3.py", protocol=CFG,
        protocol_sha256=digest(PROTOCOL, text=True), tests_registry_sha256=digest(REGISTRY, text=True),
        prior_sources_sha256=preserved, input_checks=sources.audit, verified_input_file_count=len(private_lock),
        input_collection_sha256=sha256("".join(sha+"\n" for sha in sorted(private_lock.values())).encode("ascii")).hexdigest(),
        statistical_families={"main_rate":{"slots":len(main_table),"passes":int(main_table.passes_fdr.sum())},
                              "secondary_absolute_same_cases":{"slots":len(secondary),
                                                               "passes":int(secondary.passes_fdr.sum())}},
        clinical_rate_eligibility=support, same_case_sets_guaranteed_by_shared_patient_mask=True,
        plot_font=figures["font"], heatmap_annotation_key=figures["metadata_key"],
        python=platform.python_version(), packages={name:importlib.metadata.version(name) for name in
                                                    ("numpy", "scipy", "pandas", "matplotlib", "pillow")},
        started_utc=started.isoformat(), ended_utc=datetime.now(timezone.utc).isoformat(),
        scripts_sha256={p.name:digest(p, text=True) for p in sorted((ROOT / "scripts").glob("*.py"))},
        public_outputs_sha256={p.relative_to(ROOT).as_posix():digest(p, text=p.suffix in {".md", ".csv"})
                               for p in sorted(created)},
    )
    if verify_prior() != preserved:
        raise ValueError("The preceding reports or their receipts changed")
    write_json(manifest, RESULTS / "Round3_Run_Manifest.json")
    print(f"Completed independent BH20 families. Main passed {int(main_table.passes_fdr.sum())}/20; "
          f"absolute same cases passed {int(secondary.passes_fdr.sum())}/20.", flush=True)


if __name__ == "__main__":
    main()
