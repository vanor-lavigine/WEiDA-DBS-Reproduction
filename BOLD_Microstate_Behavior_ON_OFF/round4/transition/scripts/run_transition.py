"""Run the frozen A3 transition analysis and write its reproducibility receipt."""

import importlib.metadata
import os
import platform
import sys

import pandas as pd

from transition_io import (
    BASE, CFG, CONFIG, MODULE, PRIVATE, REGISTRY, REPO, ROOT, atomic_text, collection_digest,
    digest, git, verify_base, verify_inputs, verify_registry, write_csv, write_json,
)
from transition_report import report_text
from transition_sources import load_transition_sources
from transition_stats import clinical_associations, group_statistics
from transition_summaries import describe_quality, hc_reference, summarize_m4, v6_parity


PUBLIC = ROOT / "results"
ARCHIVED_TRANSITIONS = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics/results/Transition_Matrices.csv"
PREVIOUS = [
    MODULE / "reports/Behavior_ON_OFF_Report.md",
    MODULE / "round2/reports/ON_OFF_Round2_Report.md",
    MODULE / "round3/reports/ON_OFF_Coverage_Clinical_Rate_Report.md",
    MODULE / "round4/audit/Clinical_Timing_Audit_Report.md",
    REPO / "V4_WEiDA_Statistical_Framework_Microstate/reports/WEIDA_STATISTICAL_PROTOCOL.md",
    REPO / "V4_WEiDA_Statistical_Framework_Microstate/reports/V4_STATISTICAL_RESULTS.md",
    REPO / "V4_WEiDA_Statistical_Framework_Microstate/scripts/statistics_v4.py",
    MODULE / "round3/scripts/coverage_inputs.py",
    ARCHIVED_TRANSITIONS,
]


def clinical_inputs(pairs):
    """Reuse the authenticated frozen clinical scores and verify both scan keys."""
    sys.path.insert(0, str(MODULE / "round3/scripts"))
    from coverage_inputs import load_inputs

    source = load_inputs(REPO.parent, MODULE, expected_patients=CFG["cohort"]["pairs"])
    keys = ["patient_id", "fold", "on_scan_id", "off_scan_id"]
    columns = keys + [name + "_improvement" for name in CFG["clinical_endpoints"]]
    scores = source.pairs[columns].copy()
    joined = pairs.rename(columns={"participant_id": "patient_id"})[keys].merge(
        scores, on=keys, how="outer", indicator=True, validate="one_to_one",
    )
    if len(joined) != CFG["cohort"]["pairs"] or not joined.pop("_merge").eq("both").all():
        raise ValueError("Archived manual scores failed exact patient, ON/OFF scan and fold join")
    return joined, source


def write_private(source, clinical, receipt):
    results = PRIVATE / "results"
    metrics, matrices = source.scan_metrics, source.matrices
    scan_counts = matrices.assign(edge=matrices.from_state + "_" + matrices.to_state).pivot(
        index="scan_id", columns="edge", values="count",
    ).add_prefix("N_").reset_index()
    write_csv(results / "Transition_Counts_All_Scans.csv",
              metrics.merge(scan_counts, on="scan_id", how="inner", validate="one_to_one"))
    write_csv(results / "Transition_Matrix_All_Scans.csv", matrices)
    keys = ["scan_id", "participant_id", "condition", "fold", "from_state", "to_state"]
    write_csv(results / "Transition_Probability_WEiDA.csv", matrices[[*keys, "p_weida"]])
    write_csv(results / "Transition_Probability_ExitConditional.csv", matrices[[*keys, "p_exit"]])
    write_csv(results / "Frozen_Clinical_Pairs.csv", clinical)
    write_json(PRIVATE / "Transition_Input_Hashes.json", {"file_sha256": receipt})


def result_paths(tables):
    paths = []
    for name, table in tables.items():
        path = PUBLIC / (name + ".csv")
        write_csv(path, table)
        paths.append(path)
    return paths


def main():
    verify_base()
    verify_registry()
    charts = ROOT / "figures"
    (PRIVATE / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")
    from transition_figures import render_transition_figures

    source = load_transition_sources(REPO.parent, MODULE)
    scores, clinical_source = clinical_inputs(source.pairs)
    matrices = source.matrices
    frozen_transition_hash = digest(ARCHIVED_TRANSITIONS)
    parity = v6_parity(matrices, pd.read_csv(ARCHIVED_TRANSITIONS, float_precision="round_trip"))
    quality = describe_quality(source.scan_metrics)
    receipt = verify_inputs(source.input_hashes, clinical_source.input_hashes,
                            {str(ARCHIVED_TRANSITIONS): frozen_transition_hash})
    group, comparisons, corrections = group_statistics(matrices, CFG)
    associations = clinical_associations(matrices, scores, CFG)
    m4 = summarize_m4(group, comparisons, source.scan_metrics, matrices)
    reference = hc_reference(group, comparisons)
    tables = {
        "Transition_Probability_WEiDA": group.loc[group.method.eq("WEiDA")],
        "Transition_Probability_ExitConditional": group.loc[group.method.eq("ExitConditional")],
        "Transition_Group_Comparisons": comparisons,
        "Transition_Statistical_Corrections": corrections,
        "M4_Transition_Summary": m4,
        "Transition_Clinical_Associations": associations,
        "Transition_QC_Summary": quality,
        "Transition_HC_Reference": reference,
    }
    if verify_inputs(receipt) != receipt:
        raise ValueError("The frozen input receipt changed while processing")
    write_private(source, scores, receipt)
    output = result_paths(tables)
    figures = render_transition_figures(group, comparisons, matrices, charts, PRIVATE / "figures")
    output.extend(charts / name for name in figures["public"])
    report = ROOT / "reports/A3_Transition_Dynamics_Report.md"
    atomic_text(report, report_text(group, comparisons, associations, m4, reference, quality, parity))
    output.append(report)

    verify_base()
    prefix = "python -B -X utf8 " + ROOT.relative_to(REPO).as_posix() + "/scripts/"
    manifest = {
        "source_commit": BASE,
        "source_tree": git("rev-parse", BASE + "^{tree}"),
        "branch": CFG["branch"],
        "command": prefix + "run_transition.py",
        "verify_command": prefix + "verify_transition.py",
        "protocol": CFG,
        "protocol_sha256": digest(CONFIG, normalize=True),
        "plan_sha256": digest(ROOT / "ANALYSIS_PLAN.md", normalize=True),
        "registry_sha256": digest(REGISTRY, normalize=True),
        "label_and_window_audit": source.audit,
        "archived_v6_probability_parity": parity,
        "clinical_patient_pairs": len(scores),
        "clinical_score_basis": "archived 33 manual items, five stored absolute improvement variables",
        "clinical_same_visit_full_pairs": 0,
        "verified_input_files": len(receipt),
        "input_collection_sha256": collection_digest(receipt),
        "private_receipt": "private/round4/transition/Transition_Input_Hashes.json",
        "private_outputs": {
            "scan_tables": ["Transition_Counts_All_Scans.csv", "Transition_Matrix_All_Scans.csv",
                            "Transition_Probability_WEiDA.csv", "Transition_Probability_ExitConditional.csv"],
            "figures": figures["private"],
        },
        "families": {
            method: {"slots": len(rows), "estimable": int(rows.raw_p.notna().sum()),
                     "bh_pass": int(rows.passes_fdr.sum()), "holm_pass": int(rows.passes_holm.sum())}
            for method, rows in corrections.groupby("method")
        } | {"Clinical_WEiDA": {"slots": len(associations), "estimable": int(associations.raw_p.notna().sum()),
                               "bh_pass": int(associations.passes_fdr.sum())}},
        "previous_inputs_sha256": {path.relative_to(REPO).as_posix(): digest(path, normalize=True)
                                   for path in PREVIOUS},
        "scripts_sha256": {path.name: digest(path, normalize=True)
                           for path in sorted((ROOT / "scripts").glob("*.py"))},
        "public_outputs_sha256": {path.relative_to(ROOT).as_posix(): digest(
            path, normalize=path.suffix in {".md", ".csv"},
        ) for path in sorted(output)},
        "plot_font": figures["font"],
        "plot_metadata_key": figures["metadata_key"],
        "python": platform.python_version(),
        "libraries": {package: importlib.metadata.version(package)
                      for package in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pillow")},
    }
    write_json(PUBLIC / "Transition_Run_Manifest.json", manifest)
    print(f"Completed {source.audit['scans']} scans and {len(scores)} paired participants.")
    for family, result in manifest["families"].items():
        print(f"{family}: {result}")


if __name__ == "__main__":
    main()
