"""Reproduce only the second-round ON–OFF summaries and M4 interval."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import os
from pathlib import Path
import platform

for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import pandas as pd

from round2_io import (BASE, CFG, CONFIG_PATH, ITEM_FILE, MODULE, OLD_INPUT_LOCK, PAIR_FILE, PREVIOUS_RESULTS,
                       PRIVATE, REPO, RESULTS, ROOT, digest, frozen_receipt, git, public_artifact_digests,
                       read_json, read_pairs, write_csv, write_json)


def main():
    start = datetime.now(timezone.utc)
    for folder in (ROOT / "reports", RESULTS, PRIVATE / "results", PRIVATE / "matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")
    from brain import brain, duration_patterns, load_hc
    from clinical import analyze_clinical
    from figures import render_figures
    from report_round2 import write_report

    _, legacy_hashes = frozen_receipt()
    input_paths = [PAIR_FILE, CONFIG_PATH, ROOT / "ANALYSIS_PLAN.md", PREVIOUS_RESULTS / "Run_Manifest.json"]
    input_paths += [MODULE / file for file in legacy_hashes]
    reference_files = [
        "BOLD_Microstate_Mechanistic_Decomposition/reports/V7_M4_Duration_Decomposition.md",
        "BOLD_Microstate_Mechanistic_Decomposition/reports/V7_Functional_Network_Annotation.md",
        "V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Normalization_Analysis.md",
        "V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Paired_Response_Heterogeneity.md",
    ]
    input_paths += [REPO / file for file in reference_files]
    private_hashes = {str(path): digest(path) for path in input_paths}
    pairs = read_pairs()

    dictionary, old_clinical, old_duration = (
        pd.read_csv(PREVIOUS_RESULTS / file, float_precision="round_trip")
        for file in ("Score_Item_Dictionary.csv", "Clinical_Paired_Results.csv", "Duration_Paired_Results.csv")
    )
    issues = {}
    item_scores = None
    if ITEM_FILE.is_file():
        private_hashes[str(ITEM_FILE)] = digest(ITEM_FILE)
        item_scores = pd.read_csv(ITEM_FILE, float_precision="round_trip")
    else:
        issues["clinical_composition"] = "Archived manual item rows were unavailable"
    try:
        clinical = analyze_clinical(pairs, dictionary, old_clinical, item_scores)
    except ValueError:
        if item_scores is None:
            raise
        clinical = analyze_clinical(pairs, dictionary, old_clinical)
        issues["clinical_composition"] = "Archived manual item rows could not be reconciled"

    hc, hc_audit = None, {}
    try:
        private_hashes[str(OLD_INPUT_LOCK)] = digest(OLD_INPUT_LOCK)
        old_locks = read_json(OLD_INPUT_LOCK)
        hc, hc_audit = load_hc(old_locks, REPO.parent, MODULE, pairs=pairs, private_lock=private_hashes)
    except (OSError, ValueError) as exc:
        hc = None
        issues["m4_hc_reference"] = "Frozen HC/visit sources were unavailable or did not reconcile"
        (PRIVATE / "hc_source_issue.txt").write_text(str(exc) + "\n", encoding="utf-8")

    tolerance = CFG["duration"]["zero_tolerance_s"]
    if hc is not None:
        ref = CFG["m4_reference"]
        if tolerance != ref["zero_tolerance_s"]:
            raise ValueError("The two registered numerical tolerances differ")
        patterns, states, reference, brain_private, brain_audit = brain(
            pairs, old_duration, hc, tolerance=tolerance, draws=ref["draws"], seed=ref["seed"],
        )
    else:
        patterns, states, brain_private = duration_patterns(pairs, old_duration, tolerance=tolerance)
        reference, brain_audit = None, {}

    files = {
        "Clinical_Response_Profile.csv": clinical.profile, "Clinical_Improvement_Composition.csv": clinical.composition,
        "M2_M4_Response_Patterns.csv": patterns, "M2_M4_Duration_Change_Summary.csv": states,
        "M4_HC_Reference_Summary.csv": reference,
    }
    created = []
    for filename, frame in files.items():
        if frame is not None:
            target = RESULTS / filename
            write_csv(frame, target)
            created.append(target)
    write_csv(clinical.private, PRIVATE / "results/Clinical_Changes_Paired.csv")
    write_csv(brain_private, PRIVATE / "results/M2_M4_Changes_Paired.csv")
    if hc is not None:
        write_csv(pd.DataFrame({"m4_duration_s": hc}), PRIVATE / "results/HC_M4_Values.csv")

    figures = render_figures(clinical.profile, clinical.composition, clinical.private, patterns, states, reference,
                             brain_private, hc, ROOT / "figures", PRIVATE / "figures", zero_tolerance_s=tolerance)
    created.extend(ROOT / "figures" / filename for filename in figures["public"])
    report = ROOT / "reports/ON_OFF_Round2_Report.md"
    write_report(report, clinical.profile, clinical.composition, patterns, states, reference, brain_audit, issues)
    created.append(report)

    for filename, expected in private_hashes.items():
        if digest(filename) != expected:
            raise ValueError("An input changed during round two")
    frozen_receipt()
    write_json({"file_sha256": private_hashes}, PRIVATE / "Round2_Input_Hashes.json")
    command = f"python -B -X utf8 {ROOT.relative_to(REPO).as_posix()}/scripts/"
    manifest = dict(
        reference_commit=BASE, branch=git("branch", "--show-current"),
        execution_head=git("rev-parse", "HEAD"),
        plan_commit=git("log", "-1", "--format=%H", "--", CONFIG_PATH.relative_to(REPO).as_posix()),
        protocol_sha256=digest(CONFIG_PATH, text=True), plan_sha256=digest(ROOT / "ANALYSIS_PLAN.md", text=True),
        command=command+"run_round2.py", verify_command=command+"verify_round2.py",
        protocol=CFG, prior_frozen_sha256=legacy_hashes,
        prior_interpretation_sha256={file: digest(REPO / file, text=True) for file in reference_files},
        verified_input_file_count=len(private_hashes),
        input_collection_sha256=sha256("".join(x+"\n" for x in sorted(private_hashes.values())).encode("ascii")).hexdigest(),
        private_input_digest_convention="SHA-256 of the sorted list of raw input SHA-256 hashes, each followed by LF",
        checks={"input_pairing_n": len(pairs), "clinical": clinical.audit, "v6_sources": hc_audit, "brain": brain_audit},
        missing_input_dependencies=issues, public_figure_font=figures["font"],
        python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ("numpy", "pandas", "matplotlib")},
        start_utc=start.isoformat(), finish_utc=datetime.now(timezone.utc).isoformat(),
        scripts_sha256={path.name: digest(path, text=True) for path in sorted((ROOT / "scripts").glob("*.py"))},
        public_outputs_sha256=public_artifact_digests(created),
    )
    write_json(manifest, RESULTS / "Round2_Run_Manifest.json")
    print(f"Round two prepared {len(created)} shareable outputs from {len(pairs)} patient pairs and "
          f"{0 if hc is None else len(hc)} HC scans. Missing dependencies: {len(issues)}.", flush=True)


if __name__ == "__main__":
    main()
