"""Execute the frozen transition analysis in P0–P4 order."""

import importlib.metadata
import os
import platform

from round5_io import (
    BASE, CFG, CONFIG, MODULE, PRIVATE, REPO, RESULTS, ROOT, WORKSPACE, collection_digest,
    confirm_receipts, digest, git, load_frozen_inputs, verify_base, write_csv, write_json, write_text,
)


def main():
    verify_base()
    frozen = load_frozen_inputs()
    (PRIVATE / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(PRIVATE / "matplotlib")
    output = {}
    public = []
    completed = ["verify_round4_inputs"]

    def publish(stage, tables):
        for name, table in tables.items():
            stem = name.removesuffix(".csv")
            path = RESULTS / (stem + ".csv")
            write_csv(path, table)
            output[stem] = table
            public.append(path)
        completed.append(stage)
        print(f"Completed {stage}", flush=True)

    from omnibus import p0
    from m4_dynamics import render_m4_dynamics, run_m4_dynamics
    from row_ilr import p2
    from motion_sensitivity import motion_sensitivity
    from m4_clinical import run_clinical
    from round5_reports import reports

    overall, delta = p0(frozen.matrices, frozen.v6_omnibus)
    publish("P0_whole_vector", overall)
    keyed_delta = delta.copy()
    keyed_delta.columns = [f"{state}->{target}" for state, target in delta.columns]
    write_csv(PRIVATE / "results/Omnibus_Patient_Vectors.csv", keyed_delta.reset_index())

    m4 = run_m4_dynamics(frozen.scans, frozen.matrices, frozen.comparisons,
                         seed=CFG["seed"], bootstrap=CFG["bootstrap_draws"])
    write_csv(PRIVATE / "results/M4_Dynamics_Patient_Pairs.csv", m4["private_pairs"])
    publish("P1_M4_dynamics", {key: table for key, table in m4.items() if key != "private_pairs"})

    publish("P2_row_ILR", p2(frozen.matrices))
    motion, order, private_motion, motion_lock, acquisition_audit = motion_sensitivity(
        frozen.matrices, frozen.pairs, MODULE, WORKSPACE, CFG,
    )
    for name, table in private_motion.items():
        write_csv(PRIVATE / "results" / (name + ".csv"), table)
    publish("P3_motion_order", {"Transition_Motion_Sensitivity": motion, "Transition_Order_Sensitivity": order})

    clinical = run_clinical(frozen.matrices, frozen.scores, frozen.associations, frozen.round4_protocol, CFG)
    publish("P4_clinical", clinical)

    figures = render_m4_dynamics(m4, ROOT / "figures", PRIVATE / "figures")
    public.extend(figures)
    for name, body in reports(output, frozen.v6_omnibus, acquisition_audit, frozen.audit).items():
        path = ROOT / "reports" / name
        write_text(path, body)
        public.append(path)
    completed.append("reports_figures")
    receipt = confirm_receipts(frozen.hashes, motion_lock)
    write_json(PRIVATE / "Round5_Input_Hashes.json", {"file_sha256": receipt})

    def family(frame, q="fdr_q", holm="holm_p"):
        return {
            "slots": len(frame), "estimable": int(frame.raw_p.notna().sum()),
            "bh_pass": int(frame[q].lt(CFG["alpha"]).sum()),
            "holm_pass": int(frame[holm].lt(CFG["alpha"]).sum()) if holm in frame else None,
        }

    families = {
        "P0_primary": family(output["Omnibus_Transition_Results"]),
        "P2_ILR": family(output["Transition_ILR_Omnibus"]),
        "P3_motion": family(motion.loc[motion.correction_slot.eq(True)], q="bh_q"),
        "P4_historical_spearman": family(output["M4_Clinical_Associations"]),
        "P4_joint": family(output["M4_Joint_Response_Analysis"]),
        "P4_baseline": family(output["M4_Baseline_Adjusted_Models"].query("hypothesis")),
    }
    if {key: value["slots"] for key, value in families.items()} != CFG["families"]:
        raise ValueError("Round 5 family sizes differ from the predeclared sizes")
    verify_base()
    prefix = f"python -B -X utf8 {ROOT.relative_to(REPO).as_posix()}/scripts/"
    release_metadata = {
        "source_commit": BASE, "source_tree": git("rev-parse", BASE + "^{tree}"), "branch": CFG["branch"],
        "protocol": CFG, "protocol_sha256": digest(CONFIG, text=True),
        "plan_sha256": digest(ROOT / "ANALYSIS_PLAN.md", text=True),
        "commands": {"run": prefix + "run_round5.py", "verify": prefix + "verify_round5.py"},
        "completed_steps": completed, "frozen_archive_audit": frozen.audit, "motion_order_audit": acquisition_audit,
        "verified_inputs": len(receipt), "input_collection_sha256": collection_digest(receipt),
        "private_input_receipt": "private/Round5_Input_Hashes.json",
        "families": families, "figure_metadata_key": "Round5_Annotations",
        "scripts_sha256": {f.name: digest(f, text=True) for f in sorted((ROOT / "scripts").glob("*.py"))},
        "public_outputs_sha256": {f.relative_to(ROOT).as_posix(): digest(f, text=f.suffix != ".png")
                                  for f in sorted(public)},
        "python": platform.python_version(),
        "libraries": {name: importlib.metadata.version(name) for name in
                      ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "pillow")},
    }
    write_json(RESULTS / "Round5_Run_Manifest.json", release_metadata)
    print(f"Finished {len(frozen.pairs)} intact pairs; {len(receipt)} frozen input digests.", flush=True)
    for name, record in families.items():
        print(name, record, flush=True)


if __name__ == "__main__":
    main()
