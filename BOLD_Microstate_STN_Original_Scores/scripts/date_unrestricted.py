"""Reanalyse all 25 identity- and condition-matched STN pairs."""

from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from identity_audit import CONFLICT, VERIFIED, match_visits
from pipeline_io import (CFG, PAIR_KEYS, PRIVATE, PROJECT, PROTOCOL, RESULTS, TOL,
                         digest, fingerprint, git, load_imaging, read_json, require,
                         verify_frozen_worktree, write_csv, write_json)
from regression import registry, run_models
from score_sources import read_primary

VARIANT = "date_unrestricted"
CONFIG = PROJECT / "configs/date_unrestricted.json"
OUTPUT = RESULTS / VARIANT
PATIENT = PRIVATE / VARIANT
ORIGINAL_FILES = [
    "Primary_Associations.csv", "Exploratory_Domain_Associations.csv", "Motion_and_Visit_Sensitivity.csv",
    "STN_Cohort_Flow.csv", "Score_Reconciliation_Summary.csv", "Legacy_Score_Replacement_Associations.csv",
]
COMPARE = ["n_patients", "beta", "ci95_low", "ci95_high", "permutation_p", "holm3_p", "bh9_q", "holm12_p"]


def reload_inputs(lock):
    paths = {role: Path(record["path"]).resolve(strict=True) for role, record in lock["input_files"].items()}
    require(fingerprint(paths) == lock["input_files"] and digest(PROTOCOL) == lock["protocol_sha256"],
            "Frozen input or original model protocol changed from the original V8 lock")
    imaging = load_imaging(paths)
    primary, _ = read_primary(paths["score_primary"])
    _, scores, original, _, _ = match_visits(primary, paths, imaging, private_dir=PATIENT)
    require(digest(PATIENT / "results/STN_Analysis_Pairs.csv") == lock["private_table_sha256"],
            "The date-filtered cohort did not reproduce the original V8 input byte-for-byte")
    return paths, imaging, scores, original


def eligible_pairs(scored, imaging, original, settings):
    statuses = settings["included_visit_statuses"]
    selected = scored.loc[
        scored.identity_status.eq("confirmed_code_name_condition") & scored.target.eq("STN") &
        scored.complete_score_pair & scored.visit_status.isin(statuses)
    ].dropna(subset=PAIR_KEYS)
    require(selected.visit_status.value_counts().to_dict() == statuses, "Reconstructed visits differ from the intended scope")
    roster = imaging.roster.reset_index()
    keys = selected.merge(roster, on=PAIR_KEYS, validate="one_to_one")
    require(len(keys) == len(selected) == len(roster) == settings["n_frozen_pairs"],
            "The full 25 frozen patient-and-scan keys did not match exactly")
    analysis = keys.merge(imaging.paired.reset_index(), on="participant_id", validate="one_to_one").sort_values("participant_id")
    clear_dates = analysis.loc[analysis.visit_status.ne(CONFLICT)].reset_index(drop=True)
    pd.testing.assert_frame_equal(clear_dates, original.reset_index(drop=True), check_exact=True)
    return analysis


def save_comparison(models):
    old = pd.concat([pd.read_csv(RESULTS / name, float_precision="round_trip") for name in ORIGINAL_FILES[:2]],
                    ignore_index=True)
    keys = ["endpoint", "family", "outcome", "predictor"]
    new = models.loc[models.model.eq("Primary_FD")]
    comparison = old[keys + COMPARE].merge(new[keys + COMPARE], on=keys, validate="one_to_one",
                                           suffixes=("_date_filtered", "_all_matched"))
    require(len(comparison) == len(registry()) == 12, "The paired cohort comparison is incomplete")
    comparison["beta_change_after_inclusion"] = comparison.beta_all_matched - comparison.beta_date_filtered
    write_csv(comparison, OUTPUT / "Comparison_23_vs_25.csv")
    prior_verified = pd.read_csv(RESULTS / "Motion_and_Visit_Sensitivity.csv", float_precision="round_trip")
    before = prior_verified.loc[prior_verified.model.eq("Verified_Visit")].set_index("endpoint")
    after = models.loc[models.model.eq("Verified_Visit")].set_index("endpoint")
    require(set(before.index) == set(after.index) and len(after) == len(registry()), "Verified subset is incomplete")
    np.testing.assert_allclose(before.loc[after.index, COMPARE], after[COMPARE], equal_nan=True, atol=TOL, rtol=0)


def main():
    settings = read_json(CONFIG)
    base = settings["v8_reference_commit"]
    require(git("merge-base", base, "HEAD").strip() == base, "The date-unrestricted run must start from the original V8 commit")
    verify_frozen_worktree()
    original_files = {name: digest(RESULTS / name) for name in ORIGINAL_FILES}
    original_lock = read_json(PRIVATE / "manifests/Prespecified_Inference_Lock.json")
    with threadpool_limits(limits=1):
        paths, imaging, scores, original = reload_inputs(original_lock)
        analysis = eligible_pairs(scores, imaging, original, settings)
        write_csv(analysis, PATIENT / "results/STN_Analysis_Pairs_25.csv")
        write_json({
            "config_sha256": digest(CONFIG), "parent_protocol_sha256": digest(PROTOCOL), "seed": CFG["seed"],
            "inputs": original_lock["input_files"], "visit_status_counts": analysis.visit_status.value_counts().to_dict(),
            "patient_input_sha256": digest(PATIENT / "results/STN_Analysis_Pairs_25.csv"),
        }, PATIENT / "manifests/Inference_Lock.json")
        models, _ = run_models(analysis, results_dir=OUTPUT, private_dir=PATIENT)
    save_comparison(models)
    require(fingerprint(paths) == original_lock["input_files"], "One of the frozen inputs changed during the run")
    require(all(digest(RESULTS / name) == stamp for name, stamp in original_files.items()),
            "A result from the original V8 analysis changed")
    require(len(models) == 60 and models.status.eq("OK").all(), "One or more prespecified estimates did not complete")
    require(models.permutations_valid.eq(CFG["permutations"]).all() and models.bootstrap_valid.eq(CFG["bootstrap"]).all(),
            "The resampling verification did not pass for every estimate")
    main_result = models.loc[models.model.eq("Primary_FD")]
    write_json({
        "v8_reference_commit": base, "n_date_unrestricted": len(analysis),
        "date_conflict_included": int(analysis.visit_status.eq(CONFLICT).sum()),
        "n_independently_verified": int(analysis.visit_status.eq(VERIFIED).sum()),
        "n_original_date_filtered": len(original), "models_completed": len(models),
        "valid_permutations_per_model": CFG["permutations"], "valid_bootstrap_per_model": CFG["bootstrap"],
        "original_date_filtered_inputs_and_results_verified": True, "same_day_subset_reproduced": True,
        "main_holm3_positive": int(main_result.holm3_p.lt(CFG["alpha"]).sum()),
        "exploratory_bh9_positive": int(main_result.bh9_q.lt(CFG["alpha"]).sum()),
        "overall_holm12_positive": int(main_result.holm12_p.lt(CFG["alpha"]).sum()),
    }, OUTPUT / "Verification.json")
    print(f"Completed 25-pair reanalysis, 60 model rows, 12 direct comparisons and verified same-day estimates.")


if __name__ == "__main__":
    main()
