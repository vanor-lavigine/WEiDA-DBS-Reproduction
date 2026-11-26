"""Verify the restored V8 archive and replay its date-unrestricted E8 estimate."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

FOLLOWUP = Path(__file__).resolve().parents[1]
REPO = FOLLOWUP.parent
V8 = REPO / "BOLD_Microstate_STN_Original_Scores"
ARCHIVE_COMMIT = "6b2978c97fc38bc05f3ce577b60e66941c173768"
ARCHIVE_IMPORT = "2b44ada9fc1439d33a61d6a8438fc76c248561d8"
ARCHIVE_TREE = "f6e5b5dff6dcbc40f34cac575b4d60ef1d94a7f1"
REPORT_BASE = "dc9c732dbbf7d3cd9005bbd908d75f5ffb1ce987"
PRIVATE_RECEIPT = FOLLOWUP / "private/manifests/V8_Archive_Verification_Private.json"
PUBLIC_RECEIPT = FOLLOWUP / "results/V8_Archive_Verification.json"
RESULT_NAMES = ("Primary_Associations.csv", "Exploratory_Domain_Associations.csv",
                "Motion_and_Visit_Sensitivity.csv")
COMPARE = ["n_patients", "beta", "ci95_low", "ci95_high", "permutation_p", "holm3_p", "bh9_q", "holm12_p"]

# The archived modules use imports relative to the original V8 script directory.
sys.dont_write_bytecode = True
sys.path.insert(0, str(V8 / "scripts"))
import audit_stage
import identity_audit
from date_unrestricted import eligible_pairs
from pipeline_io import CFG, digest, fingerprint, load_imaging, read_json, require, write_json
from regression import adjust_model, estimate, model_data, registry
from score_sources import AI_HEADERS, DOMAINS, ITEMS


def git(*args):
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args])


def git_blob_digest(path):
    key = path.relative_to(REPO).as_posix()
    return sha256(git("show", f"{ARCHIVE_IMPORT}:{key}")).hexdigest()


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n", na_rep="NA").encode("utf-8-sig")


def archive_tree():
    for commit in (ARCHIVE_IMPORT, "HEAD"):
        tree = git("rev-parse", f"{commit}:{V8.name}").decode("ascii").strip()
        require(tree == ARCHIVE_TREE, "V8 archive tree differs")
    require(not git("diff", "--name-only", ARCHIVE_IMPORT, "--", V8.name).strip(),
            "V8 archive has unstaged or staged changes")
    return ARCHIVE_TREE, len(git("ls-tree", "-r", ARCHIVE_TREE).splitlines())


def verify_locks(private_root):
    date_config = V8 / "configs/date_unrestricted.json"
    protocol = V8 / "configs/protocol.json"
    settings = read_json(date_config)
    older_lock_path = private_root / "manifests/Prespecified_Inference_Lock.json"
    latest_lock_path = private_root / "date_unrestricted/manifests/Inference_Lock.json"
    older, latest = map(read_json, (older_lock_path, latest_lock_path))
    require(git_blob_digest(protocol) == older["protocol_sha256"] == latest["parent_protocol_sha256"],
            "Frozen V8 protocol hash differs")
    require(git_blob_digest(date_config) == latest["config_sha256"], "Date-unrestricted settings hash differs")
    require(CFG["seed"] == older["seed"] == latest["seed"], "Frozen seed differs")
    sources = {role: Path(entry["path"]).resolve(strict=True) for role, entry in older["input_files"].items()}
    require(fingerprint(sources) == older["input_files"] == latest["inputs"], "Locked V8 source hashes differ")
    restricted = private_root / "results/STN_Analysis_Pairs.csv"
    unrestricted = private_root / "date_unrestricted/results/STN_Analysis_Pairs_25.csv"
    require(digest(restricted) == older["private_table_sha256"], "Locked 23-patient table hash differs")
    require(digest(unrestricted) == latest["patient_input_sha256"], "Locked 25-patient table hash differs")
    require(settings["included_visit_statuses"] == latest["visit_status_counts"], "Visit-status lock differs")
    return settings, sources, restricted, unrestricted, (older_lock_path, latest_lock_path)


def verify_scoring(settings, sources, restricted, unrestricted):
    item_columns = {item.name: item.primary for item in ITEMS}
    domains = {name: [item_columns[item] for item in DOMAINS[name]]
               for name in ("rigidity", "bradykinesia", "tremor")}
    require(domains == {
        "rigidity": list("IJKLM"), "bradykinesia": list("NOPQRSTUVWX"),
        "tremor": ["AG", "AH", "AI", "AJ", "AK", "AL", "AM", "AN", "AO", "AP"],
    }, "Scoring domain columns differ")
    require(len(item_columns) == 33 and set(AI_HEADERS).isdisjoint(item.secondary for item in ITEMS),
            "Exclusive 33-item mapping differs")
    outputs = {}
    public_names = {"Score_Reconciliation_Summary.csv", "STN_Cohort_Flow.csv", "Clinical_Distribution_Summary.csv"}

    def capture_public(frame, path):
        if path.name in public_names:
            outputs[path.name] = csv_bytes(frame)

    # Execute the original audit in memory. Only the three public summaries are retained.
    with (patch.object(audit_stage, "write_csv", side_effect=capture_public),
          patch.object(audit_stage, "write_json"),
          patch.object(audit_stage, "write_source_audit"),
          patch.object(identity_audit, "write_csv")):
        imaging = load_imaging(sources)
        paired, original, _ = audit_stage.run_audit(sources, imaging)
        rebuilt = eligible_pairs(paired, imaging, original, settings)

    require(sha256(csv_bytes(original)).hexdigest() == digest(restricted), "23-patient scoring replay differs")
    require(sha256(csv_bytes(rebuilt)).hexdigest() == digest(unrestricted), "25-patient scoring replay differs")
    require(set(outputs) == public_names, "Original scoring audit summaries are incomplete")
    for name, content in outputs.items():
        archived = (V8 / "results" / name).read_bytes().replace(b"\r\n", b"\n")
        require(content == archived, f"Original aggregate scoring replay differs for {name}")
    counts = read_csv(BytesIO(outputs["Score_Reconciliation_Summary.csv"])).set_index("metric")["n"]
    return {
        "scored_STN_pairs": len(paired),
        "outside_frozen": int(paired.visit_status.eq(identity_audit.OUTSIDE).sum()),
        "identity_unresolved": int(paired.visit_status.eq(identity_audit.UNRESOLVED).sum()),
        "primary_Y_inflated_rows": int(counts["primary_stn_y_differs_from_corrected_bradykinesia"]),
        "secondary_AT_inflated_rows": int(counts["secondary_stn_at_rows_with_bias"]),
        "legacy_total_score_changes": int(counts["legacy_total_score_records_changed"]),
        "legacy_paired_total_improvement_changes": int(counts["legacy_patient_improvements_changed"]),
        "legacy_comparable_domains": len(audit_stage.LEGACY_FIELDS) - 1,
        "legacy_rows_changed_any_comparable_value": int(counts["legacy_records_changed_any_comparable_value"]),
    }


def load_models(folder, n):
    primary, exploratory, sensitivity = (read_csv(folder / name) for name in RESULT_NAMES)
    main = pd.concat([primary, exploratory], ignore_index=True)
    verified = sensitivity.loc[sensitivity.model.eq("Verified_Visit")].reset_index(drop=True)
    all_models = pd.concat([main, sensitivity], ignore_index=True)
    columns = ["endpoint", "family", "outcome", "predictor", "predictor_unit", "interpretation"]
    expected = pd.DataFrame(registry())[columns].set_index("endpoint").sort_index()
    require((len(primary), len(exploratory), len(verified), len(all_models)) == (3, 9, 12, 60),
            "V8 model families are incomplete")
    require(set(all_models.model) == {model["name"] for model in CFG["models"]}, "Unexpected V8 models")
    require(main.model.eq("Primary_FD").all() and main.n_patients.eq(n).all() and verified.n_patients.eq(22).all(),
            "Archive reports a different effective sample size")
    for _, model in all_models.groupby("model"):
        pd.testing.assert_frame_equal(model[columns].set_index("endpoint").sort_index(), expected)
    require(all_models.status.eq("OK").all() and all_models.permutations_valid.eq(CFG["permutations"]).all()
            and all_models.bootstrap_valid.eq(CFG["bootstrap"]).all(), "V8 resampling receipt differs")
    corrected = adjust_model(all_models)
    correction_columns = ["holm3_p", "bh9_q", "holm12_p"]
    np.testing.assert_allclose(all_models[correction_columns], corrected[correction_columns], rtol=0, atol=CFG["tolerance"])
    return main, verified, len(all_models)


def verify_tables():
    results = V8 / "results"
    original, before, n_original = load_models(results, 23)
    latest, after, n_latest = load_models(results / "date_unrestricted", 25)
    pd.testing.assert_frame_equal(before, after, check_exact=True)
    comparison = read_csv(results / "date_unrestricted/Comparison_23_vs_25.csv").set_index("endpoint")
    require(comparison.index.is_unique and len(comparison) == len(latest), "Cross-cohort comparison is incomplete")
    for source, suffix in ((original, "date_filtered"), (latest, "all_matched")):
        expected = source.set_index("endpoint").loc[comparison.index, COMPARE]
        actual = comparison[[f"{field}_{suffix}" for field in COMPARE]].set_axis(COMPARE, axis=1)
        pd.testing.assert_frame_equal(expected, actual, check_exact=True)
    all_cohorts = [("date_filtered_23", original), ("date_unrestricted_25", latest), ("verified_22", after)]
    discoveries = {
        sample: {name: int(frame[column].lt(CFG["alpha"]).sum()) for name, column in
                 (("Holm3", "holm3_p"), ("BH9", "bh9_q"), ("Holm12", "holm12_p"))}
        for sample, frame in all_cohorts
    }
    require(not any(discoveries["date_unrestricted_25"].values()), "Date-unrestricted decisions differ")
    return latest, n_original, n_latest, discoveries


def replay_e8(unrestricted, settings, latest):
    frame = read_csv(unrestricted)
    require(frame.visit_status.value_counts().to_dict() == settings["included_visit_statuses"],
            "Private replay cohort visit counts differ")
    endpoint = next(row for row in registry() if row["endpoint"] == "E8")
    model = next(row for row in CFG["models"] if row["name"] == "Primary_FD")
    design, outcome, _, columns, sample = model_data(frame, endpoint, model)
    fit = estimate(design, outcome)
    row = latest.set_index("endpoint").loc["E8"]
    require(sample["n_patients"] == int(row.n_patients) == len(frame) == 25, "E8 sample differs")
    require(endpoint["outcome"] == "tremor" and endpoint["predictor"] == "delta_duration_4"
            and model["outcome_scale"] == "absolute" and " + ".join(columns) == row.design
            and row.effect_unit == "points per second", "E8 model or units differ")
    fields = ["beta", "ci95_low", "ci95_high", "hc3_t", "permutation_p", "permutations_valid",
              "permutations_failed", "bootstrap_valid", "bootstrap_failed"]
    np.testing.assert_allclose([fit[field] for field in fields], row[fields].astype(float), rtol=0, atol=CFG["tolerance"])
    require(fit["status"] == row.status == "OK" and fit["permutation_mode"] == row.permutation_mode,
            "E8 resampling status differs")
    return dict({field: fit[field] for field in fields}, n=sample["n_patients"], model="Primary_FD",
                design=row.design, outcome="OFF minus ON tremor score in points",
                predictor="ON minus OFF M4 mean duration in seconds", effect_unit=row.effect_unit,
                ci_method="patient-row percentile bootstrap", p_method="two-sided Freedman-Lane residual permutation",
                bh9_q=float(row.bh9_q), holm12_p=float(row.holm12_p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-private", type=Path,
                        default=REPO.parent / "WEiDA-DBS-V8" / V8.name / "private")
    args = parser.parse_args()
    stage = "restored Git tree"
    try:
        tree, files = archive_tree()
        stage = "original V8 input locks"
        settings, sources, restricted, unrestricted, locks = verify_locks(args.v8_private.resolve(strict=True))
        stage = "workbook scoring, target/identity audit and clinical concordance"
        with threadpool_limits(limits=1):
            scoring = verify_scoring(settings, sources, restricted, unrestricted)
            stage = "full-precision archived statistical tables"
            latest, n_original, n_latest, discoveries = verify_tables()
            stage = "read-only E8 replay"
            e8 = replay_e8(unrestricted, settings, latest)
        stage = "verification receipt output"
        git("check-ignore", "--quiet", "--", PRIVATE_RECEIPT.relative_to(REPO).as_posix())
        private_paths = dict(sources, original_inference_lock=locks[0], latest_inference_lock=locks[1],
                             restricted_patient_table=restricted, unrestricted_patient_table=unrestricted)
        private_fingerprints = fingerprint(private_paths)
        require(all(private_fingerprints[key] == value for key, value in read_json(locks[0])["input_files"].items()),
                "A locked source changed during verification")
        write_json({"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "inputs": private_fingerprints}, PRIVATE_RECEIPT)
        write_json({
            "status": "verified", "v8_archive_commit": ARCHIVE_COMMIT, "v8_archive_import": ARCHIVE_IMPORT,
            "v8_report_base": REPORT_BASE,
            "v8_archive_tree": tree, "v8_tracked_files": files, "locked_source_roles_verified": len(sources),
            "private_patient_tables_identical_after_original_audit": [23, 25],
            "full_precision_model_rows": {"date_filtered": n_original, "date_unrestricted": n_latest},
            "verified_22_shared_endpoints_identical": 12,
            "visit_counts_date_unrestricted": {"verified": 22, "insufficient": 1, "conflict": 2},
            "scoring": scoring, "discoveries": discoveries, "E8_replay": e8, "seed": CFG["seed"],
            "software": dict(python=sys.version.split()[0], **{package: version(package) for package in
                             ("numpy", "pandas", "scipy", "statsmodels", "openpyxl", "threadpoolctl")}),
        }, PUBLIC_RECEIPT)
    except Exception as error:
        # Private pandas diagnostics can contain patient identifiers.
        print(f"V8 verification failed during {stage} ({type(error).__name__}).", file=sys.stderr)
        return 1
    print(f"Verified {files} V8 tracked files, {len(sources)} locked sources, both patient tables and the 25-patient E8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
