#!/usr/bin/env python3
"""Validate the archived SyN9 cohort and generate server-only experiment manifests."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from run import digest, load_roi, new_output, read_table, write_json


EXPECTED_PROFILE = {
    "--file-format": "nifti",
    "--dummy-scans": "0",
    "--fd-thresh": "0",
    "--despike": "y",
    "--nuisance-regressors": "36P",
    "--smoothing": "0",
    "--lower-bpf": "0.01",
    "--upper-bpf": "0.08",
    "--bpf-order": "2",
    "--min-coverage": "0.2",
    "--atlases": "AAL1ctx",
}
CONDITION_CODE = {"HC": 1, "OFF": 2, "ON": 3}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def keyed(rows, *fields):
    result = {tuple(row[field] for field in fields): row for row in rows}
    require(len(result) == len(rows), f"Duplicate keys in {fields}")
    return result


def check_profile(tokens, source):
    for option, value in EXPECTED_PROFILE.items():
        require(tokens.count(option) == 1 and
                tokens[tokens.index(option) + 1:tokens.index(option) + 2] == [value],
                f"SyN9 option {option} differs in {source}")


def motion_metadata(folder):
    """Describe existing pre-denoising fMRIPrep motion confounds without censoring."""
    matches = list((folder / "preprocessing-evidence").glob("*_desc-confounds_timeseries.tsv"))
    if len(matches) != 1:
        return {"motion_status": "missing_or_ambiguous_confounds"}
    table = pd.read_csv(matches[0], sep="\t")
    if len(table) != 210 or not {"framewise_displacement", "dvars", "std_dvars"} <= set(table):
        return {"motion_status": "unexpected_confound_shape_or_fields"}
    result = {"motion_status": "available", "motion_source": str(matches[0])}
    for column in ("framewise_displacement", "dvars", "std_dvars"):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy()
        finite = values[np.isfinite(values)]
        result[f"{column}_mean"] = float(finite.mean()) if finite.size else None
        result[f"{column}_n"] = int(finite.size)
    return result


def adapt(args):
    batch, cohort = args.batch.resolve(), args.cohort.resolve()
    finished_path = batch / "finished.json"
    finished, plan = read_json(finished_path), read_json(batch / "plan.json")
    packing_check = read_json(args.pack_verification)
    require(digest(finished_path) == packing_check["source_batch_finish_sha256"],
            "The completed batch differs from the batch used to pack the MATLAB input")
    require((finished["phase"], finished["scans"], finished["finished_scans"]) ==
            ("finished", 69, 69), "Expected the terminal 69-scan SyN9 batch")
    check_profile(plan["profile"], "plan.json")
    scans = keyed(finished["scan_results"], "alias", "condition")
    planned = keyed(plan["scans"], "alias", "condition")
    require(scans.keys() == planned.keys() and len(scans) == 69,
            "Terminal scan inventory differs from the run plan")
    source_rows = read_table(cohort).to_dict("records")
    identities = keyed(source_rows, "run_audit_id", "condition")
    require(identities.keys() == {(row["run_audit_id"], row["condition"])
                                  for row in planned.values()},
            "Cohort and plan disagree on the audit IDs or conditions")
    subject_by_scan = {}
    for key, row in planned.items():
        identity = identities[row["run_audit_id"], row["condition"]]
        require(identity["bids_label"] == row["label"] and identity["person_id"],
                "Cohort and plan disagree on the BIDS label or subject evidence")
        subject_by_scan[key] = identity

    eligible_pairs = [row["alias"] for row in finished["pairs"] if row["eligible"] is True]
    expected = {(alias, condition) for alias in eligible_pairs for condition in ("OFF", "ON")}
    expected |= {key for key, row in scans.items() if key[1] == "HC" and row["phase"] == "PASSED"}
    roi_order = read_json(args.roi_order)
    require(len(roi_order) == len(set(roi_order)) == 78, "Expected the archived 78-ROI axis")
    with args.pack_manifest.open(encoding="utf-8-sig", newline="") as handle:
        packed = list(csv.DictReader(handle))
    require([int(row["scan_index"]) for row in packed] == list(range(1, 68)),
            "Expected the original ordered 67-row MATLAB packing manifest")
    require(keyed(packed, "alias", "condition").keys() == expected,
            "MATLAB packing and eligible SyN9 scans differ")
    require(Counter(key[1] for key in expected) == {"HC": 15, "OFF": 26, "ON": 26},
            "The analysis cohort must contain 26 complete DBS pairs and 15 controls")

    runs, metadata = [], []
    for row in packed:
        index = int(row["scan_index"])
        alias, condition = row["alias"], row["condition"]
        key = (alias, condition)
        scan, identity = scans[key], subject_by_scan[key]
        folder = batch / "scans" / alias / condition
        source = folder / "roi-timeseries.tsv"
        require(row["source"] == source.relative_to(batch).as_posix(),
                f"Unexpected time-series path at packing row {index}")
        fingerprint = digest(source)
        require(fingerprint == row["source_sha256"], f"Time series differs at packing row {index}")
        check_profile(read_json(folder / "xcpd-command.json"), f"packing row {index}")
        verification = read_json(folder / "verification.json")
        require(scan["phase"] == "PASSED" and scan["technical_pass"] is True
                and verification["technical_gate_pass"] is True
                and verification["roi_means_match_denoised_image"] is True
                and verification["coverage_minimum"] >= 0.20,
                f"Missing SyN9 extraction/coverage evidence at packing row {index}")
        require((int(row["frames"]), float(row["tr_seconds"]), int(row["condition_code"])) ==
                (210, 2.1, CONDITION_CODE[condition]) and
                (verification["frames"], verification["tr_seconds"], verification["condition"]) ==
                (210, 2.1, condition), f"Frame/TR/condition disagreement at packing row {index}")
        values, names = load_roi(source)
        require(names == roi_order == verification["atlas_region_order"]
                and values.shape == (210, 78) and np.isfinite(values).all()
                and (values.std(axis=0, ddof=1) > 0).all(),
                f"ROI axis, shape or signal check failed at packing row {index}")
        run_id = f"syn9_{index:02d}"
        runs.append({"run_id": run_id, "roi_time_series": str(source), "tr_sec": 2.1,
                     "atlas": "AAL78", "format": "nifti", "n_frames_expected": 210,
                     "n_regions_expected": 78, "packing_row": index, "source_sha256": fingerprint})
        metadata.append({
            "run_id": run_id, "subject_id": identity["person_id"],
            "cohort": "HC" if condition == "HC" else "DBS", "condition": condition,
            "mapping_source": f"{cohort}; {batch / 'plan.json'}; {finished_path}; "
                              f"{args.pack_manifest.resolve()} row {index}",
            "clinical_scale": "", "clinical_score": "", "visual_qc": identity["output_visual_qc"],
            "min_roi_coverage": verification["coverage_minimum"],
            **motion_metadata(folder),
        })
    meta = pd.DataFrame(metadata)
    conditions = meta.groupby("subject_id")["condition"].agg(lambda values: tuple(sorted(values)))
    require(Counter(conditions) == {("OFF", "ON"): 26, ("HC",): 15},
            "The source-backed identities do not resolve to 26 DBS pairs and 15 unique HC")

    output = new_output(args.out)
    pd.DataFrame(runs).to_csv(output / "runs.tsv", sep="\t", index=False)
    meta.to_csv(output / "metadata.verified.tsv", sep="\t", index=False)
    summary = {
        "data_kind": "real",
        "source_terminal_scans": len(scans),
        "source_technical_pass": sum(row["phase"] == "PASSED" for row in scans.values()),
        "source_excluded_by_phase": dict(Counter(row["phase"] for row in scans.values()
                                                 if row["phase"] != "PASSED")),
        "technical_pass_without_pair": sum(row["phase"] == "PASSED" and key not in expected
                                           for key, row in scans.items()),
        "included_scans": len(runs),
        "included_subjects": len(conditions),
        "complete_dbs_pairs": len(eligible_pairs),
        "included_hc": int(meta["condition"].eq("HC").sum()),
        "frames_per_scan": 210,
        "roi_count": 78,
        "tr_sec": 2.1,
        "roi_order_check": "all 67 raw headers exactly equal to archived MATLAB order",
        "packing_sha256_matches": len(runs),
        "source_batch_finish_sha256": digest(finished_path),
        "packing_manifest_sha256": digest(args.pack_manifest),
        "roi_order_json_sha256": digest(args.roi_order),
        "minimum_roi_coverage": float(meta["min_roi_coverage"].min()),
        "visual_qc_status": dict(Counter(meta["visual_qc"])),
        "motion_status": dict(Counter(meta["motion_status"])),
    }
    write_json(output / "input_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "cohort", "pack-manifest", "pack-verification", "roi-order", "out"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    adapt(parser.parse_args())


if __name__ == "__main__":
    main()
