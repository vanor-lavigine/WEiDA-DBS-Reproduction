"""Verify two aggregate clinical runs and export group estimates for the EEG comparison."""
import argparse
import csv
import json
import math
from pathlib import Path


KEY = ("variant", "contrast", "state", "metric")
ESTIMATES = ("r", "ci_low", "ci_high", "p_raw", "q_bh42")
FIELDS = (*KEY, "n", *ESTIMATES)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export(qc, first, repeat, preceding_stats, output):
    first_rows = {tuple(row[k] for k in KEY): row for row in first["tests"]}
    repeat_rows = {tuple(row[k] for k in KEY): row for row in repeat["tests"]}
    if len(repeat_rows) != 84 or len(repeat["tests"]) != 84 or len(first["tests"]) != 84 or first_rows.keys() != repeat_rows.keys():
        raise ValueError("The runs must have the same 84 distinct endpoints")
    if any(row["n"] != 24 for row in [*first_rows.values(), *repeat_rows.values()]):
        raise ValueError("Unexpected endpoint sample size")
    differences = {}
    for field in ESTIMATES:
        values = [abs(row[field] - first_rows[key][field]) for key, row in repeat_rows.items()]
        if not all(math.isfinite(d) for d in values) or max(values) > (1e-14 if field == "q_bh42" else 0):
            raise ValueError(f"The two runs differ for {field}")
        differences[field] = max(values)
    for field in ("primary_sensitivity", "clinical_summary", "frozen_direct_spearman", "environment"):
        if first[field] != repeat[field]:
            raise ValueError(f"The runs differ for {field}")
    with preceding_stats.open(encoding="utf-8", newline="") as stream:
        prior = {"A" + row["state"]: float(row["estimate"]) for row in csv.DictReader(stream)
                 if (row["scenario"], row["method"], row["family"]) ==
                    ("archive_main", "AAHC", "clinical_off_duration")}
    frozen = repeat["frozen_direct_spearman"]
    if prior.keys() != frozen.keys() or len(prior) != 4:
        raise ValueError("Could not find the four preceding clinical estimates")
    rank_error = max(abs(prior[state] - frozen[state]) for state in prior)
    if rank_error >= 1e-12:
        raise ValueError("Clinical rank correlations changed for the frozen direct sequences")
    if (qc["source"], qc["patients"], qc["visits"], qc["frozen_direct_frames_verified"]) != ("real", 24, 48, 10080):
        raise ValueError("Unexpected frame or sample counts")
    counts = {}
    for variant in ("peak", "direct"):
        rows = [row for row in repeat_rows.values() if row["variant"] == variant]
        if len(rows) != 42 or {row["contrast"] for row in rows} != {"off_severity", "on_severity", "on_improvement_pct"}:
            raise ValueError("Incomplete clinical contrast family")
        counts[variant] = {"tests": len(rows), "q_below_0_01": sum(row["q_bh42"] < .01 for row in rows)}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "stats.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in repeat["tests"])
    quality = {key: qc[key] for key in ("source", "patients", "visits", "tr_sec", "n_roi",
                                         "frozen_direct_frames_verified", "total_spatial_peaks", "condition_qc")}
    quality.update(contrast_families=counts, rerun_max_abs_differences=differences,
                   frozen_off_correlations_max_abs_difference=rank_error,
                   primary_sensitivity=repeat["primary_sensitivity"],
                   environment={"backfit": qc["environment"], "clinical": repeat["environment"]})
    (output / "verification.json").write_text(json.dumps(quality, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backfit_qc", type=Path)
    parser.add_argument("clinical_run", type=Path)
    parser.add_argument("clinical_repeat", type=Path)
    parser.add_argument("preceding_stats", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    export(read_json(args.backfit_qc), read_json(args.clinical_run), read_json(args.clinical_repeat),
           args.preceding_stats, args.output_dir)
