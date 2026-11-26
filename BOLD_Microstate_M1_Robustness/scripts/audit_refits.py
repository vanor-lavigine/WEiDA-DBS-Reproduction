"""Audit the existing V4 formal refit archives without rerunning model fitting."""

import argparse
from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np


MODULE = Path(__file__).resolve().parents[1]
CONTRASTS = ("OFF-HC", "ON-HC", "ON-OFF")
METRICS = ("coverage_1", "gev_1")
STATE_METRICS = ("coverage", "duration", "occurrence", "gev", "dispersion")
FEATURES = (
    tuple(f"{metric}_{state}" for metric in STATE_METRICS for state in range(1, 5))
    + tuple(f"transition_{i}_{j}" for i in range(1, 5) for j in range(1, 5))
    + ("switching_rate",)
    + tuple(f"ilr_{j}" for j in range(1, 4))
    + tuple(f"sorted_ilr_{j}" for j in range(1, 4))
)
METHODS = ("A0", "A1", "A2", "A3")
CHUNK_SIZE = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_manifest_files(root, entries, paths):
    hashes = {item["path"]: item["sha256"] for item in entries}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        body = path.read_bytes()
        if path.suffix in {".csv", ".json", ".py"}:
            body = body.replace(b"\r\n", b"\n")
        require(sha256(body).hexdigest() == hashes[relative], f"V4 freeze hash mismatch: {path.name}")


def archive_locations(v4, reliability, protocol, attempts):
    templates = reliability / "private/V4_Maps.npz"
    source = (
        v4 / "scripts/full_bootstrap.py",
        v4 / "scripts/micro_inputs.py",
        v4 / "configs/protocol.json",
        reliability / "scripts/v4_models.py",
        templates,
    )
    source_key = sha256(b"".join(path.read_bytes() for path in source)).hexdigest()[:16]
    full = v4 / "private/refits" / source_key
    earlier = sha256(json.dumps([METHODS, protocol["lambda"]]).encode("utf-8"))
    for path in (
        reliability / "private/V4_Scan_Metrics.csv",
        templates,
        reliability / "configs/inference.json",
        reliability / "configs/protocol.json",
        reliability / "scripts/v4_bootstrap.py",
        reliability / "scripts/v4_models.py",
    ):
        earlier.update(path.read_bytes())
    old = reliability / "private/bootstrap" / earlier.hexdigest()[:15]
    full_names = [f"{i:04}.npz" for i in range(0, attempts, CHUNK_SIZE)]
    old_names = [f"formal_{i:04}_{i + CHUNK_SIZE:04}.npz" for i in range(0, attempts, CHUNK_SIZE)]
    require(sorted(path.name for path in full.glob("*.npz")) == full_names,
            "Formal V4 significance chunk inventory differs from the locked sequence")
    require(sorted(path.name for path in old.glob("formal_*.npz")) == old_names,
            "Formal V4 reliability chunk inventory differs from the locked sequence")
    return [full / name for name in full_names], [old / name for name in old_names]


def load_archive(paths, schema):
    arrays = {key: [] for key in schema}
    fingerprints = {}
    group_hash = sha256()
    for path in paths:
        blob = path.read_bytes()
        digest = sha256(blob)
        fingerprints[str(path)] = digest.hexdigest()
        group_hash.update(digest.digest())
        with np.load(BytesIO(blob), allow_pickle=False) as data:
            require(set(data.files) == set(schema), "Unexpected fields in a frozen refit batch")
            for key, (shape, kind) in schema.items():
                values = data[key]
                require(values.shape == shape and values.dtype.kind == kind,
                        f"Unexpected dimension or type for archived field {key}")
                arrays[key].append(values)
    return {key: np.concatenate(value) for key, value in arrays.items()}, fingerprints, group_hash.hexdigest()


def compare_summary(full, matches, reference, attempts):
    keys = {(row["scheme"], row["contrast"], row["metric"]): row for row in reference}
    require(len(keys) == len(reference) == 2 * len(CONTRASTS) * len(FEATURES),
            "Unexpected row count in the frozen V4 bootstrap summary")
    maximum_ci_delta = 0.0
    for name in ("fixed", "full"):
        for i, contrast in enumerate(CONTRASTS):
            for j, metric in enumerate(FEATURES):
                row = keys[(name, contrast, metric)]
                values = full[name][:, i, j]
                defined = values[np.isfinite(values)]
                require(len(defined) == int(row["defined"]) and attempts == int(row["requested"]),
                        f"Frozen finite counts differ for {name} {contrast} {metric}")
                if len(defined) < 100:
                    continue
                expected = (
                    ("ci95_low", "ci95_high") if len(defined) == attempts
                    else ("matched_only_ci95_low", "matched_only_ci95_high")
                )
                expected_values = [float(row[column]) for column in expected]
                empirical = np.quantile(defined, [0.025, 0.975])
                np.testing.assert_allclose(empirical, expected_values, rtol=0, atol=1e-14)
                maximum_ci_delta = max(maximum_ci_delta, float(np.max(np.abs(empirical - expected_values))))
    distributions = []
    for i, contrast in enumerate(CONTRASTS):
        for metric in METRICS:
            values = full["full"][:, i, FEATURES.index(metric)]
            require(np.array_equal(np.isfinite(values), matches),
                    "A target M1 contrast does not coincide with the saved match indicator")
            sample = values[matches]
            low, q25, median, q75, high = np.quantile(sample, [0.025, 0.25, 0.5, 0.75, 0.975])
            distributions.append(dict(
                method="A3", state="M1", contrast=contrast, metric=metric, effect_unit="proportion_difference",
                interpretation="full_refit_matched_only", total_draws=attempts, matched_draws=len(sample),
                match_fraction=len(sample) / attempts,
                frozen_point_difference=float(keys[("full", contrast, metric)]["mean_difference"]),
                matched_only_mean=float(np.mean(sample)), matched_only_q025=low, matched_only_q25=q25,
                matched_only_median=median, matched_only_q75=q75, matched_only_q975=high,
                matched_only_positive_draws=int(np.count_nonzero(sample > 0)),
            ))
    return distributions, maximum_ci_delta


def audit(workspace):
    base = workspace / "WEiDA-DBS-V4"
    significance = base / "V4_Microstate_Statistical_Significance"
    reliability = base / "V4_BOLD_Microstate_Reliability"
    manifest = read_json(significance / "workflow/Freeze_Manifest.json")
    verify_manifest_files(significance, manifest["public_files"], [
        significance / path for path in (
            "configs/protocol.json", "scripts/full_bootstrap.py", "scripts/micro_inputs.py",
            "results/Bootstrap_Results.csv", "results/Bootstrap_Verification.json",
        )
    ])
    verify_manifest_files(workspace, manifest["local_dependencies"] + manifest["local_archive_files"], [
        reliability / path for path in (
            "configs/inference.json", "configs/protocol.json", "scripts/v4_models.py", "scripts/v4_bootstrap.py",
            "private/V4_Maps.npz", "private/V4_Scan_Metrics.csv", "results/V4_Bootstrap_Identifiability_formal.csv",
        )
    ])
    protocol = read_json(significance / "configs/protocol.json")
    old_protocol = read_json(reliability / "configs/inference.json")["bootstrap"]
    attempts = protocol["full_refit_draws"]
    require(attempts == old_protocol["formal_repetitions"] == 2000
            and protocol["full_refit_seed_base"] == old_protocol["formal_seed_base"]
            and tuple(protocol["contrasts"]) == CONTRASTS
            and (protocol["main_method"], protocol["k"], protocol["lambda"]) == ("A3", 4, 0.04),
            "Formal A3 protocols disagree")
    files, old_files = archive_locations(significance, reliability, protocol, attempts)
    current_shapes = {
        "fixed": ((CHUNK_SIZE, len(CONTRASTS), len(FEATURES)), "f"),
        "full": ((CHUNK_SIZE, len(CONTRASTS), len(FEATURES)), "f"),
        "matches": ((CHUNK_SIZE, 4), "b"),
        "status": ((CHUNK_SIZE,), "U"),
    }
    old_shapes = {
        "beta": ((CHUNK_SIZE, 2, len(METHODS), len(CONTRASTS), 23), "f"),
        "stable": ((CHUNK_SIZE, len(METHODS), 4), "b"),
        "status": ((CHUNK_SIZE,), "U"),
        "converged": ((CHUNK_SIZE, 5), "f"),
        "counts": ((CHUNK_SIZE,), "i"),
    }
    current, fingerprints, total_sha = load_archive(files, current_shapes)
    previous, old_fingerprints, old_total_sha = load_archive(old_files, old_shapes)
    matches = current["matches"]
    receipt = read_json(significance / "results/Bootstrap_Verification.json")
    actual = matches.sum(axis=0).tolist()
    history = {row["state"]: int(row["n_retrained_state_matches"]) for row in read_csv(
        reliability / "results/V4_Bootstrap_Identifiability_formal.csv") if row["method"] == "A3"}
    require(np.array_equal(matches, previous["stable"][:, METHODS.index("A3"), :]),
            "Formal archives differ in the saved A3 match flags")
    outcome_status = dict(Counter(current["status"].tolist()))
    successes = int(np.sum(current["status"] == "ok"))
    paired_matches = int(matches.all(axis=1).sum())
    require(np.array_equal(current["status"], previous["status"])
            and receipt["failures"] == outcome_status
            and successes == receipt["fully_refitted"] == receipt["draws"] == attempts
            and paired_matches == receipt["full_composition_matches"]
            and actual == receipt["matches_by_state"] == receipt["historical_a3_state_matches"]
            == [history[f"R{i}"] for i in range(1, 5)]
            and actual[0] == 1019,
            "V4 published and saved refit counts disagree")
    rows, maximum_ci_delta = compare_summary(current, matches[:, 0], read_csv(
        significance / "results/Bootstrap_Results.csv"), attempts)
    summary = dict(
        method="A3", draws=attempts, repeat_unit="participant stratified within original fold",
        same_resample_multiplicity_on_off=True, first_seed=protocol["full_refit_seed_base"],
        last_seed=protocol["full_refit_seed_base"] + attempts - 1,
        status_counts=outcome_status, matches_by_state={f"M{i}": n for i, n in enumerate(actual, 1)},
        all_four_match_count=paired_matches, m1_match_fraction=actual[0] / attempts,
        min_spatial_r=protocol["min_spatial_r"], min_bidirectional_margin=protocol["min_bidirectional_margin"],
        verified_against_original_public_summary=True, max_interval_absolute_difference=maximum_ci_delta,
        match_flags_identical_draw_by_draw_to_v4_reliability=True,
        v4_significance_archive=dict(files=len(files), group_sha256=total_sha,
                                    fields={key: list(shape) for key, (shape, _) in current_shapes.items()}),
        v4_reliability_archive=dict(files=len(old_files), group_sha256=old_total_sha,
                                   fields={key: list(shape) for key, (shape, _) in old_shapes.items()}),
        group_hash_method="SHA-256 of concatenated raw file SHA-256 bytes in draw-start order",
        inference="matched-only coverage and GEV contrast distributions; unmatched draws remain missing",
        refit_absolute_condition_readouts_available=False,
        refit_K_C_interval_available=False,
        missing_refit_K_C_inputs=["paired patient scan-level C, W, F after successful matching in each draw",
                                  "per-draw paired multiplicities linked to those readouts"],
    )
    return rows, summary, {**fingerprints, **old_fingerprints}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=MODULE.parents[1])
    args = parser.parse_args()
    rows, summary, fingerprints = audit(args.workspace.resolve(strict=True))
    results = MODULE / "results"
    manifests = MODULE / "private/manifests"
    results.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    with (results / "Refit_M1_Matched_Only.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for path, value in (
        (results / "Refit_Archive_Audit.json", summary),
        (manifests / "Refit_Input_Hashes.json", {"file_sha256": fingerprints}),
    ):
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"A3 refits: {summary['draws']} draws, {summary['matches_by_state']['M1']} matched M1. "
          "All archived contrasts match the frozen summary.")


if __name__ == "__main__":
    main()
