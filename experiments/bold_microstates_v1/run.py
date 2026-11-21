#!/usr/bin/env python3
"""Microstate-inspired BOLD experiment; original images/derivatives are read-only.

This implements a BOLD adaptation, not EEG A-D identification. See PROTOCOL.md.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from core import prepare_bold, spherical_kmeans, assign_states, sequence_metrics


METADATA_COLUMNS = ["run_id", "subject_id", "cohort", "condition", "mapping_source",
                    "clinical_scale", "clinical_score"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n", encoding="utf-8")


def new_output(path):
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def read_table(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def prepare_index(args):
    source = Path(args.index).resolve()
    table = read_table(source)
    needed = {"roi_time_series", "atlas", "format"}
    if not needed.issubset(table.columns):
        raise ValueError("results.tsv must contain: " + ", ".join(sorted(needed)))
    output = new_output(args.out)
    rows, audit, seen = [], [], set()
    for number, row in table.iterrows():
        reason = ""
        if row["atlas"] != args.atlas:
            reason = "different_atlas"
        elif row["format"].lower() != args.format:
            reason = "different_format"
        elif "label" in table and row["label"] != "COMPUTATIONAL_CHECKS_PASS":
            reason = "upstream_computational_status_not_pass"
        path = Path(row["roi_time_series"])
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        if not reason and str(path) in seen:
            reason = "duplicate_roi_time_series_path"
        if not reason and not path.is_file():
            reason = "time_series_file_not_found"
        if not reason:
            seen.add(str(path))
            # A technical identifier, deliberately not an inferred participant ID.
            run_id = "run_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
            rows.append({"run_id": run_id, "roi_time_series": str(path),
                         "tr_sec": args.tr, "atlas": row["atlas"],
                         "format": row["format"], "source_row": int(number) + 2,
                         "n_frames_expected": row.get("n_frames", ""),
                         "n_regions_expected": row.get("n_regions", ""),
                         "source_index": str(source)})
        audit.append({"source_row": int(number) + 2, "roi_time_series": str(path),
                      "status": "excluded" if reason else "selected", "reason": reason})
    columns = ["run_id", "roi_time_series", "tr_sec", "atlas", "format", "source_row",
               "n_frames_expected", "n_regions_expected", "source_index"]
    pd.DataFrame(rows, columns=columns).to_csv(output / "runs.tsv", sep="\t", index=False)
    metadata = pd.DataFrame("", index=range(len(rows)), columns=METADATA_COLUMNS)
    if rows:
        metadata["run_id"] = [r["run_id"] for r in rows]
    metadata.to_csv(output / "metadata.template.tsv", sep="\t", index=False)
    pd.DataFrame(audit).to_csv(output / "index_audit.tsv", sep="\t", index=False)
    summary = {"stage": "index", "selected_runs": len(rows), "atlas": args.atlas,
               "format": args.format, "tr_sec": args.tr, "index_sha256": digest(source),
               "subject_count": None, "note": "run counts are not participant counts"}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


def load_roi(path, selected=None):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        raw_header = next(csv.reader(handle, delimiter="\t"), [])
    if not raw_header or any(not name.strip() for name in raw_header):
        raise ValueError("empty ROI header or unnamed column")
    if len(raw_header) != len(set(raw_header)):
        raise ValueError("duplicate ROI names in source TSV header")
    table = pd.read_csv(path, sep="\t", na_values=["n/a", "NA", "na"])
    names = list(map(str, table.columns))
    if len(names) != len(set(names)):
        raise ValueError("duplicate ROI names")
    # Drop only an explicitly named time/index column, not the first anatomical ROI.
    if names and names[0].lower() in {"time", "frame", "index", "unnamed: 0"}:
        table = table.iloc[:, 1:]
    table.columns = table.columns.astype(str)
    if selected is not None:
        absent = [name for name in selected if name not in table.columns]
        if absent:
            raise ValueError("selected ROI names missing: " + ", ".join(absent[:5]))
        table = table.loc[:, selected]
    try:
        x = table.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("non-numeric ROI column; do not silently remove anatomical columns") from exc
    return x, list(table.columns)


def verified_metadata(path):
    if path is None:
        return None
    meta = read_table(path)
    required = {"run_id", "subject_id", "cohort", "condition", "mapping_source"}
    if not required.issubset(meta.columns):
        raise ValueError("metadata missing identity/evidence columns")
    for col in required:
        meta[col] = meta[col].str.strip()
    if meta["run_id"].duplicated().any():
        raise ValueError("duplicate run_id in metadata; resolve mapping without duplicating subjects")
    if meta["run_id"].eq("").any():
        raise ValueError("empty run_id in metadata")
    meta["cohort"] = meta["cohort"].str.upper()
    meta["condition"] = meta["condition"].str.upper()
    return meta.set_index("run_id", drop=False)


def fit(args):
    manifest = Path(args.runs).resolve()
    rows = read_table(manifest)
    required = {"run_id", "roi_time_series", "tr_sec", "atlas", "format"}
    if not required.issubset(rows.columns):
        raise ValueError("runs.tsv must contain run_id, roi_time_series, tr_sec")
    if rows["run_id"].duplicated().any():
        raise ValueError("runs.tsv contains duplicate run identifiers")
    if args.limit:
        rows = rows.iloc[:args.limit].copy()
    for column in ("atlas", "format"):
        if column in rows and rows[column].nunique() > 1:
            raise ValueError("one fit must use a single " + column)
    metadata = verified_metadata(args.metadata)
    if args.mode == "analysis" and metadata is None:
        raise ValueError("analysis needs source-backed identity metadata; pilot does not")
    if args.mode == "analysis" and args.limit:
        raise ValueError("--limit is for technical pilots, not an incomplete analysis cohort")
    selected = None
    if args.roi_list:
        selected = [line.strip() for line in Path(args.roi_list).read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
        if len(selected) < 2 or len(set(selected)) != len(selected):
            raise ValueError("ROI list needs at least two unique exact labels")
    output = new_output(args.out)
    (output / "sequences").mkdir()
    audit, records, roi_names = [], [], None
    content_seen = {}
    numeric_seen = {}
    for row in rows.to_dict("records"):
        run_id = row["run_id"]
        record = {"run_id": run_id, "status": "excluded", "reason": ""}
        try:
            if args.mode == "analysis":
                if run_id not in metadata.index:
                    raise ValueError("no_source_backed_identity_mapping")
                m = metadata.loc[run_id]
                if not m["subject_id"] or not m["mapping_source"]:
                    raise ValueError("unresolved_subject_or_mapping_evidence")
                allowed = ((m["cohort"] == "DBS" and m["condition"] in {"ON", "OFF"})
                           or (m["cohort"] == "HC" and m["condition"] == "HC"))
                if not allowed:
                    raise ValueError("outside_DBS_HC_discovery_cohort")
            path = Path(row["roi_time_series"])
            if not path.is_absolute():
                path = manifest.parent / path
            path = path.resolve()
            x, names = load_roi(path, selected)
            if roi_names is not None:
                if set(names) != set(roi_names) or len(names) != len(roi_names):
                    raise ValueError("ROI_definition_differs_from_common_feature_space")
                x = x[:, [names.index(n) for n in roi_names]]
                names = roi_names
            expected = row.get("n_frames_expected", "")
            if expected and len(x) != int(expected):
                raise ValueError("time_axis_differs_from_upstream_index")
            expected_regions = row.get("n_regions_expected", "")
            if selected is None and expected_regions and x.shape[1] != int(expected_regions):
                raise ValueError("ROI_count_differs_from_upstream_index")
            tr = float(row["tr_sec"])
            if not np.isfinite(tr) or tr <= 0:
                raise ValueError("invalid_TR")
            prepared = prepare_bold(x)
            if int(prepared["valid"].sum()) < args.k:
                raise ValueError("too_few_valid_frames")
            fingerprint = digest(path)
            if fingerprint in content_seen:
                raise ValueError("duplicate_input_content_of_" + content_seen[fingerprint])
            numeric_fingerprint = hashlib.sha256(np.ascontiguousarray(x, dtype="<f8").tobytes()).hexdigest()
            if numeric_fingerprint in numeric_seen:
                raise ValueError("duplicate_numeric_content_of_" + numeric_seen[numeric_fingerprint])
            content_seen[fingerprint] = run_id
            numeric_seen[numeric_fingerprint] = run_id
            if roi_names is None:
                roi_names = names
            subject = run_id
            if args.mode == "analysis":
                subject = str(metadata.loc[run_id, "subject_id"])
            records.append({"run_id": run_id, "subject": subject, "tr": tr,
                            "prepared": prepared, "path": path,
                            "sha256": fingerprint, "numeric_sha256": numeric_fingerprint, "source": row})
            record.update(status="included", n_frames=len(x), n_rois=x.shape[1],
                          n_valid=int(prepared["valid"].sum()), sha256=records[-1]["sha256"])
        except (ValueError, OSError, KeyError) as exc:
            record["reason"] = str(exc)
        audit.append(record)
    pd.DataFrame(audit).to_csv(output / "input_audit.tsv", sep="\t", index=False)
    if not records:
        write_json(output / "summary.json", {"stage": "fit", "status": "no_usable_runs"})
        raise ValueError("no usable runs; see input_audit.tsv")
    if args.mode == "analysis":
        # If one condition failed input checks, do not use the other as an independent subject.
        by_subject = {}
        for rec in records:
            by_subject.setdefault(rec["subject"], []).append(rec)
        retained = []
        for subject, group in by_subject.items():
            mapped = [metadata.loc[r["run_id"]] for r in group]
            cohorts = {r["cohort"] for r in mapped}
            conditions = sorted(r["condition"] for r in mapped)
            good = (cohorts == {"DBS"} and conditions == ["OFF", "ON"])
            good = good or (cohorts == {"HC"} and conditions == ["HC"])
            if good:
                retained.extend(group)
            else:
                for rec in group:
                    for item in audit:
                        if item["run_id"] == rec["run_id"]:
                            item.update(status="excluded", reason="incomplete_or_duplicate_subject_conditions")
        records = retained
        pd.DataFrame(audit).to_csv(output / "input_audit.tsv", sep="\t", index=False)
        if not records:
            raise ValueError("no complete DBS pairs or single-run HC observations after input checks")
    counts = {}
    for rec in records:
        counts[rec["subject"]] = counts.get(rec["subject"], 0) + 1
    train = np.concatenate([r["prepared"]["unit_vectors"][r["prepared"]["valid"]] for r in records])
    weights = np.concatenate([np.full(int(r["prepared"]["valid"].sum()),
                                      1.0 / (counts[r["subject"]] * int(r["prepared"]["valid"].sum())))
                              for r in records])
    print("Fitting %d runs, %d valid frames, %d ROI, K=%d" %
          (len(records), len(train), train.shape[1], args.k), flush=True)
    model = spherical_kmeans(train, k=args.k, n_init=args.n_init, max_iter=args.max_iter,
                             seed=args.seed, weights=weights, polarity=args.polarity)
    if model.get("n_occupied_clusters", args.k) != args.k:
        write_json(output / "summary.json", {"stage": "fit", "status": "degenerate_clusters",
                                              "k": args.k, "data_kind": args.data_kind})
        raise ValueError("fewer occupied clusters than K; inspect input rather than inventing empty states")
    centers = model["centers"]
    np.save(output / "templates.npy", centers)
    pd.DataFrame(centers, columns=roi_names, index=["M%d" % (k + 1) for k in range(args.k)]).to_csv(
        output / "templates.tsv", sep="\t", index_label="state")
    (output / "roi_labels.txt").write_text("\n".join(roi_names) + "\n")
    metric_rows, transitions, fit_rows, provenance = [], [], [], []
    for rec in records:
        prepared = rec["prepared"]
        assignment = assign_states(prepared["unit_vectors"], centers, valid=prepared["valid"],
                                   polarity=args.polarity)
        labels = assignment["labels"]
        sequence = pd.DataFrame({"frame": np.arange(len(labels)),
                                 "time_sec": np.arange(len(labels)) * rec["tr"],
                                 "state": np.where(labels >= 0, labels + 1, -1),
                                 "similarity": assignment["similarity"],
                                 "polarity_sign": assignment["polarity_sign"],
                                 "frame_norm": prepared["frame_norm"]})
        sequence.to_csv(output / "sequences" / (rec["run_id"] + ".tsv"), sep="\t", index=False,
                        na_rep="n/a")
        measured = sequence_metrics(labels, rec["tr"], args.k)
        for item in measured["states"]:
            metric_rows.append(dict(item, run_id=rec["run_id"], tr_sec=rec["tr"],
                                    n_valid_frames=int(prepared["valid"].sum())))
        for kind in ("frame", "episode"):
            for a in range(args.k):
                for b in range(args.k):
                    transitions.append({"run_id": rec["run_id"], "definition": kind,
                                        "from_state": a + 1, "to_state": b + 1,
                                        "count": measured[kind + "_transition_counts"][a, b],
                                        "probability": measured[kind + "_transition_probabilities"][a, b]})
        fit_rows.append({"run_id": rec["run_id"],
                         "mean_selected_similarity": float(np.nanmean(assignment["similarity"])),
                         "n_valid_frames": int(prepared["valid"].sum()),
                         "weight_per_run": 1.0 / counts[rec["subject"]]})
        np.savez_compressed(output / "sequences" / (rec["run_id"] + "_scaling.npz"),
                            temporal_mean=prepared["temporal_mean"], temporal_std=prepared["temporal_std"])
        provenance.append({"run_id": rec["run_id"], "input_path": str(rec["path"]),
                           "sha256": rec["sha256"], "numeric_sha256": rec["numeric_sha256"]})
    pd.DataFrame(metric_rows).to_csv(output / "metrics.tsv", sep="\t", index=False, na_rep="n/a")
    pd.DataFrame(transitions).to_csv(output / "transitions.tsv", sep="\t", index=False, na_rep="n/a")
    pd.DataFrame(fit_rows).to_csv(output / "run_fit.tsv", sep="\t", index=False)
    stability = []
    for repeat in range(args.stability_repeats):
        other = spherical_kmeans(train, k=args.k, n_init=min(args.n_init, 10), max_iter=args.max_iter,
                                  seed=args.seed + repeat + 1, weights=weights, polarity=args.polarity)
        similarity = centers @ other["centers"].T
        if args.polarity == "unsigned":
            similarity = np.abs(similarity)
        left, right = linear_sum_assignment(-similarity)
        for a, b in zip(left, right):
            stability.append({"repeat": repeat + 1, "reference_state": int(a) + 1,
                              "matched_state": int(b) + 1, "template_cosine": float(similarity[a, b])})
    pd.DataFrame(stability, columns=["repeat", "reference_state", "matched_state", "template_cosine"]).to_csv(
        output / "initialization_stability.tsv", sep="\t", index=False)
    summary = {"stage": "fit", "status": "completed", "mode": args.mode,
               "data_kind": args.data_kind,
               "atlas": str(rows["atlas"].iloc[0]), "format": str(rows["format"].iloc[0]),
               "representation": "temporally_zscored_unit_norm_ROI_BOLD",
               "polarity": args.polarity, "k": args.k, "seed": args.seed,
               "n_init": args.n_init, "included_runs": len(records),
               "max_iter": args.max_iter, "n_iter": model["n_iter"],
               "verified_subjects": len(counts) if args.mode == "analysis" else None,
               "n_complete_dbs_pairs": (len({r["subject"] for r in records
                                              if metadata.loc[r["run_id"], "cohort"] == "DBS"})
                                          if args.mode == "analysis" else None),
               "n_hc": (len({r["subject"] for r in records
                              if metadata.loc[r["run_id"], "cohort"] == "HC"})
                          if args.mode == "analysis" else None),
               "n_rois": len(roi_names), "n_valid_frames": len(train),
               "roi_scope": "explicit_label_list" if selected else "all_atlas_columns",
               "roi_list_sha256": digest(args.roi_list) if selected else None,
               "manifest_sha256": digest(manifest), "objective": model["objective"],
               "initialization_objectives": model["run_objectives"],
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "inference": "not_run_by_fit", "no_EEG_class_correspondence": True,
               "note": "Initialization stability is not test-retest or independent replication."}
    if args.mode == "analysis":
        identity_columns = ["run_id", "subject_id", "cohort", "condition", "mapping_source"]
        metadata.loc[[r["run_id"] for r in records], identity_columns].to_csv(
            output / "identity_used.tsv", sep="\t", index=False)
    write_json(output / "summary.json", summary)
    write_json(output / "input_provenance.json", provenance)
    plot_overview(output, centers, records, args.k)
    print(json.dumps({k: summary[k] for k in ["status", "mode", "included_runs", "n_rois", "k"]}))


def plot_overview(output, centers, records, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
    lim = float(np.max(np.abs(centers)))
    im = axes[0].imshow(centers, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[0].set(ylabel="State", xlabel="ROI (see roi_labels.txt)", title="BOLD spatial templates")
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels(["M%d" % (i + 1) for i in range(k)])
    fig.colorbar(im, ax=axes[0], label="Unit-norm template weight")
    first = records[0]
    seq = pd.read_csv(output / "sequences" / (first["run_id"] + ".tsv"), sep="\t")
    observed_states = seq["state"].astype(float).where(seq["state"] >= 0, np.nan)
    axes[1].step(seq["time_sec"], observed_states, where="post")
    axes[1].set(xlabel="Time (s)", ylabel="State", title="One run: observed BOLD state sequence")
    axes[1].set_yticks(range(1, k + 1))
    fig.savefig(output / "overview.png", dpi=160)
    plt.close(fig)


def infer(args):
    from stats import analyze
    source = Path(args.fit_dir).resolve()
    summary = json.loads((source / "summary.json").read_text())
    if summary.get("mode") != "analysis" or summary.get("polarity") != "signed" or summary.get("k") != 4:
        raise ValueError("prespecified inference requires a signed K=4 analysis fit; pilot/controls are descriptive")
    if summary.get("data_kind") != "synthetic" and (summary.get("atlas") != "AAL78" or
                                                     summary.get("format") != "nifti" or
                                                     summary.get("n_rois") != 78):
        raise ValueError("v1 real-data inference uses the 78-ROI AAL78 NIfTI-derived SyN9 cohort; other inputs are descriptive")
    metrics = pd.read_csv(source / "metrics.tsv", sep="\t", na_values=["n/a"])
    metadata = verified_metadata(args.metadata).reset_index(drop=True)
    # Fit and inference must refer to the same verified cohort and metadata file.
    provenance = json.loads((source / "input_provenance.json").read_text())
    included = {item["run_id"] for item in provenance}
    metadata = metadata.loc[metadata["run_id"].isin(included)].copy()
    if metadata["run_id"].duplicated().any():
        raise ValueError("duplicate metadata run identifiers")
    original = read_table(source / "identity_used.tsv")
    identity_keys = ["run_id", "subject_id", "cohort", "condition"]
    for column in identity_keys:
        metadata[column] = metadata[column].str.strip()
    for column in ["cohort", "condition"]:
        metadata[column] = metadata[column].str.upper()
    current = metadata.loc[:, identity_keys].sort_values("run_id").reset_index(drop=True)
    previous = original.loc[:, identity_keys].sort_values("run_id").reset_index(drop=True)
    if not current.equals(previous):
        raise ValueError("identity/condition mapping differs from model fit; refit on the corrected cohort")
    output = new_output(args.out)
    result = analyze(metrics, metadata, output, k=4, n_permutations=args.permutations,
                     n_bootstrap=args.bootstrap, seed=args.seed, clinical_scale=args.clinical_scale)
    result["data_kind"] = summary.get("data_kind", "unspecified")
    result["roi_scope"] = summary.get("roi_scope")
    result["atlas"] = summary.get("atlas")
    result["metadata_sha256"] = digest(args.metadata)
    write_json(output / "analysis_summary.json", result)
    print(json.dumps(clean_json(result), ensure_ascii=False))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    index = sub.add_parser("index", help="read existing XCP-D results.tsv; create an explicit input manifest")
    index.add_argument("--index", required=True)
    index.add_argument("--out", required=True)
    index.add_argument("--atlas", default="AAL78")
    index.add_argument("--format", choices=["cifti", "nifti"], default="nifti")
    index.add_argument("--tr", type=float, default=2.1)
    index.set_defaults(func=prepare_index)
    f = sub.add_parser("fit", help="fit common BOLD templates and describe every included run")
    f.add_argument("--runs", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--mode", choices=["pilot", "analysis"], default="pilot")
    f.add_argument("--data-kind", choices=["real", "synthetic"], default="real")
    f.add_argument("--metadata")
    f.add_argument("--roi-list")
    f.add_argument("--k", type=int, default=4)
    f.add_argument("--n-init", type=int, default=100)
    f.add_argument("--max-iter", type=int, default=300)
    f.add_argument("--seed", type=int, default=20261121)
    f.add_argument("--polarity", choices=["signed", "unsigned"], default="signed")
    f.add_argument("--stability-repeats", type=int, default=2)
    f.add_argument("--limit", type=int)
    f.set_defaults(func=fit)
    s = sub.add_parser("stats", help="patient-level paired and clinical inference from the prespecified fit")
    s.add_argument("--fit-dir", required=True)
    s.add_argument("--metadata", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--clinical-scale")
    s.add_argument("--permutations", type=int, default=10000)
    s.add_argument("--bootstrap", type=int, default=2000)
    s.add_argument("--seed", type=int, default=20261121)
    s.set_defaults(func=infer)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(2)
