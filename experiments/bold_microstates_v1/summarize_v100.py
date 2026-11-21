#!/usr/bin/env python3
"""Verify and export group-level results from the V100 SyN9 analysis."""

import argparse
from collections import Counter
from itertools import groupby
from pathlib import Path
from shutil import copyfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from adapt_syn9 import read_json, require
from run import new_output, write_json
from stats import METRICS


CONDITIONS = ("HC", "OFF", "ON")
COLORS = ("#2A6F97", "#E69A32", "#548966", "#9C5573")
STATS_COLUMNS = (
    "family", "state", "metric", "family_size", "clinical_scale", "n_pool",
    "n_subjects", "n_missing_endpoint", "estimate_type", "estimate", "mean_off",
    "mean_on", "effect_dz", "ci_low", "ci_high", "ci_method", "n_bootstrap_valid",
    "p_raw", "p_holm", "n_permutations", "status", "reason", "comparison",
)


def tsv(path):
    return pd.read_csv(path, sep="\t", na_values=["n/a"])


def check_primary(fit, runs, metadata, centers, metrics, transitions):
    """Recompute assignment, segment metrics and transition counts on the real fits."""
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1, rtol=0, atol=1e-12)
    observed_weights = tsv(fit / "run_fit.tsv").merge(
        metadata[["run_id", "subject_id"]], on="run_id", validate="one_to_one")
    require(len(observed_weights) == 67, "Missing run training weights")
    np.testing.assert_allclose(observed_weights.groupby("subject_id")["weight_per_run"].sum(), 1)
    input_audit = tsv(fit / "input_audit.tsv").merge(
        runs[["run_id", "source_sha256"]], on="run_id", validate="one_to_one")
    require(len(input_audit) == 67 and input_audit["status"].eq("included").all()
            and input_audit["sha256"].eq(input_audit["source_sha256"]).all(),
            "Fitted files do not match all audited source fingerprints")
    labels_by_condition = {condition: [] for condition in CONDITIONS}
    condition_by_run = metadata.set_index("run_id")["condition"]
    max_metric_error = 0.0
    min_frame_norm = np.inf
    for row in runs.itertuples(index=False):
        series = tsv(row.roi_time_series)
        require(series.columns.tolist() == (fit / "roi_labels.txt").read_text().splitlines(),
                "Fitted ROI order differs from the audited series")
        signal = series.to_numpy()
        standardized = (signal - signal.mean(axis=0)) / signal.std(axis=0, ddof=1)
        norms = np.linalg.norm(standardized, axis=1)
        scores = standardized @ centers.T / norms[:, None]
        actual = tsv(fit / "sequences" / f"{row.run_id}.tsv")
        np.testing.assert_array_equal(actual["frame"], np.arange(210))
        np.testing.assert_allclose(actual["time_sec"], np.arange(210) * 2.1, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(actual["state"], scores.argmax(axis=1) + 1)
        np.testing.assert_allclose(actual["similarity"], scores.max(axis=1), atol=1e-12, rtol=0)
        min_frame_norm = min(min_frame_norm, float(norms.min()))
        labels = actual["state"].to_numpy()
        labels_by_condition[condition_by_run.loc[row.run_id]].append(labels)
        lengths = {state: [] for state in range(1, 5)}
        for state, segment in groupby(labels):
            lengths[state].append(sum(1 for _ in segment))
        for state, segments in lengths.items():
            expected = [sum(segments) / 210, len(segments) * 60 / 441,
                        np.mean(segments) * 2.1 if segments else np.nan]
            entry = metrics.loc[(metrics["run_id"] == row.run_id) & (metrics["state"] == state)]
            observed = entry.loc[:, list(METRICS)].to_numpy()[0]
            np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
            errors = np.abs(observed - expected)
            max_metric_error = max(max_metric_error, *errors[np.isfinite(errors)])
        transition_counts = Counter(zip(labels[:-1], labels[1:]))
        run_transitions = transitions.loc[transitions["run_id"].eq(row.run_id)]
        require(len(run_transitions) == 32, "Expected both complete 4×4 transition grids")
        for item in run_transitions.itertuples(index=False):
            count = transition_counts[item.from_state, item.to_state]
            if item.definition == "episode" and item.from_state == item.to_state:
                count = 0
            require(item.count == count, "Transition counts do not match adjacent valid frames")
    continuous = metrics.dropna(subset=["duration_sec"])
    identity_error = (continuous["coverage"]
                      - continuous["duration_sec"] * continuous["occurrence_per_min"] / 60).abs().max()
    require(identity_error < 1e-12, "The duration/occurrence/coverage identity failed")
    return labels_by_condition, {
        "assignment_verified_frames": 67 * 210,
        "metric_rows_recomputed": len(metrics),
        "transition_counts_recomputed": len(transitions),
        "min_standardized_frame_norm": min_frame_norm,
        "max_metric_recalculation_error": max_metric_error,
        "max_metric_identity_error": float(identity_error),
        "subjects_with_equal_total_training_weight": int(metadata["subject_id"].nunique()),
    }


def check_inference(stats, metrics, metadata):
    require(Counter(stats["family"]) ==
            {"on_off": 12, "clinical_change": 12, "clinical_off_duration": 4},
            "The inference output must retain the three complete planned families")
    require(not stats.duplicated(["family", "state", "metric"]).any(), "Duplicate inference endpoint")
    df = metrics.merge(metadata, on="run_id", validate="many_to_one")
    dbs = df.loc[df["cohort"].eq("DBS")]
    for row in stats.loc[stats["family"].eq("on_off")].itertuples(index=False):
        wide = dbs.loc[dbs["state"].eq(row.state)].pivot(
            index="subject_id", columns="condition", values=row.metric).dropna()
        require(len(wide) == row.n_subjects, "Endpoint-specific pair count failed")
        np.testing.assert_allclose(row.estimate, (wide["ON"] - wide["OFF"]).mean(),
                                   rtol=0, atol=1e-12, equal_nan=True)
    for _, family in stats.groupby("family"):
        raw = family["p_raw"].fillna(1).to_numpy()
        order = np.argsort(raw)
        expected = np.empty(len(raw))
        expected[order] = np.minimum(1, np.maximum.accumulate(raw[order] * np.arange(len(raw), 0, -1)))
        expected[family["p_raw"].isna()] = np.nan
        np.testing.assert_allclose(family["p_holm"], expected, rtol=0, atol=1e-12, equal_nan=True)


def plot_states(path, centers, labels_by_condition):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), layout="constrained",
                             gridspec_kw={"height_ratios": [1.75, 1, 1, 1]})
    extent = float(np.max(np.abs(centers)))
    image = axes[0].imshow(centers, aspect="auto", cmap="RdBu_r", vmin=-extent, vmax=extent)
    axes[0].set(title="Signed K=4 BOLD spatial templates", xlabel="AAL78 ROI index",
                yticks=range(4), yticklabels=[f"M{i}" for i in range(1, 5)])
    axes[0].set_xticks(np.arange(0, 78, 7), np.arange(1, 79, 7))
    fig.colorbar(image, ax=axes[0], label="Template weight")
    time = np.arange(211) * 2.1
    for ax, condition in zip(axes[1:], CONDITIONS):
        labels = np.stack(labels_by_condition[condition])
        counts = np.stack([(labels == state).mean(axis=0) for state in range(1, 5)])
        ax.stackplot(time, np.column_stack((counts, counts[:, -1])), step="post",
                     colors=COLORS, labels=[f"M{i}" for i in range(1, 5)])
        ax.set(xlim=(0, 441), ylim=(0, 1), yticks=[0, 0.5, 1],
               ylabel=f"{condition} fraction", title=f"{condition} · {len(labels)} scans")
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Seconds from scan start")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_paired(path, stats):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, layout="constrained")
    titles = ("Coverage (percentage points)", "Occurrence (events/min)", "Duration (s)")
    paired = stats.loc[stats["family"].eq("on_off")]
    for ax, metric, title, scale in zip(axes, METRICS, titles, (100, 1, 1)):
        values = paired.loc[paired["metric"].eq(metric)].sort_values("state")
        for row in values.itertuples(index=False):
            ax.hlines(row.state, row.ci_low * scale, row.ci_high * scale, color=COLORS[row.state - 1])
            ax.scatter(row.estimate * scale, row.state, color=COLORS[row.state - 1], s=32, zorder=2)
        ax.axvline(0, color="0.5", linewidth=1, linestyle="--", zorder=0)
        ax.set(title=title, xlabel="Mean paired ON − OFF", yticks=range(1, 5),
               yticklabels=[f"M{i}" for i in range(1, 5)])
    axes[0].invert_yaxis()
    fig.suptitle("STN-DBS paired differences · marginal 95% patient bootstrap intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(run_dir, destination):
    metadata = tsv(run_dir / "input/metadata.verified.tsv")
    runs = tsv(run_dir / "input/runs.tsv")
    primary = run_dir / "analysis_k4"
    centers = np.load(primary / "templates.npy", allow_pickle=False)
    primary_metrics = tsv(primary / "metrics.tsv")
    trans = tsv(primary / "transitions.tsv")
    labels, checks = check_primary(primary, runs, metadata, centers, primary_metrics, trans)
    stats = pd.read_csv(run_dir / "inference_k4/stats.csv").loc[:, list(STATS_COLUMNS)]
    check_inference(stats, primary_metrics, metadata)
    cases = ("analysis_k4", "pilot_k4", "analysis_k5", "analysis_k6", "analysis_k7", "unsigned_k4")
    fit_rows, descriptive, stability, maps = [], [], [], []
    for case in cases:
        directory = run_dir / case
        if case == "unsigned_k4" and not (directory / "summary.json").is_file():
            continue
        fit = read_json(directory / "summary.json")
        require((fit["status"], fit["data_kind"], fit["included_runs"], fit["n_valid_frames"],
                 fit["n_rois"], fit["seed"], fit["n_init"], fit["max_iter"]) ==
                ("completed", "real", 67, 14070, 78, 20261121, 100, 300), f"Incomplete fit {case}")
        check = tsv(directory / "input_audit.tsv")
        require(set(check.loc[check["status"].eq("included"), "run_id"]) == set(runs["run_id"]),
                f"Cohort changed in {case}")
        other = np.load(directory / "templates.npy", allow_pickle=False)
        match = centers @ other.T
        if fit["polarity"] == "unsigned":
            match = np.abs(match)
        ref, alt = linear_sum_assignment(-match)
        for i, j in zip(ref, alt):
            maps.append({"fit": case, "k4_state": i+1, "other_state": j+1,
                         "similarity": match[i, j], "method": "absolute_cosine"
                         if fit["polarity"] == "unsigned" else "signed_cosine"})
        repeats = tsv(directory / "initialization_stability.tsv")
        stability.append(repeats.assign(fit=case))
        group = tsv(directory / "metrics.tsv").merge(
            metadata[["run_id", "condition"]], on="run_id", validate="many_to_one")
        metrics = group.melt(id_vars=["condition", "state"], value_vars=list(METRICS),
                             var_name="metric", value_name="value")
        grouped = metrics.groupby(["condition", "state", "metric"])["value"]
        descriptive.append(grouped.agg(n="count", mean="mean", sd="std", median="median",
                                       q25=lambda x: x.quantile(0.25),
                                       q75=lambda x: x.quantile(0.75)).reset_index().assign(fit=case))
        loss = np.asarray(fit["initialization_objectives"])
        fit_rows.append({"fit": case, "mode": fit["mode"], "k": fit["k"],
                         "polarity": fit["polarity"], "n_runs": fit["included_runs"],
                         "n_valid_frames": fit["n_valid_frames"], "n_init": len(loss),
                         "best_n_iter": fit["n_iter"], "best_loss": fit["objective"],
                         "median_loss": np.median(loss), "max_loss": loss.max(),
                         "stability_min": repeats["template_cosine"].min(),
                         "stability_median": repeats["template_cosine"].median(),
                         "stability_max": repeats["template_cosine"].max()})

    probabilities = trans.merge(metadata[["run_id", "condition"]], on="run_id", validate="many_to_one")
    trans_summary = probabilities.groupby(
        ["condition", "definition", "from_state", "to_state"]
    )["probability"].agg(n_runs="count", mean_probability="mean").reset_index()
    motion = metadata.melt(
        id_vars="condition", value_vars=["framewise_displacement_mean", "dvars_mean", "std_dvars_mean"],
        var_name="metric", value_name="run_mean")
    motion_summary = motion.groupby(["condition", "metric"])["run_mean"].agg(
        n="count", median="median", q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75)).reset_index()
    output = new_output(destination)
    for name, frame in {
        "stats": stats, "condition_metrics": pd.concat(descriptive, ignore_index=True),
        "transitions": trans_summary, "motion": motion_summary,
        "fit_summary": pd.DataFrame(fit_rows), "template_matches": pd.DataFrame(maps),
        "initialization_stability": pd.concat(stability, ignore_index=True),
    }.items():
        frame.to_csv(output / f"{name}.csv", index=False, na_rep="NA")
    write_json(output / "numerical_checks.json", checks)
    copyfile(run_dir / "input/input_summary.json", output / "input_summary.json")
    copyfile(primary / "templates.tsv", output / "templates.tsv")
    plot_states(output / "states.png", centers, labels)
    plot_paired(output / "paired_effects.png", stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    export(args.run_dir.resolve(), args.out)
