"""Relate whole-sequence BOLD switching rates to WEiDA S2 local metastability."""

import argparse
import hashlib
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from core import sequence_metrics
from matched24_analysis import shared_interior
from matched24_cells import PARENT, TEMPLATE_SHA
from matched24_data import load_study
from peak_states import backfit
from run import write_json
from same_band_analysis import SOURCES
from same_band_stats import BOOTSTRAPS, PERMUTATIONS, association
from stats import METRICS, holm_adjust
from verify_same_band import amplitude_labels, partial_from_correlations


TR = 2.1
SEED = 20261123


def switching_rate(labels, tr=TR):
    """Count changes per minute within one continuous window of valid labels."""
    labels = np.asarray(labels)
    valid_time_axis = np.isfinite(tr) and tr > 0 and labels.ndim == 1 and len(labels) >= 2
    if not valid_time_axis or not np.isin(labels, np.arange(4)).all():
        raise ValueError("Expected at least two consecutive, valid four-state labels and a positive TR")
    changes = np.count_nonzero(labels[1:] != labels[:-1])
    return changes * 60 / (len(labels) * tr)


def read_templates(axis):
    template_file = PARENT / "shared_templates.tsv"
    digest = hashlib.sha256(template_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if digest != TEMPLATE_SHA:
        raise ValueError("Shared templates differ from the matched 24-patient analysis")
    templates = pd.read_csv(template_file, sep="\t", index_col=[0, 1], float_precision="round_trip")
    np.testing.assert_array_equal(templates.columns, axis)
    for method in ("WEiDA", "micro"):
        np.testing.assert_array_equal(templates.loc[method].index, [1, 2, 3, 4])
    return templates.loc["WEiDA"].to_numpy(), templates.loc["micro"].to_numpy()


def measure_scans(data_root):
    study = load_study(data_root)
    phase, amplitude = read_templates(study.axis)
    s2_center = np.array([float(f"{weight:.5g}") for weight in phase[1]])
    scans = []
    for patient in range(len(study.person_ids)):
        for condition_index, condition in enumerate(("OFF", "ON")):
            visits = [study.visits[i, patient, condition_index] for i in range(len(SOURCES))]
            labels = [backfit(visit.peaks, amplitude)[0] for visit in visits]
            window = shared_interior(labels)
            n_frames = int(window.sum())
            for source, visit, sequence in zip(SOURCES, visits, labels):
                np.testing.assert_array_equal(sequence, amplitude_labels(visit.wave, amplitude))
                features = visit.features[window]
                state = cdist(features, phase, "sqeuclidean").argmin(axis=1)
                s2 = features[state == 1]
                if len(s2) < 2:
                    raise ValueError("Fewer than two S2 frames in a frozen analysis window")
                variance = np.square(s2 - s2_center).sum() / (features.shape[1] * (len(s2) - 1))
                rate = switching_rate(sequence[window])
                metrics = sequence_metrics(np.where(window, sequence, -1), TR, 4)["states"]
                episode_rate_sum = sum(row["occurrence_per_min"] for row in metrics)
                np.testing.assert_allclose(rate, episode_rate_sum - 60 / (n_frames * TR), atol=1e-12, rtol=0)
                row = dict(source=source, patient=patient, condition=condition,
                           frames=n_frames, s2_frames=len(s2), fd=visit.peaks.fd,
                           s2_variance=variance, switching_rate=rate)
                for number, values in enumerate(metrics, 1):
                    for metric in METRICS:
                        row[f"{number}_{metric}"] = values[metric]
                scans.append(row)
    return pd.DataFrame(scans).set_index(["source", "patient", "condition"]).sort_index()


def validate_baseline(deltas):
    """Recheck all published counts and coefficients using the common complete cases."""
    reference = pd.read_csv(PARENT / "associations.csv", dtype={"source": str})
    expected = {(source, state, metric) for source in SOURCES
                for state in range(1, 5) for metric in METRICS}
    keys = reference[["source", "state", "metric"]].itertuples(index=False, name=None)
    if set(keys) != expected or len(reference) != len(expected):
        raise ValueError("Incomplete reference table for the matched analysis")
    errors = []
    for row in reference.itertuples(index=False):
        endpoint = f"{row.state}_{row.metric}"
        cols = ["s2_variance", endpoint, "fd"]
        complete = np.logical_and.reduce([
            deltas.xs(source)[cols].notna().all(axis=1).to_numpy() for source in SOURCES
        ])
        values = deltas.xs(row.source)[cols].to_numpy()[complete]
        if len(values) != row.n_pairs:
            raise ValueError("Historical and reconstructed complete-case counts differ")
        rho = partial_from_correlations(*values.T)
        np.testing.assert_allclose(rho, row.partial_r, atol=1e-10, rtol=0)
        errors.append(abs(rho - row.partial_r))
    return float(max(errors))


def plot_result(result, path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "matched24-switch-rate"}):
        fig, ax = plt.subplots(figsize=(6.4, 2.3), layout="constrained")
        for y, row in enumerate(result.itertuples(index=False)):
            ax.plot([row.ci_low, row.ci_high], [y, y], color="#226c79", linewidth=2.6)
            ax.plot(row.partial_r, y, marker="o", color="#164453", markersize=7)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=1, zorder=0)
        ax.set(xlim=(-1, 1), ylim=(1.65, -0.65), yticks=[0, 1],
               yticklabels=["105", "V100 SyN9"], xlabel="Partial Spearman correlation (95% bootstrap CI)",
               title="Microstate switching rate and S2 local metastability")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        fig.savefig(path, metadata={"Date": None})
        plt.close(fig)


def run(data_root, output):
    if output.exists():
        raise FileExistsError(output)
    scans = measure_scans(data_root)
    counts = scans.groupby(level="source").agg(visits=("frames", "size"), valid_frames=("frames", "sum"))
    for source in SOURCES:
        np.testing.assert_array_equal(counts.loc[source], [48, 8650])
    off = scans.xs("OFF", level="condition")
    on = scans.xs("ON", level="condition")
    pd.testing.assert_index_equal(off.index, on.index)
    deltas = on - off
    reference_error = validate_baseline(deltas)
    rows = []
    for source_index, source in enumerate(SOURCES):
        values = deltas.xs(source)[["s2_variance", "switching_rate", "fd"]].to_numpy()
        if values.shape != (24, 3) or not np.isfinite(values).all():
            raise ValueError("Expected complete data from the 24 shared patients")
        effect = association(*values.T, seed=[SEED, source_index])
        if effect["status"] != "ok":
            raise ValueError("The switching-rate association could not be estimated")
        np.testing.assert_allclose(effect["partial_r"], partial_from_correlations(*values.T), atol=1e-12)
        rows.append(dict(source=source, **effect,
                         mean_switches_per_min_off=off.xs(source).switching_rate.mean(),
                         mean_switches_per_min_on=on.xs(source).switching_rate.mean()))
    result = pd.DataFrame(rows)
    result["p_holm_two"] = holm_adjust(result.p_raw.to_numpy())
    verification = dict(data_kind="real", shared_patients=24, visits_per_source=48,
                        valid_frames_per_source=8650, baseline_coefficients_verified=24,
                        max_abs_baseline_r_error=reference_error, permutations=PERMUTATIONS,
                        bootstrap_draws=BOOTSTRAPS, seed=SEED, python=platform.python_version(),
                        packages={name: version(name) for name in (
                            "numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl")})
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staging = Path(temp) / "results"
        staging.mkdir()
        result.to_csv(staging / "associations.csv", index=False)
        write_json(staging / "verification.json", verification)
        plot_result(result, staging / "switch_rate.svg")
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root, args.output)
