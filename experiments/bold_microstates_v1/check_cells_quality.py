"""Sensitivity of the matched 4x4 heatmap to peak ties and archived ROI coverage."""

import argparse
import hashlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from matched24_analysis import shared_interior
from matched24_cells import BOOT_DRAWS, PARENT, TEMPLATE_SHA, cell_tests
from matched24_data import load_study
from peak_states import backfit, interpolate_peaks
from run import new_output, write_json
from same_band_analysis import SOURCES
from same_band_stats import SEED, SHIFT_DRAWS, joint_counts_by_shift, zero_motion_weights


def read_quality(root, people):
    """Match archived QC by scan, visit and alias before applying the 78-region rule."""
    aux = root / "V100脑深部刺激WEiDA复现_20261118/代码与辅助"
    qc_dir = aux / "影像质控对照"
    roster = (
        pd.read_csv(aux / "同人群精确度对照/同人群清单.csv").query("condition in ('OFF','ON')").copy()
    )
    ident = pd.read_csv(
        root / "WEiDA-raw-audit-private-20261116/own_pipeline_20261116/stn_replication/cohort.tsv",
        sep="\t",
    ).set_index("bids_label", verify_integrity=True)
    roster["person"] = ident.loc[roster.scan, "person_id"].to_numpy()
    np.testing.assert_array_equal(roster.condition, ident.loc[roster.scan, "condition"])
    qc_path = qc_dir / "逐扫描影像质控.csv"
    roi_path = qc_dir / "逐扫描逐脑区覆盖.csv"
    scans = roster.merge(
        pd.read_csv(qc_path),
        left_on=["scan", "alias", "condition"],
        right_on=["原始BIDS", "V100别名", "条件"],
        validate="one_to_one",
    )
    assert len(scans) == 48 and scans.groupby("person").size().eq(2).all()
    for visit in ("OFF", "ON"):
        np.testing.assert_array_equal(scans.loc[scans.condition == visit, "person"], people)
    regions = roster.merge(
        pd.read_csv(roi_path),
        left_on=["scan", "condition"],
        right_on=["原始BIDS", "条件"],
        validate="one_to_many",
    )
    assert len(regions) == 48 * 78 and not regions.duplicated(["scan", "AAL78脑区"]).any()
    summary = {}
    for label, src in zip(("105", "V100最新"), SOURCES):
        assert scans[label + "_检查状态"].eq("CHECKED").all()
        below = regions[label + "_覆盖率百分值"] < 20
        assert regions[label + "_覆盖率百分值"].between(0, 100).all()
        missing = below.groupby(regions.scan).sum().reindex(scans.scan)
        weakest = regions.groupby("scan")[label + "_覆盖率百分值"].min().reindex(scans.scan)
        np.testing.assert_allclose(weakest, scans[label + "_最低覆盖率百分值"], atol=1e-6, rtol=0)
        np.testing.assert_array_equal(78 - missing.to_numpy(), scans[label + "_达到20覆盖脑区数"])
        summary[src] = dict(
            mask_dice_median=scans[label + "_dice"].median(),
            mask_hd95_median_mm=scans[label + "_hd95_mm"].median(),
            weakest_roi_median_pct=scans[label + "_最低覆盖率百分值"].median(),
            scans_all_78_at_20pct=int((missing == 0).sum()),
            low_roi_scan_counts=regions.loc[below, "AAL78脑区"].value_counts().to_dict(),
        )
    scans["all_covered"] = (
        scans[["105_达到20覆盖脑区数", "V100最新_达到20覆盖脑区数"]].eq(78).all(axis=1)
    )
    eligible = scans.groupby("person").all_covered.all().reindex(people).to_numpy()
    summary.update(
        common_grid="MNI152NLin2009cAsym 2 mm",
        scans=len(scans),
        eligible_pairs=int(eligible.sum()),
        scans_dice_higher=int((scans["V100最新_dice"] > scans["105_dice"]).sum()),
        scans_hd95_lower=int((scans["V100最新_hd95_mm"] < scans["105_hd95_mm"]).sum()),
        scans_weakest_roi_higher=int(
            (scans["V100最新_最低覆盖率百分值"] > scans["105_最低覆盖率百分值"]).sum()
        ),
    )
    return eligible, summary


def midpoint_labels(scan, templates):
    """Compare the two assignments for equidistant frames bracketed by different states."""
    original, direct, _ = backfit(scan, templates)
    similarity = np.abs(scan.units @ templates.T)
    peaks = scan.peaks
    peak_labels = direct[peaks]
    valid_frames = scan.gfp > 0
    spatial, n_ties = interpolate_peaks(len(original), peaks, peak_labels, valid_frames, similarity)

    # Independently enumerate midpoints within each uninterrupted stretch of valid frames.
    invalid_seen = np.cumsum(~valid_frames)[peaks]
    candidates = (np.diff(invalid_seen) == 0) & (np.diff(peaks) % 2 == 0) & (np.diff(peak_labels) != 0)
    assert candidates.sum() == n_ties
    midpoints = ((peaks[1:] + peaks[:-1]) // 2)[candidates]
    neighbors = np.column_stack([peak_labels[:-1][candidates], peak_labels[1:][candidates]])
    choices = np.sort(neighbors, axis=1)
    best = similarity[midpoints[:, None], choices].argmax(axis=1)
    winners = choices[np.arange(len(midpoints)), best]
    scored = spatial[midpoints] >= 0
    np.testing.assert_array_equal(spatial[midpoints[scored]], winners[scored])
    return original, spatial, midpoints


def run(root, out):
    study = load_study(root)
    eligible, quality = read_quality(root, study.person_ids)
    template_file = PARENT / "shared_templates.tsv"
    assert hashlib.sha256(template_file.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == TEMPLATE_SHA
    templates = pd.read_csv(template_file, sep="\t", index_col=[0, 1], float_precision="round_trip")
    np.testing.assert_array_equal(templates.columns, study.axis)
    phase, amp = (templates.loc[m].sort_index().to_numpy() for m in ("WEiDA", "micro"))
    tables = {
        label: [[], []]
        for label in ("original", "original_common", "spatial_midpoint", "coverage_subset")
    }
    motion = np.empty((2, 24, 2))
    changed = np.zeros(2, dtype=int)
    for p in range(24):
        for visit in range(2):
            original, spatial, phases, tied = [], [], [], []
            for si in range(2):
                visit_data = study.visits[si, p, visit]
                base, spatial_labels, midpoints = midpoint_labels(visit_data.peaks, amp)
                tied.append(midpoints)
                original.append(base)
                spatial.append(spatial_labels)
                phases.append(cdist(visit_data.features, phase, "sqeuclidean").argmin(1))
                motion[si, p, visit] = visit_data.peaks.fd
            baseline = shared_interior(original)
            common = baseline & shared_interior(spatial)
            ix = np.flatnonzero(common)
            assert len(ix) > 1 and (np.diff(ix) == 1).all()
            for si in range(2):
                differences = np.flatnonzero(common & (original[si] != spatial[si]))
                assert np.isin(differences, tied[si]).all()
                changed[si] += len(differences)
                for name, labels, mask in (
                    ("original", original, baseline),
                    ("original_common", original, common),
                    ("spatial_midpoint", spatial, common),
                ):
                    counts = joint_counts_by_shift(labels[si][mask], phases[si][mask]).transpose(0, 2, 1)
                    tables[name][si].append(counts)
                    if name == "original" and eligible[p]:
                        tables["coverage_subset"][si].append(counts)
    reference = pd.read_csv(PARENT.parent / "cooccurrence_cells/results/cells.csv")
    rows, patient_s2 = [], {}
    delta_fd = motion[:, :, 1] - motion[:, :, 0]
    for name, data in tables.items():
        selected = eligible if name == "coverage_subset" else np.ones(24, dtype=bool)
        frames = sum(len(t) for t in data[0])
        np.testing.assert_array_equal([len(t) for t in data[0]], [len(t) for t in data[1]])
        patient_s2[name] = []
        for si, src in enumerate(SOURCES):
            results, values = cell_tests(data[si], delta_fd[si, selected], [SEED, si, 0, 99])
            if name == "original":
                ref = reference.loc[reference.source == src].sort_values(["weida", "micro"])
                for col in (
                    "raw_frames",
                    "observed_pct",
                    "shift_expected_pct",
                    "excess_pp",
                    "extreme",
                    "p_upper",
                    "holm_16",
                ):
                    np.testing.assert_allclose([r[col] for r in results], ref[col], rtol=0, atol=1e-11)
            patient_s2[name].append(values[:, 0])
            rows.extend(
                dict(scenario=name, source=src, n_pairs=int(selected.sum()), frames=frames, **r)
                for r in results
            )
    # Paired change in S2-M1 excess co-occurrence on identical frames.
    bootstrap_seed = [SEED, 24, 16, 3]
    draws = np.random.default_rng(bootstrap_seed).integers(24, size=(BOOT_DRAWS, 24))
    estimates, boots = [], []
    for si, src in enumerate(SOURCES):
        diff = patient_s2["spatial_midpoint"][si] - patient_s2["original_common"][si]
        estimates.append(zero_motion_weights(delta_fd[si]) @ diff)
        boots.append((zero_motion_weights(delta_fd[si, draws]) * diff[draws]).sum(1))
    radius = np.quantile(np.max(np.abs(np.stack(boots, axis=1) - estimates), axis=1), 0.95)
    directory = new_output(out)
    pd.DataFrame(rows).to_csv(directory / "all_cells.csv", index=False)
    write_json(
        directory / "summary.json",
        dict(
            data_kind="real",
            quality=quality,
            changed_midpoint_frames=dict(zip(SOURCES, changed)),
            shared_template="matched24_primary/results/shared_templates.tsv",
            midpoint_minus_original_s2_m1_pp=dict(zip(SOURCES, estimates)),
            simultaneous_95_ci_pp={
                src: [effect - radius, effect + radius] for src, effect in zip(SOURCES, estimates)
            },
            simultaneous_95_bootstrap_radius_pp=radius,
            bootstrap_draws=len(draws),
            bootstrap_seed=bootstrap_seed,
            frame_rule="intersection of both pipelines and both boundary rules for midpoint comparison",
            rotation_draws=SHIFT_DRAWS,
            family_n=16,
            python=platform.python_version(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            pandas=pd.__version__,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root, args.out)
