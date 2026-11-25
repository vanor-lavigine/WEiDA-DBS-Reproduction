"""Verify the frozen scans and recover features from archival labels and maps."""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

CONDITIONS = {1: "HC", 2: "OFF", 3: "ON"}
STATE = range(1, 5)
EDGES = [(i, j) for i in STATE for j in STATE if i != j]
AXIS = [f"ROI_{i}" for i in range(1, 79)]
MS_KEYS = ("coverage", "duration", "occurrence", "gev")
ARCHIVES = (
    "Group_Level_Summary", "All_Methods_Comparison", "HC_vs_ON_All_Results",
    "HC_vs_OFF_All_Results", "ON_vs_OFF_All_Results", "Common_PD_Effect",
    "HC_Both_Conditions_Significant",
)


def unit_maps(values):
    centered = values - values.mean(axis=1, keepdims=True)
    lengths = np.linalg.norm(centered, axis=1, keepdims=True)
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("Nonfinite or constant spatial map")
    return centered / lengths


def validate_roster(frame):
    if len(frame) != 65 or frame.scan.nunique() != 65:
        raise ValueError("Expected 65 distinct scans")
    group = frame.groupby("condition").subject.agg(list).to_dict()
    if {g: len(ids) for g, ids in group.items()} != {"HC": 15, "OFF": 25, "ON": 25}:
        raise ValueError("Freeze must contain HC=15, OFF=25 and ON=25")
    if any(len(set(ids)) != len(ids) for ids in group.values()):
        raise ValueError("Duplicate subject within a condition")
    if set(group["ON"]) != set(group["OFF"]) or set(group["HC"]) & set(group["ON"]):
        raise ValueError("Patient pairing or HC separation failed")


def read_v3_archive(v3):
    tables = {key: pd.read_csv(v3 / "results" / f"{key}.csv", low_memory=False) for key in ARCHIVES}
    all_methods = tables["All_Methods_Comparison"]
    views = {"HC_vs_ON_All_Results": "C1", "HC_vs_OFF_All_Results": "C2", "ON_vs_OFF_All_Results": "C3",
             "Common_PD_Effect": "C4", "HC_Both_Conditions_Significant": "both"}
    for name, contrast in views.items():
        expected = all_methods.loc[all_methods.contrast.eq(contrast)].reset_index(drop=True)
        actual = tables[name].reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_dtype=False, check_exact=False)
    return tables


def adjacency(labels):
    valid = labels >= 0
    eligible = valid[:-1] & valid[1:]
    count = np.zeros((4, 4), dtype=int)
    np.add.at(count, (labels[:-1][eligible], labels[1:][eligible]), 1)
    return count


def probabilities(counts, leaving_only=False):
    counts = counts.copy()
    if leaving_only:
        np.fill_diagonal(counts, 0)
    denominator = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, denominator, out=np.full((4, 4), np.nan), where=denominator > 0)


def label_features(labels, tr):
    """Same elapsed-time and segment definitions for either state representation."""
    from features_v3 import segments

    exposure = np.count_nonzero(labels >= 0) * tr
    if not exposure:
        raise ValueError("Empty effective scan")
    runs = segments(labels)
    count = adjacency(labels)
    leaving = probabilities(count, leaving_only=True)
    native = probabilities(count)
    row = {}
    for state in STATE:
        intervals = [(end - start) * tr for start, end, tag in runs if tag == state - 1]
        internal = [(end - start) * tr for start, end, tag in runs
                    if tag == state - 1 and start > 0 and end < len(labels)
                    and labels[start - 1] >= 0 and labels[end] >= 0]
        row[f"coverage_{state}"] = np.count_nonzero(labels == state - 1) * tr / exposure
        row[f"duration_{state}"] = np.mean(intervals) if intervals else np.nan
        row[f"occurrence_{state}"] = len(intervals) / exposure
        row[f"native_duration_{state}"] = np.mean(internal) if internal else np.nan
    coverage = np.array([row[f"coverage_{state}"] for state in STATE])
    nonzero = coverage[coverage > 0]
    row.update(duration_s=exposure, coverage_entropy=-(nonzero @ np.log(nonzero)) / np.log(4),
               global_pooled_duration=exposure / len(runs), global_occurrence=len(runs) / exposure,
               switching_rate=(count.sum() - count.trace()) / exposure)
    for i, j in EDGES:
        row[f"transition_{i}_{j}"] = leaving[i - 1, j - 1]
        row[f"native_transition_{i}_{j}"] = native[i - 1, j - 1]
    return row, native


def microstate_features(v3, scans, cfg):
    from features_v3 import measure
    from inference_correspondence import correspondence
    from input_data import transform
    from models_v3 import prepare

    features = pd.read_csv(v3 / "private" / "Subject_Level_Features.csv", low_memory=False)
    source = features.loc[features.experiment.eq(cfg["v3_method"]) & features.fit.eq(cfg["v3_fit"])]
    validate_roster(source)
    if not source.n_states.eq(4).all() or source.groupby("subject").fold.nunique().max() != 1:
        raise ValueError("Expected K=4 and one held-out fold for each participant")
    maps = pd.read_csv(v3 / "results" / "Group_Templates.csv")
    maps = maps.loc[maps.experiment.eq(cfg["v3_method"])]
    internal, order, qualified = correspondence(maps, cfg["min_abs_spatial_r"],
                                                cfg["min_bidirectional_margin"])
    maps = {int(fold): rows.sort_values("state")[AXIS].to_numpy()
            for fold, rows in maps.loc[maps.fold.ge(0)].groupby("fold")}
    scan_data = {row["scan"]: row for row in scans}
    if set(source.scan) != scan_data.keys():
        raise ValueError("V3 Feature table does not cover the frozen scans")
    output = []
    for row in source.sort_values("scan").to_dict("records"):
        scan = scan_data[row["scan"]]
        assert (row["condition"], row["subject"]) == (scan["condition"], scan["subject"])
        np.testing.assert_allclose(row["fd"], scan["fd"], rtol=0, atol=1e-14)
        path = v3 / "private" / "backfits" / cfg["v3_method"] / cfg["v3_fit"] / f"{row['scan']}.npz"
        with np.load(path) as backfit:
            labels = backfit["labels"]
            family = backfit["family"]
            valid = backfit["valid"]
            archived_signed = backfit["signed_similarity"]
        assert len(labels) == len(scan["x"]) and np.array_equal(family, np.arange(4))
        assert np.isin(labels, [-1, 0, 1, 2, 3]).all() and np.array_equal(valid, labels >= 0)
        prepared = prepare(transform(scan["x"], "F1"), "N0")
        np.testing.assert_array_equal(prepared.valid, valid)
        native_bank = maps[row["fold"]]
        signed = prepared.units @ native_bank.T
        np.testing.assert_allclose(signed[valid], archived_signed[valid], atol=4e-8, rtol=0)
        np.testing.assert_array_equal(np.abs(signed[valid]).argmax(axis=1), labels[valid])
        raw = measure(prepared, labels, np.abs(signed))
        verified = [f"{key}_{state}" for key in MS_KEYS for state in STATE]
        verified += [f"transition_{i}_{j}" for i, j in EDGES]
        verified += ["duration_s", "global_pooled_duration", "global_occurrence", "switching_rate"]
        np.testing.assert_allclose([raw[k] for k in verified], [row[k] for k in verified],
                                   atol=1e-12, rtol=0, equal_nan=True)

        centered_bank = unit_maps(native_bank)
        assigned_r = (prepared.units[valid] * centered_bank[labels[valid]]).sum(axis=1)
        distance = np.sqrt(2 * (1 - np.clip(np.abs(assigned_r), 0, 1)))
        coverage = np.array([row[f"coverage_{i}"] for i in STATE])
        nonzero = coverage[coverage > 0]
        new = {key: row[key] for key in ("scan", "subject", "condition", "fd", "fold", "duration_s",
                                        "global_pooled_duration", "global_occurrence", "switching_rate")}
        new.update(coverage_entropy=-(nonzero @ np.log(nonzero)) / np.log(4),
                   total_gev=sum(row[f"gev_{i}"] for i in STATE), global_dispersion=distance.mean())
        for i, original in enumerate(order[row["fold"]], 1):
            take = labels[valid] == original
            for key in MS_KEYS:
                new[f"{key}_{i}"] = row[f"{key}_{original + 1}"] if qualified else np.nan
            new[f"dispersion_{i}"] = distance[take].mean() if qualified and take.any() else np.nan
        for i, j in EDGES:
            a, b = order[row["fold"]][[i - 1, j - 1]] + 1
            new[f"transition_{i}_{j}"] = row[f"transition_{a}_{b}"] if qualified else np.nan
        output.append(new)
    banks = {fold: unit_maps(bank[order[fold]]) for fold, bank in maps.items()}
    return pd.DataFrame(output), banks, internal, qualified


def assert_close(a, b, tol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=tol, equal_nan=True)


def weida_features(root, scans, manifest, cfg):
    primary = root / "WEiDA_STN25_HC15_K4_20261117"
    original_manifest = pd.read_csv(primary / "input_manifest.csv")
    comparison = root / "WEiDA_STN25_HC15_K4_R2020b_20261117"
    pd.testing.assert_frame_equal(original_manifest, manifest)
    pd.testing.assert_frame_equal(pd.read_csv(comparison / "input_manifest.csv"), manifest)
    directory = next(root.glob("WEiDA_MATLAB*20261119")) / cfg["weida_release"]
    seed = loadmat(directory / f"seed{cfg['weida_seed']:02d}.mat")
    features = loadmat(directory / "features.mat")
    original = loadmat(primary / "author_k4.mat")
    for key in ("Vemp", "TR", "IDX", "P", "LT", "Time_all", "ind_sort"):
        np.testing.assert_array_equal(original[key], seed[key])
    np.testing.assert_array_equal(original["Time_all"], features["Time_all"])
    assert features["release"].item() == cfg["weida_release"] and seed["seed"].item() == cfg["weida_seed"]
    assert original["TR"].item() == features["TR"].item() == cfg["TR_seconds"]
    cells = loadmat(primary / "inputs.mat")["ts_cell"]
    seed_cells = loadmat(comparison / "inputs.mat")["ts_cell"]
    assert cells.shape == seed_cells.shape == (65, 2)
    centers = original["Vemp"]
    archive = primary / "WEiDA4_atlasAAL78_DBS"
    def csv(name):
        return np.loadtxt(archive / f"{name}.csv", delimiter=",")

    def verify_csv(expected, observed):
        # MATLAB csvwrite exports five significant digits. Use the MAT values for inference.
        np.testing.assert_allclose(expected, observed, rtol=5e-5, atol=1e-12, equal_nan=True)

    verify_csv(centers, csv("Vemp").T)
    labels = original["IDX"].ravel().astype(int) - 1
    np.testing.assert_array_equal(labels, csv("State_ts").astype(int) - 1)
    vectors = features["WEiDA_Eig"]
    assert len(labels) == 13647 and vectors.shape == (len(labels), 78) and np.isfinite(vectors).all()
    deviations = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    np.testing.assert_array_equal(deviations.argmin(axis=1), labels)
    centroid_error = max(np.abs(vectors[labels == i].mean(axis=0) - center).max()
                         for i, center in enumerate(centers))
    assert centroid_error < 1e-12

    counts = manifest.frames.to_numpy()
    expected_clock = np.stack([np.repeat(np.arange(1, 66), counts), np.repeat(manifest.condition, counts)])
    np.testing.assert_array_equal(original["Time_all"], expected_clock)
    pp = {g: csv(f"P{g}emp") for g in CONDITIONS}
    dd = {g: csv(f"LT{g}emp") for g in CONDITIONS}
    permutations = original["ind_sort"].ravel().astype(int) - 1
    it = list(zip(np.r_[0, np.cumsum(counts)[:-1]], np.cumsum(counts)))
    output, mapping = [], []
    by_id = {s["scan"]: s for s in scans}
    for pos, ((start, end), entry) in enumerate(zip(it, manifest.itertuples())):
        scan = by_id[entry.scan]
        cond = int(entry.condition)
        assert scan["roi_sha256"] == entry.sha256 and scan["condition"] == CONDITIONS[cond]
        assert scan["subject"] == (entry.scan if cond == 1 else entry.pair_id)
        x = scan["x"][entry.source_index_zero_based_start:]
        assert x.shape == (end - start, 78)
        np.testing.assert_array_equal(x.T, cells[pos, 0])
        np.testing.assert_array_equal(cells[pos, 0], seed_cells[pos, 0])
        assert int(cells[pos, 1].item()) == int(seed_cells[pos, 1].item()) == cond
        state = labels[start:end]
        row, native = label_features(state, cfg["TR_seconds"])
        occupancy = [row[f"coverage_{i}"] for i in STATE]
        lifetime = [row[f"native_duration_{i}"] for i in STATE]
        verify_csv(occupancy, pp[cond][pos])
        verify_csv(lifetime, dd[cond][pos])
        assert_close(occupancy, original["P"][cond - 1, pos, 0, permutations])
        assert_close(lifetime, original["LT"][cond - 1, pos, 0, permutations])
        verify_csv(np.nan_to_num(native, nan=0), csv(f"Ptrans{pos + 1}"))
        for i in STATE:
            use = state == i - 1
            row[f"var_{i}"] = deviations[start:end, i - 1][use].sum() / (78 * (use.sum() - 1)) if use.sum() > 1 else np.nan
        output.append({**{k: scan[k] for k in ("scan", "subject", "condition", "fd")}, **row})
        mapping.append(dict(archive_row=pos + 1, scan=entry.scan, subject=scan["subject"], condition=scan["condition"],
                            frames=end - start, frame_start=entry.source_index_zero_based_start,
                            roi_sha256=entry.sha256, exact_roi_values=True, fo_dt_transition_verified=True))
    data = pd.DataFrame(output)
    validate_roster(data)
    files = [primary / "author_k4.mat", primary / "inputs.mat", comparison / "inputs.mat",
             directory / f"seed{cfg['weida_seed']:02d}.mat", directory / "features.mat"]
    return data, centers, pd.DataFrame(mapping), float(centroid_error), files


def cross_method_matching(banks, centers, v3_qualified, cfg):
    rows, orders = [], []
    for fold, bank in sorted(banks.items()):
        signed = bank @ unit_maps(centers).T
        scores = np.abs(signed)
        i, j = linear_sum_assignment(-scores)
        orders.append(j)
        other = scores.copy()
        other[i, j] = -np.inf
        margins = scores[i, j] - np.maximum(other.max(axis=1), other.max(axis=0)[j])
        for left in range(4):
            for right in range(4):
                selected = right == j[left]
                margin = margins[left] if selected else np.nan
                adequate = selected and scores[left, right] >= cfg["min_abs_spatial_r"] and margin >= cfg["min_bidirectional_margin"]
                rows.append(dict(fold=fold, microstate=left + 1, weida_state=right + 1,
                                 spatial_r=signed[left, right], abs_spatial_r=scores[left, right],
                                 hungarian_selected=selected, bidirectional_margin=margin, pair_pass=adequate))
    table = pd.DataFrame(rows)
    same = all(np.array_equal(orders[0], j) for j in orders[1:])
    passed = bool(v3_qualified and same and table.loc[table.hungarian_selected, "pair_pass"].all())
    table = table.assign(same_order_in_all_folds=same, state_match_pass=passed)
    return table, dict(zip(STATE, orders[0] + 1)) if passed else {}
