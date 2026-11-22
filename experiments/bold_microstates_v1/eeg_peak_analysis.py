"""Run the prespecified GFP-peak/unsigned comparison on both archived real BOLD inputs."""

import argparse
import platform
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from compare_legacy_105 import REPORTS, family, read_inputs, score
from core import assign_states, prepare_bold, sequence_metrics
from peak_states import K, backfit, group_maps, match_to_signed, prepare_scan
from run import new_output, write_json
from stats import METRICS


def train(items, reference):
    counts = Counter(person for person, scan in items)
    maps = [scan.individual for person, scan in items]
    weights = [1 / counts[person] for person, scan in items]
    raw, fit = group_maps(maps, weights)
    aligned, correlations, gap = match_to_signed(reference, raw)
    return aligned, correlations, {**fit, "people": len(counts), "assignment_gap_sum_abs_r": gap}


def metrics(key, fd, labels):
    measured = sequence_metrics(labels, 2.1, K)
    if np.any(labels >= 0):
        np.testing.assert_allclose(sum(s["coverage"] for s in measured["states"]), 1)
    return [dict(row, subject_id=key[0], condition=key[1], fd=fd) for row in measured["states"]]


def segment(prepared, templates):
    peak_rows, direct_rows, diagnostics, sequences = [], [], [], {}
    for key, scan in prepared.items():
        peak, direct, info = backfit(scan, templates)
        peak_rows.extend(metrics(key, scan.fd, peak))
        direct_rows.extend(metrics(key, scan.fd, direct))
        diagnostics.append(dict(condition=key[1], **info))
        sequences[key] = peak
    return pd.DataFrame(peak_rows), pd.DataFrame(direct_rows), pd.DataFrame(diagnostics), sequences


def paired_condition_summary(measured, contrasts):
    """Summarize each endpoint over the same complete patient pairs used for inference."""
    patients = measured.loc[measured.condition.isin(["OFF", "ON"])]
    long = patients.melt(id_vars=["subject_id", "state", "condition"], value_vars=METRICS,
                         var_name="metric")
    visits = long.pivot(index=["state", "metric", "subject_id"], columns="condition", values="value")
    complete = visits[["OFF", "ON"]].dropna()
    means = complete.groupby(["state", "metric"]).agg(
        counted_pairs=("OFF", "size"), off_mean=("OFF", "mean"), on_mean=("ON", "mean"))
    effects = contrasts.set_index(["state", "metric"], verify_integrity=True)
    means = means.reindex(effects.index)
    np.testing.assert_array_equal(means.counted_pairs.fillna(0), effects.n_pairs)
    estimated = effects.original_delta.notna()
    np.testing.assert_allclose((means.on_mean - means.off_mean)[estimated],
                               effects.original_delta[estimated], rtol=0, atol=1e-12)
    columns = ["original_delta", "adjusted_delta", "adjusted_ci_low", "adjusted_ci_high",
               "adjusted_p_raw", "adjusted_p_holm"]
    return pd.concat([effects.n_pairs, means.off_mean, means.on_mean, effects[columns]], axis=1)


def aggregate(frame, subset):
    rows = []
    groups = [("all", frame), *frame.groupby("condition", sort=True)]
    for group, part in groups:
        for metric in frame.columns.drop("condition"):
            x = part[metric].dropna()
            rows.append(dict(analysis=subset, condition=group, metric=metric, scans=len(x),
                             median=x.median(), q25=x.quantile(.25), q75=x.quantile(.75),
                             minimum=x.min(), maximum=x.max()))
    return rows


def transition_of_labels(source, raw, reference, sequences):
    counts = {c: np.zeros((K, K), dtype=int) for c in ("OFF", "ON", "HC")}
    for key, labels in sequences.items():
        x = prepare_bold(raw[key].values)
        old = assign_states(x["unit_vectors"], reference, x["valid"], "signed")["labels"]
        use = (old >= 0) & (labels >= 0)
        np.add.at(counts[key[1]], (old[use], labels[use]), 1)
    counts["all"] = sum(counts.values())
    return [dict(source=source, condition=c, old_signed_state=i+1, new_unsigned_state=j+1,
                 frames=int(table[i,j]), fraction_of_old=table[i,j]/table[i].sum() if table[i].sum() else np.nan,
                 fraction_of_new=table[i,j]/table[:,j].sum() if table[:,j].sum() else np.nan)
            for c, table in counts.items() for i in range(K) for j in range(K)]


def method_change(peak, baseline):
    keys = ["subject_id", "condition", "state"]
    new = peak.set_index(keys).sort_index()
    old = baseline.set_index(keys).sort_index()
    pd.testing.assert_index_equal(new.index, old.index)
    np.testing.assert_array_equal(new.fd, old.fd)
    change = new[["fd", *METRICS]].copy()
    change.loc[:, list(METRICS)] = new[list(METRICS)] - old[list(METRICS)]
    return change.reset_index()


def baseline_check(table, archived):
    actual = table.set_index(["state", "metric"]).sort_index()
    expected = archived.set_index(["state", "metric"]).sort_index()
    pd.testing.assert_index_equal(actual.index, expected.index)
    cols = ["n_pairs", "original_delta", "original_p_holm", "adjusted_delta", "adjusted_ci_low",
            "adjusted_ci_high", "adjusted_p_raw", "adjusted_p_holm", "n_extreme"]
    np.testing.assert_allclose(actual[cols], expected[cols], rtol=1e-11, atol=1e-12)


def spatial_rows(kind, correlation):
    rows = []
    for i, row in enumerate(correlation):
        order = np.argsort(-np.abs(row), kind="stable")
        for j, value in enumerate(row):
            rows.append(dict(comparison=kind, reference=i+1, target=j+1, spatial_r=value, abs_r=abs(value),
                             assigned_diagonal=i == j, best_free_target=order[0]+1,
                             best_minus_next_abs_r=abs(row[order[0]])-abs(row[order[1]])))
    return rows


def main(root, destination):
    signed = pd.read_csv(REPORTS / "templates.tsv", sep="\t", index_col=0)
    old_105 = pd.read_csv(REPORTS / "legacy_105/results/templates_105.tsv", sep="\t", index_col=0)
    pd.testing.assert_index_equal(signed.columns, old_105.columns)
    pd.testing.assert_index_equal(signed.index, old_105.index)
    sources = ("105", "SyN9")
    references = dict(zip(sources, (old_105.to_numpy(), signed.to_numpy())))
    raw = dict(zip(sources, read_inputs(root, list(signed.columns))))
    archived = pd.read_csv(REPORTS / "legacy_105/results/paired_results.csv")
    old_name = {"105": "legacy_refit_25", "SyN9": "syn9_original_26"}
    prepared, templates, tables, fits, matched, shared_labels = {}, {}, {}, {}, {}, {}
    condition_summaries = {}
    diagnostics, crosswalk, spatial = [], [], []
    for src in sources:
        prepared[src] = {key: prepare_scan(s.values, s.mean_fd) for key, s in raw[src].items()}
        maps = [(key[0], scan) for key, scan in prepared[src].items()]
        templates[src], corr, fits[src] = train(maps, references[src])
        spatial.extend(spatial_rows(src + "_old_to_peak", corr))
        spatial.extend(spatial_rows(src + "_old_to_old", np.corrcoef(references[src])))
        print(f"Fitted {src}: {len(maps)} scans.", flush=True)
        peak, direct, diag, sequences = segment(prepared[src], templates[src])
        diagnostics.extend(aggregate(diag, src + "_full"))
        crosswalk.extend(transition_of_labels(src, raw[src], references[src], sequences))
        base = score(raw[src], references[src])
        for variant, frame in [("old_signed", base), ("peak_interp", peak), ("peak_direct", direct),
                               ("paired_method_change", method_change(peak, base))]:
            tables[src, variant] = family(frame)[0]
        condition_summaries[src] = paired_condition_summary(peak, tables[src, "peak_interp"])
        baseline_check(tables[src, "old_signed"], archived.loc[archived.analysis.eq(old_name[src])])
    spatial.extend(spatial_rows("105_peak_to_SyN9_peak", templates["105"] @ templates["SyN9"].T))

    common = sorted(raw["105"].keys() & raw["SyN9"].keys())
    assert len(common) == 63 and sum(c == "OFF" for _, c in common) == 24
    trimmed = 0
    for src in sources:
        matched[src] = {}
        for key in common:
            offset = raw["105"][key].first_frame - raw[src][key].first_frame
            assert offset in (0, 1)
            scan = raw[src][key]
            matched[src][key] = prepare_scan(scan.values[offset:], scan.mean_fd) if offset else prepared[src][key]
            trimmed += offset
    pooled = [(key[0], matched[src][key]) for key in common for src in sources]
    templates["shared"], corr, fits["shared"] = train(pooled, references["SyN9"])
    spatial.extend(spatial_rows("SyN9_old_to_shared_peak", corr))
    for src in sources:
        peak, _, diag, shared_labels[src] = segment(matched[src], templates["shared"])
        tables[src, "common24_peak"] = family(peak)[0]
        diagnostics.extend(aggregate(diag, src + "_common"))

    paired_qc = []
    for key in common:
        a, b = (matched[src][key] for src in sources)
        u, v = (shared_labels[src][key] for src in sources)
        both = (u >= 0) & (v >= 0)
        coincidence = np.intersect1d(a.peaks, b.peaks).size
        separation = np.abs(a.peaks[:, None] - b.peaks)
        paired_qc.append(dict(condition=key[1], peak_difference_105_minus_SyN9=len(a.peaks)-len(b.peaks),
                              gfp_timecourse_r=np.corrcoef(a.gfp, b.gfp)[0,1],
                              identical_peak_jaccard=coincidence/(len(a.peaks)+len(b.peaks)-coincidence),
                              fraction_105_peaks_with_SyN9_within_one_TR=(separation.min(axis=1) <= 1).mean(),
                              fraction_SyN9_peaks_with_105_within_one_TR=(separation.min(axis=0) <= 1).mean(),
                              both_retained_fraction=both.mean(),
                              common_label_agreement=np.mean(u[both] == v[both]) if both.any() else np.nan))
    diagnostics.extend(aggregate(pd.DataFrame(paired_qc), "paired_cross_pipeline"))
    spatial_frame = pd.DataFrame(spatial)
    out = new_output(destination)
    stats = pd.concat(tables, names=["source", "variant", "row"]).droplevel("row")
    stats.to_csv(out / "paired_stats.csv", na_rep="NA")
    pd.concat(condition_summaries, names=["source"]).to_csv(out / "condition_summary.csv", na_rep="NA")
    pd.DataFrame(diagnostics).to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(crosswalk).to_csv(out / "frame_crosswalk.csv", index=False)
    spatial_frame.to_csv(out / "spatial_matches.csv", index=False)
    all_maps = {k: pd.DataFrame(v, columns=signed.columns, index=signed.index) for k,v in templates.items()}
    pd.concat(all_maps, names=["template_set", "state"]).to_csv(out / "templates.tsv", sep="\t")
    write_json(out / "summary.json", {"data_kind": "real", "k": K, "tr_sec": 2.1, "matched_scans": len(common),
        "matched_pairs": 24, "aligned_initial_frames": trimmed, "fit": fits, "verified_old_endpoints": 24,
        "seed": 20261121, "n_flips": 10000, "n_bootstrap": 2000, "python": platform.python_version(),
        "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__})
    print(f"Completed {len(stats)} endpoint rows, including the verified baselines.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    main(args.data_root, args.out)
