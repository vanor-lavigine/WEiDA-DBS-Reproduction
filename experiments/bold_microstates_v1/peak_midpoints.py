"""Resolve tied peak-interpolated frames using the frame's actual BOLD spatial map."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from compare_legacy_105 import family, read_inputs
from eeg_peak_analysis import aggregate, metrics
from peak_states import backfit, interpolate_peaks, prepare_scan
from run import digest, new_output, write_json


def main(root, primary, destination):
    templates = pd.read_csv(primary / "templates.tsv", sep="\t", index_col=[0,1])
    axis = list(templates.columns)
    baseline = pd.read_csv(primary / "paired_stats.csv")
    stats, diagnostics = {}, []
    for source, data in zip(("105", "SyN9"), read_inputs(root, axis)):
        centers = templates.loc[source].to_numpy()
        rows, original_rows, changes = [], [], []
        for key, scan in data.items():
            prepared = prepare_scan(scan.values, scan.mean_fd)
            original, _, _ = backfit(prepared, centers)
            sim = np.abs(prepared.units @ centers.T)
            p = prepared.peaks
            states = sim[p].argmax(axis=1)
            new, n_ties = interpolate_peaks(len(sim), p, states, prepared.gfp > 0, sim)
            eligible = ((np.diff(p) % 2) == 0) & (states[:-1] != states[1:])
            assert int(eligible.sum()) == n_ties
            pos = ((p[:-1] + p[1:]) // 2)[eligible]
            lo = np.minimum(states[:-1], states[1:])[eligible]
            hi = np.maximum(states[:-1], states[1:])[eligible]
            switched = np.count_nonzero(sim[pos, hi] > sim[pos, lo])
            changes.append(dict(condition=key[1], ambiguous_midpoints=n_ties,
                                switches_before_edge_removal=switched,
                                fraction_switched_before_edges=switched/n_ties if n_ties else np.nan,
                                changed_frames_fraction=(original != new).mean(),
                                exposure_change_frames=int((new >= 0).sum() - (original >= 0).sum())))
            rows.extend(metrics(key, scan.mean_fd, new))
            original_rows.extend(metrics(key, scan.mean_fd, original))
        original_stats = family(pd.DataFrame(original_rows))[0].set_index(["state","metric"]).sort_index()
        old = baseline.loc[(baseline.source==source)&(baseline.variant=="peak_interp")]
        cols = ["n_pairs", "adjusted_delta", "adjusted_p_holm", "n_extreme"]
        np.testing.assert_allclose(original_stats[cols], old.set_index(["state","metric"]).sort_index()[cols],
                                   rtol=0, atol=1e-12)
        stats[source] = family(pd.DataFrame(rows))[0]
        diagnostics.extend(aggregate(pd.DataFrame(changes), source))
    out = new_output(destination)
    pd.concat(stats, names=["source", "row"]).droplevel("row").to_csv(out / "paired_stats.csv", na_rep="NA")
    pd.DataFrame(diagnostics).to_csv(out / "diagnostics.csv", index=False)
    write_json(out / "summary.json", {"data_kind": "real", "design": "post_hoc_midpoint_sensitivity",
               "primary_templates_sha256": digest(primary / "templates.tsv"), "original_rows_reproduced": 24})
    print("Completed the two twelve-endpoint spatial midpoint comparisons.")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", required=True, type=Path)
    cli.add_argument("--primary", required=True, type=Path)
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    main(args.data_root, args.primary, args.out)
