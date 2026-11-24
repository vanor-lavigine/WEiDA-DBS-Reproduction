"""Apply frozen BOLD AAHC templates with direct and peak-interpolated assignments."""
import argparse
import json
import platform
from importlib.metadata import version
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from core import sequence_metrics
from peak_states import interpolate_peaks, prepare_scan
from run import digest, load_roi


def backfit(index, training, output):
    roster = pd.read_csv(training / "fit_roster.tsv", sep="\t").set_index("run_id", verify_integrity=True)
    pairs = roster.groupby("person").condition.agg(list)
    if len(pairs) != 24 or not all(sorted(v) == ["OFF", "ON"] for v in pairs):
        raise ValueError("Expected 24 paired patients")
    runs = pd.read_csv(index / "input/runs.tsv", sep="\t").set_index("run_id", verify_integrity=True)
    metadata = pd.read_csv(index / "input/metadata.verified.tsv", sep="\t",
                           usecols=["run_id", "condition", "cohort", "framewise_displacement_mean"]
                           ).set_index("run_id", verify_integrity=True)
    regions = json.loads((index / "aal78-order.json").read_text())
    fit = training / "fit"
    maps = pd.read_csv(fit / "templates.tsv", sep="\t", float_precision="round_trip")
    maps = maps[maps.method.eq("AAHC")].sort_values("state")
    if maps.state.tolist() != [1, 2, 3, 4] or maps.columns[2:].tolist() != regions:
        raise ValueError("Frozen maps and anatomical axis disagree")
    centers = maps[regions].to_numpy()
    archived = pd.read_csv(fit / "sequences.tsv", sep="\t")
    archived = archived[archived.method.eq("AAHC")].pivot(index="run_id", columns="frame", values="state")
    if set(archived.index) != set(roster.index) or archived.columns.tolist() != list(range(210)):
        raise ValueError("Unexpected frozen sequence axis")
    rows, sequences, exposure = [], [], []
    for rid, visit in roster.sort_index().iterrows():
        run = runs.loc[rid]
        ident = metadata.loc[rid]
        path = Path(run.roi_time_series)
        if (visit.condition != ident.condition or ident.cohort != "DBS" or run.tr_sec != 2.1
                or digest(path) != run.source_sha256 or run.source_sha256 != visit.roi_digest):
            raise ValueError("Frozen scan identity or input digest changed")
        fd = float(ident.framewise_displacement_mean)
        if not np.isfinite(fd) or fd < 0:
            raise ValueError("Invalid mean framewise displacement")
        values, names = load_roi(path)
        if values.shape != (210, 78) or set(names) != set(regions):
            raise ValueError("Unexpected ROI matrix")
        scan = prepare_scan(values[:, [names.index(name) for name in regions]], fd)
        if not np.all(scan.gfp > 0):
            raise ValueError("Original fit used 210 valid frames")
        similarity = np.abs(scan.units @ centers.T)
        direct = similarity.argmax(axis=1)
        np.testing.assert_array_equal(direct + 1, archived.loc[rid].to_numpy())
        interp, _ = interpolate_peaks(len(direct), scan.peaks, direct[scan.peaks], scan.gfp > 0, similarity)
        for method, labels in (("peak", interp), ("direct", direct)):
            measured = sequence_metrics(labels, 2.1, 4)
            valid = measured["n_valid_frames"]
            if valid == 0:
                raise ValueError("No assigned frames for this protocol")
            segments = [(state, sum(1 for _ in group)) for state, group in groupby(labels) if state >= 0]
            for item in measured["states"]:
                lengths = [n for state, n in segments if state + 1 == item["state"]]
                if (len(lengths), sum(lengths)) != (item["n_episodes"], item["n_state_frames"]):
                    raise ValueError("Independent episode counter disagrees")
                rows.append(dict(variant=method, run_id=rid, mean_fd=fd, n_valid_frames=valid, **item))
            sequences.append(pd.DataFrame({"run_id":rid, "variant":method, "frame":np.arange(210),
                                           "state":np.where(labels >= 0, labels + 1, -1)}))
            exposure.append(dict(variant=method, condition=visit.condition, frames=valid,
                                 episodes=len(segments), peaks=len(scan.peaks)))
    info = pd.DataFrame(exposure)
    summary = {
        "source":"real", "patients":24, "visits":48, "tr_sec":2.1, "n_roi":78,
        "frozen_direct_frames_verified":int(archived.size),
        "total_spatial_peaks":int(info[info.variant.eq("peak")].peaks.sum()),
        "condition_qc":info.groupby(["variant","condition"]).agg(
            visits=("frames","size"), usable_frames=("frames","sum"), min_frames=("frames","min"),
            max_frames=("frames","max"), mean_frames=("frames","mean"), total_episodes=("episodes","sum")
        ).reset_index().to_dict("records"),
        "environment":{"python":platform.python_version(), **{p:version(p) for p in ("numpy","scipy","pandas")}},
    }
    output.mkdir(exist_ok=False)
    pd.DataFrame(rows).to_csv(output / "metrics.tsv", sep="\t", index=False, na_rep="n/a")
    pd.concat(sequences, ignore_index=True).to_csv(output / "sequences.tsv.gz", sep="\t", index=False)
    (output / "qc.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backfit(args.index, args.training, args.output)
