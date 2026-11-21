"""Repeat signed K4 and the paired FD sensitivity comparison on archived 105 BOLD."""

import argparse
import gzip
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment

from adapt_syn9 import CONDITION_CODE, require
from core import assign_states, prepare_bold, sequence_metrics
from fd_sensitivity import adjusted_endpoint
from run import digest, fit, load_roi, new_output, parser as run_parser, write_json
from stats import METRICS, _bootstrap_mean, _rng, holm_adjust, signflip_pvalue


SEED, FLIPS, BOOTS = 20261121, 10000, 2000
REPORTS = Path(__file__).resolve().parents[2] / "reports/bold_microstates_v1"


@dataclass
class Scan:
    values: np.ndarray
    mean_fd: float
    first_frame: int


def compressed(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def mean_fd(trace):
    fd = np.asarray(trace["motion_fd"], dtype=float)
    require(trace["frames"] == 210 and fd.shape == (210,) and np.isnan(fd[0])
            and np.isfinite(fd[1:]).all() and (fd[1:] >= 0).all(), "Incomplete motion trace")
    return float(fd[1:].mean())


def read_inputs(root, axis):
    audits = root / "WEiDA-raw-audit-private-20261116"
    ident = pd.read_csv(audits / "own_pipeline_20261116/stn_replication/cohort.tsv",
                        sep="\t").set_index("bids_label", verify_integrity=True)
    aux = root / "V100脑深部刺激WEiDA复现_20261118/代码与辅助"
    evidence = aux / "两套预处理对照"
    same_raw = pd.read_csv(evidence / "逐扫描对照.csv").set_index("original_scan", verify_integrity=True)
    require(same_raw.raw_bold_and_t1w_match.eq(True).all(), "Raw scan correspondence is unresolved")

    archive = compressed(evidence / "105归档FC与删帧记录.json.gz")["scans"]
    capture = compressed(evidence / "105处理阶段实时核验.json.gz")["scans"]
    captured = {r["label"]: r for r in capture}
    require(len(captured) == len(capture), "Duplicate scan in historical capture")
    old = pd.read_csv(root / "WEiDA_STN25_HC15_K4_20261117/input_manifest.csv", keep_default_na=False)
    require(old.scan.is_unique and len(old) == 65, "Unexpected historical input manifest")
    series = root / "STN-DBS_105_AAL78_20261117/PD_DBS_zcc/derivatives/fmripost"
    legacy = {}
    for row in old.itertuples(index=False):
        identity, hist, saved = ident.loc[row.scan], captured[row.scan], archive[row.scan]
        condition = identity.condition
        require(hist["condition"] == condition and CONDITION_CODE[condition] == row.condition
                and (condition == "HC" or row.pair_id == identity.person_id),
                "Historical membership conflicts with confirmed scan mapping")
        path = series / row.scan / f"{row.scan}_BOLD_AAL1ctx_MNI_basic36_sm6.csv"
        require(digest(path) == row.sha256 == hist["artifacts"]["series"] == saved["legacy_series_sha256"],
                "Historical ROI file checksum differs")
        trace = hist["confounds"]
        start = row.source_index_zero_based_start
        require(start in (0, 1) and row.frames + start == saved["confounds_rows"] == 210
                and trace["sha256"] == saved["confounds_sha256"]
                and trace["initial_outliers"] == saved["non_steady_state_rows_zero_based"] == list(range(start)),
                "Historical time axis or motion source differs")
        frame = pd.read_csv(path, index_col=0, float_precision="round_trip")
        require(list(frame.columns) == axis and frame.shape == (row.frames, 78)
                and np.array_equal(frame.index, np.arange(row.frames)), "Historical ROI or frame axis differs")
        key = (identity.person_id, condition)
        require(key not in legacy, "Duplicate historical subject and condition")
        legacy[key] = Scan(frame.to_numpy(), mean_fd(trace), start)

    capture = compressed(evidence / "V100处理阶段实时核验.json.gz")["scans"]
    captured = {(r["alias"], r["condition"]): r for r in capture}
    require(len(captured) == len(capture), "Duplicate scan in SyN9 capture")
    pack = pd.read_csv(aux / "复现输入清单.csv")
    syn9 = {}
    review = audits / "syn9_stn_hc_batch/review"
    for row in pack.itertuples(index=False):
        saved = captured[row.alias, row.condition]
        label = saved["label"]
        identity, match = ident.loc[label], same_raw.loc[label]
        require(identity.condition == match.condition == row.condition and match.v100_alias == row.alias
                and saved["phase"] == "PASSED", "SyN9 or raw-scan mapping differs")
        path = review / row.source
        require(digest(path) == saved["artifacts"]["roi-timeseries.tsv"] == row.source_sha256,
                "SyN9 ROI file checksum differs")
        values, labels = load_roi(path, selected=axis)
        require(labels == axis and values.shape == (210, 78) and row.frames == 210 and row.tr_seconds == 2.1,
                "SyN9 ROI or frame axis differs")
        key = (identity.person_id, row.condition)
        require(key not in syn9, "Duplicate SyN9 subject and condition")
        syn9[key] = Scan(values, mean_fd(saved["confounds"]), 0)

    for dataset, n in [(legacy, 25), (syn9, 26)]:
        groups = {cond: {person for person, c in dataset if c == cond} for cond in CONDITION_CODE}
        require(groups["OFF"] == groups["ON"] and len(groups["OFF"]) == n and len(groups["HC"]) == 15
                and groups["HC"].isdisjoint(groups["OFF"]), "Unresolved cohort")
    require(len(set(legacy) & set(syn9)) == 63, "Unexpected overlap of the two sources")
    return legacy, syn9


def fit_legacy(legacy, axis):
    # The normal fit entry point writes individual labels only inside the temporary study directory.
    with TemporaryDirectory(prefix="bold_105_k4_") as tmp:
        tmp = Path(tmp)
        runs, meta = [], []
        for i, ((person, condition), scan) in enumerate(legacy.items()):
            run = f"run-{i + 1:03d}"
            path = tmp / f"{run}.tsv"
            pd.DataFrame(scan.values, columns=axis).to_csv(path, sep="\t", index=False, float_format="%.17g")
            # The temporary TSV must present the same numeric matrix to the pipeline as the source CSV.
            converted, labels = load_roi(path)
            require(labels == axis, "Temporary ROI axis differs")
            np.testing.assert_allclose(converted, scan.values, atol=1e-14, rtol=1e-14)
            runs.append(dict(run_id=run, roi_time_series=path.name, tr_sec=2.1, atlas="AAL78",
                             format="nifti", n_frames_expected=len(scan.values), n_regions_expected=78))
            meta.append(dict(run_id=run, subject_id=person, condition=condition,
                             cohort="HC" if condition == "HC" else "DBS", mapping_source="verified 105 roster"))
        pd.DataFrame(runs).to_csv(tmp / "runs.tsv", sep="\t", index=False)
        pd.DataFrame(meta).to_csv(tmp / "meta.tsv", sep="\t", index=False)
        output = tmp / "fit"
        args = run_parser().parse_args(["fit", "--runs", str(tmp / "runs.tsv"), "--metadata", str(tmp / "meta.tsv"),
            "--out", str(output), "--mode", "analysis", "--data-kind", "real", "--polarity", "signed",
            "--k", "4", "--n-init", "100", "--max-iter", "300", "--stability-repeats", "2", "--seed", str(SEED)])
        fit(args)
        details = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        require(details["included_runs"] == 65 and details["n_valid_frames"] == 13647
                and details["n_complete_dbs_pairs"] == 25 and details["n_hc"] == 15, "Incomplete historical fit")
        stability = pd.read_csv(output / "initialization_stability.tsv", sep="\t")
        centers = np.load(output / "templates.npy", allow_pickle=False)
        measured = pd.read_csv(output / "metrics.tsv", sep="\t").merge(pd.DataFrame(meta), on="run_id",
                                                                       validate="many_to_one")
        return centers, measured, {"n_iter": details["n_iter"], "objective": details["objective"],
                                   "min_repeat_cosine": stability.template_cosine.min(),
                                   "valid_frames": details["n_valid_frames"]}


def score(scans, centers, starts=None):
    rows = []
    for key, scan in scans.items():
        if starts is not None and key not in starts:
            continue
        start = scan.first_frame if starts is None else starts[key]
        values = scan.values[start - scan.first_frame:]
        require(start >= scan.first_frame and len(values) == 210 - start, "Unequal original-frame support")
        prepared = prepare_bold(values)
        assignment = assign_states(prepared["unit_vectors"], centers, valid=prepared["valid"], polarity="signed")
        require(prepared["valid"].all(), "Unexpected invalid BOLD frame")
        states = sequence_metrics(assignment["labels"], 2.1, 4)["states"]
        require(np.isclose(sum(r["coverage"] for r in states), 1), "Invalid coverage sum")
        for r in states:
            if r["coverage"] > 0:
                np.testing.assert_allclose(r["duration_sec"] * r["occurrence_per_min"] / 60, r["coverage"])
            rows.append(dict(r, subject_id=key[0], condition=key[1], fd=scan.mean_fd))
    return pd.DataFrame(rows)


def family(frame):
    groups = frame[frame.condition != "HC"].pivot(index="subject_id", columns=["condition", "state"],
                                                  values=["fd", *METRICS]).sort_index()
    motion = groups["fd", "ON", 1] - groups["fd", "OFF", 1]
    rows = []
    for state in range(1, 5):
        for index, metric in enumerate(METRICS):
            dy = groups[metric, "ON", state] - groups[metric, "OFF", state]
            use = np.isfinite(dy)
            x, y = motion[use].to_numpy(), dy[use].to_numpy()
            require(len(y) >= 3 and np.isfinite(x).all(), "Too few finite pairs for the planned analysis")
            def rng(purpose):
                return _rng(SEED, 1, state, index, purpose)
            raw = signflip_pvalue(y, FLIPS, rng(0))
            low, high, _ = _bootstrap_mean(y, BOOTS, rng(1))
            adjusted = adjusted_endpoint(x, y, FLIPS, BOOTS, rng(0), rng(1))
            if adjusted["status"] == "ok":
                jack = []
                for omit in range(len(x)):
                    keep = np.arange(len(x)) != omit
                    jack.append(np.polynomial.polynomial.polyfit(x[keep], y[keep], 1)[0])
                intercept = np.polynomial.polynomial.polyfit(x, y, 1)[0]
                np.testing.assert_allclose([intercept, np.linalg.norm(np.asarray(jack) - intercept)],
                                           [adjusted["adjusted_delta"], adjusted["intercept_hc3_se"]], atol=1e-11)
            rows.append(dict(state=state, metric=metric, n_pairs=len(y), family_size=12,
                             original_delta=y.mean(), original_ci_low=low, original_ci_high=high,
                             original_p_raw=raw, **adjusted))
    result = pd.DataFrame(rows)
    for prefix in ("original", "adjusted"):
        result[f"{prefix}_p_holm"] = holm_adjust(result[f"{prefix}_p_raw"])
    return result, {"n_pairs": len(motion), "mean_delta_fd_mm": motion.mean(),
                    "median_delta_fd_mm": motion.median(), "on_higher": int((motion > 0).sum())}


def main(root, destination):
    reference = pd.read_csv(REPORTS / "templates.tsv", sep="\t", index_col=0)
    require(list(reference.index) == ["M1", "M2", "M3", "M4"], "Unexpected reference templates")
    axis, ref = list(reference.columns), reference.to_numpy()
    legacy, syn9 = read_inputs(root, axis)
    centers, fitted, fit_info = fit_legacy(legacy, axis)
    similarity = ref @ centers.T
    left, perm = linear_sum_assignment(-similarity)
    np.testing.assert_array_equal(left, np.arange(4))
    mapped = centers[perm]
    scored = score(legacy, mapped)
    fitted["state"] = fitted.state.map({old + 1: new + 1 for new, old in enumerate(perm)})
    keys = ["subject_id", "condition", "state"]
    np.testing.assert_allclose(scored.set_index(keys).sort_index()[list(METRICS)],
                               fitted.set_index(keys).sort_index()[list(METRICS)], atol=1e-12)
    starts = {key: scan.first_frame for key, scan in legacy.items() if key in syn9 and key[1] != "HC"}
    require(len(starts) == 48, "Expected 24 shared patient pairs")
    tables, motions = {}, {}
    for label, frame in [("legacy_refit_25", scored), ("syn9_original_26", score(syn9, ref)),
                         ("legacy_common_24", score(legacy, ref, starts)), ("syn9_common_24", score(syn9, ref, starts))]:
        tables[label], motions[label] = family(frame)
    published = pd.read_csv(REPORTS / "fd_sensitivity/fd_adjusted_stats.csv").set_index(["state", "metric"])
    check = tables["syn9_original_26"].set_index(["state", "metric"]).loc[published.index]
    for column in ("status", "reason"):
        np.testing.assert_array_equal(check[column].fillna(""), published[column].fillna(""))
    numeric = published.columns.drop(["status", "reason"])
    np.testing.assert_allclose(check[numeric].to_numpy(dtype=float), published[numeric].to_numpy(dtype=float),
                               rtol=1e-11, atol=1e-12)

    output = new_output(destination)
    pd.concat(tables, names=["analysis", "row"]).droplevel("row").to_csv(output / "paired_results.csv",
                                                                        na_rep="NA")
    pd.DataFrame(mapped, columns=axis, index=reference.index).to_csv(output / "templates_105.tsv", sep="\t")
    matches = [dict(syn9_state=i + 1, local_cluster=j + 1, signed_cosine=similarity[i, j],
                    selected=j == perm[i]) for i in range(4) for j in range(4)]
    pd.DataFrame(matches).to_csv(output / "template_matches.csv", index=False)
    summary = {"data_kind": "real", "fit": fit_info, "motion": motions, "short_legacy_scans":
               sum(scan.first_frame for scan in legacy.values()), "shared_omitted_initial_frames": sum(starts.values()),
               "matched_signed_cosines": similarity[left, perm], "syn9_original_rows_verified": len(check),
               "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
               "pandas": pd.__version__, "matplotlib": matplotlib.__version__}
    write_json(output / "summary.json", summary)
    print("Completed four twelve-endpoint comparisons; published SyN9 table reproduced.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    main(args.data_root, args.out)
