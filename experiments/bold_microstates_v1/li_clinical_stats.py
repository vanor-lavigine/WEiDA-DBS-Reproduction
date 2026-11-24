"""Compare fixed BOLD dynamics to the paired motor assessments used in the EEG protocol."""
import argparse
import io
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from clinical_archive_rerun import ENGLISH_DOMAINS, VERSIONS, clinical_visits


def correlation(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    a, b = np.asarray(x)[good], np.asarray(y)[good]
    n = len(a)
    if n < 4 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"n":n, "r":None, "ci_low":None, "ci_high":None, "p_raw":None}
    result = stats.pearsonr(a, b)
    ci = result.confidence_interval(.95)
    np.testing.assert_allclose(result.statistic, np.corrcoef(a, b)[0, 1], rtol=0, atol=2e-14)
    return {"n":n, "r":float(result.statistic), "ci_low":float(ci.low),
            "ci_high":float(ci.high), "p_raw":float(result.pvalue)}


def primary_sensitivity(x, y, fd, equal_class_duration):
    x, y, fd = map(np.asarray, (x, y, fd))
    if not all(np.isfinite(v).all() for v in (x, y, fd)) or len(x) < 4 or np.ptp(fd) == 0:
        raise ValueError("Incomplete primary or motion-adjusted observations")
    rng = np.random.default_rng(20261124)
    indices = np.stack([rng.permutation(len(y)) for _ in range(10000)])
    a, b = x - x.mean(), y - y.mean()
    observed = abs(float(a @ b))
    null = np.abs(b[indices] @ a)
    permutation_p = (1 + int(np.count_nonzero(null >= observed - 1e-12))) / (1 + len(null))
    design = np.column_stack([np.ones(len(fd)), fd])
    residual = lambda v: v - design @ np.linalg.lstsq(design, v, rcond=None)[0]
    pr = float(stats.pearsonr(residual(x), residual(y)).statistic)
    df = len(x) - 3
    t = pr * np.sqrt(df / (1 - pr**2))
    leave_one_out = [stats.pearsonr(np.delete(x, i), np.delete(y, i)).statistic for i in range(len(x))]
    return {"n":len(x), "spearman_rho":float(stats.spearmanr(x, y).statistic),
            "permutation_p_10000":permutation_p, "partial_r_off_fd":pr,
            "partial_p_off_fd":float(2 * stats.t.sf(abs(t), df)), "partial_df":df,
            "leave_one_out_min":float(min(leave_one_out)), "leave_one_out_max":float(max(leave_one_out)),
            "equal_class_weight":correlation(equal_class_duration, y)}


def scan_features(data):
    if data.duplicated(["run_id", "state"]).any() or not data.groupby("run_id").state.size().eq(4).all():
        raise ValueError("Missing or duplicated states")
    run_axis = sorted(data.run_id.unique())
    features = {}
    for name in ("duration_sec", "occurrence_per_min", "coverage"):
        wide = data.pivot(index="run_id", columns="state", values=name).loc[run_axis, [1, 2, 3, 4]]
        for state in wide:
            label = "occurrence_per_sec" if name == "occurrence_per_min" else name
            features[(f"A{state}", label)] = wide[state] / 60 if name == "occurrence_per_min" else wide[state]
    sums = data.groupby("run_id")[["n_state_frames", "n_episodes", "coverage"]].sum().loc[run_axis]
    duration = 2.1 * sums.n_state_frames / sums.n_episodes
    features[("all", "duration_sec")] = duration
    features[("all", "occurrence_per_sec")] = 1 / duration
    np.testing.assert_allclose(sums.coverage, 1, atol=1e-12)
    np.testing.assert_array_equal(sums.n_state_frames, data.groupby("run_id").n_valid_frames.first().loc[run_axis])
    np.testing.assert_allclose(sum(features[(f"A{s}", "occurrence_per_sec")] for s in range(1, 5)),
                               1 / duration, atol=1e-12)
    equal_weight = pd.concat([features[(f"A{s}", "duration_sec")] for s in range(1, 5)], axis=1).mean(axis=1, skipna=False)
    return features, equal_weight


def analyze(metrics, visits, expected_patients=24):
    if (len(visits)!=2*expected_patients or visits.person.nunique()!=expected_patients or not visits.run_id.is_unique
            or visits.score.isna().any() or len(metrics)!=16*expected_patients or set(metrics.variant)!={"direct","peak"}):
        raise ValueError("Unexpected visit roster or metrics")
    by_condition = visits.pivot(index="person", columns="condition", values="run_id").sort_index()
    scores = visits.pivot(index="person", columns="condition", values="score").reindex(by_condition.index)
    if set(scores.columns)!={"OFF","ON"} or scores.isna().any().any() or scores.OFF.le(0).any():
        raise ValueError("Missing paired scores or undefined relative improvement")
    benefit = (scores.OFF - scores.ON) / scores.OFF * 100
    contrasts = (("off_severity","OFF",scores.OFF), ("on_severity","ON",scores.ON),
                 ("on_improvement_pct","ON",benefit))
    rows, sensitivity, frozen_ranks = [], {}, {}
    for variant in ("peak", "direct"):
        data = metrics[metrics.variant.eq(variant)]
        if set(data.run_id) != set(visits.run_id):
            raise ValueError("Run identities disagree with the frozen roster")
        features, equal_weight = scan_features(data)
        fd = data.groupby("run_id").mean_fd.first()
        family = []
        for condition_test, condition, outcome in contrasts:
            runs = by_condition[condition].tolist()
            for (state, feature), values in features.items():
                x = values.loc[runs].to_numpy()
                y = outcome.to_numpy()
                row = correlation(x, y)
                family.append(dict(variant=variant, contrast=condition_test, state=state, metric=feature, **row))
                if condition_test=="off_severity" and (state,feature)==("all","duration_sec"):
                    sensitivity[variant] = primary_sensitivity(x, y, fd.loc[runs], equal_weight.loc[runs])
                if variant=="direct" and condition_test=="off_severity" and state!="all" and feature=="duration_sec":
                    frozen_ranks[state] = float(stats.spearmanr(x, y, nan_policy="omit").statistic)
        if len(family) != 42:
            raise ValueError("Incomplete family of tests")
        raw = np.array([1.0 if row["p_raw"] is None else row["p_raw"] for row in family])
        order = np.argsort(raw, kind="stable")
        ranked = raw[order] * len(raw) / np.arange(1, len(raw) + 1)
        adjusted = np.empty_like(raw)
        adjusted[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
        for row, q in zip(family, adjusted):
            row["q_bh42"] = None if row["p_raw"] is None else float(q)
        rows.extend(family)
    return {"tests":rows,"primary_sensitivity":sensitivity,"frozen_direct_spearman":frozen_ranks,
            "clinical_summary":{"patients":len(scores),"scale":"UPDRS3",
                                "mean_improvement_points":float((scores.OFF-scores.ON).mean()),
                                "mean_individual_improvement_pct":float(benefit.mean())}}


def aggregate(payload, project, private_tmp=None):
    visits = pd.DataFrame(payload["visits"])
    if not visits.scan_key.is_unique:
        raise ValueError("Duplicate clinical visit key")
    primary, duplicate = [clinical_visits(project, folder, series, payload["salt"])
                          for folder, series in VERSIONS[:2]]
    pd.testing.assert_series_equal(primary.score, duplicate.score)
    np.testing.assert_array_equal(primary.score, primary[list(ENGLISH_DOMAINS)].sum(axis=1))
    visits["score"] = visits.scan_key.map(primary.score)
    measures = pd.read_csv(io.StringIO(payload["metrics_tsv"]), sep="\t", na_values=["n/a"])
    result = analyze(measures, visits)
    result["environment"] = {"python":platform.python_version(), **{p:version(p) for p in ("numpy","scipy","pandas")}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(json.load(sys.stdin), args.project), allow_nan=False))
