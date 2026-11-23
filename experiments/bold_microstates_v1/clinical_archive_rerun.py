"""Compute aggregate v1 statistics beside the private clinical score archive."""
import argparse
import hashlib
import hmac
import io
import json
import platform
import re
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stats import analyze


VERSIONS = (
    ("WEiDA4-1_atlas1_group0", "ts_fmripost_nodbs_AAL78_all"),
    ("WEiDA3-42_atlas1_group2", "ts_fmripost_nodbs_AAL78_all"),
    ("WEiDA4-3_atlas1_group2", "ts_fmripost_nodbs_AAL78"),
)
CHINESE_DOMAINS = ("言语", "面部表情", "强直", "运动徐缓", "起立", "步态", "步态冻结", "姿势平稳度", "姿势", "姿态型震颤", "动作型震颤", "静止型震颤")
ENGLISH_DOMAINS = ("speak", "facial expression", "Ankylosis", "bradykinesia", "stand up", "gait", "freezing of gait", "Posture stability", "posture", "Postural tremor", "Action tremor", "Stationary tremor")
SHARE_FIELDS = ("family", "state", "metric", "family_size", "n_pool", "n_subjects",
                "n_missing_endpoint", "estimate_type", "estimate", "mean_off", "mean_on",
                "ci_low", "ci_high", "p_raw", "p_holm", "status", "reason", "n_bootstrap_valid")


def scan_key(label, salt):
    return hmac.new(bytes.fromhex(salt), label.lower().encode(), hashlib.sha256).hexdigest()


def clinical_visits(project, result_folder, series_folder, salt):
    """Recover score-row identity from both imaging fingerprints and the named scan list."""
    source = project / "data" / series_folder
    archive = project / "result" / result_folder
    pattern = "_BOLD_AAL1+PD25_MNI_basic36_sm6.csv"
    labels = sorted(f.name.split("_", 1)[0] for f in source.glob("sub-STN*" + pattern))
    hc = len(list(source.glob("sub-HC*" + pattern)))
    names = [re.fullmatch(r"sub-STN(off|on)(\d+)", label, re.I) for label in labels]
    if len(names) != 50 or not all(names):
        raise ValueError("Expected the 50 audited STN source runs")
    conditions = [m[1].lower() for m in names]
    numbers = [int(m[2]) for m in names]
    if conditions != ["off"]*25 + ["on"]*25 or numbers[:25] != numbers[25:] or len(set(numbers)) != 25:
        raise ValueError("Unexpected ordering or identity in source filenames")
    joined = {}
    tables = [pd.read_csv(archive/f"scl_{name}_stn.csv", float_precision="round_trip")
              for name in ("P", "LT")]
    domains = CHINESE_DOMAINS if CHINESE_DOMAINS[0] in tables[0] else ENGLISH_DOMAINS
    score_cols = ["UPDRS3", *domains]
    np.testing.assert_array_equal(tables[0][score_cols], tables[1][score_cols])
    features = np.column_stack([table.filter(regex=r"^(?:LPMS|LTMS)[1-9]$").to_numpy() for table in tables])
    if len(features) != 50 or not np.isfinite(features).all():
        raise ValueError("Incomplete P/LT fingerprint")
    for group, condition in ((2, "off"), (3, "on")):
        reference = np.column_stack([np.loadtxt(archive/f"{name}{group}emp.csv", delimiter=",")
                                     for name in ("P", "LT")])
        matches = np.isclose(features[:, None, :], reference[None, :, :], atol=1e-8, rtol=0).all(axis=2)
        row_index, matrix_index = np.nonzero(matches)
        if len(row_index) != 25 or len(set(row_index)) != 25 or len(set(matrix_index)) != 25:
            raise ValueError("Nonunique score-to-scan match")
        for row, pos in zip(row_index, matrix_index - hc):
            if row in joined or not 0 <= pos < 50 or conditions[pos] != condition:
                raise ValueError("Score row conflicts with original stimulation condition")
            joined[row] = pos
    if len(joined) != len(set(joined.values())) or len(joined) != 50:
        raise ValueError("Incomplete one-to-one score assignment")
    ordered = tables[0].iloc[sorted(joined, key=joined.get)][score_cols].copy()
    ordered.index = [scan_key(label, salt) for label in labels]
    ordered.columns = ["score", *ENGLISH_DOMAINS]
    if not ordered.index.is_unique or not np.isfinite(ordered).all().all():
        raise ValueError("Incomplete or duplicate clinical data")
    return ordered


def motion_summary(metadata):
    complete = metadata[metadata.clinical_score.notna()].copy()
    off = complete[complete.condition.eq("OFF")].set_index("subject_id").sort_index()
    on = complete[complete.condition.eq("ON")].set_index("subject_id").sort_index()
    pd.testing.assert_index_equal(off.index, on.index)
    return {
        "n": len(off), "mean_fd_off_mm": off.fd.mean(), "mean_fd_on_mm": on.fd.mean(),
        "rho_off_fd_off_score": float(spearmanr(off.fd, off.clinical_score).statistic),
        "rho_delta_fd_improvement": float(spearmanr(on.fd-off.fd, off.clinical_score-on.clinical_score).statistic)
    }


def check_effects(metrics, metadata, stats, draw_severity=False):
    """Verify each effect from visit pivots, independently of the inference loop."""
    joined = metrics.merge(metadata[["run_id", "subject_id", "condition", "clinical_score"]],
                           on="run_id", how="left", validate="many_to_one")
    dbs = joined[joined.condition.isin(["OFF", "ON"])]
    errors = []
    bins = []
    for row in stats.itertuples(index=False):
        state = dbs[dbs.state.eq(row.state)]
        image = state.pivot(index="subject_id", columns="condition", values=row.metric)
        score = state.pivot(index="subject_id", columns="condition", values="clinical_score")
        if row.family == "on_off":
            values = (image.ON-image.OFF).dropna()
            estimate = values.mean()
            n = len(values)
        else:
            score = score.dropna()
            if row.family == "clinical_change":
                x, y = image.ON-image.OFF, score.OFF-score.ON
            elif row.family == "clinical_off_duration":
                x, y = image.OFF, score.OFF
            else:
                raise ValueError("Unknown comparison family")
            obs = pd.DataFrame({"x":x, "y":y}).dropna()
            n = len(obs)
            estimate = float(spearmanr(obs.x, obs.y).statistic)
            if draw_severity and row.family == "clinical_off_duration":
                # Build descriptive thirds without splitting ties in OFF duration.
                ranks = obs.x.rank(method="average", pct=True)
                thirds = pd.cut(ranks, [0, 1/3, 2/3, 1], labels=["lower","middle","upper"])
                for third, group in obs.groupby(thirds, observed=True):
                    if len(group) < 5:
                        raise ValueError("Too few observations for an aggregate plot bin")
                    bins.append(dict(state=row.state, third=str(third), n=len(group),
                                     mean_duration_sec=group.x.mean(), mean_archived_score=group.y.mean()))
        if n != row.n_subjects or not np.isfinite(estimate):
            raise ValueError("Independent check found different available cases")
        errors.append(abs(estimate-float(row.estimate)))
    if not errors or max(errors) > 1e-12:
        raise ValueError("Independent check found a different effect")
    return max(errors), bins


def aggregate(payload, project, private_tmp=None):
    frozen = payload["verified"]
    if (frozen["data_kind"], frozen["polarity"], frozen["k"], frozen["mode"],
        frozen["n_complete_dbs_pairs"], frozen["n_valid_frames"]) != ("real","signed",4,"analysis",26,14070):
        raise ValueError("Unexpected source fit")
    metrics = pd.read_csv(io.StringIO(payload["metrics_tsv"]), sep="\t", na_values=["n/a"])
    meta = pd.DataFrame(payload["visits"]).copy()
    if (len(meta) != 67 or meta.run_id.nunique()!=67 or len(metrics)!=268 or
        set(meta.run_id)!=set(metrics.run_id) or not meta.scan_key.is_unique or not np.isfinite(meta.fd).all()):
        raise ValueError("Imaging input differs from the frozen 67-run cohort")
    meta["mapping_source"] = "verified frozen fit identity and original BOLD/T1w lineage"
    archives = [clinical_visits(project, folder, data, payload["salt"]) for folder, data in VERSIONS]
    primary, duplicate, alternate = archives
    pd.testing.assert_series_equal(primary.score, duplicate.score)
    for table in (primary, duplicate):
        np.testing.assert_array_equal(table.score, table[list(ENGLISH_DOMAINS)].sum(axis=1))
    pd.testing.assert_frame_equal(primary[list(ENGLISH_DOMAINS)], alternate[list(ENGLISH_DOMAINS)], check_dtype=False)
    agreed = primary.score.eq(alternate.score)
    if int(agreed.sum()) != 36:
        raise ValueError("Clinical archive differs from the previous version audit")
    meta = meta.rename(columns={"person":"subject_id"})
    agree_ids = meta[meta.cohort.eq("DBS")].assign(matches=meta.scan_key.map(agreed).eq(True)).groupby("subject_id").matches.all()
    agreed_people = set(agree_ids.index[agree_ids])
    if len(agreed_people) != 17:
        raise ValueError("Expected 17 completely concordant pairs")
    pools = (("archive_main", primary.score, None, 24), ("archive_alternate", alternate.score, None, 24),
             ("concordant_pairs", primary.score, agreed_people, 17))
    output = []
    motion = {}
    verification = []
    grouped = []
    for tag, scores, eligible, n in pools:
        metadata = meta.copy()
        metadata["clinical_score"] = metadata.scan_key.map(scores)
        if eligible is not None:
            metadata.loc[~metadata.subject_id.isin(eligible), "clinical_score"] = np.nan
        metadata["clinical_scale"] = np.where(metadata.clinical_score.notna(), "UPDRS3", "")
        if metadata.clinical_score.notna().sum() != n*2:
            raise ValueError("Incomplete clinical pairs")
        motion[tag] = motion_summary(metadata)
        with TemporaryDirectory(dir=private_tmp) as work:
            result = analyze(metrics, metadata, work, k=4, clinical_scale="UPDRS3",
                             n_permutations=10000, n_bootstrap=2000, seed=20261121)
            if result["n_eligible_pairs"] != 26 or result["n_clinically_eligible_pairs"] != n:
                raise ValueError("The frozen statistics engine produced unexpected sample sizes")
            stats = pd.read_csv(Path(work)/"stats.csv", keep_default_na=False)
        if len(stats) != 28:
            raise ValueError("Incomplete test families")
        if tag != "archive_main":
            stats = stats[stats.family.ne("on_off")]
        error, points = check_effects(metrics, metadata, stats, draw_severity=tag=="archive_main")
        verification.append(dict(scenario=tag, estimates_checked=len(stats), max_abs_difference=error))
        grouped.extend(points)
        for record in stats[list(SHARE_FIELDS)].to_dict("records"):
            output.append(dict(scenario=tag, **record))
    return {"summary":{"data_kind":"real","template_scans":67,"stn_pairs":26,
                       "scored_pairs":24,"concordant_pairs":17,"same_day_and_medication_verified_pairs":0},
            "environment":{"python":platform.python_version(), **{p:version(p) for p in ("numpy","pandas","scipy")}},
            "stats":output,"motion":motion,"effect_verification":verification,"severity_bins":grouped}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--private-tmp", type=Path, required=True)
    opts = parser.parse_args()
    print(json.dumps(aggregate(json.load(sys.stdin), opts.project, opts.private_tmp), allow_nan=False))
