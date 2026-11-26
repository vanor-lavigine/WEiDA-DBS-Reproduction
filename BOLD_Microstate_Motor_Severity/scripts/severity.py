"""Estimate same-condition motor severity associations from frozen scan features."""

import argparse
from pathlib import Path
import sys

import numpy as np
from openpyxl import load_workbook
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
BASE = "3c32098232e1c5a6f2ddcfeefe5bd5b979047a11"
PREVIOUS = REPO / "BOLD_Microstate_Behavior_Association"
PRIVATE = PROJECT / "private"
RESULTS = PROJECT / "results"
REGISTRY = RESULTS / "Endpoint_Registry.csv"
PRIOR_RESULTS = PREVIOUS / "results" / "Archived_Score_Associations.csv"
LEGACY_TABLE = REPO / "reports/bold_microstates_v1/eeg_replacement_105/clinical.csv"
STAGE_KEYS = (
    "off_gev_1", "on_gev_1", "off_duration_4", "on_duration_4",
    "off_occurrence_1", "on_occurrence_1", "off_switching_rate", "on_switching_rate",
    "off_gev_3", "off_dispersion_3", "off_occurrence_3",
)
PRIOR_KEYS = ("delta_gev_1", "delta_duration_4", "delta_occurrence_1", "delta_switching_rate")

sys.path.insert(0, str(PREVIOUS / "scripts"))
from io_sources import (ORIGINAL_FILES, PAIR_KEYS, PROTOCOL, REFERENCE, SCORE_KEYS, WORKSPACE,
                        digest, environment, fingerprint, load_frozen, read_json, read_table,
                        require, scan_lookup, verified_scores, write_json, write_result)
from association import association


HISTORICAL_STATUS = "archived_unverified_visit"
VALIDATED_STATUS = "verified_visit"
STAGE = "same_condition_severity"
PRIOR = "paired_change_improvement"


def read_registry(path=REGISTRY):
    d = read_table(path)
    require(tuple(d.endpoint) == STAGE_KEYS
            and d.endpoint.eq(d.condition.str.lower() + "_" + d.metric).all(),
            "Registry differs from the locked sequence of eleven endpoints")
    source_tables = {k: read_table(REPO / k) for k in d.imaging_result_file.unique()}
    for r in d.itertuples(index=False):
        expected = source_tables[r.imaging_result_file]
        v6 = r.imaging_release == "V6"
        keys = expected["endpoint" if v6 else "metric"].eq(r.metric) & expected.contrast.eq(r.imaging_contrast)
        require(int(keys.sum()) == 1, "Registry source cannot be uniquely located")
        observed = expected[keys].iloc[0]
        require(r.imaging_family == observed["family" if v6 else "correction_family"] and
                r.imaging_family_size == observed.family_size, "Registry imaging family differs from its source")
        pairs = [
            (r.imaging_effect, observed["mean_difference" if v6 else "effect_size"]),
            (r.imaging_permutation_holm_p, observed.permutation_holm_p),
            (r.imaging_fd_holm_p, observed["fd_holm_p" if v6 else "fd_permutation_holm_p"]),
        ]
        if v6:
            pairs.append((r.imaging_dvars_holm_p, observed.dvars_holm_p))
        else:
            require(pd.isna(r.imaging_dvars_holm_p), "Registry assigns DVARS evidence to a V4 row")
            pairs.extend([(r.imaging_all_scalar_holm_p, observed.all_scalar_permutation_holm_p),
                          (r.imaging_all_scalar_fd_holm_p, observed.all_scalar_fd_permutation_holm_p)])
        require(np.allclose([x[0] for x in pairs], [x[1] for x in pairs], rtol=0, atol=1e-14),
                "Registry statistics differ from the archived group comparison")
    return d


def one_row_per_scan(pairs, scans, metric_names):
    expected = scan_lookup(pairs)[["participant_id", "scan", "condition"]]
    scans = scans.rename(columns={"subject": "participant_id"})
    keys = ["participant_id", "scan", "condition"]
    kept = scans[keys + ["fd", "fold", "valid_frames", "coverage_4", "occurrence_4"] + metric_names]
    joined = expected.merge(kept, on=keys, how="inner", validate="one_to_one")
    require(len(joined) == len(expected) == len(scans), "Same-condition scan join is incomplete")
    return joined.sort_values(["participant_id", "condition"]).reset_index(drop=True)


def check_units(scans):
    tr = PROTOCOL["cohort"]["tr_seconds"]
    exposure = tr * scans.valid_frames
    for field in ("occurrence_1", "occurrence_3", "switching_rate"):
        count = (scans[field] * exposure).dropna()
        require(count.ge(0).all() and count.le(scans.loc[count.index, "valid_frames"] + 1e-10).all()
                and np.allclose(count, np.rint(count), rtol=0, atol=1e-10),
                "Occurrence or switching rate disagrees with the original rate per second")
    for field in ("gev_1", "gev_3"):
        require(scans[field].dropna().between(0., 1.).all(), "Frozen GEV must be in proportion units")
    require(scans.dispersion_3.dropna().between(0., np.sqrt(2)).all(), "Spatial dispersion is outside its range")
    compatible = scans.duration_4.notna() & scans.coverage_4.notna() & scans.occurrence_4.gt(0)
    expected = scans.loc[compatible, "coverage_4"] / scans.loc[compatible, "occurrence_4"]
    require(np.allclose(scans.loc[compatible, "duration_4"], expected, rtol=0, atol=1e-10),
            "M4 mean dwell time disagrees with coverage divided by occurrence per second")


def merge_scores(pairs, scans, wide_scores, visit_status):
    frozen_keys = pairs.reset_index()[PAIR_KEYS]
    require(wide_scores[PAIR_KEYS].notna().all().all() and wide_scores.participant_id.is_unique,
            "Historical score input contains incomplete or duplicated pair keys")
    require(wide_scores.visit_evidence.eq(visit_status).all(), "Score evidence label does not match the analysis path")
    matched = wide_scores[PAIR_KEYS + SCORE_KEYS].merge(frozen_keys, on=PAIR_KEYS, how="inner", validate="one_to_one")
    require(len(matched) == len(wide_scores), "A score record has a wrong patient, ON scan, or OFF scan")
    sides = []
    for condition in ("ON", "OFF"):
        low = condition.lower()
        side = matched[["participant_id", f"{low}_scan_id", f"updrs_iii_{low}"]].copy()
        side.columns = ["participant_id", "scan", "score"]
        score = pd.to_numeric(side.score, errors="raise")
        require((score.isna() | (np.isfinite(score) & score.ge(0))).all(), "A score is invalid")
        sides.append(side.assign(score=score, condition=condition))
    return scans.merge(pd.concat(sides, ignore_index=True), on=["participant_id", "scan", "condition"],
                       how="left", validate="one_to_one")


def read_inputs(behavior_private, v6_private, registry):
    stamp = behavior_private / "manifests/Input_Hashes.json"
    recorded = read_json(stamp)
    archived_digest = recorded["inputs"]
    if v6_private is None:
        v6_private = Path(archived_digest["v6_manifest"]["file"]).parents[1]
    score_file = behavior_private / "archived_unverified" / REFERENCE / "Archived_Score_Input.csv"
    link_file = behavior_private / "evidence/Archive_Scan_Source_Index.csv"
    files = {f"v6_{k}": v6_private / v for k, v in ORIGINAL_FILES.items()}
    for key, path in files.items():
        require(digest(path) == archived_digest[key]["sha256"],
                f"Frozen imaging input differs from the behavior analysis audit ({key})")
    files.update({"previous_audit": stamp, "archived_score_pairs": score_file, "archived_score_links": link_file,
                  "prior_change_results": PRIOR_RESULTS, "endpoint_registry": REGISTRY,
                  "locked_plan": PROJECT / "ANALYSIS_PLAN.md"})
    for relative in registry.imaging_result_file.unique():
        files[relative] = REPO / relative
    if LEGACY_TABLE.is_file():
        files["legacy_42_test_family"] = LEGACY_TABLE
    workbook = Path(archived_digest["reference_score_row_crosswalk"]["file"])
    if workbook.is_file():
        require(digest(workbook) == archived_digest["reference_score_row_crosswalk"]["sha256"],
                "Score crosswalk is not the workbook recorded by the prior behavior run")
        files["legacy_score_crosswalk"] = workbook
    before = fingerprint(files)
    pairs, counts, _ = load_frozen(v6_private)
    scans = one_row_per_scan(pairs, read_table(files["v6_scans"]), list(registry.metric.unique()))
    check_units(scans)
    score_data = read_table(score_file)
    observations = merge_scores(pairs, scans, score_data, HISTORICAL_STATUS)
    link = read_table(link_file)
    link = link.loc[link.source.eq(REFERENCE), ["participant_id", "scan", "condition", "score"]]
    require(not link.duplicated(["participant_id", "scan", "condition"]).any(),
            "Historical score-to-scan evidence contains duplicate keys")
    original = observations[["participant_id", "scan", "condition", "score"]].merge(
        link, on=["participant_id", "scan", "condition"], suffixes=("", "_archived"), validate="one_to_one"
    )
    require(len(original) == len(observations) == len(link) and
            np.allclose(original.score, original.score_archived, rtol=0, atol=0, equal_nan=True),
            "Scoring values or condition keys differ from the archived crosswalk")
    return pairs, scans, observations, counts, files, before


def run_stage(observations, registry, source, visit_evidence):
    rows = []
    for spec in registry.itertuples(index=False):
        data = observations.loc[observations.condition.eq(spec.condition)]
        x = data[spec.metric].to_numpy(float)
        y, fd = data.score.to_numpy(float), data.fd.to_numpy(float)
        complete = np.isfinite(x) & np.isfinite(y)
        rec = association(x, y, fd)
        rec["n_patients"] = rec.pop("n_pairs")
        rec["fd_n_patients"] = rec.pop("fd_n_pairs")
        rec["score_tie_excess"] = rec.pop("improvement_tie_excess")
        rec["bootstrap_constant_score"] = rec.pop("bootstrap_constant_improvement")
        rec["reason"] = rec["reason"].replace("complete pairs", "complete patients").replace("score improvement", "motor score")
        rec["fd_reason"] = rec["fd_reason"].replace("complete FD pairs", "FD-complete patients").replace("score improvement", "motor score")
        rows.append(dict(endpoint=spec.endpoint, analysis_type=STAGE, condition=spec.condition, metric=spec.metric,
                         label=spec.label, priority=spec.priority, score_source=source, visit_evidence=visit_evidence,
                         unit=spec.unit, display_unit=spec.display_unit, display_multiplier=spec.display_multiplier,
                         n_cohort=len(data), n_missing_feature=int((~np.isfinite(x)).sum()),
                         n_missing_score=int((~np.isfinite(y)).sum()),
                         n_missing_both=int((~np.isfinite(x) & ~np.isfinite(y)).sum()),
                         fd_missing_among_complete=int((~np.isfinite(fd) & complete).sum()), **rec))
    stage = pd.DataFrame(rows)
    add_corrections(stage, "permutation_p", STAGE_KEYS, "11")
    add_corrections(stage, "fd_p", STAGE_KEYS, "11", prefix="fd_")
    stage["family_size"] = len(STAGE_KEYS)
    return stage


def add_corrections(table, column, keys, size_tag, prefix=""):
    require(tuple(table.endpoint) == tuple(keys), "Correction input does not match the fully specified family")
    raw = table[column].to_numpy(float)
    usable = np.isfinite(raw)
    require(((raw[usable] >= 0) & (raw[usable] <= 1)).all(), "Invalid P value before correction")
    for method, name in (("holm", "holm"), ("fdr_bh", "bh")):
        corrected = multipletests(np.where(usable, raw, 1.), method=method)[1]
        table[f"{prefix}{name}{size_tag}_p"] = np.where(usable, corrected, np.nan)


def cumulative(stage, previous):
    previous = previous.loc[previous.score_source.eq(REFERENCE) & previous.analysis_path.eq(HISTORICAL_STATUS)].copy()
    require(tuple(previous.endpoint) == PRIOR_KEYS and previous.family_size.eq(4).all(),
            "Prior results do not describe the four locked change associations")
    previous = previous.rename(columns={"n_pairs": "n_patients", "permutation_holm_p": "holm4_p",
                                        "permutation_fdr_p": "bh4_p", "analysis_path": "visit_evidence"})
    previous = previous.assign(analysis_type=PRIOR, condition="ON-OFF",
                               metric=previous.endpoint.str.removeprefix("delta_"))
    columns = ["endpoint", "analysis_type", "condition", "metric", "status", "reason", "visit_evidence",
               "score_source", "n_patients", "rho", "ci95_low", "ci95_high", "raw_p", "permutation_p"]
    cum = pd.concat([stage[columns + ["holm11_p", "bh11_p"]],
                     previous[columns + ["holm4_p", "bh4_p"]]], ignore_index=True)
    add_corrections(cum, "permutation_p", STAGE_KEYS + PRIOR_KEYS, "15")
    cum["cumulative_family_size"] = 15
    return cum


def historical_m4(workbook, observations, clinical):
    """Recompute one legacy Pearson row; obtain the remaining family P values from its archive."""
    require(len(clinical) == 42 and len(clinical.drop_duplicates(["band", "state", "metric", "contrast"])) == 42,
            "Legacy 42-outcome family is missing rows")
    target = (clinical.band.eq("weida_band") & clinical.state.eq("M4")
              & clinical.metric.eq("occurrence_s") & clinical.contrast.eq("on_score"))
    require(int(target.sum()) == 1, "Legacy ON M4 result must be unique")
    old = clinical.loc[target].iloc[0]
    base = dict(evidence_class="prior_result_recalculation", condition="ON", band_hz="0.04-0.07",
                metric="legacy_m4_occurrence_rate", score_source=REFERENCE, visit_evidence=HISTORICAL_STATUS,
                method="Pearson with Fisher 95% CI", family_size=42,
                archived_n=int(old.n), archived_r=old.r, archived_raw_p=old.p_raw, archived_bh42_p=old.q_bh42)
    if workbook is None:
        return pd.DataFrame([dict(base, status="SOURCE_UNAVAILABLE", n_patients=0, r=np.nan, ci95_low=np.nan,
                                  ci95_high=np.nan, raw_p=np.nan, bh42_p=np.nan)])
    with workbook.open("rb") as handle:
        wb = load_workbook(handle, read_only=True, data_only=True)
        try:
            sheet = wb["原表与访视"]
            require(REFERENCE in str(sheet["A2"].value), "Legacy workbook references a different score source")
            headers = next(sheet.iter_rows(min_row=4, max_row=4, values_only=True))
            indices = [headers.index(label) for label in
                       ("访视", "105 扫描编号", "原 M4\n（次/分）", "UPDRS3\n运动总分")]
            rows = [tuple(r[k] for k in indices) for r in sheet.iter_rows(min_row=5, values_only=True)
                    if r[indices[0]] == "ON"]
        finally:
            wb.close()
    old_scan = pd.DataFrame(rows, columns=["condition", "scan", "m4_per_minute", "legacy_score"])
    old_scan["scan"] = old_scan.scan.str.casefold()
    recent = observations.loc[observations.condition.eq("ON"), ["condition", "scan", "score"]].copy()
    recent["scan"] = recent.scan.str.casefold()
    joined = old_scan.merge(recent, on=["scan", "condition"], how="inner", validate="one_to_one")
    require(len(joined) == len(old_scan) == len(recent) == int(old.n), "Legacy score cohort uses different scan keys")
    require(np.allclose(joined.legacy_score, joined.score, rtol=0, atol=0), "Legacy and current ON scores differ")
    x = joined.m4_per_minute.to_numpy(float) / 60.
    y = joined.score.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "A legacy association input is missing")
    result = stats.pearsonr(x, y)
    ci = result.confidence_interval(.95)
    raw_family = clinical.p_raw.to_numpy(float, copy=True)
    raw_family[target.to_numpy()] = result.pvalue
    bh = stats.false_discovery_control(np.where(np.isfinite(raw_family), raw_family, 1.))[target.to_numpy()][0]
    exact = np.allclose([result.statistic, ci.low, ci.high, result.pvalue, bh],
                        [old.r, old.ci_low, old.ci_high, old.p_raw, old.q_bh42], rtol=0, atol=1e-12)
    return pd.DataFrame([dict(base, status="REPRODUCED" if exact else "NUMERICAL_DIFFERENCE", n_patients=len(joined),
                              r=result.statistic, ci95_low=ci.low, ci95_high=ci.high,
                              raw_p=result.pvalue, bh42_p=bh)])


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-private", type=Path,
                        default=WORKSPACE / "WEiDA-DBS-Behavior" / PREVIOUS.name / "private")
    parser.add_argument("--v6-private", type=Path, help="Override the V6 private root recorded in the behavior audit")
    parser.add_argument("--verified-visits", type=Path)
    parser.add_argument("--verified-evidence", type=Path)
    return parser.parse_args()


def main():
    args = options()
    registry = read_registry()
    pairs, scans, obs, counts, files, before = read_inputs(args.behavior_private, args.v6_private, registry)
    verified_path = args.verified_visits or args.behavior_private / "inputs/Verified_Visit_UPDRS.csv"
    evidence_path = args.verified_evidence or args.behavior_private / "inputs/Verified_Visit_Evidence.csv"
    verified, evidence_files, _ = verified_scores(verified_path, evidence_path, pairs)
    before.update(fingerprint(evidence_files))
    files.update(evidence_files)
    stage = run_stage(obs, registry, REFERENCE, HISTORICAL_STATUS)
    c = cumulative(stage, read_table(PRIOR_RESULTS))
    stage = stage.merge(c[["endpoint", "holm15_p", "bh15_p"]], on="endpoint", validate="one_to_one")
    legacy = historical_m4(files.get("legacy_score_crosswalk"), obs, read_table(LEGACY_TABLE))
    require(fingerprint(files) == before, "A frozen or analysis input changed during execution")

    cols = ["endpoint", "analysis_type", "condition", "metric", "label", "priority", "score_source",
            "visit_evidence", "n_patients", "family_size"]
    fd_cols = ["fd_missing_among_complete", *[k for k in stage if k.startswith("fd_") and k != "fd_missing_among_complete"]]
    write_result(stage[cols + fd_cols], RESULTS / "Motor_Severity_FD_Sensitivity.csv")
    write_result(stage, RESULTS / "Motor_Severity_Associations.csv")
    write_result(c, RESULTS / "Cumulative_15_Associations.csv")
    write_result(legacy, RESULTS / "Legacy_ON_M4_Recheck.csv")
    write_result(obs, PRIVATE / "results/Same_Condition_Patient_Input.csv")
    if len(verified):
        verified_obs = merge_scores(pairs, scans, verified, VALIDATED_STATUS)
        verified_stage = run_stage(verified_obs, registry, "adjudicated_visit_records", VALIDATED_STATUS)
        write_result(verified_stage, RESULTS / "Verified_Visit_Motor_Severity_Associations.csv")
        write_result(verified_obs, PRIVATE / "results/Verified_Visit_Patient_Input.csv")
    write_json({"base_commit": BASE, "files": before}, PRIVATE / "manifests/Input_Hashes.json")
    write_json({"base_commit": BASE, "imaging_commit": PROTOCOL["imaging_commit"], "cohort": counts,
                "score_source": REFERENCE, "visit_evidence": HISTORICAL_STATUS, "verified_pairs": len(verified),
                "software": environment(), "seed": PROTOCOL["seed"], "permutations": PROTOCOL["permutations"],
                "bootstrap": PROTOCOL["bootstrap"], "stage_family_size": 11, "cumulative_family_size": 15,
                "inputs_unchanged": True}, RESULTS / "Run_Metadata.json")

    from severity_figures import draw_figures
    draw_figures(registry, stage, obs, PROJECT / "figures", PRIVATE / "figures")
    print(f"Completed {len(stage)} same-condition and {len(c)} cumulative associations.")
    print(f"Patients per stage endpoint: {stage.n_patients.min()}–{stage.n_patients.max()}.")
    print(f"Visit-verified patient pairs: {len(verified)}. Legacy check: {legacy.status.iloc[0]}.")


if __name__ == "__main__":
    main()
