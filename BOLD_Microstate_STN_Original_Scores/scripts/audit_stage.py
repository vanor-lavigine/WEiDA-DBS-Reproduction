"""Score provenance, identity audit and the archived WEiDA4-1 comparison."""

import numpy as np
import pandas as pd

from identity_audit import CONFLICT, INSUFFICIENT, OUTSIDE, UNRESOLVED, VERIFIED, match_visits
from pipeline_io import ARCHIVE, CFG, PAIR_KEYS, PRIVATE, RESULTS, write_csv, write_json, require
from score_sources import (ITEM_NAMES, normalize_study_code, read_primary, read_secondary,
                           reconcile, write_source_audit)

LEGACY_FIELDS = {
    "motor_total": "UPDRS3", "speech": "speak", "facial_expression": "facial expression",
    "rigidity": "Ankylosis", "bradykinesia": "bradykinesia", "arising": "stand up",
    "gait": "gait", "gait_freezing": "freezing of gait", "postural_stability": "Posture stability",
    "posture": "posture", "postural_tremor": "Postural tremor", "kinetic_tremor": "Action tremor",
    "rest_tremor": "Stationary tremor", "tremor": "tremor",
}


def reconstruct_old_scores(paths, imaging):
    paired = pd.read_csv(paths["archived_score_pairs"]).sort_values("participant_id").reset_index(drop=True)
    roster = imaging.roster.reset_index()
    require(paired.visit_evidence.eq("archived_unverified_visit").all(), "Historical scoring status differs")
    pd.testing.assert_frame_equal(paired[PAIR_KEYS], roster[PAIR_KEYS], check_exact=True)
    link = pd.read_csv(paths["archived_score_links"])
    link = link.loc[link.source.eq(ARCHIVE)].sort_values("source_row").reset_index(drop=True)
    raw = pd.read_csv(paths["archived_score_domains"], float_precision="round_trip")
    require(len(link) == len(raw) == 2 * len(paired), "Historical source rows are incomplete")
    np.testing.assert_array_equal(link.source_row, np.arange(2, len(link) + 2))
    np.testing.assert_array_equal(raw["研究编号"].map(normalize_study_code), link.mapped_study_code.map(normalize_study_code))
    np.testing.assert_array_equal(raw.UPDRS3, link.score)
    archive_roster = pd.concat([
        paired[["participant_id", f"{side}_scan_id", f"updrs_iii_{side}"]].rename(
            columns={f"{side}_scan_id": "scan", f"updrs_iii_{side}": "paired_total"}).assign(condition=side.upper())
        for side in ("off", "on")
    ])
    audit = link.merge(archive_roster, on=["participant_id", "scan", "condition"], validate="one_to_one")
    require(len(audit) == len(link), "Historical score and scan keys differ")
    np.testing.assert_array_equal(audit.score, audit.paired_total)
    old = link[["participant_id", "scan", "condition", "source_row", "mapped_study_code"]].rename(
        columns={"scan": "scan_id", "condition": "dbs_condition", "source_row": "legacy_source_row"})
    old = old.join(raw.rename(columns={original: f"legacy_{new}" for new, original in LEGACY_FIELDS.items()})[
        [f"legacy_{x}" for x in LEGACY_FIELDS]])
    paired = paired.assign(legacy_motor_improvement=paired.updrs_iii_off - paired.updrs_iii_on)
    return old, paired


def compare_legacy(long, paired, old_long, old_pairs, second):
    connected = long.loc[long.participant_id.notna()]
    cols = ["participant_id", "scan_id", "dbs_condition", "study_code", "source_row", *LEGACY_FIELDS]
    span = connected[cols].merge(old_long, on=["participant_id", "scan_id", "dbs_condition"], validate="one_to_one")
    require(len(span) == len(connected), "Some matched V8 rows have no old WEiDA4-1 record")
    historical_codes = span.mapped_study_code.map(normalize_study_code)
    require(historical_codes.str.fullmatch(r"STN-?\d+").all(), "Unexpected archived study-code format")
    require(span.study_code.eq(historical_codes.str.replace("-", "", regex=False)).all(),
            "Original and historical study codes differ after separator normalization")
    for attr in LEGACY_FIELDS:
        span[f"legacy_difference_{attr}"] = span[attr] - span[f"legacy_{attr}"]
    difference = [f"legacy_difference_{attr}" for attr in LEGACY_FIELDS]
    is_different = span[difference].ne(0) & span[difference].notna()
    span["legacy_any_available_difference"] = is_different.any(axis=1)
    changes = paired.dropna(subset=["participant_id"]).merge(old_pairs, on=PAIR_KEYS, validate="one_to_one")
    changes["legacy_difference_improvement"] = changes.motor_total_improvement - changes.legacy_motor_improvement
    usable = changes.loc[changes.complete_score_pair & changes.visit_status.isin([VERIFIED, INSUFFICIENT])]
    rows = [
        dict(metric="legacy_scan_key_records_compared", n=len(span), denominator=len(old_long)),
        dict(metric="legacy_total_score_records_changed", n=int(is_different.legacy_difference_motor_total.sum()), denominator=len(span)),
        dict(metric="legacy_records_changed_any_comparable_value", n=int(is_different.any(axis=1).sum()), denominator=len(span)),
        dict(metric="legacy_patients_with_any_changed_value", n=int(span.loc[is_different.any(axis=1), "participant_id"].nunique()), denominator=len(changes)),
        dict(metric="legacy_patient_improvements_changed",
             n=int(changes.legacy_difference_improvement.dropna().ne(0).sum()),
             denominator=int(changes.legacy_difference_improvement.notna().sum())),
        dict(metric="eligible_patient_improvements_changed",
             n=int(usable.legacy_difference_improvement.dropna().ne(0).sum()),
             denominator=int(usable.legacy_difference_improvement.notna().sum())),
    ]
    for attr in LEGACY_FIELDS:
        rows.append(dict(metric=f"legacy_comparison_{attr}_mismatch", n=int(is_different[f"legacy_difference_{attr}"].sum()),
                         denominator=int(span[f"legacy_difference_{attr}"].notna().sum())))
    audit_cols = ["primary_source_row", "secondary_source_row", "join_status", "item_mismatch_count",
                  "primary_y_cached", "primary_y_formula", "primary_y_minus_bradykinesia",
                  "secondary_at_cached", "secondary_at_formula", "secondary_at_minus_motor_total",
                  "secondary_ai_sum", "secondary_at_minus_motor_equals_ai"]
    score_audit = second.loc[second.primary_target.eq("STN"), audit_cols]
    base = long[["study_code", "participant_id", "dbs_condition", "assessment_date", "source_row",
                 "identity_status", "visit_status", "motor_total", "bradykinesia"]]
    base = base.merge(score_audit, left_on="source_row", right_on="primary_source_row", validate="one_to_one", how="left")
    base = base.merge(span[["source_row", "legacy_source_row", *difference]], on="source_row", validate="one_to_one", how="left")
    write_csv(base, PRIVATE / "results/Score_discrepancies.csv")
    write_csv(span, PRIVATE / "results/Legacy_vs_Original_Scores.csv")
    write_csv(changes[[*PAIR_KEYS, "visit_status", "motor_total_improvement", "legacy_motor_improvement",
                       "legacy_difference_improvement"]], PRIVATE / "results/Legacy_vs_Original_Improvement.csv")
    return rows, old_pairs


def cohort_flow(primary, paired, analysis, master):
    def count(key, n_people, n_records=None, note=""):
        return dict(stage=key, n_people=n_people, n_records=n_records, note=note)

    nominal = master.demographic_raw_target.eq("STN")
    statuses = paired.visit_status.value_counts()
    complete = paired.complete_score_pair.fillna(False)
    rows = [
        count("Primary workbook people and records", primary["people"], primary["rows"]),
        count("STN scored in primary", len(paired), primary["stn_visits"]),
        count("STN with complete total-score pairs", int(complete.sum()), int(complete.sum()) * 2),
        count("Demographic nominal STN label", int(nominal.sum()), note="Literal target column before checking remarks"),
        count("Demographic confirmed STN after target remarks", int(master.demographic_target.eq("STN").sum())),
        count("Demographic GPi", int(master.demographic_target.eq("GPi").sum())),
        count("Demographic mixed/asymmetric", int(master.demographic_target.eq("mixed_or_asymmetric").sum())),
        count("Demographic target other", int(master.demographic_target.eq("other").sum())),
        count("Demographic target unknown", int(master.demographic_target.eq("unknown").sum())),
        count("Original score and demographic names match exactly", int(paired.demographic_name_agreement.fillna(False).sum())),
        count("Demographic name variant linked via unique study code and original acquisition name", int(
            (paired.identity_status.ne(UNRESOLVED) & ~paired.demographic_name_agreement.fillna(False)).sum())),
        count("STN identity confirmed from original named acquisition roster", int(paired.identity_status.ne(UNRESOLVED).sum())),
        count("STN matched both frozen scans", int(paired.participant_id.notna().sum()), int(paired.participant_id.notna().sum()) * 2),
        count("Outside frozen cohort", int(statuses.get(OUTSIDE, 0))),
        count("Unresolved identity", int(statuses.get(UNRESOLVED, 0))),
        count("Date conflict; excluded", int(statuses.get(CONFLICT, 0)), int(statuses.get(CONFLICT, 0)) * 2),
        count("Independent same-date evidence on both conditions", int(statuses.get(VERIFIED, 0)), int(statuses.get(VERIFIED, 0)) * 2),
        count("Exact identity and condition; visit evidence insufficient", int(statuses.get(INSUFFICIENT, 0))),
        count("Main clinical association eligible", len(analysis), len(analysis) * 2),
    ]
    for reason, n in paired.loc[paired.visit_status.eq(OUTSIDE), "inclusion_reason"].value_counts().items():
        rows.append(count(reason, int(n)))
    return pd.DataFrame(rows)


def clinical_distributions(paired, analysis):
    rows = []
    for label, selection in [("All original STN pairs", paired), ("Association eligible", analysis)]:
        for field in CFG["descriptive_domains"]:
            for side, column in [("OFF", f"off_{field}"), ("ON", f"on_{field}"),
                                 ("OFF minus ON", f"{field}_improvement"),
                                 ("Percent improvement", f"{field}_percent_improvement")]:
                data = selection[column].dropna()
                n_zero = data.eq(0).sum()
                rows.append(dict(sample=label, outcome=field, readout=side, n=len(data),
                                 n_missing=len(selection) - len(data), n_unique=data.nunique(), n_zero=n_zero,
                                 zero_fraction=n_zero / len(data) if len(data) else np.nan,
                                 mean=data.mean(), median=data.median(), sd=data.std(ddof=1)))
    return pd.DataFrame(rows)


def run_audit(paths, imaging):
    primary, primary_receipt = read_primary(paths["score_primary"])
    secondary, secondary_receipt = read_secondary(paths["score_secondary"])
    second, audit = reconcile(primary, secondary)
    receipts = {"primary": primary_receipt, "secondary": secondary_receipt}
    write_source_audit(primary, secondary, second, receipts, audit, private_dir=PRIVATE)
    long, paired, analysis, master, demo_scope = match_visits(primary, paths, imaging)
    stn = primary.loc[primary.target.eq("STN")]
    require(len(ITEM_NAMES) == 33 and len(stn) == primary_receipt["stn_visits"], "Unexpected clinical item scope")
    write_json({"demographics": demo_scope}, PRIVATE / "manifests/Demographics_Read_Receipt.json")
    old, old_pairs = reconstruct_old_scores(paths, imaging)
    legacy_rows, old_pairs = compare_legacy(long, paired, old, old_pairs, second)
    count_data = {
        "primary_stn_dated_rows": primary_receipt["stn_dated_rows"],
        "primary_stn_complete_33_item_rows": primary_receipt["stn_rows_with_all_33_items"],
        "primary_stn_y_differs_from_corrected_bradykinesia": primary_receipt["stn_y_differs_from_pure_bradykinesia"],
        "secondary_stn_records_exact_33_items": audit["stn_records_exact_33_clinical_items"],
        "secondary_stn_at_rows_with_formula_includes_ai": audit["stn_at_formula_includes_ai_rows"],
        "secondary_stn_at_rows_with_bias": audit["stn_at_total_mismatch"],
        "secondary_stn_at_bias_exact_same_row_ai_sum": audit["stn_at_mismatch_explained_by_own_ai_sum"],
    }
    rows = [dict(metric=k, n=n, denominator=len(stn)) for k, n in count_data.items()]
    rows.extend([
        dict(metric="secondary_stn_at_patients_affected", n=audit["stn_at_affected_people"], denominator=len(paired)),
        dict(metric="secondary_stn_at_off_rows_affected", n=audit["stn_at_mismatch_by_dbs"].get("OFF", 0), denominator=len(paired)),
        dict(metric="secondary_stn_at_on_rows_affected", n=audit["stn_at_mismatch_by_dbs"].get("ON", 0), denominator=len(paired)),
        dict(metric="secondary_stn_at_bias_absolute_range", n=audit["stn_at_total_mismatch"], denominator=len(stn),
             min_points=audit["stn_at_mismatch_min"], max_points=audit["stn_at_mismatch_max"]),
        dict(metric="primary_stn_y_excess_range", n=primary_receipt["stn_y_differs_from_pure_bradykinesia"],
             denominator=len(stn), min_points=primary_receipt["stn_y_excess_min"], max_points=primary_receipt["stn_y_excess_max"]),
        dict(metric="secondary_blank_target_extra_candidates", n=audit["unmatched_secondary_blank_target_records"]),
        dict(metric="stn_medication_off_rows", n=int(stn.medication_state.eq("med_off").sum()), denominator=len(stn)),
    ])
    rows.extend(legacy_rows)
    write_csv(pd.DataFrame(rows), RESULTS / "Score_Reconciliation_Summary.csv")
    write_csv(cohort_flow(primary_receipt, paired, analysis, master), RESULTS / "STN_Cohort_Flow.csv")
    write_csv(clinical_distributions(paired, analysis), RESULTS / "Clinical_Distribution_Summary.csv")
    return paired, analysis, old_pairs
