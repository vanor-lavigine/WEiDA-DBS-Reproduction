"""Build public aggregates and local patient-level worksheets."""

import pandas as pd

from timing_rules import GROUPS, STATUSES, present


def row(metric, n, denominator, evidence):
    return {
        "metric": metric, "n": int(n), "denominator": int(denominator),
        "percent": round(100 * n / denominator, 2), "evidence": evidence,
    }


def summaries(visits, groups, orders, hc):
    output = []
    for condition in ("ON", "OFF", "both"):
        part = visits if condition == "both" else visits.loc[visits.condition.eq(condition)]
        for status in STATUSES:
            output.append(dict(condition=condition, **row(status, part.visit_match_status.eq(status).sum(), len(part), "assessment-scan link")))
        for field in ("date_difference_hours", "calendar_date_difference_days"):
            output.append(dict(condition=condition, **row(field + "_computable", part[field].notna().sum(), len(part), "documented temporal precision")))
        for relation in ("before", "after", "same_recorded_time", "unknown"):
            output.append(dict(condition=condition, **row("score_" + relation, part.assessment_relative_to_scan.eq(relation).sum(), len(part), "score relative to acquisition")))
    for group in GROUPS:
        output.append(dict(condition="patient_pair", **row("group_" + group, groups.group.eq(group).sum(), len(groups), "both DBS conditions combined")))
    link_summary = pd.DataFrame(output)

    order_summary = pd.DataFrame([
        row(status, orders.scan_order.eq(status).sum(), len(orders), "DBS scan order from common-clock time")
        for status in ("ON first", "OFF first", "same recorded time", "unverified")
    ] + [
        row("same_calendar_day_and_known_order", (
            orders.OFF_minus_ON_hours.notna() & orders.ON_scan_date.eq(orders.OFF_scan_date)
        ).sum(), len(orders), "resolved scan dates")
    ])
    hc_summary = pd.DataFrame([
        row(label, hc[key].eq(True).sum(), len(hc), description)
        for label, key, description in (
            ("scan_identity_confirmed", "scan_identity_confirmed", "frozen HC scan linked to source cohort"),
            ("motor_assessment_documented", "motor_assessment_documented", "documented original assessment"),
            ("measured_zero_verified", "measured_zero_verified", "actual measured score of zero"),
            ("same_instrument_confirmed", "same_instrument_confirmed", "shared instrument and scoring rules verified"),
            ("same_visit_confirmed", "same_visit_confirmed", "original assessment and BOLD visit linked"),
            ("usable_as_clinical_reference", "usable_as_clinical_reference", "measured zero, comparable scale and visit verified"),
        )
    ])
    medication_rows = []
    scan_meds = {}
    for condition in ("ON", "OFF"):
        part = visits.loc[visits.condition.eq(condition)]
        scan_meds[condition] = part.set_index("patient_id").scan_medication_state
        tests = {
            "score_medication_OFF": part.score_medication_state.eq("OFF"),
            "scan_medication_documented": part.scan_medication_state.map(present),
            "scan_DBS_condition_verified": part.scan_condition_confirmed.eq(True),
            "stim_switch_time_documented": part.stim_switch_datetime.map(present),
            "stabilization_hours_verified": part.stabilization_hours.notna(),
            "in_scan_stimulation_documented": part.stim_change_during_scan.map(present),
        }
        for label, verified in tests.items():
            medication_rows.append(dict(condition=condition, **row(label, verified.sum(), len(part), "condition-specific evidence")))
    on, off = scan_meds["ON"], scan_meds["OFF"]
    meds_both = on.map(present) & off.map(present)
    medication_rows.extend([
        dict(condition="patient_pair", **row("scan_medication_comparable", meds_both.sum(), len(groups), "both scan medication states documented")),
        dict(condition="patient_pair", **row("scan_medication_same_confirmed", (meds_both & on.eq(off)).sum(), len(groups), "same medication state at both scans")),
    ])
    return {
        "Clinical_Scan_Visit_Matching.csv": link_summary,
        "HC_Clinical_Score_Audit.csv": hc_summary,
        "Scan_Order_Audit.csv": order_summary,
        "Medication_Stimulation_Audit.csv": pd.DataFrame(medication_rows),
        "Timing_Difference_Summary.csv": time_summary(visits, orders),
    }


def time_summary(visits, orders):
    rows = []
    for condition in ("ON", "OFF", "both"):
        part = visits if condition == "both" else visits.loc[visits.condition.eq(condition)]
        for field in ("date_difference_hours", "calendar_date_difference_days"):
            rows.append(describe(condition, field, part[field], len(part)))
    rows.append(describe("patient_pair", "OFF_minus_ON_hours", orders.OFF_minus_ON_hours, len(orders)))
    return pd.DataFrame(rows)


def describe(condition, metric, vector, total):
    values = pd.to_numeric(vector.dropna())
    return {
        "condition": condition,
        "metric": metric,
        "n": len(values),
        "denominator": total,
        "min": round(values.min(), 3),
        "p25": round(values.quantile(.25), 3),
        "median": round(values.median(), 3),
        "p75": round(values.quantile(.75), 3),
        "max": round(values.max(), 3),
    }


def patient_worksheets(visits, groups):
    wide_fields = [
        "scan_id", "scan_datetime", "scan_date", "scan_precision", "scan_roster_candidate_date",
        "score_datetime", "score_date", "original_score_date_raw",
        "score_precision", "scan_medication_state", "score_medication_state", "assessment_source",
        "date_difference_hours", "calendar_date_difference_days", "visit_match_status",
        "visit_match_reason", "assessment_relative_to_scan", "assessment_order_evidence",
        "scan_evidence_source", "score_evidence_source",
    ]
    medicine_fields = [
        "scan_id", "scan_medication_state", "score_medication_state", "scan_condition_confirmed",
        "stim_switch_datetime", "stabilization_hours", "stim_change_during_scan",
        "scan_evidence_source", "score_evidence_source",
    ]

    def pivot(fields):
        table = visits.pivot(index=["patient_id", "study_code"], columns="condition", values=fields)
        table.columns = [f"{condition}_{name}" for name, condition in table.columns]
        return table.reset_index()

    comparison = pivot(wide_fields)
    comparison = comparison.rename(columns={f"{c}_scan_medication_state": f"{c}_medication_state"
                                            for c in ("ON", "OFF")})
    comparison["stimulation_state"] = "DBS ON and DBS OFF"
    comparison = comparison.merge(groups[["patient_id", "group"]], on="patient_id", validate="one_to_one")
    medication = pivot(medicine_fields)
    for condition in ("ON", "OFF"):
        medication[f"{condition}_stimulation_state"] = condition
    return comparison, medication
