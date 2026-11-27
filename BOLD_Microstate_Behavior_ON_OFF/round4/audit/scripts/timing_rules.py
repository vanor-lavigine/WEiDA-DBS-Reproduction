"""Keep calendar-day evidence and common-clock evidence separate."""

from datetime import date, datetime

import pandas as pd


STATUSES = (
    "Confirmed same visit",
    "Confirmed same day",
    "Near-time assessment",
    "Different visit",
    "Date unavailable",
    "Identity uncertain",
)
GROUPS = ("A", "B", "C")
KEYS = ["patient_id", "condition", "scan_id"]


def present(value):
    return not pd.isna(value) and str(value).strip().casefold() not in ("", "na", "unknown", "unavailable")


def documented(value):
    return not pd.isna(value) and value == True  # Accept Python and pandas booleans.


def timestamp(value):
    if not present(value) or ("T" not in str(value) and " " not in str(value)):
        return None
    return datetime.fromisoformat(str(value))


def time_difference(later_value, earlier_value, shared_clock):
    later, earlier = timestamp(later_value), timestamp(earlier_value)
    if later is None or earlier is None:
        return None
    aware_later, aware_earlier = later.utcoffset() is not None, earlier.utcoffset() is not None
    if aware_later != aware_earlier or (not aware_later and not shared_clock):
        return None
    return (later - earlier).total_seconds() / 3600


def classify(row):
    """Evaluate the strength of the evidence for one DBS condition."""
    trust_columns = (
        "identity_confirmed", "condition_confirmed", "score_source_confirmed",
        "scan_identity_confirmed", "scan_condition_confirmed",
    )
    failed = [name for name in trust_columns if not documented(row.get(name))]
    verified = not failed
    clock_difference = None
    day_difference = None
    score_date, scan_date = row.get("score_date"), row.get("scan_date")
    dates_complete = present(score_date) and present(scan_date)
    date_conflict = documented(row.get("score_date_unresolved")) or documented(row.get("scan_date_unresolved"))
    if verified and dates_complete and not date_conflict:
        day_difference = (date.fromisoformat(score_date) - date.fromisoformat(scan_date)).days
        clock_difference = time_difference(
            row.get("score_datetime"), row.get("scan_datetime"), documented(row.get("score_scan_shared_clock"))
        )
    same_visit = documented(row.get("same_visit_documented"))
    different_visit = documented(row.get("different_visit_documented"))
    if same_visit and different_visit:
        raise ValueError("Conflicting direct visit evidence requires source adjudication")
    if not verified:
        status, reason = STATUSES[5], "|".join(failed)
    elif same_visit:
        status, reason = STATUSES[0], "direct_visit_link"
    elif different_visit:
        status, reason = STATUSES[3], "distinct_visit_documented"
    elif day_difference == 0:
        status, reason = STATUSES[1], "same_calendar_day_no_direct_visit_link"
    elif documented(row.get("near_time_documented")):
        status, reason = STATUSES[2], "temporal_proximity_documented_no_direct_visit_link"
    elif day_difference is not None:
        status, reason = STATUSES[3], "independently_documented_dates_differ"
    else:
        date_reason = row.get("score_date_unavailable_reason")
        if date_conflict:
            date_reason = "unresolved_date_conflict"
        status, reason = STATUSES[4], date_reason if present(date_reason) else "date_evidence_incomplete"
    if clock_difference is not None:
        timing = "before" if clock_difference < 0 else "after" if clock_difference > 0 else "same_recorded_time"
        timing_evidence = "comparable_clocks"
    else:
        direct_order = row.get("score_relative_to_scan_documented")
        if verified and present(direct_order):
            timing, timing_evidence = direct_order, "direct_order_record"
        elif day_difference:
            timing, timing_evidence = ("before" if day_difference < 0 else "after"), "distinct_calendar_days"
        else:
            timing, timing_evidence = "unknown", "insufficient_temporal_precision"
    return {
        "visit_match_status": status,
        "visit_match_reason": reason,
        "date_difference_hours": clock_difference,
        "calendar_date_difference_days": day_difference,
        "assessment_relative_to_scan": timing,
        "assessment_order_evidence": timing_evidence,
        "core_link_verified": verified,
    }


def link_visits(scores, scans, expected_pairs):
    stn = scans.loc[scans.condition.isin(("ON", "OFF"))]
    joined = stn.merge(scores, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if len(joined) != expected_pairs * 2 or not joined["_merge"].eq("both").all():
        raise ValueError("Frozen patient, DBS condition and scan keys did not join exactly")
    if not joined.groupby(["patient_id", "condition"]).size().unstack().eq(1).all().all():
        raise ValueError("Each frozen patient must contribute exactly one ON and one OFF visit")
    if not joined.groupby("patient_id").study_code.nunique().eq(1).all():
        raise ValueError("Research-code linkage is inconsistent across DBS conditions")
    joined = joined.drop(columns="_merge").sort_values(KEYS, ignore_index=True)
    visits = pd.concat([joined, pd.DataFrame([classify(row) for row in joined.to_dict("records")])], axis=1)
    groups = visits.pivot(index="patient_id", columns="condition", values="visit_match_status")
    groups["group"] = "B"
    groups.loc[groups[["ON", "OFF"]].eq(STATUSES[0]).all(axis=1), "group"] = "A"
    groups.loc[groups[["ON", "OFF"]].eq(STATUSES[5]).any(axis=1), "group"] = "C"
    return visits.merge(groups[["group"]], on="patient_id", validate="many_to_one"), groups.reset_index()


def scan_order(scans):
    rows = []
    for patient, records in scans.groupby("patient_id"):
        visits = records.set_index("condition")
        on, off = (visits.loc[cond] for cond in ("ON", "OFF"))
        same_basis = (
            present(on.get("scan_clock_basis")) and on.get("scan_clock_basis") == off.get("scan_clock_basis")
        )
        delay = time_difference(off.get("scan_datetime"), on.get("scan_datetime"), same_basis)
        order = "unverified"
        if delay is not None:
            order = "ON first" if delay > 0 else "OFF first" if delay < 0 else "same recorded time"
        rows.append({
            "patient_id": patient,
            "ON_scan_id": on.scan_id, "OFF_scan_id": off.scan_id,
            "ON_scan_datetime": on.get("scan_datetime"), "OFF_scan_datetime": off.get("scan_datetime"),
            "ON_scan_date": on.get("scan_date"), "OFF_scan_date": off.get("scan_date"),
            "ON_scan_precision": on.get("scan_precision"), "OFF_scan_precision": off.get("scan_precision"),
            "ON_evidence_source": on.get("scan_evidence_source"), "OFF_evidence_source": off.get("scan_evidence_source"),
            "scan_order": order, "OFF_minus_ON_hours": delay,
        })
    return pd.DataFrame(rows)
