"""Read frozen scan keys and original acquisition evidence without changing either."""

from base64 import b64decode
import csv
from datetime import date, datetime, time
from hashlib import sha256
import io
import json
from pathlib import PurePosixPath
import re

from openpyxl import load_workbook
import pandas as pd


CLOCK_BASIS = "original_dicom_acquisition_local_clock"
ROSTERS = ("demographics_Chs.csv", "demographics.csv", "demographics_HC_STNon_STNoff.csv")
CAPTURED_CLOCK = re.compile(r"(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d{1,6}))?")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def scan_key(label):
    return str(label).strip().casefold().removeprefix("sub-")


def unique_index(rows, key, label):
    index = {}
    for row in rows:
        value = key(row)
        if value in index:
            raise ValueError(f"Duplicate key in {label}")
        index[value] = row
    return index


def clock(value):
    match = CAPTURED_CLOCK.fullmatch(str(value).strip())
    if match is None:
        return None
    hour, minute, second = map(int, match.groups()[:3])
    fraction = match.group(4) or ""
    try:
        return time(hour, minute, second, int(fraction.ljust(6, "0")))
    except ValueError:
        return None


def date_value(value, fmt):
    try:
        return datetime.strptime(str(value).strip(), fmt).date()
    except ValueError:
        return None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rosters(path, crosscheck_path):
    embedded = read_json(path)["source_file_records"]
    verified = read_json(crosscheck_path)["rosters"]
    rosters = {}
    for name in ROSTERS:
        entry, = (row for label, row in embedded.items() if label.endswith(f"/code/{name}"))
        payload = b64decode(entry["data_bytes_b64"], validate=True)
        expected = verified[f"code/{name}"]
        require(sha256(payload).hexdigest() == entry["sha256"] == expected["sha256"],
                "Original acquisition roster failed archived digest cross-check")
        try:
            decoded = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = payload.decode("gb18030")
        rosters[name] = unique_index(csv.DictReader(io.StringIO(decoded)),
                                     lambda row: scan_key(row["subject_id"]), name)
        require(len(rosters[name]) == expected["record_count"],
                "Original acquisition roster failed archived row-count check")
    return rosters


def read_frozen(cohort_path, pairs_path):
    cohort = pd.read_csv(cohort_path, usecols=["subject", "scan", "condition"], dtype=str)
    pair_columns = ["participant_id", "on_scan_id", "off_scan_id"]
    pairs = pd.read_csv(pairs_path, usecols=pair_columns, dtype=str)
    require(len(pairs) == 25 and pairs.participant_id.is_unique and not pairs.isna().any().any(),
            "Expected 25 unique complete frozen patient pairs")
    require(cohort.condition.value_counts().to_dict() == {"ON": 25, "OFF": 25, "HC": 15},
            "Frozen ON/OFF/HC counts disagree with the audited cohort")
    require(cohort.scan.map(scan_key).is_unique, "Frozen scan keys must be unique")
    from_pairs = {(r.participant_id, r.on_scan_id, "ON") for r in pairs.itertuples()}
    from_pairs |= {(r.participant_id, r.off_scan_id, "OFF") for r in pairs.itertuples()}
    patient = cohort.loc[cohort.condition.ne("HC"), ["subject", "scan", "condition"]]
    require(from_pairs == set(patient.itertuples(index=False, name=None)),
            "Frozen patient condition/scan keys disagree between pair and cohort sources")
    return cohort[["subject", "scan", "condition"]]


def read_sidecars(path, cohort_keys):
    records = [r for r in read_json(path)["scan_records"]
               if r["dataset"] == "DBS" and r["modality"] == "bold"
               and scan_key(r["subject_label"]) in cohort_keys]
    return unique_index(records, lambda row: scan_key(row["subject_label"]),
                        "original DBS BOLD sidecars")


def read_workbook(path):
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheet = workbook["48次访视"]
        header, = sheet.iter_rows(min_row=4, max_row=4, values_only=True)
        column_names = ("DBS 条件", "DICOM 采集日期时间", "评分日期时间", "最后一次服药时间",
                        "停药时长（小时）", "原始 BOLD 旁车（相对 S1）", "名册 scan 日期",
                        "DICOM 减名册（天）")
        positions = {name: header.index(name) for name in column_names}
        rows = [{name: cells[index] for name, index in positions.items()}
                for cells in sheet.iter_rows(min_row=5, values_only=True)
                if cells[positions["DBS 条件"]] in ("ON", "OFF")]
    finally:
        workbook.close()
    return unique_index(rows, lambda row: scan_key(PurePosixPath(row[column_names[5]]).parts[0]),
                        "secondary scan visit audit workbook")


def sidecar_path(record):
    return record["bids_relative_path"].removesuffix(".nii.gz") + ".json"


def captured_date(key, condition, sidecar, repeated_sidecar, visit, source, matched, detail, workbook):
    """Recover an acquisition date supported by a stored original DICOM header match."""
    if visit is None:
        return None, None, None, None, False
    require(visit["condition"] == condition and scan_key(visit["label"]) == key,
            "Original BOLD discovery capture has a conflicting condition key")
    require(len(visit["json"]) == 1, "Original BOLD discovery capture has ambiguous sidecars")
    discovery, = visit["json"]
    acquisition = clock(discovery["fields"].get("AcquisitionTime"))
    original = clock(sidecar["json_metadata"].get("AcquisitionTime")) if sidecar else None
    replayed = clock(repeated_sidecar["json_metadata"].get("AcquisitionTime")) if repeated_sidecar else None
    clocks_agree = acquisition is not None and acquisition == original == replayed
    sidecars_agree = (sidecar is not None and repeated_sidecar is not None
                      and discovery["file"] == sidecar_path(sidecar) == sidecar_path(repeated_sidecar))
    if not clocks_agree or not sidecars_agree:
        return None, None, None, None, False

    candidate = date_value(source["scan"], "%Y/%m/%d")
    if matched and detail is None:
        # The source header reread matched an actual AcquisitionDate and AcquisitionTime
        # in the named patient's DICOM files. Its archive indexes the match by exact scan key.
        acquired = candidate
        evidence = "original_dicom_date_clock_match"
        folders = matched
    elif detail is not None and not matched:
        matches = detail["candidates"]
        require(len(matches) == 1 and matches[0]["series_match"] and matches[0]["description_match"],
                "Roster date conflict has no unique corroborating original DICOM match")
        match, = matches
        acquired = date_value(match["day"], "%Y%m%d")
        require(acquired is not None and acquired != candidate,
                "Expected an original DICOM date distinct from the roster date")
        evidence = "original_dicom_date_clock_series_match_resolved_roster_conflict"
        folders = [match["folder"]]
    else:
        return None, None, None, None, False

    require(acquired is not None, "Archived original DICOM match lacks a valid acquisition date")
    instant = datetime.combine(acquired, acquisition)
    check = False
    if workbook is not None:
        workbook_value = workbook["DICOM 采集日期时间"]
        workbook_roster = workbook["名册 scan 日期"]
        check = (workbook["DBS 条件"] == condition
                 and workbook["原始 BOLD 旁车（相对 S1）"] == discovery["file"]
                 and isinstance(workbook_value, datetime)
                 and abs((workbook_value - instant).total_seconds()) <= 0.0005
                 and isinstance(workbook_roster, (date, datetime))
                 and (workbook_roster.date() if isinstance(workbook_roster, datetime) else workbook_roster) == candidate
                 and workbook["DICOM 减名册（天）"] == (acquired - candidate).days)
        require(check, "Secondary audit workbook disagrees with original acquisition capture")
    return instant, evidence, folders, discovery["fields"]["AcquisitionTime"], check


def _load(workspace, module):
    # All sources are existing local archives. Their exact paths/hashes belong in the private manifest.
    archive = workspace / "WEiDA-raw-audit-private-20261116"
    capture = archive / "visit24_20261124"
    frozen = workspace / "WEiDA-DBS-V6" / "V6_DBS_ON_OFF_Microstate_Dynamics" / "private"
    paths = {
        "frozen_pair_manifest": frozen / "manifests" / "ON_OFF_Pair_Manifest.csv",
        "frozen_cohort_manifest": frozen / "results" / "Frozen_Cohort_Features.csv",
        "original_named_acquisition_rosters": archive / "source105_evidence.json",
        "original_roster_digest_crosscheck": archive / "source105_dbs_roster_check.json",
        "original_bids_inventory_capture": archive / "source105_inventory_original_capture.json",
        "repeated_bids_inventory_capture": archive / "source105_inventory.json",
        "original_bids_timing_discovery": capture / "discover.private.json",
        "original_acquisition_date_candidates": capture / "sources.private.json",
        "original_dicom_date_clock_header_matches": capture / "review.private.json",
        "original_dicom_series_validated_date_conflicts": capture / "confirm.private.json",
    }
    audit_workbook, = workspace.glob("V100*/结论/**/105原始访视逐人核查.xlsx")
    paths["secondary_visit_workbook_crosscheck"] = audit_workbook
    locked_paths = read_json(module / "private" / "Input_Hashes.json")
    hashes = {str(path.resolve()): sha256(path.read_bytes()).hexdigest() for path in paths.values()}
    for label in ("frozen_pair_manifest", "frozen_cohort_manifest", "original_named_acquisition_rosters",
                  "original_roster_digest_crosscheck", "secondary_visit_workbook_crosscheck"):
        path = paths[label].resolve()
        expected = [digest for ref, digest in locked_paths.items()
                    if str(ref).replace("\\", "/").casefold().endswith(f"/{path.name}".casefold())]
        require(len(expected) == 1 and hashes[str(path)] == expected[0],
                f"Input digest differs from the previous behavior analysis for {label}")

    cohort = read_frozen(paths["frozen_cohort_manifest"], paths["frozen_pair_manifest"])
    rosters = read_rosters(paths["original_named_acquisition_rosters"],
                           paths["original_roster_digest_crosscheck"])
    named = rosters[ROSTERS[0]]
    keys = set(cohort.scan.map(scan_key))
    original_sidecars = read_sidecars(paths["original_bids_inventory_capture"], keys)
    repeated_sidecars = read_sidecars(paths["repeated_bids_inventory_capture"], keys)
    discovery = read_json(paths["original_bids_timing_discovery"])
    discovered = unique_index(discovery["visits"], lambda row: scan_key(row["label"]),
                              "original BOLD timing discovery")
    source = read_json(paths["original_acquisition_date_candidates"])
    review = read_json(paths["original_dicom_date_clock_header_matches"])
    matches = {scan_key(key): folders for key, folders in review["raw_matches"].items()}
    extra = read_json(paths["original_dicom_series_validated_date_conflicts"])["extra"]
    conflicts = {scan_key(key): details for key, details in extra["raw_details"].items()}
    workbook = read_workbook(paths["secondary_visit_workbook_crosscheck"])
    require((matches.keys() | conflicts.keys()) == discovered.keys() == workbook.keys()
            and not matches.keys() & conflicts.keys(),
            "Original matched DICOM capture keys disagree with discovery and derivative workbook")

    rows = []
    xlsx_count = 0
    for subject, scan, condition in cohort.itertuples(index=False, name=None):
        key = scan_key(scan)
        pd_scan = condition != "HC"
        originals = [roster[key] for roster in rosters.values() if key in roster]
        expected_label = "stn" + condition.casefold() if pd_scan else "hc"
        expected_stim = condition.casefold() if pd_scan else ""
        confirmed_condition = bool(originals) and all(
            r["condition"].strip().casefold() == expected_label
            and r["DBS_status"].strip().casefold() == expected_stim for r in originals)
        expected_study = re.fullmatch(r"stn(?:on|off)(\d+)", key)
        named_original = named.get(key, {})
        person_key = "".join(named_original.get("name_Chs", "").casefold().split())
        candidate_raw = named_original.get("scan")
        candidates = {d for r in originals if (d := date_value(r["scan"], "%Y/%m/%d")) is not None}
        sidecar = original_sidecars.get(key)
        other_sidecar = repeated_sidecars.get(key)
        sidecar_key_agrees = (sidecar is not None and other_sidecar is not None
                              and sidecar_path(sidecar) == sidecar_path(other_sidecar))
        instant, evidence, folders, matched_clock, xlsx_check = captured_date(
            key, condition, sidecar, other_sidecar, discovered.get(key), source["scans"].get(key),
            matches.get(key), conflicts.get(key), workbook.get(key))
        xlsx_count += xlsx_check
        bid_clock_raw = sidecar["json_metadata"].get("AcquisitionTime") if sidecar else None
        acquisition_clock = clock(bid_clock_raw)
        candidate = next(iter(candidates)) if len(candidates) == 1 else None
        rows.append({
            "patient_id": subject if pd_scan else None,
            "scan_id": scan,
            "condition": condition,
            "scan_date": instant.date().isoformat() if instant is not None else None,
            "scan_datetime": instant.isoformat(timespec="microseconds") if instant is not None else None,
            "scan_precision": ("fractional_second" if instant is not None
                               else "time_of_day_only" if acquisition_clock is not None else "unavailable"),
            "scan_clock_basis": CLOCK_BASIS if instant is not None else None,
            "scan_evidence_source": evidence or ("original_bids_acquisition_time_only"
                                                  if acquisition_clock is not None else "no_acquisition_clock"),
            "scan_identity_confirmed": person_key != "" and sidecar_key_agrees,
            "scan_condition_confirmed": confirmed_condition and sidecar_key_agrees,
            "scan_roster_candidate_date": candidate.isoformat() if candidate is not None else None,
            "scan_roster_date_raw": candidate_raw,
            "scan_roster_date_disagreement_resolved": key in conflicts and instant is not None,
            "scan_date_unresolved": instant is None and (len(candidates) > 1 or key in conflicts),
            "scan_bids_acquisition_time_raw": bid_clock_raw,
            "scan_bids_sidecar_source": sidecar_path(sidecar) if sidecar is not None else None,
            "scan_dicom_confirmed_clock_raw": matched_clock,
            "scan_dicom_evidence_folders": json.dumps(folders, ensure_ascii=False) if folders else None,
            "scan_medication_state": None,
            "scan_last_medication_datetime": None,
            "scan_medication_elapsed_hours": None,
            "recorded_dbs_state": condition if pd_scan and confirmed_condition else None,
            "stim_switch_datetime": None,
            "stabilization_hours": None,
            "stim_change_during_scan": None,
            "_person_key": person_key,
            "_study_key": expected_study[1] if expected_study is not None else None,
        })

    frame = pd.DataFrame(rows)
    patient = frame[frame.condition.ne("HC")]
    consistent = {key for key, pair in patient.groupby("patient_id")
                  if pair.scan_identity_confirmed.all() and pair["_person_key"].nunique() == 1
                  and pair["_study_key"].nunique() == 1}
    pd_mask = frame.condition.ne("HC")
    frame.loc[pd_mask, "scan_identity_confirmed"] &= frame.loc[pd_mask, "patient_id"].isin(consistent)
    frame = frame.drop(columns=["_person_key", "_study_key"])
    patient = frame.loc[pd_mask]
    hc = frame.loc[~pd_mask]

    summary = {
        "frozen_patient_scans": len(patient),
        "frozen_hc_scans": len(hc),
        "patient_identity_confirmed_scans": int(patient.scan_identity_confirmed.sum()),
        "patient_condition_confirmed_scans": int(patient.scan_condition_confirmed.sum()),
        "patient_dicom_timestamp_scans": int(patient.scan_datetime.notna().sum()),
        "patient_sidecar_clock_scans": int(patient.scan_bids_acquisition_time_raw.map(clock).notna().sum()),
        "dicom_date_matches_roster_scans": int(patient.scan_evidence_source.eq("original_dicom_date_clock_match").sum()),
        "dicom_date_disagrees_roster_scans": int(patient.scan_roster_date_disagreement_resolved.sum()),
        "dicom_bids_exact_clock_matches": int(patient.scan_dicom_confirmed_clock_raw.notna().sum()),
        "workbook_timestamp_crosschecks": int(xlsx_count),
        "patient_scan_medication_documented_scans": int(patient.scan_medication_state.notna().sum()),
        "patient_switch_documented_scans": int(patient.stim_switch_datetime.notna().sum()),
        "patient_stabilization_documented_scans": int(patient.stabilization_hours.notna().sum()),
        "patient_stim_continuity_documented_scans": int(patient.stim_change_during_scan.notna().sum()),
        "hc_dicom_timestamp_scans": int(hc.scan_datetime.notna().sum()),
        "hc_sidecar_clock_scans": int(hc.scan_bids_acquisition_time_raw.map(clock).notna().sum()),
        "hc_roster_candidate_dates": int(hc.scan_roster_candidate_date.notna().sum()),
        "historic_source_coverage": {
            "matched_raw_dicom_headers_read": review["summary"]["raw_headers"]["files"],
            "archived_candidate_xlsx_examined": review["summary"]["examined"][".xlsx"],
            "archived_review_unreadable_directories": source["summary"]["permission_or_stat"].get("PermissionError", 0),
            "archived_bold_scans_tsv_records": discovery["summary"]["scanstsv"],
            "archived_structured_reports_screened": sum(len(r["sr_records"]) for r in extra["archives"].values()
                                                        if "sr_records" in r),
        },
        "source_roles": list(paths),
    }
    return patient, hc, summary, hashes


def load_scan_times(workspace, module, private):
    """Return 50 patient scan records, public-safe counts, and private input digests."""
    patient, _, summary, hashes = _load(workspace, module)
    return patient, summary, hashes


def load_hc_scan_times(workspace, module, private):
    """Return HC acquisition evidence separately from HC clinical scores."""
    _, hc, summary, hashes = _load(workspace, module)
    return hc, {k: v for k, v in summary.items() if k.startswith("hc_")}, hashes
