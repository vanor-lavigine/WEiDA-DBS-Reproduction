"""Trace HC assessment evidence and preserve unavailable scores as missing."""

from base64 import b64decode
import csv
from datetime import datetime, time, timedelta
from hashlib import file_digest, sha256
import io
import json
from pathlib import Path
import re
import sys

from openpyxl import load_workbook
import pandas as pd


SCORE_MODULE = "BOLD_Microstate_STN_Original_Scores"
HC_SHEET = "健康参照名单"
HC_HEADER = ("扫描日期", "类别", "姓名", "扫描时间", "性别", "年龄", "身高", "体重", "文化程度", "利手")
WORKBOOK_NAMES = {
    "manual_item_workbook": "PD开关机fMRI评分_202011_gr_20201228.xlsx",
    "improvement_crosscheck_workbook": "PD开关机fMRI评分_20201218改善率..xlsx",
    "demographics_workbook": "PD-开关机人口学及刺激参数_2023.07.31.xlsx",
}
WORKBOOK_LOCK_ROLES = {
    "manual_item_workbook": "score_secondary",
    "improvement_crosscheck_workbook": "score_primary",
    "demographics_workbook": "score_demographics",
}


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _hash_input(path, workspace, hashes, expected=None):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(workspace) or not path.is_file():
        raise ValueError("HC audit source is outside the workspace or is missing")
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("HC audit source differs from the archived receipt")
    hashes[path.as_posix()] = digest
    return path


def _input_from_lock(role, receipts, workspace, hashes):
    receipt = receipts[role]
    return _hash_input(receipt["path"], workspace, hashes, receipt["sha256"])


def _raw_workbooks(workspace, receipts, hashes):
    paths = {}
    for role, filename in WORKBOOK_NAMES.items():
        base = re.escape(Path(filename).stem.rstrip("."))
        suffix = r"(?:\s*[\(（]\d+[\)）])?"
        pattern = re.compile(base + r"\.*" + suffix + r"\.*\.xlsx", re.I)
        candidates = sorted(path for path in workspace.iterdir() if pattern.fullmatch(path.name))
        if not candidates:
            raise FileNotFoundError(f"Missing HC audit workbook role: {role}")
        receipt = receipts[WORKBOOK_LOCK_ROLES[role]]["sha256"]
        paths[role] = []
        for candidate in candidates:
            checked = _hash_input(candidate, workspace, hashes)
            paths[role].append((checked, hashes[checked.as_posix()] == receipt))
        if role != "improvement_crosscheck_workbook" and not any(flag for _, flag in paths[role]):
            raise ValueError(f"No copy matches the archived receipt for {role}")
    return paths


def _scan_key(value):
    return str(value).strip().casefold()


def _capture_roster(path, check_path, frozen_hc):
    members = _read_json(path)["source_file_records"]
    entry, = [record for key, record in members.items() if key.endswith("/code/demographics_Chs.csv")]
    content = b64decode(entry["data_bytes_b64"], validate=True)
    checked = _read_json(check_path)["rosters"]["code/demographics_Chs.csv"]["sha256"]
    if sha256(content).hexdigest() != checked or entry["sha256"] != checked:
        raise ValueError("The named acquisition archive failed its independent checksum")
    for codec in ("utf-8-sig", "gb18030"):
        try:
            records = list(csv.DictReader(io.StringIO(content.decode(codec))))
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("The named acquisition archive encoding is unavailable")
    found = {}
    for record in records:
        key = _scan_key(record["subject_id"])
        if key in frozen_hc:
            if key in found or record["condition"].strip().casefold() != "hc":
                raise ValueError("The frozen HC key has missing or conflicting group provenance")
            found[key] = record
    if set(found) != frozen_hc:
        raise ValueError("The named acquisition archive lacks part of the frozen HC cohort")
    return found, members


def _sheet_inventory(source, aliases, normalize_person, demographics=False, data_only=False):
    by_name = {}
    wb = load_workbook(source, read_only=True, data_only=data_only)
    if demographics and HC_SHEET not in wb.sheetnames:
        wb.close()
        raise ValueError("The demographic workbook health roster sheet is missing")
    inventory = {}
    try:
        for sheet in wb:
            rows = iter(sheet.values)
            if demographics and sheet.title == HC_SHEET:
                inventory[sheet.title] = (sheet.sheet_state, 0)
                if tuple(next(rows)) != HC_HEADER:
                    raise ValueError("Health roster fields changed and need review")
                for row_number, row in enumerate(rows, start=2):
                    if not normalize_person(row[2]) or str(row[1]).strip().casefold() not in {"nc", "hc"}:
                        continue
                    name = normalize_person(row[2])
                    if name in by_name:
                        raise ValueError("Ambiguous HC name in the health roster")
                    by_name[name] = (row_number, row[0], row[3])
                continue
            keyed_rows = 0
            for row in rows:
                keyed_rows += any(isinstance(value, str) and normalize_person(value) in aliases
                                  for value in row)
            inventory[sheet.title] = (sheet.sheet_state, keyed_rows)
    finally:
        wb.close()
    return by_name, inventory


def _registered_inventory(source, aliases, normalize_person, demographics=False):
    roster, inventory = _sheet_inventory(source, aliases, normalize_person, demographics)
    _, cached = _sheet_inventory(source, aliases, normalize_person, demographics, data_only=True)
    if any(count for _, count in (*inventory.values(), *cached.values())):
        raise ValueError("A linked HC row in a candidate sheet requires source interpretation")
    return roster, {
        "worksheets_checked": len(inventory),
        "hidden_worksheets_checked": sum(state != "visible" for state, _ in inventory.values()),
        "keyed_hc_rows_outside_health_roster": 0,
        "formula_and_cached_value_layers_checked": True,
    }


def _historical_exports(paths, scan_keys, aliases, normalize_person):
    links = pd.read_csv(paths["archived_score_links"], dtype=str)
    pairs = pd.read_csv(paths["archived_score_pairs"], dtype=str)
    domains = pd.read_csv(paths["archived_score_domains"], dtype=str)
    if not links.condition.isin(["ON", "OFF"]).all() or any(links.scan.map(_scan_key).isin(scan_keys)):
        raise ValueError("The historical patient score export has an unexpected cohort")
    if pairs[["on_scan_id", "off_scan_id"]].map(_scan_key).isin(scan_keys).to_numpy().any():
        raise ValueError("The historical patient pair export has an unexpected HC scan link")
    if domains["研究编号"].map(normalize_person).isin(aliases).any():
        raise ValueError("The historical patient score domain export has an unexpected HC key")
    return {
        "condition_keyed_patient_assessment_rows": len(links),
        "archived_patient_pairs": len(pairs),
        "archived_score_domain_rows": len(domains),
        "keyed_hc_score_rows": 0,
    }


def _clock(value):
    if isinstance(value, datetime):
        return value.time().isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, timedelta) and value < timedelta(days=1):
        return str(value)
    if isinstance(value, str) and re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?", value.strip()):
        return value.strip()
    return None


def load_hc_scores(workspace, module, private):
    """Return fifteen private HC records, an aggregate summary, and private input hashes."""
    workspace = Path(workspace).resolve(strict=True)
    # Read existing normalizers. The runner owns writing the returned private records.
    score_scripts = Path(module).resolve().parent / SCORE_MODULE / "scripts"
    sys.path.insert(0, str(score_scripts))
    try:
        from identity_audit import normalized_date
        from score_sources import normalize_person
    finally:
        sys.path.remove(str(score_scripts))

    hashes = {}
    lock = _hash_input(workspace / "WEiDA-DBS-V8" / SCORE_MODULE
                       / "private/manifests/Prespecified_Inference_Lock.json", workspace, hashes)
    receipts = _read_json(lock)["input_files"]
    roles = ("v6_cohort", "original_acquisition_capture", "original_acquisition_capture_check",
             "acquisition_workbook", "archived_score_links", "archived_score_pairs", "archived_score_domains")
    paths = {role: _input_from_lock(role, receipts, workspace, hashes) for role in roles}
    frozen = pd.read_csv(paths["v6_cohort"], usecols=["scan", "condition"], dtype=str)
    scans = frozen.loc[frozen.condition.eq("HC"), "scan"]
    keys = set(scans.map(_scan_key))
    if len(scans) != 15 or len(keys) != 15 or scans.isna().any():
        raise ValueError("The A3 frozen HC cohort must have fifteen unique scan keys")
    capture, members = _capture_roster(paths["original_acquisition_capture"],
                                       paths["original_acquisition_capture_check"], keys)
    names = {normalize_person(record["name_Chs"]) for record in capture.values()}
    if not all(names) or len(names) != len(keys):
        raise ValueError("A frozen HC scan lacks an unambiguous original acquisition name")
    aliases = names | {normalize_person(value) for key in capture for value in (key, key.removeprefix("sub-"))}
    aliases |= {normalize_person(record["name"]) for record in capture.values()} - {""}

    workbooks = _raw_workbooks(workspace, receipts, hashes)
    source_inventory = {}
    for role, copies in workbooks.items():
        reports = []
        for path, matches_receipt in copies:
            is_demographics = role == "demographics_workbook"
            roster, report = _registered_inventory(path, aliases, normalize_person, is_demographics)
            report["matches_archived_receipt"] = matches_receipt
            reports.append(report)
            if is_demographics and matches_receipt:
                health_rows = roster
        source_inventory[role] = reports

    if not names.issubset(health_rows):
        raise ValueError("A frozen HC lacks a named row on the demographic HC roster")
    representative_sheet, = list(workspace.glob("105*/师姐评分表/STN患者评分与微状态对应.xlsx"))
    representative_sheet = _hash_input(representative_sheet, workspace, hashes)
    for role, path in (("historical_visit_trace_workbook", paths["acquisition_workbook"]),
                       ("historical_patient_score_workbook", representative_sheet)):
        _, report = _registered_inventory(path, aliases, normalize_person)
        source_inventory[role] = [report]
    for key, record in members.items():
        if not key.casefold().endswith(".xlsx"):
            continue
        content = b64decode(record["data_bytes_b64"], validate=True)
        if sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("An embedded acquisition workbook differs from its capture checksum")
        _, report = _registered_inventory(io.BytesIO(content), aliases, normalize_person)
        source_inventory.setdefault("embedded_acquisition_workbooks", []).append(report)
    export_audit = _historical_exports(paths, keys, aliases, normalize_person)

    rows = []
    for scan in scans:
        original = capture[_scan_key(scan)]
        roster_row, roster_date, roster_time = health_rows[normalize_person(original["name_Chs"])]
        recorded_roster_date = normalized_date(roster_date)
        original_date = normalized_date(original["scan"])
        roster_clock = _clock(roster_time)
        precision = "unavailable"
        if recorded_roster_date:
            precision = "date_and_clock" if roster_clock else "date_only"
        rows.append({
            "scan_id": scan,
            "scan_identity_confirmed": True,
            "original_acquisition_date": original_date,
            "roster_scan_date": recorded_roster_date,
            "roster_scan_time": roster_clock,
            "roster_scan_precision": precision,
            "roster_same_scan_day": recorded_roster_date is not None
                                    and recorded_roster_date == original_date,
            "motor_assessment_documented": False,
            "motor_score": pd.NA,
            "measured_zero_verified": False,
            "same_instrument_confirmed": False,
            "same_day_assessment_confirmed": False,
            "same_visit_confirmed": False,
            "assessment_datetime": pd.NA,
            "assessment_time_precision": "unavailable",
            "assessment_source": "unavailable",
            "assessment_missing_reason": "no_linked_original_HC_motor_assessment_in_checked_sources",
            "usable_as_clinical_reference": False,
            "evidence_source": f"frozen_A3_cohort; named_acquisition_archive; health_roster_row_{roster_row}",
        })
    frame = pd.DataFrame.from_records(rows)
    audit = {
        "frozen_hc": len(frame),
        "scan_identity_confirmed": int(frame.scan_identity_confirmed.sum()),
        "health_roster_rows": len(health_rows),
        "health_roster_rows_outside_frozen": len(health_rows) - len(frame),
        "roster_scan_date_agreement": int(frame.roster_same_scan_day.sum()),
        "health_roster_scan_clock_recorded": int(frame.roster_scan_time.notna().sum()),
        "health_roster_motor_score_fields": 0,
        "motor_assessment_documented": int(frame.motor_assessment_documented.sum()),
        "motor_score_available": int(frame.motor_score.notna().sum()),
        "measured_zero_verified": int(frame.measured_zero_verified.sum()),
        "same_instrument_confirmed": int(frame.same_instrument_confirmed.sum()),
        "same_day_assessment_confirmed": int(frame.same_day_assessment_confirmed.sum()),
        "same_visit_confirmed": int(frame.same_visit_confirmed.sum()),
        "usable_as_clinical_reference": int(frame.usable_as_clinical_reference.sum()),
        "zero_provenance": "all_zero_claim_unverified_no_source_score_or_imputation_found",
        "candidate_workbooks": source_inventory,
        "archived_clinical_exports": export_audit,
    }
    return frame, audit, hashes
