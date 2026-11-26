"""Verify stimulation targets, subject/scan keys, and dated acquisition evidence."""

from base64 import b64decode
import csv
from datetime import date, datetime
from hashlib import sha256
import io
import re

import numpy as np
from openpyxl import load_workbook
import pandas as pd

from pipeline_io import CFG, PAIR_KEYS, PRIVATE, digest, read_json, require, write_csv
from score_sources import normalize_person, normalize_study_code

DEMO_SHEET = "PD-DBS开关机fMRI患者总名单"
SCAN = re.compile(r"sub-stn(on|off)(\d+)", re.I)
ASYMMETRIC = re.compile(r"(?:不[对對][称稱].{0,4}靶[点點]|靶[点點].{0,4}不[对對][称稱])")
VERIFIED = "verified_identity_condition_visit"
INSUFFICIENT = "verified_identity_condition_visit_insufficient"
CONFLICT = "documented_visit_date_conflict"
UNRESOLVED = "identity_unresolved"
OUTSIDE = "outside_frozen_imaging"
DOMAIN_FIELDS = CFG["descriptive_domains"]


def normalized_date(value):
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", format="mixed")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def study_scan(study, condition):
    match = re.fullmatch(r"STN(\d+)", study, re.I)
    return f"sub-stn{condition.casefold()}{match[1]}" if match and condition in ("ON", "OFF") else None


def target_from_record(label, comment):
    raw = re.sub(r"\s+", "", str(label or "")).casefold()
    tags = set(re.findall(r"stn|gpi|vim|psa", raw))
    if ASYMMETRIC.search(f"{label or ''} {comment or ''}") or len(tags) > 1:
        return "mixed_or_asymmetric"
    if raw in ("stn", "b-stn"):
        return "STN"
    if raw in ("gpi", "b-gpi"):
        return "GPi"
    return "other" if raw else "unknown"


def demographics(path):
    workbook = load_workbook(path, data_only=True)
    try:
        sheet = workbook[DEMO_SHEET]
        require(sheet["E2"].value == "靶点" and sheet["F2"].value == "左侧" and sheet["J2"].value == "右侧",
                "Unexpected demographics or stimulation headers")
        merged_values = {}
        for block in sheet.merged_cells.ranges:
            top = sheet.cell(block.min_row, block.min_col)
            for row, column in block.cells:
                if row >= 4 and column <= 18:
                    merged_values[row, column] = top.value
        rows = []
        for row in range(4, sheet.max_row + 1):
            # Only a real name cell starts a patient record. Merged continuation
            # rows are attached to that record and never treated as new people.
            name = sheet.cell(row, 2).value
            if not name:
                continue
            name_range = next((g for g in sheet.merged_cells.ranges
                               if g.min_col == g.max_col == 2 and g.min_row == row), None)
            last = name_range.max_row if name_range is not None else row
            def value(r, c):
                return merged_values.get((r, c), sheet.cell(r, c).value)

            labels = list(dict.fromkeys(str(value(r, 5) or "") for r in range(row, last + 1)))
            comments = list(dict.fromkeys(str(value(r, 18) or "") for r in range(row, last + 1)))
            label = " / ".join(t for t in labels if t)
            comment = " / ".join(t for t in comments if t)
            settings = [{sheet.cell(3, c).coordinate.rstrip("0123456789"): value(r, c) for c in range(6, 14)}
                        for r in range(row, last + 1)]
            rows.append(dict(demographic_source_row=row, demographic_last_row=last, demographic_name=name,
                             demographic_name_key=normalize_person(name), study_code=normalize_study_code(value(row, 1)),
                             demographic_raw_target=label, demographic_target=target_from_record(label, comment),
                             demographic_remarks=comment, stimulation_F_to_M=repr(settings)))
        out = pd.DataFrame(rows)
        require(out.demographic_name_key.is_unique, "Demographic name anchors are duplicated")
        return out, dict(file=path.name, sheet=DEMO_SHEET, header_rows=[2, 3], rows_read=[4, sheet.max_row],
                         columns_read="A:R", merged_ranges=len(sheet.merged_cells.ranges))
    finally:
        workbook.close()


def captured_roster(paths):
    archived = read_json(paths["original_acquisition_capture"])["source_file_records"]
    _, match = next((k, v) for k, v in archived.items() if k.endswith("/code/demographics_Chs.csv"))
    content = b64decode(match["data_bytes_b64"], validate=True)
    reread = read_json(paths["original_acquisition_capture_check"])["rosters"]["code/demographics_Chs.csv"]
    require(sha256(content).hexdigest() == match["sha256"] == reread["sha256"],
            "Original named acquisition roster failed both historical digest checks")
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Unknown encoding of original Chinese acquisition roster")
    rows = {}
    for entry in csv.DictReader(io.StringIO(text)):
        key = entry["subject_id"].strip().casefold()
        pattern = SCAN.fullmatch(key)
        if pattern is None:
            continue
        require(key not in rows and entry["DBS_status"].casefold() == pattern[1].casefold()
                and entry["condition"].casefold() == "stn" + pattern[1].casefold(),
                "Original STN roster duplicates or condition conflicts")
        rows[key] = dict(original_study_code="STN" + pattern[2], original_name=entry["name_Chs"],
                         original_name_key=normalize_person(entry["name_Chs"]),
                         original_scan_date=normalized_date(entry["scan"]), scan_id=entry["subject_id"],
                         original_condition=pattern[1].upper())
    return rows


def dicom_dates(path):
    book = load_workbook(path, read_only=True, data_only=True)
    result = {}
    try:
        sheet = book["48次访视"]
        for sheet_row, row in enumerate(sheet.iter_rows(min_row=5, values_only=True), start=5):
            if len(row) < 12 or row[1] not in ("ON", "OFF"):
                continue
            search = SCAN.search(str(row[11]))
            require(search is not None and search[1].upper() == row[1], "DICOM audit scan and condition disagree")
            key = search[0].casefold()
            require(key not in result, "Repeated scan in the DICOM acquisition audit")
            result[key] = dict(dicom_date=normalized_date(row[2]), acquisition_workbook_row=sheet_row,
                               dicom_audit_condition=row[1], dicom_locator=row[11])
    finally:
        book.close()
    return result


def outside_note(scans, transfer_path):
    transfer = {row["path"]: row["sha256"] for row in read_json(transfer_path)["files"]}
    root = transfer_path.parents[1] / "PD_DBS_zcc"
    found = 0
    zero_roi_scan = False
    for scan_id in scans:
        if not scan_id:
            continue
        relative = f"derivatives/fmripost/{scan_id}/{scan_id}_BOLD_AAL1ctx_MNI_basic36_sm6.csv"
        if relative not in transfer:
            continue
        series = root / relative
        require(digest(series) == transfer[relative], "An archived time series differs from its transfer receipt")
        values = pd.read_csv(series, index_col=0).to_numpy(float)
        zero_roi_scan |= bool(np.any(np.all(values == 0, axis=0)))
        found += 1
    if zero_roi_scan:
        return "outside frozen cohort; archived AAL series contains entirely zero ROI columns"
    if found < len(scans):
        return "outside frozen cohort; full pair absent from the verified AAL transfer"
    return "outside frozen cohort; archived series exist but no frozen pair is registered"


def score_visit_status(assessment, archive, dicom, in_frozen):
    if not in_frozen:
        return OUTSIDE
    if any(value and assessment and value != assessment for value in (archive, dicom)):
        return CONFLICT
    if assessment and assessment == dicom:
        return VERIFIED
    return INSUFFICIENT


def match_visits(primary, paths, imaging, private_dir=PRIVATE):
    master, read_scope = demographics(paths["score_demographics"])
    write_csv(master, private_dir / "results/Demographic_Target_Review.csv")
    master_by_code = {code: frame for code, frame in master.loc[master.study_code.ne("")].groupby("study_code")}
    acquisitions = captured_roster(paths)
    acquisition_clock = dicom_dates(paths["acquisition_workbook"])
    from_image = {}
    for pair in imaging.roster.reset_index().itertuples(index=False):
        for side in ("ON", "OFF"):
            scan_id = getattr(pair, f"{side.lower()}_scan_id")
            from_image[scan_id.casefold()] = {"participant_id": pair.participant_id, "image_condition": side}
    require(set(from_image).issubset(acquisitions), "A frozen PD scan is absent from the original named acquisition roster")
    require(set(acquisition_clock).issubset(acquisitions), "A DICOM audit scan is absent from the original roster")
    scored = primary.loc[primary.target.eq("STN")].copy()
    matched = []
    for original in scored.to_dict("records"):
        code, cond = original["study_code"], original["dbs_condition"]
        key = study_scan(code, cond)
        archive = acquisitions.get(key, {})
        clock = acquisition_clock.get(key, {})
        imaging_entry = from_image.get(key, {})
        master_match = master_by_code.get(code, pd.DataFrame())
        demographic = master_match.iloc[0].to_dict() if len(master_match) == 1 else {}
        verified_identity = bool(archive and demographic and
                                 archive["original_study_code"] == code and archive["original_condition"] == cond and
                                 archive["original_name_key"] == original["person_key"] and
                                 demographic["demographic_target"] == "STN")
        if imaging_entry:
            require(imaging_entry["image_condition"] == cond, "Frozen imaging DBS condition differs from scoring")
        source_date = normalized_date(original["assessment_date"])
        if verified_identity:
            status = score_visit_status(source_date, archive["original_scan_date"], clock.get("dicom_date"),
                                        bool(imaging_entry))
        else:
            status = UNRESOLVED
        evidence = dict(original, **archive, **clock, **imaging_entry)
        fields = ("demographic_source_row", "demographic_name", "demographic_target", "demographic_raw_target")
        evidence.update({key: demographic.get(key) for key in fields})
        evidence.update(
            identity_status="confirmed_code_name_condition" if verified_identity else UNRESOLVED,
            demographic_name_agreement=(demographic.get("demographic_name_key") == original["person_key"]),
            scale_version=CFG["score_version"], visit_status=status, assessment_date=source_date)
        matched.append(evidence)
    long = pd.DataFrame(matched)
    require(not long.empty, "No scored STN records found")
    rows = []
    for code, group in long.groupby("study_code", sort=True):
        rec = dict(study_code=code, temporary_person_id=f"score-record-{len(rows) + 1:02}",
                   n_scored_rows=len(group), complete_score_pair=False, identity_status=UNRESOLVED,
                   visit_status=UNRESOLVED, inclusion_reason="Source conditions duplicated or incomplete")
        if len(group) == 2 and set(group.dbs_condition) == {"ON", "OFF"}:
            sides = group.set_index("dbs_condition")
            off, on = sides.loc["OFF"], sides.loc["ON"]
            for side, row in [("off", off), ("on", on)]:
                for key in ("name", "source_row", "scan_id", "assessment_date", "original_scan_date", "dicom_date",
                            "visit_status", "medication_state", *DOMAIN_FIELDS):
                    rec[f"{side}_{key}"] = row.get(key)
            rec.update(complete_score_pair=bool(pd.notna(off.motor_total) and pd.notna(on.motor_total)),
                       demographic_name_agreement=bool(group.demographic_name_agreement.all()),
                       target="STN", scale_version=CFG["score_version"])
            if group.identity_status.ne(UNRESOLVED).all():
                rec["identity_status"] = "confirmed_code_name_condition"
                present = group.participant_id.dropna()
                if len(present) == 2:
                    require(present.nunique() == 1, "ON and OFF score rows map to different frozen participants")
                    rec["participant_id"] = present.iloc[0]
                    both = group.visit_status
                    if both.eq(CONFLICT).any():
                        rec.update(visit_status=CONFLICT, inclusion_reason="At least one assessment and acquisition date differs")
                    elif both.eq(VERIFIED).all():
                        rec.update(visit_status=VERIFIED, inclusion_reason="Both original score dates match independent DICOM dates")
                    else:
                        rec.update(visit_status=INSUFFICIENT, inclusion_reason="Dated roster corroborates the pair but DICOM date evidence is incomplete")
                elif len(present) == 0:
                    rec.update(visit_status=OUTSIDE, inclusion_reason=outside_note([off.scan_id, on.scan_id], paths["frozen_aal_transfer"]))
                else:
                    rec.update(visit_status=OUTSIDE, inclusion_reason="Only one scan appears in the frozen roster")
            for field in DOMAIN_FIELDS:
                before, after = rec.get(f"off_{field}"), rec.get(f"on_{field}")
                d = before - after if pd.notna(before) and pd.notna(after) else np.nan
                rec[f"{field}_improvement"] = d
                rec[f"{field}_percent_improvement"] = 100 * d / before if pd.notna(before) and before > 0 else np.nan
        rows.append(rec)
    paired = pd.DataFrame(rows)
    require(paired.participant_id.dropna().is_unique, "More than one original score pair maps to a frozen patient")
    connected = paired.dropna(subset=PAIR_KEYS)
    evidence = connected.merge(imaging.roster.reset_index(), on=PAIR_KEYS, validate="one_to_one")
    require(len(evidence) == len(connected), "Score pairs do not match both V6 scan keys")
    usable = evidence.loc[evidence.complete_score_pair & evidence.visit_status.isin([VERIFIED, INSUFFICIENT])]
    analysis = usable.merge(imaging.paired.reset_index(), on="participant_id", validate="one_to_one")
    long = long.merge(paired[["study_code", "temporary_person_id"]], on="study_code", validate="many_to_one")
    crosswalk = ["study_code", "participant_id", "scan_id", "name", "assessment_date", "source_file", "source_sheet",
                 "source_row", "dbs_condition", "original_name", "original_scan_date", "dicom_date", "dicom_locator",
                 "acquisition_workbook_row", "demographic_source_row", "demographic_name", "demographic_target",
                 "demographic_name_agreement", "identity_status", "visit_status"]
    write_csv(long, private_dir / "results/STN_scores_long.csv")
    write_csv(paired, private_dir / "results/STN_scores_paired.csv")
    write_csv(long.reindex(columns=crosswalk), private_dir / "results/STN_identity_crosswalk.csv")
    write_csv(analysis.sort_values("participant_id"), private_dir / "results/STN_Analysis_Pairs.csv")
    return long, paired, analysis.sort_values("participant_id"), master, read_scope
