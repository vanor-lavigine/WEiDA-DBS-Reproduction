"""Audit the original score dates and recheck frozen clinical-to-scan identity keys."""

from collections import Counter
from contextlib import closing
from datetime import datetime, time
from pathlib import Path
import re
import sys

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles.numbers import is_datetime
from openpyxl.utils.cell import column_index_from_string

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import clinical


RAW_SOURCE_ROLES = ("score_secondary", "score_primary", "score_demographics")
IDENTITY_ROLES = ("original_acquisition_capture", "original_acquisition_capture_check")
MEDICATION = {"med_off": "OFF", "med_on": "ON"}
ASSESSMENT_HEADER = "评估时间"
VISIT_ANNOTATION = re.compile(
    r"访视|同期|同一天|当天|同步|同次|扫描|影像|MRI|BOLD|"
    r"评估前|评估后|评分前|评分后|\d{1,2}[:：]\d{2}|\d+[分钟小时]", re.I
)
SCAN_DATE_NOTE = re.compile(r"时间.{0,7}是.{0,16}扫描.{0,12}(?:时间|日期)")
UNCERTAIN_NOTE = re.compile(r"不是|是不是|[?？]|是否|可能|应该|疑似|估计|大概|或许|不清楚")


def _trusted_sources(workspace, behavior):
    workspace = Path(workspace).resolve(strict=True)
    v8_lock = clinical._validated_path(
        workspace / "WEiDA-DBS-V8" / clinical.V8_MODULE / clinical.LOCK, workspace
    )
    inputs = clinical.v8.read_json(v8_lock)["input_files"]
    hash_log = {str(v8_lock): clinical.v8.digest(v8_lock)}
    paths = {}
    comparator_archive_match = True
    for role in (*RAW_SOURCE_ROLES, *IDENTITY_ROLES):
        evidence = inputs[role]
        if role in RAW_SOURCE_ROLES:
            try:
                path = clinical._workbook(workspace, clinical.RAW_NAMES[role], evidence["sha256"])
            except ValueError:
                if role != "score_primary":
                    raise
                # Only the exact historically named comparator is admissible when its bytes drift.
                path = clinical._validated_path(evidence["path"], workspace)
        else:
            path = clinical._validated_path(evidence["path"], workspace)
        actual = clinical.v8.digest(path)
        if actual != evidence["sha256"]:
            if role != "score_primary":
                raise ValueError(f"The frozen clinical audit receipt changed ({role})")
            comparator_archive_match = False
        paths[role] = path
        hash_log[str(path)] = actual

    private = behavior / "private"
    round2 = clinical._validated_path(private / "round2/Round2_Input_Hashes.json", workspace)
    archived = clinical._validated_path(
        private / "clinical/clinical_identity/results/STN_scores_long.csv", workspace
    )
    frozen = clinical._validated_path(private / "results/Main_Pairs.csv", workspace)
    round2_digests = {
        Path(path).resolve(): digest for path, digest in
        clinical.v8.read_json(round2)["file_sha256"].items()
    }
    hash_log[str(round2)] = clinical.v8.digest(round2)
    for path in (archived, frozen):
        actual = clinical.v8.digest(path)
        if actual != round2_digests.get(path):
            raise ValueError(f"The archived clinical audit receipt changed ({path.name})")
        hash_log[str(path)] = actual
    for source in (clinical, clinical.score, clinical.identity, clinical.v8):
        path = Path(source.__file__).resolve()
        hash_log[str(path)] = clinical.v8.digest(path)
    return paths, archived, frozen, hash_log, comparator_archive_match


def _assessment_clock(cell):
    value = cell.value
    if cell.data_type == "f":
        return None, None, "unavailable"
    if isinstance(value, datetime):
        # Excel presents date-only serials as midnight datetimes. A declared clock
        # format or a stored nonzero time fraction supplies precision beyond the date.
        date_value = value.date().isoformat()
        if value.time() != time.min or is_datetime(cell.number_format) in {"datetime", "time"}:
            return date_value, value.isoformat(), "datetime"
        return date_value, None, "date"
    date_value = clinical.score.normalize_assessment_date(value)
    if date_value is not None:
        return date_value, None, "date"
    if isinstance(value, str) and re.fullmatch(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?",
        value.strip(),
    ):
        try:
            clock = datetime.fromisoformat(value.strip())
        except ValueError:
            return None, None, "unavailable"
        return clock.date().isoformat(), clock.isoformat(), "datetime"
    return None, None, "unavailable"


def _raw_dates(path, layout, source_rows):
    dates = {}
    notes = Counter()
    with closing(load_workbook(path, read_only=True, data_only=False)) as workbook:
        sheet = workbook[layout.sheet]
        date_col = column_index_from_string(layout.date) - 1
        for row_number, row in enumerate(sheet.iter_rows(), start=1):
            if row_number not in source_rows:
                continue
            cell = row[date_col]
            day, clock, precision = _assessment_clock(cell)
            # Use the calendar-day value shown by Excel for a date-only cell.
            # Its native Python midnight timestamp is an import artifact.
            raw_value = day if precision == "date" and isinstance(cell.value, datetime) else str(cell.value)
            dates[row_number] = {
                "date": day, "datetime": clock, "precision": precision,
                "cell": cell.coordinate, "raw_value": raw_value, "number_format": cell.number_format,
            }
            if layout == clinical.score.SECONDARY:
                for column, role in (("C", "remarks"), ("AU", "assessor")):
                    value = row[column_index_from_string(column) - 1].value
                    if value is not None:
                        notes[role] += 1
                        if VISIT_ANNOTATION.search(str(value)):
                            notes["possible_timing_annotations"] += 1
    if set(dates) != source_rows:
        raise ValueError("The frozen score rows do not exist in the original assessment sheet")
    return dates, notes


def _comparison_inventory(path):
    count_by_precision = Counter()
    date_columns = 0
    candidate_note_cells = 0
    with closing(load_workbook(path, read_only=True, data_only=False)) as workbook:
        for sheet in workbook.worksheets:
            headers = next(sheet.iter_rows(min_row=2, max_row=2, values_only=True))
            matches = [idx for idx, header in enumerate(headers) if header == ASSESSMENT_HEADER]
            date_columns += len(matches)
            for row in sheet.iter_rows(min_row=4):
                for col in matches:
                    if row[col].value is not None:
                        count_by_precision[_assessment_clock(row[col])[2]] += 1
                candidate_note_cells += sum(
                    isinstance(cell.value, str) and cell.data_type != "f"
                    and bool(VISIT_ANNOTATION.search(cell.value)) for cell in row
                )
        return {
            "worksheets_reviewed": len(workbook.worksheets),
            "assessment_columns_reviewed": date_columns,
            "recorded_assessment_cells_by_precision_including_repeated_tables": dict(count_by_precision),
            "candidate_timing_annotation_cells": candidate_note_cells,
        }


def _demographic_inventory(path, frozen_names):
    schedule = Counter()
    with closing(load_workbook(path, read_only=True, data_only=True)) as workbook:
        for sheet in workbook.worksheets:
            header = next(sheet.iter_rows(max_row=1, values_only=True))
            if header[:4] != ("扫描日期", "类别", "姓名", "扫描时间"):
                continue
            schedule["scan_sheets"] += 1
            for row in sheet.iter_rows(min_row=2, values_only=True):
                if not clinical.score.normalize_person(row[2]):
                    continue
                schedule["scan_schedule_rows"] += 1
                schedule["scan_schedule_clocks"] += int(row[3] is not None)
                schedule["frozen_stn_names"] += int(clinical.score.normalize_person(row[2]) in frozen_names)
    return {
        "scan_sheets": schedule["scan_sheets"],
        "scan_schedule_rows": schedule["scan_schedule_rows"],
        "scan_schedule_clocks": schedule["scan_schedule_clocks"],
        "frozen_stn_names": schedule["frozen_stn_names"],
    }


def _workbook_comments(paths):
    comments = {}
    matching_proxy_notes = {}
    for role in RAW_SOURCE_ROLES:
        notes = {}
        # Read-only mode suppresses cell comments in openpyxl. The ordinary
        # workbook reader exposes them and the workbook is closed without saving.
        with closing(load_workbook(paths[role], read_only=False, data_only=False)) as workbook:
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        if cell.comment is not None:
                            notes[(sheet.title, cell.coordinate)] = cell.comment.text
        comments[role] = notes
        matching_proxy_notes[role] = {
            ref: note for ref, note in notes.items()
            if SCAN_DATE_NOTE.search(clinical.score.normalized_text(note))
            and not UNCERTAIN_NOTE.search(note)
        }
    return comments, matching_proxy_notes


def _long_keys(pairs):
    return [
        {
            "patient_id": row["patient_id"], "study_code": row.get("study_code"),
            "scan_id": row[f"{condition.lower()}_scan_id"], "condition": condition,
        }
        for row in pairs.to_dict("records") for condition in ("ON", "OFF")
    ]


def _key_set(rows, include_code=False):
    fields = ("patient_id", "condition", "scan_id", "study_code") if include_code else (
        "patient_id", "condition", "scan_id"
    )
    return {tuple(str(row[field]).casefold() for field in fields) for row in rows}


def _indexed(frame, columns):
    return frame.set_index(list(columns), verify_integrity=True).to_dict("index")


def _archive_agrees(current, archived):
    return (
        current["person_key"] == archived["person_key"]
        and current["assessment_date"] == archived["assessment_date"]
        and current["medication_state"] == archived["medication_state"]
        and int(current["source_row"]) == int(archived["source_row"])
        and current["n_items_valid"] == archived["n_items_valid"] == 33
        and current["motor_total"] == archived["motor_total"]
        and all(current[f"item_{name}"] == archived[f"item_{name}"] for name in clinical.score.ITEM_NAMES)
    )


def _source_cell(path, row, cell):
    return f"{path} | {row['source_sheet']}!{cell}"


def load_score_audit(workspace, behavior, pairs):
    """Return one private row per frozen condition, a public-safe audit and source hashes."""
    behavior = Path(behavior).resolve(strict=True)
    paths, archive, frozen, hashes, comparator_match = _trusted_sources(workspace, behavior)
    original, orig_receipt = clinical.score.read_secondary(paths["score_secondary"])
    comparator, comp_receipt = clinical.score.read_primary(paths["score_primary"])
    clinical._dictionary(orig_receipt)
    _, corroboration = clinical._crosscheck(original, comparator)
    recorded = pd.read_csv(archive)
    frozen_rows = _long_keys(pd.read_csv(frozen, dtype=str))
    supplied_rows = _long_keys(pairs)
    include_code = "study_code" in pairs
    if len(frozen_rows) != len(supplied_rows) or _key_set(frozen_rows, include_code) != _key_set(
        supplied_rows, include_code
    ):
        raise ValueError("The independently reconstructed scan keys differ from the frozen patient table")

    paired_cols = ("study_code", "dbs_condition")
    sheets = [_indexed(table.loc[table.target.eq("STN")], paired_cols) for table in (original, comparator)]
    original_by_key, comparator_by_key = sheets
    archive_by_key = _indexed(recorded, paired_cols)
    pair_keys = {(row["study_code"], row["condition"]) for row in frozen_rows}
    orig_dates, orig_notes = _raw_dates(paths["score_secondary"], clinical.score.SECONDARY, {
        original_by_key[key]["source_row"] for key in pair_keys
    })
    comp_dates, _ = _raw_dates(paths["score_primary"], clinical.score.PRIMARY, {
        comparator_by_key[key]["source_row"] for key in pair_keys
    })
    if orig_notes.get("possible_timing_annotations", 0):
        raise ValueError("Candidate timing text in the authoritative score sheet needs direct source adjudication")

    demographics, demographic_receipt = clinical.identity.demographics(paths["score_demographics"])
    demographics = _indexed(demographics.loc[demographics.study_code.isin(
        {row["study_code"] for row in frozen_rows}
    )], ("study_code",))
    roster = clinical.identity.captured_roster(paths)
    comments, proxy_notes = _workbook_comments(paths)
    result = []
    for base in frozen_rows:
        code, condition = base["study_code"], base["condition"]
        key = code, condition
        raw, comparison, prior = (lookup[key] for lookup in (*sheets, archive_by_key))
        ori_date, cmp_date = orig_dates[raw["source_row"]], comp_dates[comparison["source_row"]]
        roster_row = roster.get(str(base["scan_id"]).casefold(), {})
        demo = demographics.get(code, {})
        identity_confirmed = (
            bool(raw["person_key"])
            and raw["person_key"] == comparison["person_key"] == roster_row.get("original_name_key")
            and roster_row.get("original_study_code") == code
            and str(prior["scan_id"]).casefold() == str(base["scan_id"]).casefold()
            and str(prior["participant_id"]).casefold() == str(base["patient_id"]).casefold()
            and demo.get("demographic_target") == "STN"
        )
        condition_confirmed = (
            roster_row.get("original_condition") == condition == prior["image_condition"]
            and clinical.score.normalize_dbs(raw["dbs_condition_raw"]) == condition
            and clinical.score.normalize_dbs(comparison["dbs_condition_raw"]) == condition
        )
        dates_agree = ori_date["date"] == cmp_date["date"]
        clocks_agree = (
            not (ori_date["datetime"] and cmp_date["datetime"])
            or ori_date["datetime"] == cmp_date["datetime"]
        )
        date_ok = dates_agree and clocks_agree
        row_clock = ori_date if ori_date["precision"] == "datetime" else cmp_date
        if row_clock["precision"] != "datetime":
            row_clock = ori_date
        orig_comment_key = raw["source_sheet"], ori_date["cell"]
        comp_comment_key = comparison["source_sheet"], cmp_date["cell"]
        orig_comment = comments["score_secondary"].get(orig_comment_key)
        comp_comment = comments["score_primary"].get(comp_comment_key)
        scan_date_proxy = (
            orig_comment_key in proxy_notes["score_secondary"]
            or comp_comment_key in proxy_notes["score_primary"]
        )
        score_date_available = date_ok and not scan_date_proxy and row_clock["date"] is not None
        orig_ref = _source_cell(paths["score_secondary"], raw, ori_date["cell"])
        comp_ref = _source_cell(paths["score_primary"], comparison, cmp_date["cell"])
        layout = clinical.score.SECONDARY
        result.append({
            **base,
            "score_date": row_clock["date"] if score_date_available else None,
            "score_datetime": row_clock["datetime"] if score_date_available else None,
            "score_precision": row_clock["precision"] if score_date_available else "unavailable",
            "score_date_unresolved": not date_ok,
            "score_date_is_annotated_scan_proxy": scan_date_proxy,
            "score_date_unavailable_reason": (
                "assessment_cell_comment_identifies_scan_date" if scan_date_proxy else None
            ),
            "score_medication_state": MEDICATION.get(raw["medication_state"], "unknown"),
            "score_medication_state_raw": raw["medication_state_raw"],
            "score_dbs_state_raw": raw["dbs_condition_raw"],
            "assessment_source": f"{paths['score_secondary'].name} | {raw['source_sheet']}",
            "score_evidence_source": f"{orig_ref} | crosscheck: {comp_ref}",
            "score_medication_source": _source_cell(
                paths["score_secondary"], raw, f"{layout.med}{raw['source_row']}"
            ),
            "score_condition_source": _source_cell(
                paths["score_secondary"], raw, f"{layout.dbs}{raw['source_row']}"
            ),
            "score_recomputed_archive_source": str(archive),
            "original_score_date_raw": ori_date["raw_value"],
            "original_score_date_number_format": ori_date["number_format"],
            "original_score_date_cell_comment": orig_comment,
            "comp_score_date_raw": cmp_date["raw_value"],
            "comp_score_date_number_format": cmp_date["number_format"],
            "comp_score_date_cell_comment": comp_comment,
            "score_visit_id": None,
            "score_scan_shared_clock": False,
            "same_visit_documented": False,
            "different_visit_documented": False,
            "near_time_documented": False,
            "score_relative_to_scan_documented": None,
            "identity_confirmed": identity_confirmed,
            "condition_confirmed": condition_confirmed,
            "score_source_confirmed": _archive_agrees(raw, prior),
            "score_original_acquisition_roster_evidence": str(paths["original_acquisition_capture"]),
            "demographic_name_agreement": demo.get("demographic_name_key") == raw["person_key"],
            "historical_score_visit_status": prior["visit_status"],
        })
    frame = pd.DataFrame(result)
    summary = {
        "frozen_score_rows": len(frame),
        "raw_stn_rows_per_workbook": {
            "authoritative": int(original.target.eq("STN").sum()),
            "comparison": int(comparator.target.eq("STN").sum()),
        },
        "raw_stn_rows_dual_source_exact": int(corroboration["stn_records_exact_33_clinical_items"]),
        "frozen_score_rows_archived_33_items_and_total_exact": int(frame.score_source_confirmed.sum()),
        "frozen_score_rows_identity_confirmed": int(frame.identity_confirmed.sum()),
        "frozen_score_rows_condition_confirmed": int(frame.condition_confirmed.sum()),
        "frozen_score_rows_by_precision": dict(Counter(frame.score_precision)),
        "score_dates_reclassified_as_acquisition": int(frame.score_date_is_annotated_scan_proxy.sum()),
        "frozen_score_rows_by_medication": dict(Counter(frame.score_medication_state)),
        "direct_score_scan_visit_links": int(frame.same_visit_documented.sum()),
        "demographic_name_discrepancy_patients": int(frame.loc[~frame.demographic_name_agreement, "patient_id"].nunique()),
        "historical_date_labels_by_visit": dict(Counter(frame.historical_score_visit_status)),
        "comparison_archive_sha_match": comparator_match,
        "original_score_source_inventory": {
            "authoritative": {
                "file": paths["score_secondary"].name, "sheet": orig_receipt["source_sheet"],
                "assessment_column": clinical.score.SECONDARY.date,
                "assessment_header": ASSESSMENT_HEADER,
            },
            "comparison": {
                "file": paths["score_primary"].name, "sheet": comp_receipt["source_sheet"],
                "assessment_column": clinical.score.PRIMARY.date,
                "assessment_header": ASSESSMENT_HEADER,
                **_comparison_inventory(paths["score_primary"]),
            },
            "demographics": {"file": paths["score_demographics"].name, "sheet": demographic_receipt["sheet"]},
        },
        "demographic_schedule": _demographic_inventory(paths["score_demographics"], {
            original_by_key[key]["person_key"] for key in pair_keys
        }),
        "annotation_scope": {
            "remarks": orig_notes["remarks"], "assessor": orig_notes["assessor"],
            "possible_timing_annotations": orig_notes["possible_timing_annotations"],
            "workbook_cell_comments_by_source": {role: len(notes) for role, notes in comments.items()},
            "scan_date_proxy_comments_by_source_including_repeated_tables": {
                role: len(notes) for role, notes in proxy_notes.items()
            },
        },
    }
    return frame, summary, hashes
