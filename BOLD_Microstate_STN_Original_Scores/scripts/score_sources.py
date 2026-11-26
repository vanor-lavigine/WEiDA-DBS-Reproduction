"""Read dated motor-score workbooks and reconcile entries on explicit visit keys."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import unicodedata

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string


MODULE_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FILE = "PD开关机fMRI评分_20201218改善率..xlsx"
SECONDARY_FILE = "PD开关机fMRI评分_202011_gr_20201228.xlsx"
PRIMARY_SHEET = "开关机fMRI实际纳入患者.UPDRS-III(原始数据)"
SECONDARY_SHEET = "开关机fMRI实际纳入患者.UPDRS-III"


@dataclass(frozen=True)
class Item:
    name: str
    primary: str
    secondary: str
    header: tuple[str, ...]


# The secondary workbook moves the spontaneous movement item after posture.
ITEMS = (
    Item("speech", "G", "H", ("言语",)),
    Item("facial_expression", "H", "I", ("面部表情",)),
    Item("rigidity_neck", "I", "J", ("强直", "脖子")),
    Item("rigidity_arm_right", "J", "K", ("上肢:右",)),
    Item("rigidity_arm_left", "K", "L", ("上肢:左",)),
    Item("rigidity_leg_right", "L", "M", ("下肢:右",)),
    Item("rigidity_leg_left", "M", "N", ("下肢:左",)),
    Item("finger_tapping_right", "N", "O", ("手指拍打", "右")),
    Item("finger_tapping_left", "O", "P", ("左",)),
    Item("hand_movements_right", "P", "S", ("手掌运动", "右")),
    Item("hand_movements_left", "Q", "T", ("左",)),
    Item("pronation_supination_right", "R", "U", ("前臂回旋", "右")),
    Item("pronation_supination_left", "S", "V", ("左",)),
    Item("toe_tapping_right", "T", "W", ("脚趾拍地", "右")),
    Item("toe_tapping_left", "U", "X", ("左",)),
    Item("leg_agility_right", "V", "Y", ("脚灵敏度", "右")),
    Item("leg_agility_left", "W", "Z", ("左",)),
    Item("body_bradykinesia", "X", "AI", ("全身自发性的动作评估", "身体动作迟缓")),
    Item("arising", "AB", "AC", ("起立",)),
    Item("gait", "AC", "AD", ("步态",)),
    Item("gait_freezing", "AD", "AF", ("步态冻结", "步态冻结的评估")),
    Item("postural_stability", "AE", "AG", ("姿势平稳度",)),
    Item("posture", "AF", "AH", ("姿势",)),
    Item("postural_tremor_right", "AG", "AJ", ("姿态性震颤", "右")),
    Item("postural_tremor_left", "AH", "AK", ("左",)),
    Item("kinetic_tremor_right", "AI", "AL", ("动作性震颤", "右")),
    Item("kinetic_tremor_left", "AJ", "AM", ("左",)),
    Item("rest_tremor_arm_right", "AK", "AN", ("静止型震颤", "上肢:右")),
    Item("rest_tremor_arm_left", "AL", "AO", ("上肢:左",)),
    Item("rest_tremor_leg_right", "AM", "AP", ("下肢:右",)),
    Item("rest_tremor_leg_left", "AN", "AQ", ("下肢:左",)),
    Item("rest_tremor_lip_jaw", "AO", "AR", ("嘴唇/下巴",)),
    Item("rest_tremor_constancy", "AP", "AS", ("持续性",)),
)
ITEM_NAMES = tuple(item.name for item in ITEMS)
DOMAINS = {
    "rigidity": ITEM_NAMES[2:7],
    "bradykinesia": ITEM_NAMES[7:18],
    "postural_tremor": ITEM_NAMES[23:25],
    "kinetic_tremor": ITEM_NAMES[25:27],
    "rest_tremor": ITEM_NAMES[27:33],
    "tremor": ITEM_NAMES[23:33],
    "speech": ("speech",),
    "facial_expression": ("facial_expression",),
    "arising": ("arising",),
    "gait": ("gait",),
    "gait_freezing": ("gait_freezing",),
    "postural_stability": ("postural_stability",),
    "posture": ("posture",),
}
AI_HEADERS = {
    "Q": ("AI预测手指拍打", "右"),
    "R": ("左",),
    "AA": ("AI预测腿部灵活性", "右"),
    "AB": ("左",),
    "AE": ("AI预测步态",),
}
MATCH_FIELDS = ("person_key", "assessment_date", "dbs_condition", "medication_state")


@dataclass(frozen=True)
class Layout:
    kind: str
    sheet: str
    header_rows: int
    date: str
    dbs: str
    med: str
    target: str
    last_col: str


PRIMARY = Layout("primary", PRIMARY_SHEET, 3, "C", "D", "E", "F", "AP")
SECONDARY = Layout("secondary", SECONDARY_SHEET, 2, "D", "E", "F", "G", "AT")


def normalized_text(value) -> str:
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def normalize_person(value) -> str:
    return re.sub(r"\s+", "", normalized_text(value)).casefold()


def normalize_study_code(value) -> str:
    return re.sub(r"\s+", "", normalized_text(value)).upper()


def normalize_dbs(value) -> str:
    text = normalized_text(value).casefold()
    if text in {"dbs off", "dbs-off", "off", "关机", "dbs关机"}:
        return "OFF"
    if text in {"dbs on", "dbs-on", "on", "开机", "dbs开机"}:
        return "ON"
    return "unknown"


def normalize_medication(value) -> str:
    text = normalized_text(value).casefold()
    if text in {"med off", "med-off", "停药"}:
        return "med_off"
    if text in {"med on", "med-on", "服药"}:
        return "med_on"
    return "unknown"


def normalize_target(value) -> str:
    text = normalized_text(value).casefold()
    if not text:
        return "unknown"
    if ("stn" in text and "gpi" in text) or any(word in text for word in ("混合", "不对称", "mixed", "asymmetr")):
        return "mixed"
    if text in {"b-stn", "stn"}:
        return "STN"
    if text in {"b-gpi", "gpi"}:
        return "GPi"
    return "other"


def normalize_assessment_date(value) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    match = re.fullmatch(r"(\d{4})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", normalized_text(value))
    if match is not None:
        try:
            return date(*(int(part) for part in match.groups())).isoformat()
        except ValueError:
            return None
    return None


def numeric(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def clean_item(value) -> tuple[int | None, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, "missing"
    if not numeric(value):
        return None, "not_finite_numeric"
    if not 0 <= value <= 4:
        return None, "outside_0_to_4"
    if value != int(value):
        return None, "not_integer"
    return int(value), "ok"


def subtotal(scores, members):
    selected = [scores[item] for item in members]
    return sum(selected) if all(value is not None for value in selected) else None


def _value(row, column):
    return row[column_index_from_string(column) - 1]


def _headers(sheet, layout):
    rows = tuple(sheet.iter_rows(min_row=1, max_row=layout.header_rows,
                                 max_col=column_index_from_string(layout.last_col), values_only=True))
    audit_columns = ("Y",) if layout.kind == "primary" else (*AI_HEADERS, "AT")
    use = ("A", "B", layout.date, layout.dbs, layout.med, layout.target,
           *(getattr(item, layout.kind) for item in ITEMS), *audit_columns)
    columns = {name: tuple(text for row in rows if (text := normalized_text(_value(row, name)))) for name in use}
    metadata = {"B": ("姓名",), layout.date: ("评估时间",), layout.dbs: ("手术情况",),
                layout.med: ("服药情况",), layout.target: ("靶点",)}
    for column, expected in metadata.items():
        if columns[column] != expected:
            raise ValueError(f"Unexpected {layout.kind} metadata header at {column}")
    for item in ITEMS:
        column = getattr(item, layout.kind)
        expected = ("运动徐缓", *item.header) if layout.kind == "primary" and column == "N" else item.header
        if columns[column] != expected:
            raise ValueError(f"Unexpected {layout.kind} header for {item.name} at {column}")
    if layout.kind == "secondary":
        if any(columns[column] != expected for column, expected in AI_HEADERS.items()):
            raise ValueError("Unexpected secondary AI audit headers")
        if columns["AT"] != ("UPDRS总分",):
            raise ValueError("Unexpected secondary AT header")
    return columns


def _column_range(start, stop):
    return set(range(column_index_from_string(start), column_index_from_string(stop) + 1))


def _validate_item_map():
    observed = {kind: {column_index_from_string(getattr(item, kind)) for item in ITEMS}
                for kind in ("primary", "secondary")}
    expected = {
        "primary": _column_range("G", "X") | _column_range("AB", "AP"),
        "secondary": _column_range("H", "AS") - {column_index_from_string(key) for key in AI_HEADERS},
    }
    if len(ITEM_NAMES) != 33 or len(set(ITEM_NAMES)) != 33 or observed != expected:
        raise ValueError("The item map must cover exactly 33 specified clinical columns in each workbook")
    partition = Counter(item for key, group in DOMAINS.items() if key != "tremor" for item in group)
    if partition != Counter(ITEM_NAMES):
        raise ValueError("The original subtotals must partition all 33 clinical items before tremor aggregation")


def _formula(value):
    return value if isinstance(value, str) and value.startswith("=") else None


def _primary_formula_audit(values, formulas, number, scores):
    y = _value(values, "Y")
    formula = _formula(_value(formulas, "Y"))
    expected_formula = "=" + "+".join(f"{column}{number}" for column in "JKLMNOPQRSTUVWX")
    expected_total = subtotal(scores, ITEM_NAMES[3:18])
    pure_brady = subtotal(scores, DOMAINS["bradykinesia"])
    return {
        "y_cached": y,
        "y_formula": formula,
        "y_formula_adds_j_to_x": bool(formula and formula.replace(" ", "").upper() == expected_formula),
        "y_recomputed_j_to_x": expected_total,
        "y_matches_j_to_x": bool(numeric(y) and expected_total is not None and y == expected_total),
        "y_minus_bradykinesia": y - pure_brady if numeric(y) and pure_brady is not None else None,
        "y_limb_rigidity_contribution": subtotal(scores, ITEM_NAMES[3:7]),
    }


def _secondary_formula_audit(values, formulas, number, motor_total):
    at = _value(values, "AT")
    literal = _value(formulas, "AT")
    formula = _formula(literal)
    ai_fields = {column: _value(values, column) for column in AI_HEADERS}
    # Excel SUM over a range ignores blank and text cells. This is an audit calculation only.
    ai_sum = sum(value for value in ai_fields.values() if numeric(value))
    offset = at - motor_total if numeric(at) and motor_total is not None else None
    includes_ai = bool(formula and re.fullmatch(
        rf"=SUM\(\$?H\$?{number}:\$?AS\$?{number}\)", formula.replace(" ", ""), re.I))
    return {
        "at_cached": at,
        "at_formula": formula,
        "at_kind": "formula" if formula else ("stored_value" if literal is not None else "blank"),
        "at_formula_includes_ai": includes_ai,
        "at_minus_motor_total": offset,
        "at_minus_motor_equals_ai": None if offset is None else math.isclose(offset, ai_sum, abs_tol=1e-10),
        "ai_sum": ai_sum,
        "ai_text_cells": sum(isinstance(value, str) and bool(value.strip()) for value in ai_fields.values()),
        "ai_values_json": json.dumps(ai_fields, ensure_ascii=False, default=str),
    }


def _record(values, formulas, layout, number, file_name):
    raw_items = {item.name: _value(values, getattr(item, layout.kind)) for item in ITEMS}
    scores, issues = {}, {}
    for item in ITEMS:
        column = getattr(item, layout.kind)
        scores[item.name], issue = clean_item(raw_items[item.name])
        if _formula(_value(formulas, column)):
            scores[item.name], issue = None, "clinical_formula_cell"
        if issue != "ok":
            issues[f"{column}{number}"] = issue
    totals = {"motor_total": subtotal(scores, ITEM_NAMES)}
    totals.update({name: subtotal(scores, group) for name, group in DOMAINS.items()})
    date_value = _value(values, layout.date)
    name, study = _value(values, "B"), _value(values, "A")
    target, dbs, med = (_value(values, label) for label in (layout.target, layout.dbs, layout.med))
    audit = (_primary_formula_audit(values, formulas, number, scores) if layout.kind == "primary"
             else _secondary_formula_audit(values, formulas, number, totals["motor_total"]))
    return {
        "person_key": normalize_person(name),
        "name": name,
        "study_code": normalize_study_code(study),
        "study_code_raw": study,
        "assessment_date": normalize_assessment_date(date_value),
        "assessment_date_raw": date_value,
        "dbs_condition": normalize_dbs(dbs),
        "dbs_condition_raw": dbs,
        "medication_state": normalize_medication(med),
        "medication_state_raw": med,
        "target": normalize_target(target),
        "target_raw": target,
        "scale_version": "unverified",
        "source_file": file_name,
        "source_sheet": layout.sheet,
        "source_row": number,
        "n_items_valid": sum(value is not None for value in scores.values()),
        "item_issue_cells_json": json.dumps(issues, ensure_ascii=False),
        **{f"item_{key}": value for key, value in raw_items.items()},
        **totals,
        **audit,
    }


def _complete_stn_pairs(stn):
    return sum(len(group) == 2 and set(group.dbs_condition) == {"ON", "OFF"} and group.motor_total.notna().all()
               for _, group in stn.groupby("person_key"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path, layout):
    _validate_item_map()
    path = Path(path).resolve(strict=True)
    sha_before = _sha256(path)
    with (closing(load_workbook(path, read_only=True, data_only=False)) as original,
          closing(load_workbook(path, read_only=True, data_only=True)) as cached):
        sheet_names = original.sheetnames
        headers = _headers(original[layout.sheet], layout)
        start = layout.header_rows + 1
        last_col = column_index_from_string(layout.last_col)
        stored = cached[layout.sheet].iter_rows(min_row=start, max_col=last_col, values_only=True)
        formulas = original[layout.sheet].iter_rows(min_row=start, max_col=last_col, values_only=True)
        records = [_record(values, literal, layout, number, path.name)
                   for number, (values, literal) in enumerate(zip(stored, formulas, strict=True), start)
                   if normalize_person(_value(values, "B"))]
    if _sha256(path) != sha_before:
        raise ValueError(f"The {layout.kind} workbook changed while it was being read")
    if not records:
        raise ValueError(f"No named score rows found in the {layout.kind} workbook")
    frame = pd.DataFrame.from_records(records)
    stn = frame.loc[frame.target.eq("STN")]
    receipt = {
        "path": str(path),
        "sha256": sha_before,
        "sheet_names": sheet_names,
        "source_sheet": layout.sheet,
        "header_rows": [1, layout.header_rows],
        "actual_score_read_range": f"A{start}:{layout.last_col}{records[-1]['source_row']}",
        "observed_headers_by_column": headers,
        "clinical_item_columns": {item.name: getattr(item, layout.kind) for item in ITEMS},
        "domains_by_clinical_item": {domain: list(group) for domain, group in DOMAINS.items()},
        "rows": len(frame),
        "people": int(frame.person_key.nunique()),
        "rows_by_target": dict(Counter(frame.target)),
        "stn_people": int(stn.person_key.nunique()),
        "stn_visits": len(stn),
        "stn_dbs_condition": dict(Counter(stn.dbs_condition)),
        "stn_medication_state": dict(Counter(stn.medication_state)),
        "stn_dated_rows": int(stn.assessment_date.notna().sum()),
        "stn_complete_on_off_pairs": int(_complete_stn_pairs(stn)),
        "stn_rows_with_all_33_items": int(stn.n_items_valid.eq(33).sum()),
    }
    if layout.kind == "primary":
        offsets = stn.y_minus_bradykinesia.dropna()
        nonzero = offsets[offsets.ne(0)]
        receipt.update(stn_y_adds_j_to_x=int(stn.y_formula_adds_j_to_x.sum()),
                       stn_y_cache_matches_j_to_x=int(stn.y_matches_j_to_x.sum()),
                       stn_y_differs_from_pure_bradykinesia=len(nonzero),
                       stn_y_excess_min=int(nonzero.min()) if len(nonzero) else None,
                       stn_y_excess_max=int(nonzero.max()) if len(nonzero) else None)
    else:
        blank = frame.target_raw.isna()
        existing = set(frame.loc[~blank, "person_key"])
        receipt["target_blank_candidates"] = int(blank.sum())
        receipt["people_already_named_with_target_blank_candidates"] = len(set(frame.loc[blank, "person_key"]) & existing)
    return frame, receipt


def read_primary(path) -> tuple[pd.DataFrame, dict]:
    return _read(path, PRIMARY)


def read_secondary(path) -> tuple[pd.DataFrame, dict]:
    return _read(path, SECONDARY)


def _match_key(record):
    person, when, stimulus, medicine = (record[field] for field in MATCH_FIELDS)
    if person and isinstance(when, str) and when and stimulus in {"ON", "OFF"} and medicine in {"med_on", "med_off"}:
        return (person, when, stimulus, medicine)
    return None


def _index(records):
    index = defaultdict(list)
    for record in records:
        key = _match_key(record)
        if key is not None:
            index[key].append(record)
    return index


def _same_value(x, y):
    if pd.isna(x) or pd.isna(y):
        return bool(pd.isna(x) and pd.isna(y))
    return bool(x == y)


def _audit_join(primary, secondary, status):
    chosen = primary if primary is not None else secondary
    primary = primary or {}
    secondary = secondary or {}
    both = bool(primary and secondary)
    mismatches = [item for item in ITEM_NAMES if not _same_value(primary.get(f"item_{item}"),
                                                               secondary.get(f"item_{item}"))] if both else None
    domain_mismatches = [field for field in ("motor_total", *DOMAINS) if not _same_value(
        primary.get(field), secondary.get(field))] if both else None
    return {
        **{field: chosen[field] for field in (*MATCH_FIELDS, "name", "study_code")},
        "join_status": status,
        "primary_source_row": primary.get("source_row"),
        "primary_source_file": primary.get("source_file"),
        "primary_source_sheet": primary.get("source_sheet"),
        "secondary_source_row": secondary.get("source_row"),
        "secondary_source_file": secondary.get("source_file"),
        "secondary_source_sheet": secondary.get("source_sheet"),
        "primary_target": primary.get("target"),
        "secondary_target": secondary.get("target"),
        "target_match": None if not both else primary["target"] == secondary["target"],
        "item_mismatch_count": None if mismatches is None else len(mismatches),
        "item_mismatch_names": None if mismatches is None else "|".join(mismatches),
        "domain_mismatch_names": None if domain_mismatches is None else "|".join(domain_mismatches),
        "primary_motor_total": primary.get("motor_total"),
        "secondary_motor_total": secondary.get("motor_total"),
        "primary_y_cached": primary.get("y_cached"),
        "primary_y_formula": primary.get("y_formula"),
        "primary_y_minus_bradykinesia": primary.get("y_minus_bradykinesia"),
        "secondary_at_cached": secondary.get("at_cached"),
        "secondary_at_formula": secondary.get("at_formula"),
        "secondary_at_kind": secondary.get("at_kind"),
        "secondary_at_formula_includes_ai": secondary.get("at_formula_includes_ai"),
        "secondary_at_minus_motor_total": secondary.get("at_minus_motor_total"),
        "secondary_ai_sum": secondary.get("ai_sum"),
        "secondary_at_minus_motor_equals_ai": secondary.get("at_minus_motor_equals_ai"),
    }


def reconcile(primary, secondary) -> tuple[pd.DataFrame, dict]:
    first, second = primary.to_dict("records"), secondary.to_dict("records")
    first_index, second_index = _index(first), _index(second)
    seen, reports = set(), []
    for p in first:
        key = _match_key(p)
        if key is None:
            reports.append(_audit_join(p, None, "primary_missing_visit_key"))
        elif len(first_index[key]) > 1 or len(second_index.get(key, ())) > 1:
            reports.append(_audit_join(p, None, "ambiguous_visit_key"))
        elif key not in second_index:
            reports.append(_audit_join(p, None, "primary_only"))
        else:
            s = second_index[key][0]
            reports.append(_audit_join(p, s, "matched"))
            seen.add(s["source_row"])
    for s in second:
        if s["source_row"] not in seen:
            key = _match_key(s)
            if key is None:
                status = "secondary_missing_visit_key"
            elif len(second_index[key]) > 1 or len(first_index.get(key, ())) > 1:
                status = "ambiguous_visit_key"
            else:
                status = "secondary_only"
            reports.append(_audit_join(None, s, status))

    joined = pd.DataFrame.from_records(reports)
    stn = joined.loc[joined.primary_target.eq("STN") & joined.join_status.eq("matched")]
    offsets = stn.secondary_at_minus_motor_total
    biased = stn.loc[offsets.notna() & offsets.ne(0)]
    summary = {
        "records_by_join_status": dict(Counter(joined.join_status)),
        "primary_stn_records": int(primary.target.eq("STN").sum()),
        "stn_records_matched_exact_key": len(stn),
        "stn_records_exact_33_clinical_items": int(stn.item_mismatch_count.eq(0).sum()),
        "stn_records_exact_all_derived_domains": int(stn.domain_mismatch_names.eq("").sum()),
        "stn_records_equal_target": int(stn.target_match.eq(True).sum()),
        "stn_at_cached_total_available": int(stn.secondary_at_cached.notna().sum()),
        "stn_at_formula_includes_ai_rows": int(stn.secondary_at_formula_includes_ai.eq(True).sum()),
        "stn_at_stored_values": int(stn.secondary_at_kind.eq("stored_value").sum()),
        "stn_at_total_mismatch": len(biased),
        "stn_at_affected_people": int(biased.person_key.nunique()),
        "stn_at_mismatch_by_dbs": dict(Counter(biased.dbs_condition)),
        "stn_at_mismatch_min": float(biased.secondary_at_minus_motor_total.min()) if len(biased) else None,
        "stn_at_mismatch_max": float(biased.secondary_at_minus_motor_total.max()) if len(biased) else None,
        "stn_at_mismatch_explained_by_own_ai_sum": int(biased.secondary_at_minus_motor_equals_ai.eq(True).sum()),
        "stn_at_mismatch_with_formula_including_ai": int(biased.secondary_at_formula_includes_ai.eq(True).sum()),
        "unmatched_secondary_blank_target_records": int((joined.primary_target.isna() & joined.secondary_target.eq("unknown")).sum()),
    }
    return joined, summary


def write_source_audit(primary, secondary, joined, receipts, summary, private_dir=MODULE_ROOT / "private"):
    private_dir = Path(private_dir)
    results, manifests = private_dir / "results", private_dir / "manifests"
    results.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    primary.loc[primary.target.eq("STN")].to_csv(results / "Primary_Source_Scores.csv", encoding="utf-8-sig", index=False)
    kept_targets = {"STN", "unknown", "mixed", "other"}
    secondary.loc[secondary.target.isin(kept_targets)].to_csv(results / "Secondary_Source_Scores.csv",
                                                            encoding="utf-8-sig", index=False)
    joined.loc[joined.primary_target.eq("STN") | joined.secondary_target.isin(kept_targets)].to_csv(
        results / "Worksheet_Score_Discrepancies.csv", encoding="utf-8-sig", index=False)
    (manifests / "Score_Workbook_Read_Receipt.json").write_text(
        json.dumps({"workbooks": receipts, "reconciliation": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    primary, p_receipt = read_primary(args.source_root / PRIMARY_FILE)
    secondary, s_receipt = read_secondary(args.source_root / SECONDARY_FILE)
    joined, summary = reconcile(primary, secondary)
    write_source_audit(primary, secondary, joined, {"primary": p_receipt, "secondary": s_receipt}, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
