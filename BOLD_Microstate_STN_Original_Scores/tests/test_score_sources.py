from datetime import date
from pathlib import Path
import sys

from openpyxl.utils import column_index_from_string
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from score_sources import ITEMS, PRIMARY, SECONDARY, _record, clean_item, normalize_dbs, normalize_medication, reconcile


def visit(layout, row_number, *, condition="DBS off", med="med off", when=date(2020, 1, 2), missing=None):
    raw = {
        "speech": 0, "rigidity_neck": 2, "rigidity_arm_right": 3, "finger_tapping_right": 1,
        "body_bradykinesia": 2, "gait": 4, "postural_tremor_right": 2, "kinetic_tremor_left": 1,
        "rest_tremor_constancy": 1,
    }
    values = [None] * column_index_from_string(layout.last_col)

    def put(column, value):
        values[column_index_from_string(column) - 1] = value

    for item in ITEMS:
        put(getattr(item, layout.kind), None if item.name == missing else raw.get(item.name, 0))
    for col, value in {"A": "TEST1", "B": "Example Name", layout.date: when, layout.dbs: condition,
                       layout.med: med, layout.target: "B-STN"}.items():
        put(col, value)
    if layout is PRIMARY:
        put("Y", 6)
    else:
        for col, value in {"Q": 2, "R": "2", "AA": 4, "AE": 3, "AT": 25}.items():
            put(col, value)
    formulas = values.copy()
    if layout is PRIMARY:
        formulas[column_index_from_string("Y") - 1] = "=" + "+".join(
            f"{col}{row_number}" for col in "JKLMNOPQRSTUVWX")
    else:
        formulas[column_index_from_string("AT") - 1] = f"=SUM(H{row_number}:AS{row_number})"
    return _record(values, formulas, layout, row_number, "source.xlsx")


def test_semantic_item_mapping_excludes_ai_and_four_limb_rigidity_from_bradykinesia():
    primary = visit(PRIMARY, 14)
    secondary = visit(SECONDARY, 29)
    for record in (primary, secondary):
        assert record["n_items_valid"] == 33
        assert record["motor_total"] == 16
        assert record["rigidity"] == 5
        assert record["bradykinesia"] == 3
        assert record["tremor"] == 4
        assert record["gait"] == 4
    assert primary["y_formula_adds_j_to_x"]
    assert primary["y_matches_j_to_x"]
    assert primary["y_minus_bradykinesia"] == 3
    assert secondary["at_formula_includes_ai"]
    assert secondary["ai_sum"] == 9
    assert secondary["ai_text_cells"] == 1
    assert secondary["at_minus_motor_total"] == 9
    assert secondary["at_minus_motor_equals_ai"]


def test_missing_and_invalid_clinical_values_do_not_become_zeros():
    record = visit(PRIMARY, 4, missing="rest_tremor_constancy")
    assert record["n_items_valid"] == 32
    assert record["motor_total"] is None and record["tremor"] is None
    assert record["rigidity"] == 5
    assert clean_item(0) == (0, "ok")
    for invalid in (None, "", "0", True, 2.5, -1, 5, float("nan")):
        assert clean_item(invalid)[0] is None


def test_visit_reconciliation_preserves_condition_and_medication_as_separate_keys():
    off = visit(PRIMARY, 12)
    on = visit(PRIMARY, 13, condition="DBS on")
    kept = visit(SECONDARY, 27, condition="DBS on")
    extra = visit(SECONDARY, 22, med="med on")
    joined, summary = reconcile(pd.DataFrame([off, on]), pd.DataFrame([extra, kept]))
    match = joined.loc[joined.join_status.eq("matched")].iloc[0]
    assert match.primary_source_row == 13
    assert match.secondary_source_row == 27
    assert match.item_mismatch_count == 0
    assert summary["records_by_join_status"] == {"primary_only": 1, "matched": 1, "secondary_only": 1}
    assert normalize_dbs("med off") == "unknown"
    assert normalize_medication("DBS off") == "unknown"


def test_duplicate_and_missing_visit_keys_cannot_create_matches():
    original = visit(PRIMARY, 10)
    no_date = visit(PRIMARY, 11, when=None)
    secondary = [visit(SECONDARY, 12), visit(SECONDARY, 13), visit(SECONDARY, 14, when=None)]
    rows, stats = reconcile(pd.DataFrame([original, no_date]), pd.DataFrame(secondary))
    assert stats["stn_records_matched_exact_key"] == 0
    assert rows.join_status.value_counts().to_dict() == {
        "ambiguous_visit_key": 3, "primary_missing_visit_key": 1, "secondary_missing_visit_key": 1
    }
