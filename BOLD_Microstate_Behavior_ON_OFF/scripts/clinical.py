"""Reconstruct patient-paired scores from the specified 33 human-rated items."""

from collections import Counter
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from openpyxl.utils.cell import column_index_from_string, get_column_letter

MODULE = Path(__file__).resolve().parents[1]
V8_MODULE = "BOLD_Microstate_STN_Original_Scores"
LOCK = "private/manifests/Prespecified_Inference_Lock.json"
V8_PAIR = "private/date_unrestricted/results/STN_Analysis_Pairs_25.csv"
V8_PAIR_LOCK = "private/date_unrestricted/manifests/Inference_Lock.json"
sys.path.insert(0, str(MODULE.parent / V8_MODULE / "scripts"))
import identity_audit as identity
import pipeline_io as v8
import score_sources as score

BEHAVIOR_METRICS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
RAW_NAMES = {
    "score_secondary": "PD开关机fMRI评分_202011_gr_20201228.xlsx",
    "score_primary": "PD开关机fMRI评分_20201218改善率..xlsx",
    "score_demographics": "PD-开关机人口学及刺激参数_2023.07.31.xlsx",
}
SPECS = {
    "total": (("H", "P"), ("S", "Z"), ("AC", "AD"), ("AF", "AS")),
    "rigidity": (("J", "N"),),
    "bradykinesia": (("O", "P"), ("S", "Z"), ("AI", "AI")),
    "tremor": (("AJ", "AS"),),
    "gait_balance": (("AD", "AD"), ("AF", "AG")),
}
GAIT_ITEMS = ("gait", "gait_freezing", "postural_stability")
VERSION_LABEL = "原表运动检查总分，33项重算，正式版本待核实"
DIRECTION_REFERENCE = "https://www.movementdisorders.org/MDS-Files1/Resources/PDFs/MDS-UPDRS.pdf"


def _expanded(ranges):
    return [get_column_letter(pos) for first, last in ranges
            for pos in range(column_index_from_string(first), column_index_from_string(last) + 1)]


def _validated_path(path, workspace):
    file = Path(path).resolve(strict=True)
    if not file.is_relative_to(workspace) or not file.is_file():
        raise ValueError("Clinical input is outside the authorized workspace or is missing")
    return file


def _workbook(workspace, name, expected_hash):
    base = Path(name).stem.rstrip(".")
    extension = r"\.xlsx"
    suffix = r"(?:\s*[\(（]\d+[\)）])?"
    filename = re.compile(re.escape(base) + r"\.*" + suffix + r"\.*" + extension, re.I)
    candidates = [p for p in workspace.iterdir() if p.is_file() and filename.fullmatch(p.name)]
    matched = [_validated_path(p, workspace) for p in candidates if v8.digest(p) == expected_hash]
    if not matched:
        raise ValueError(f"No matching, V8-audited workbook for {name}")
    return sorted(matched, key=lambda p: (p.name != name, p.name))[0]


def _trusted_inputs(workspace):
    archive = workspace / "WEiDA-DBS-V8" / V8_MODULE
    historic_lock = _validated_path(archive / LOCK, workspace)
    original = v8.read_json(historic_lock)
    hashes = {historic_lock.as_posix(): v8.digest(historic_lock)}
    paths = {}
    for role, evidence in original["input_files"].items():
        expected = evidence["sha256"]
        path = (_workbook(workspace, RAW_NAMES[role], expected) if role in RAW_NAMES
                else _validated_path(evidence["path"], workspace))
        if v8.digest(path) != expected:
            raise ValueError(f"The V8 archived input receipt no longer matches ({role})")
        paths[role] = path
        hashes[path.as_posix()] = expected
    prior_lock = _validated_path(archive / V8_PAIR_LOCK, workspace)
    prior_pairs = _validated_path(archive / V8_PAIR, workspace)
    hashes[prior_lock.as_posix()] = v8.digest(prior_lock)
    prior_checksum = v8.read_json(prior_lock)["patient_input_sha256"]
    if v8.digest(prior_pairs) != prior_checksum:
        raise ValueError("The archived date-unrestricted V8 data no longer match their receipt")
    hashes[prior_pairs.as_posix()] = prior_checksum
    for module in (score, identity, v8):
        path = Path(module.__file__).resolve()
        hashes[path.as_posix()] = v8.digest(path)
    return paths, hashes, prior_pairs


def _dictionary(receipt):
    by_column = {item.secondary: item for item in score.ITEMS}
    definitions = {domain: _expanded(ranges) for domain, ranges in SPECS.items()}
    expected = {domain: [by_column[column].name for column in cols] for domain, cols in definitions.items()}
    if (set(definitions["total"]) != set(by_column) or len(definitions["total"]) != 33
            or any(set(expected[domain]) != set(score.DOMAINS[domain]) for domain in ("rigidity", "bradykinesia", "tremor"))
            or tuple(expected["gait_balance"]) != GAIT_ITEMS or by_column["AI"].name != "body_bradykinesia"
            or set(score.AI_HEADERS).intersection(definitions["total"])):
        raise ValueError("The frozen clinical workbook parser disagrees with the declared scoring columns")
    header = receipt["observed_headers_by_column"]
    items = []
    for col in definitions["total"]:
        item = by_column[col]
        items.append(dict(column=col, item=item.name, observed_header=list(header[col]),
                          comparison_workbook_column=item.primary,
                          domains=[domain for domain, cols in definitions.items() if col in cols]))
    return {
        "file": RAW_NAMES["score_secondary"], "sheet": receipt["source_sheet"],
        "scale_label": VERSION_LABEL, "original_sheet_rating_label": "UPDRS-III",
        "rating_direction": "higher_is_worse", "allowed_item_values": list(range(5)),
        "rating_direction_reference": DIRECTION_REFERENCE,
        "domains": {name: {"columns": cols, "item_count": len(cols), "items": expected[name]}
                    for name, cols in definitions.items()},
        "items": items, "excluded_prediction_columns": {
            col: list(header[col]) for col in score.AI_HEADERS
        },
        "excluded_cached_totals": {"scoring_sheet": ["AT"], "comparison_raw_sheet": ["Y"]},
        "gait_balance_scope": "预定义探索性的步态、步态冻结及姿势平稳度三项和",
    }


def _crosscheck(first, second):
    # V8 uses the opposite workbook numbering. Its join verifies both raw sources by the same four keys.
    joined, summary = score.reconcile(second, first)
    domains = ("person_key", "assessment_date", "dbs_condition", "medication_state")
    keep = lambda frame: frame.loc[frame.target.eq("STN"), [*domains, "study_code"]]
    checked = keep(first).merge(keep(second), on=list(domains), how="outer", validate="one_to_one",
                                indicator=True, suffixes=("_first", "_comparison"))
    if (len(checked) != 54 or not checked["_merge"].eq("both").all()
            or not checked.study_code_first.eq(checked.study_code_comparison).all()
            or not checked.study_code_first.str.fullmatch(r"STN\d+").all()
            or summary["stn_records_exact_33_clinical_items"] != len(checked)):
        raise ValueError("The scored STN identity or item cross-check disagrees across the original workbooks")
    return joined, summary


def _rebuild_pairs(long, scored, dictionary):
    verified = long.loc[long.identity_status.eq("confirmed_code_name_condition")].copy()
    if (len(verified) != 54 or not verified.medication_state.eq("med_off").all()
            or verified.assessment_date.isna().any() or not verified.n_items_valid.eq(33).all()):
        raise ValueError("The source STN medication, date or complete-item cohort differs from the audit")
    if (verified.groupby("study_code").dbs_condition.nunique().ne(2).any()
            or verified.duplicated(["study_code", "dbs_condition"]).any()):
        raise ValueError("Each eligible STN study code must have one verified ON and one verified OFF scoring visit")
    for metric, definition in dictionary["domains"].items():
        cols = [f"item_{item}" for item in definition["items"]]
        verified[metric] = verified[cols].sum(axis=1, min_count=len(cols))
    paired = (scored.set_index("study_code", verify_integrity=True)[
        ["participant_id", "on_scan_id", "off_scan_id", "visit_status", "inclusion_reason",
         "on_visit_status", "off_visit_status", "on_assessment_date", "off_assessment_date",
         "on_medication_state", "off_medication_state"]].rename(columns={"participant_id": "patient_id"})
              .copy())
    for metric in BEHAVIOR_METRICS:
        sides = verified.pivot(index="study_code", columns="dbs_condition", values=metric)
        if not np.isfinite(sides[["ON", "OFF"]].to_numpy()).all() or set(paired.index) != set(sides.index):
            raise ValueError("A reconstructed clinical score or its condition pairing is unavailable")
        paired[f"{metric}_on"] = sides.ON
        paired[f"{metric}_off"] = sides.OFF
        paired[f"{metric}_improvement"] = sides.OFF - sides.ON
        paired[f"{metric}_improvement_rate"] = (sides.OFF - sides.ON).div(sides.OFF.where(sides.OFF.ne(0)))
    paired["date_verified"] = paired.visit_status.eq(identity.VERIFIED)
    return paired.reset_index()


def _check_prior(pairs, file):
    old = pd.read_csv(file, dtype={k: str for k in v8.PAIR_KEYS})
    mapped = pairs.rename(columns={"patient_id": "participant_id"})
    check = mapped.merge(old, on=v8.PAIR_KEYS, how="outer", validate="one_to_one", indicator=True,
                         suffixes=("_current", "_v8"))
    if not check["_merge"].eq("both").all() or not check.visit_status_current.eq(check.visit_status_v8).all():
        raise ValueError("The reconstructed paired identities or date labels differ from the archived V8 analysis")
    for metric in BEHAVIOR_METRICS[:4]:
        history = "motor_total" if metric == "total" else metric
        for name, reference in ((f"{metric}_on", f"on_{history}"), (f"{metric}_off", f"off_{history}"),
                                (f"{metric}_improvement", f"{history}_improvement")):
            if name == reference:
                name, reference = f"{name}_current", f"{reference}_v8"
            if not np.array_equal(check[name].to_numpy(float), check[reference].to_numpy(float)):
                raise ValueError(f"A recomputed {metric} score differs from the archived V8 score")
    for side in ("on", "off"):
        reference = sum(check[f"{side}_{item}"] for item in GAIT_ITEMS)
        if not np.array_equal(check[f"gait_balance_{side}"].to_numpy(float), reference.to_numpy(float)):
            raise ValueError("The recomputed gait/balance score differs from the archived component scores")
    return len(check)


def load_clinical(workspace, private):
    """Return matched pairs, all score pairs, a public audit, dictionary and source fingerprints."""
    workspace, private = Path(workspace).resolve(), Path(private).resolve()
    if not private.is_relative_to(MODULE / "private"):
        raise ValueError("The private clinical output must be inside this experiment")
    paths, hashes, prior_pairs = _trusted_inputs(workspace)
    first, receipt_first = score.read_secondary(paths["score_secondary"])
    second, receipt_second = score.read_primary(paths["score_primary"])
    dictionary = _dictionary(receipt_first)
    check, reconciliation = _crosscheck(first, second)
    frozen = v8.load_imaging(paths)
    long, old_pairs, _, demographics, demo_scope = identity.match_visits(first, paths, frozen, private / "clinical_identity")
    pairs = _rebuild_pairs(long, old_pairs, dictionary)
    eligible = pairs.loc[pairs.patient_id.notna()].sort_values("patient_id").reset_index(drop=True)
    if len(pairs) != 27 or len(eligible) != len(frozen.roster) or eligible.date_verified.sum() != 22:
        raise ValueError("The reconstructed clinical, imaging and independently dated scopes differ from V8")
    archived_score_pairs = _check_prior(eligible, prior_pairs)

    v8.write_csv(pairs, private / "Clinical_Score_Pairs_27.csv")
    v8.write_csv(eligible, private / "Clinical_Imaging_Pairs_25.csv")
    v8.write_csv(check.loc[check.primary_target.eq("STN")], private / "Clinical_Workbook_Reconciliation.csv")
    audit = {
        "score_version": VERSION_LABEL,
        "score_direction": "higher_is_worse",
        "score_direction_reference": DIRECTION_REFERENCE,
        "improvement": "OFF minus ON; positive values indicate lower severity at ON",
        "improvement_rate": "(OFF minus ON) divided by OFF; no value when OFF equals zero",
        "scoring_workbook": {
            "file": paths["score_secondary"].name, "sheet": receipt_first["source_sheet"],
            "read_range": receipt_first["actual_score_read_range"], "stn_visits": receipt_first["stn_visits"],
            "stn_all_33_items_valid": receipt_first["stn_rows_with_all_33_items"],
            "stn_medication_by_visit": receipt_first["stn_medication_state"],
        },
        "comparison_workbook": {
            "file": paths["score_primary"].name, "sheet": receipt_second["source_sheet"],
            "read_range": receipt_second["actual_score_read_range"],
            "stn_old_y_mixes_limb_rigidity_count": receipt_second["stn_y_differs_from_pure_bradykinesia"],
            "stn_old_y_formula_includes_limb_rigidity_count": receipt_second["stn_y_adds_j_to_x"],
        },
        "demographics": {
            **demo_scope, "participants_by_target": dict(Counter(demographics.demographic_target)),
            "exact_names_and_study_codes": int(old_pairs.demographic_name_agreement.sum()),
            "adjudicated_name_variants": int((~old_pairs.demographic_name_agreement).sum()),
        },
        "reconciliation": reconciliation,
        "cohorts": {
            "confirmed_stn_clinical_pairs": len(pairs), "matched_imaging_pairs": len(eligible),
            "date_verified_pairs": int(eligible.date_verified.sum()),
            "matched_pair_visit_statuses": dict(Counter(eligible.visit_status)),
            "unmatched_scoring_pair_reasons": dict(Counter(pairs.loc[pairs.patient_id.isna(), "inclusion_reason"])),
            "archived_v8_pairs_and_scores_exact": archived_score_pairs,
            "all_clinical_pairs_zero_off_for_rate": {m: int(pairs[f"{m}_off"].eq(0).sum()) for m in BEHAVIOR_METRICS},
            "matched_pairs_zero_off_for_rate": {m: int(eligible[f"{m}_off"].eq(0).sum()) for m in BEHAVIOR_METRICS},
        },
    }
    return eligible, pairs, audit, dictionary, hashes
