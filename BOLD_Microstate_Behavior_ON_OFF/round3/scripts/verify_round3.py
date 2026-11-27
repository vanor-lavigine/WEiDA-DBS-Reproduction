"""Verify the third-round release independently against the frozen pairs."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import rankdata, t

from round3_io import CFG, PRIVATE, REPO, RESULTS, ROOT, digest, git, read_json, verify_prior

STATES = ("M1", "M2", "M3", "M4")
ENDPOINTS = ("total", "rigidity", "bradykinesia", "tremor", "gait_balance")
KEYS = ["patient_id", "on_scan_id", "off_scan_id"]
CELLS = [(state, endpoint) for state in STATES for endpoint in ENDPOINTS]
CELL_KEYS = ["state", "clinical_endpoint"]
LONG_KEYS = ["patient_id", *CELL_KEYS]
FIELDS = ["delta_coverage", "improvement_rate", "absolute_improvement"]
RATE_TABLE = "Coverage_Clinical_Rate_Associations.csv"
POINT_TABLE = "Coverage_Clinical_Absolute_Associations.csv"
INPUT_LOCK = "Round3_Input_Hashes.json"
MANIFEST = "Round3_Run_Manifest.json"
PRIVATE_WIDE = "Coverage_Clinical_Pairs.csv"
PRIVATE_LONG = "Association_Eligible_Pairs.csv"
REPORT = "ON_OFF_Coverage_Clinical_Rate_Report.md"
REDACTED_COLUMNS = {
    "patient_id", "participant_id", "subject", "scan", "scan_id", "on_scan_id",
    "off_scan_id", "study_code", "assessment_date", "birth_date", "name",
}
TEXT_SUFFIXES = {".md", ".csv", ".py", ".json", ".txt"}
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(path, dtype=None):
    return pd.read_csv(path, dtype=dtype, float_precision="round_trip")


def same(actual, expected):
    require(np.allclose(actual, expected, atol=TOL, rtol=0, equal_nan=True),
            "A saved statistic differs from the independently reconstructed value")


def bh(p):
    """Benjamini–Hochberg step-up correction using all registered slots."""
    p = np.asarray(p, dtype=float)
    keep = np.isfinite(p)
    full = np.where(keep, p, 1.)
    order = np.argsort(full, kind="stable")
    scaled = full[order] * len(full) / np.arange(1, len(full)+1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1].clip(0, 1)
    result = np.empty(len(full))
    result[order] = monotone
    return np.where(keep, result, np.nan)


def rank_test(x, y, minimum_n):
    """Spearman from averaged ranks and the two-sided t approximation."""
    n = len(x)
    if n < minimum_n or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan
    rx, ry = rankdata(x), rankdata(y)
    center_x, center_y = rx - rx.mean(), ry - ry.mean()
    rho = float(np.clip(np.dot(center_x, center_y)
                        / np.sqrt(np.dot(center_x, center_x) * np.dot(center_y, center_y)), -1, 1))
    if abs(rho) == 1:
        return rho, 0.
    test = rho * np.sqrt((n-2)/((1-rho)*(1+rho)))
    return rho, float(2*t.sf(abs(test), n-2))


def source_path(lock, suffix):
    ending = Path(suffix).parts
    candidates = [Path(path) for path in lock if Path(path).parts[-len(ending):] == ending]
    require(len(candidates) == 1, f"A unique locked {ending[-1]} source is required")
    return candidates[0]


def load_source_pairs(lock, expected_n):
    saved = table(source_path(lock, "private/results/Main_Pairs.csv"),
                  dtype={name: str for name in KEYS})
    visits = table(source_path(lock, "private/results/Frozen_Cohort_Features.csv"),
                   dtype={"scan": str, "subject": str})
    roster = table(source_path(lock, "private/manifests/ON_OFF_Pair_Manifest.csv"),
                   dtype={"participant_id": str, "on_scan_id": str, "off_scan_id": str})
    roster = roster.rename(columns={"participant_id": "patient_id"})
    for frame in (saved, roster):
        require(len(frame) == expected_n, "Frozen participant pair count differs from the protocol")
        require(not frame[KEYS].isna().any().any() and all(frame[k].is_unique for k in KEYS),
                "Frozen ON/OFF pair keys are missing or duplicated")
        require(set(frame.on_scan_id).isdisjoint(frame.off_scan_id),
                "A frozen scan occupies both visit roles")
    paired = saved.merge(roster[[*KEYS, "on_valid_frames", "off_valid_frames"]],
                         on=KEYS, how="inner", validate="one_to_one")
    require(len(paired) == expected_n, "Clinical and frozen V6 pair keys disagree")
    require(visits.scan.is_unique and not visits.scan.isna().any(), "V6 has invalid scan keys")
    visits = visits.set_index("scan", verify_integrity=True)
    coverage = [f"coverage_{i}" for i in range(1, 5)]
    for visit in ("on", "off"):
        key = f"{visit}_scan_id"
        scan_keys = paired[key].tolist()
        require(set(scan_keys) == set(visits.index[visits.condition.eq(visit.upper())]),
                "Matched scan set differs from the frozen ON/OFF group")
        matched = visits.loc[scan_keys]
        require(np.array_equal(matched.subject, paired.patient_id)
                and np.array_equal(matched.fold, paired.fold)
                and np.array_equal(matched.valid_frames, paired[f"{visit}_valid_frames"]),
                "Frozen patient, fold or common-window scan metadata disagree")
        values = matched[coverage].to_numpy(float)
        require(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all(),
                "Coverage must be a fraction in [0,1]")
        same(values.sum(axis=1), np.ones(expected_n))
        paired[[f"{feature}_{visit}" for feature in coverage]] = values
    delta = paired[[f"{name}_on" for name in coverage]].to_numpy() - paired[
        [f"{name}_off" for name in coverage]].to_numpy()
    same(delta.sum(axis=1), np.zeros(expected_n))
    paired[[f"delta_coverage_{i}" for i in range(1, 5)]] = delta

    for endpoint in ENDPOINTS:
        on, off = (paired[f"{endpoint}_{visit}"].to_numpy(float) for visit in ("on", "off"))
        require(not np.any(off < 0) and not np.any(on < 0), "Clinical source contains negative scores")
        improvement = off-on
        rate = np.divide(improvement, off, out=np.full(len(off), np.nan), where=off > 0)
        same(improvement, paired[f"{endpoint}_improvement"])
        same(rate, paired[f"{endpoint}_improvement_rate"])
        require(np.array_equal(np.sign(rate[off > 0]), np.sign(improvement[off > 0]), equal_nan=True),
                "Clinical score and rate signs disagree")
    return paired


def verify_saved_wide(source):
    path = PRIVATE / "results" / PRIVATE_WIDE
    wide = table(path, dtype={name: str for name in KEYS})
    source = source.set_index(KEYS, verify_integrity=True).sort_index()
    wide = wide.set_index(KEYS, verify_integrity=True).sort_index()
    require(wide.index.equals(source.index), "Saved joined table has incorrect patient/scan combinations")
    columns = ["fold"]
    columns += [f"coverage_{i}_{side}" for side in ("off", "on") for i in range(1, 5)]
    columns += [f"delta_coverage_{i}" for i in range(1, 5)]
    columns += [f"{endpoint}_{suffix}" for endpoint in ENDPOINTS
                for suffix in ("off", "on", "improvement", "improvement_rate")]
    same(wide[columns], source[columns])


def reconstruct_cells(source, minimum_n):
    rows, patients = [], []
    for state, endpoint in CELLS:
        x = source[f"delta_coverage_{state[1:]}"].to_numpy(float)
        off = source[f"{endpoint}_off"].to_numpy(float)
        rate = source[f"{endpoint}_improvement_rate"].to_numpy(float)
        absolute = source[f"{endpoint}_improvement"].to_numpy(float)
        eligible = np.isfinite(x) & np.isfinite(rate) & np.isfinite(absolute) & (off > 0)
        missing = int((~eligible & ~(off == 0)).sum())
        n_zero = int((off == 0).sum())
        measure = {}
        for analysis, y in (("rate", rate), ("absolute", absolute)):
            rho, p = rank_test(x[eligible], y[eligible], minimum_n)
            measure[f"{analysis}_rho"], measure[f"{analysis}_raw_p"] = rho, p
        rows.append(dict(state=state, clinical_endpoint=endpoint, n=int(eligible.sum()), n_off_zero=n_zero,
                         n_missing=missing, n_excluded=n_zero+missing, **measure))
        selected = source.loc[eligible, ["patient_id"]].copy()
        selected["state"], selected["clinical_endpoint"] = state, endpoint
        selected[FIELDS] = np.column_stack((x[eligible], rate[eligible], absolute[eligible]))
        patients.append(selected)
    expected = pd.DataFrame(rows).set_index(CELL_KEYS, verify_integrity=True)
    for analysis in ("rate", "absolute"):
        expected[f"{analysis}_fdr_q"] = bh(expected[f"{analysis}_raw_p"])
    return expected, pd.concat(patients, ignore_index=True)


def verify_same_cases(expected):
    path = PRIVATE / "results" / PRIVATE_LONG
    actual = table(path, dtype={"patient_id": str}).set_index(LONG_KEYS, verify_integrity=True).sort_index()
    expected = expected.set_index(LONG_KEYS, verify_integrity=True).sort_index()
    require(actual.index.equals(expected.index),
            "Rate and matched absolute decrement do not share the exact expected pair membership")
    same(actual[FIELDS], expected[FIELDS])


def verify_correlation_tables(expected, alpha):
    main = None
    for file, analysis in ((RATE_TABLE, "rate"), (POINT_TABLE, "absolute")):
        rows = table(RESULTS / file)
        require(list(zip(rows.state, rows.clinical_endpoint, strict=True)) == CELLS,
                "A correlation table differs from the ordered 20-cell registry")
        actual = rows.set_index(CELL_KEYS, verify_integrity=True)
        require(actual.bh_family_size.eq(len(CELLS)).all(), "The BH family must keep all 20 slots")
        for col in ("n", "n_off_zero", "n_missing", "n_excluded"):
            same(actual[col], expected[col])
        for col in ("rho", "raw_p", "fdr_q"):
            same(actual[col], expected[f"{analysis}_{col}"])
        significant = expected[f"{analysis}_fdr_q"].lt(alpha)
        require(actual.passes_fdr.tolist() == significant.tolist(), "FDR pass flags disagree with q<0.05")
        unavailable = ~np.isfinite(actual.raw_p)
        directions = np.select([actual.rho.gt(0), actual.rho.lt(0), actual.rho.eq(0)],
                               ["positive", "negative", "zero"], default="NA")
        require(np.array_equal(actual.direction.fillna("NA"), directions),
                "The published direction differs from the sign of the Spearman coefficient")
        labels = np.select([unavailable, significant, actual.raw_p.lt(alpha)],
                           ["unestimable", "fdr_supported", "nominal_only"], default="no_evidence")
        require(np.array_equal(actual.evidence, labels), "A public evidence label disagrees with raw P and BH q")
        require(actual.loc[unavailable, "reason"].fillna("").ne("").all(),
                "An unestimable test is missing its stated reason")
        require(actual.clinical_definition.tolist() == [CFG["clinical_definitions"][endpoint]
                                                       for _, endpoint in CELLS],
                "A clinical endpoint definition differs from the locked protocol")
        if analysis == "rate":
            main = rows
    return main

def verify_logged_artifacts(source, lock):
    manifest = read_json(RESULTS / MANIFEST)
    require(manifest["reference_commit"] == CFG["reference_commit"], "Release base SHA differs from the protocol")
    require(tuple(CFG["states"]) == STATES and tuple(CFG["clinical_endpoints"]) == ENDPOINTS,
            "The locked state or clinical axis differs from the registered 4×5 analysis")
    require(CFG["statistics"]["bh_family_size"] == len(CELLS), "The protocol must retain all twenty BH slots")
    require(manifest["protocol_sha256"] == digest(ROOT / "configs/protocol.json", text=True),
            "The saved protocol digest differs")
    require(manifest["prior_sources_sha256"] == verify_prior(), "The preceding rounds' locked digests differ")

    registry = table(ROOT / "configs/test_registry.csv")
    require(list(zip(registry.state, registry.clinical_endpoint, strict=True)) == CELLS
            and registry.main.eq("rate").all() and registry.secondary.eq("absolute_same_cases").all(),
            "The saved correlation registry differs from the two fixed 20-cell families")
    for name, expected in manifest["scripts_sha256"].items():
        path = (ROOT / "scripts" / name).resolve()
        require(path.is_relative_to(ROOT / "scripts") and digest(path, text=True) == expected,
                "A released script differs from its recorded digest")

    outputs = manifest["public_outputs_sha256"]
    required = {
        f"results/{RATE_TABLE}", f"results/{POINT_TABLE}", "results/Coverage_Change_Summary.csv",
        "results/Clinical_Rate_Eligibility.csv", f"reports/{REPORT}", "figures/Coverage_Clinical_Rate_Heatmap.png",
    }
    require(required == set(outputs), "The release manifest differs from the expected aggregate outputs")
    private_points = PRIVATE / "figures/Coverage_Clinical_Rate_Scatter_Individual.png"
    require(private_points.is_file(), "The local 20-panel scatter figure is missing")
    with Image.open(private_points) as img:
        require(img.format == "PNG" and img.width > 1000 and img.height > 500,
                "The private scatter output has an invalid figure size or format")
        img.verify()
    release_paths = []
    for name, expected in outputs.items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), "A release path is missing or outside round three")
        require(digest(path, text=path.suffix in TEXT_SUFFIXES) == expected,
                "An aggregate output differs from its recorded digest")
        release_paths.append(path)
    for name, expected in lock.items():
        path = Path(name)
        require(path.is_absolute() and digest(path) == expected, "A locked round-three source was modified")

    relative_private = PRIVATE.relative_to(REPO).as_posix()
    for path in (private_points, PRIVATE / INPUT_LOCK, PRIVATE / "results" / PRIVATE_WIDE,
                 PRIVATE / "results" / PRIVATE_LONG):
        git("check-ignore", "--", path.relative_to(REPO).as_posix())
    require(not git("ls-files", "--", relative_private), "A round-three private file is tracked in Git")

    identifiers = set(source[KEYS].astype(str).to_numpy().ravel())
    if "study_code" in source:
        identifiers.update(source.study_code.dropna().astype(str))
    risk_keys = [key.lower() for key in identifiers if len(key) >= 5 and any(c.isalpha() for c in key)]
    for path in [*release_paths, RESULTS / MANIFEST]:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        raw = path.read_text(encoding="utf-8-sig")
        lowered = raw.lower()
        require(not any(key in lowered for key in risk_keys), "A public artifact contains a private identifier")
        require(not re.search(r"(?<![\w:/])[A-Za-z]:(?:\\|/(?!/))|/(?:mnt|home|data_main|Users)/", raw),
                "A public artifact contains a private absolute path")
        if path.suffix == ".csv":
            header = table(path).columns
            require(not REDACTED_COLUMNS.intersection(header), "A public CSV contains a patient-level field")


def _prob(value):
    return f"{value:.4g}" if np.isfinite(value) else "NA"


def _rho(value):
    return f"{value:+.3f}" if np.isfinite(value) else "NA"


def verify_heatmap(main):
    path = ROOT / "figures/Coverage_Clinical_Rate_Heatmap.png"
    with Image.open(path) as image:
        require(image.format == "PNG", "The public heatmap must be PNG")
        metadata = image.info.get("Round3_Annotations")
        require(isinstance(metadata, str), "The public heatmap is missing its display annotation audit")
        records = json.loads(metadata)
        image.verify()
    require(records == [
        {"state": row.state, "clinical_endpoint": row.clinical_endpoint, "n": int(row.n),
         "rho_display": _rho(row.rho), "q_display": _prob(row.fdr_q), "passes_fdr": bool(row.passes_fdr)}
        for row in main.itertuples(index=False)
    ], "Public heatmap labels differ from the released association table")

def verify_descriptive_tables(source):
    summary = table(RESULTS / "Coverage_Change_Summary.csv").set_index("state", verify_integrity=True)
    require(tuple(summary.index) == STATES, "Coverage summary must retain all four states")
    threshold = CFG["imaging"]["direction_tolerance"]
    for state, row in summary.iterrows():
        idx = state[1:]
        require(row.n_pairs == len(source), "Coverage denominator differs from the frozen pair count")
        off, on, change = (source[col].to_numpy(float)
                           for col in (f"coverage_{idx}_off", f"coverage_{idx}_on", f"delta_coverage_{idx}"))
        for values, label in ((off, "off"), (on, "on"), (change, "delta")):
            stats = [values.mean(), np.median(values), *np.percentile(values, [25, 75])]
            same(row[[f"{label}_mean", f"{label}_median", f"{label}_q25", f"{label}_q75"]].to_numpy(float), stats)
        directions = ((change > threshold).sum(), (np.abs(change) <= threshold).sum(), (change < -threshold).sum())
        same(row[["n_increased", "n_unchanged", "n_decreased"]].to_numpy(float), directions)

    eligibility = table(RESULTS / "Clinical_Rate_Eligibility.csv").set_index("clinical_endpoint", verify_integrity=True)
    require(tuple(eligibility.index) == ENDPOINTS, "Clinical eligibility must include all five endpoints")
    for endpoint, row in eligibility.iterrows():
        on, off = (source[f"{endpoint}_{visit}"].to_numpy(float) for visit in ("on", "off"))
        absolute = np.isfinite(on) & np.isfinite(off)
        zero = off == 0
        rate = absolute & (off > 0)
        require(row.clinical_definition == CFG["clinical_definitions"][endpoint],
                "Clinical eligibility misstates an endpoint definition")
        missing = ~rate & ~zero
        expected = [len(source), rate.sum(), zero.sum(), missing.sum(),
                    (missing & ~np.isfinite(off)).sum(),
                    (missing & np.isfinite(off) & ~np.isfinite(on)).sum(), absolute.sum()]
        same(row[["n_pairs", "n_valid_rate", "n_off_zero", "n_missing", "n_missing_off_score",
                  "n_missing_on_score", "n_valid_absolute"]].to_numpy(float), expected)


REPORT_LABELS = {
    "运动总分": "total",
    "强直": "rigidity",
    "运动徐缓": "bradykinesia",
    "震颤": "tremor",
    "步态与平衡": "gait_balance",
}


def report_endpoint(label):
    if label in ENDPOINTS:
        return label
    candidates = [endpoint for display, endpoint in REPORT_LABELS.items() if display in label]
    return candidates[0] if len(candidates) == 1 else None


def report_number(value):
    return float(value.replace("−", "-")) if value not in {"NA", "—"} else np.nan


def verify_report(main):
    report = (ROOT / "reports" / REPORT).read_text(encoding="utf-8-sig")
    rows = []
    for line in report.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip().replace("*", "") for cell in line.strip().strip("|").split("|")]
        if len(cells) != 9 or cells[0] not in STATES:
            continue
        endpoint = report_endpoint(cells[1])
        if endpoint:
            rows.append((cells[0], endpoint, *cells[2:]))
    require([(row[0], row[1]) for row in rows] == CELLS,
            "The report must retain the twenty registered main tests in canonical order")
    absolute = table(RESULTS / POINT_TABLE)
    for report_row, rate_row, point_row in zip(rows, main.itertuples(index=False), absolute.itertuples(index=False), strict=True):
        same(report_number(report_row[2]), rate_row.n)
        shown = [report_number(report_row[i]) for i in (3, 4, 5, 7)]
        expected = [float(f"{value:.6g}") for value in (rate_row.rho, rate_row.raw_p, rate_row.fdr_q, point_row.rho)]
        same(shown, expected)


def main():
    lock = read_json(PRIVATE / INPUT_LOCK)["file_sha256"]
    source = load_source_pairs(lock, CFG["cohort"]["n_pairs"])
    verify_logged_artifacts(source, lock)
    verify_saved_wide(source)
    expected, eligible = reconstruct_cells(source, CFG["statistics"]["minimum_n"])
    verify_same_cases(eligible)
    main_table = verify_correlation_tables(expected, CFG["statistics"]["alpha"])
    verify_descriptive_tables(source)
    verify_heatmap(main_table)
    verify_report(main_table)
    print("Round three verified frozen inputs, paired coverage closure and rate eligibility.")
    print("Independently reproduced both Spearman BH20 families, identical patient sets, and public annotations.")


if __name__ == "__main__":
    main()
