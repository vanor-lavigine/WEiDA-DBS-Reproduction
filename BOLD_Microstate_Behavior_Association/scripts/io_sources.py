"""Read frozen features and verify score-to-scan links before computing correlations."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
import platform
import re
import secrets
import sys

import numpy as np
from openpyxl import load_workbook
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
V6_NAME = "V6_DBS_ON_OFF_Microstate_Dynamics"
sys.path.insert(0, str(REPO / V6_NAME / "scripts"))
from v6_common import CFG as V6_CONFIG, WORKSPACE, digest, read_json, write_csv, write_json


PROTOCOL_PATH = PROJECT / "configs/protocol.json"
PROTOCOL = read_json(PROTOCOL_PATH)
PRIVATE = PROJECT / "private"
ENDPOINTS = tuple(PROTOCOL["endpoints"])
FIELDS = [item["field"] for item in ENDPOINTS]
PAIR_KEYS = ["participant_id", "on_scan_id", "off_scan_id"]
SCORE_KEYS = ["updrs_iii_on", "updrs_iii_off"]
ARCHIVES = tuple(PROTOCOL["archive_versions"])
REFERENCE = ARCHIVES[0]
ARCHIVE_FILE = "scl_P_stn.csv"
SCORE_AUDIT_NAME = "source105_score_verify_join_20261123.json"
AUDIT_DIR = WORKSPACE / "WEiDA-raw-audit-private-20261116"
BEHAVIOR_DIR = WORKSPACE / "105微状态分析" / "师姐评分表"
ORIGINAL_FILES = {
    "manifest": "manifests/ON_OFF_Pair_Manifest.csv",
    "cohort": "results/Frozen_Cohort_Features.csv",
    "scans": "results/ON_OFF_Subject_Features.csv",
    "differences": "results/ON_OFF_Paired_Differences.csv",
}
SCAN_PATTERN = re.compile(r"sub-stn(on|off)(\d+)", re.I)
NONEXPORTED = "Current run has no readable row-level copy of this historical export"
VERIFY_MISSING = "No verified score-to-scan visit records and supporting evidence index"


@dataclass
class Sources:
    pairs: pd.DataFrame
    archived: dict[str, pd.DataFrame]
    verified: pd.DataFrame
    archive_stats: dict
    consistency: pd.Index
    audit: dict
    files: dict[str, Path]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_same(left, right, message):
    require(np.allclose(left, right, rtol=0, atol=1e-12, equal_nan=True), message)


def read_table(path):
    return pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")


def write_result(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="NA", encoding="utf-8-sig", lineterminator="\n")


def fingerprint(files):
    return {key: {"file": str(path.resolve()), "sha256": digest(path)}
            for key, path in sorted(files.items()) if path.is_file()}


def environment():
    modules = ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "openpyxl")
    return {"python": platform.python_version(), **{key: version(key) for key in modules}}


def load_frozen(private_root):
    files = {f"v6_{key}": private_root / rel for key, rel in ORIGINAL_FILES.items()}
    m, cohort, scans, saved = (read_table(files[f"v6_{name}"]) for name in ORIGINAL_FILES)
    counts = PROTOCOL["cohort"]
    n = counts["pd_pairs"]
    require(len(m) == n and m[PAIR_KEYS].notna().all().all() and m.participant_id.is_unique,
            "Frozen patient manifest must contain unique complete pair keys")
    require(len(set(m.on_scan_id) | set(m.off_scan_id)) == 2 * n,
            "Frozen ON/OFF scan keys must be unique across all pairs")
    for field, condition in (("on_scan_id", "on"), ("off_scan_id", "off")):
        require(m[field].map(lambda x: bool(SCAN_PATTERN.fullmatch(x)) and
                             SCAN_PATTERN.fullmatch(x)[1].lower() == condition).all(),
                "Frozen scan-key condition conflicts with the paired manifest")
    require(len(cohort) == counts["scans"] and cohort.scan.is_unique and
            cohort.condition.value_counts().to_dict() == {"HC": counts["hc"], "ON": n, "OFF": n},
            "Frozen cohort does not match the locked scan counts")
    cohort_pd = cohort.loc[cohort.condition.isin(["ON", "OFF"])].sort_values("scan").reset_index(drop=True)
    scans = scans.sort_values("scan").reset_index(drop=True)
    require(scans.equals(cohort_pd), "Frozen subject-level scan features differ from the PD cohort rows")
    require(len(saved) == n and saved.participant_id.is_unique and
            set(saved.participant_id) == set(m.participant_id), "Frozen difference table has inconsistent keys")
    m = m.set_index("participant_id").sort_index()
    aligned = {}
    for condition in ("ON", "OFF"):
        side = scans.loc[scans.condition.eq(condition)].set_index("subject")
        require(side.index.is_unique and set(side.index) == set(m.index),
                "Images do not map one-to-one to participants within each condition")
        side = side.loc[m.index]
        require(side.scan.eq(m[f"{condition.lower()}_scan_id"]).all(),
                "Manifest and source scan labels are misaligned")
        check_same(side.fd, m[f"{condition.lower()}_mean_fd"], "Frozen FD and pair manifest differ")
        aligned[condition] = side
    require(aligned["ON"].fold.eq(aligned["OFF"].fold).all(), "An ON/OFF pair crosses frozen folds")

    # Rates are episode or transition counts divided by elapsed seconds at the frozen exposure.
    seconds = counts["tr_seconds"] * scans.valid_frames
    for field in ("occurrence_1", "switching_rate"):
        implied = (scans[field] * seconds).dropna()
        check_same(implied, np.rint(implied), "Frozen rates do not have the documented per-second units")
    gev = scans.gev_1.dropna()
    require(gev.between(0, 1).all(), "Frozen GEV is outside its proportion scale")
    readout = aligned["ON"][FIELDS + ["fd"]] - aligned["OFF"][FIELDS + ["fd"]]
    prior = saved.set_index("participant_id").loc[m.index, readout.columns]
    check_same(readout, prior, "Reconstructed ON-minus-OFF differences do not match frozen values")
    paired = m.join(prior.rename(columns={key: f"delta_{key}" for key in prior.columns}))
    meta = {
        "frozen_scans": len(cohort), "frozen_pd_scans": len(scans), "frozen_pd_pairs": len(m),
        "hc_scans": int(cohort.condition.eq("HC").sum()),
        "valid_trs": int(cohort.valid_frames.sum()),
        "scan_clock_pairs": int(m[["on_scan_time", "off_scan_time"]].notna().all(axis=1).sum()),
        "on_first_pairs": int(m.scan_order.eq("ON before OFF").sum()),
        "verified_medication_pairs": int(m[["on_medication_hours", "off_medication_hours"]].notna().all(axis=1).sum()),
    }
    return paired, meta, files


def scan_lookup(pairs):
    shots = pairs.reset_index()[PAIR_KEYS].melt(id_vars="participant_id", var_name="side", value_name="scan")
    shots["condition"] = shots.side.map({"on_scan_id": "ON", "off_scan_id": "OFF"})
    shots["scan_key"] = shots.scan.str.casefold()
    require(shots.scan_key.is_unique, "Duplicate case-insensitive scan key")
    return shots


def archive_pairs(long, pairs):
    long = long.assign(scan_key=long.scan.str.casefold())
    require(long.scan_key.is_unique, "Historical scan crosswalk is not one-to-one")
    map_keys = scan_lookup(pairs).drop(columns=["scan", "side"])
    merged = long.merge(map_keys, on=["scan_key", "condition"], validate="one_to_one", how="inner")
    require(len(merged) == len(long), "Historical score row conflicts with frozen scan or DBS condition")
    if "mapped_study_code" in merged:
        require(merged.mapped_study_code.notna().all()
                and merged.groupby("mapped_study_code").participant_id.nunique().eq(1).all()
                and merged.groupby("participant_id").mapped_study_code.nunique().eq(1).all(),
                "Historical study codes do not map one-to-one to frozen participants")
    scores = merged.pivot(index="participant_id", columns="condition", values="score")
    scores = scores.reindex(columns=["ON", "OFF"]).set_axis(SCORE_KEYS, axis=1)
    return pairs.reset_index()[PAIR_KEYS].merge(scores, on="participant_id", validate="one_to_one"), merged


def rowwise_archive(csv_file, workbook_file, pairs):
    """Recover the frozen scan link through a previously fingerprint-audited workbook."""
    raw = read_table(csv_file)
    with workbook_file.open("rb") as book_file:
        workbook = load_workbook(book_file, read_only=True, data_only=True)
        try:
            sheet = workbook["原表与访视"]
            note = str(sheet["A2"].value)
            physical_rows = list(enumerate(sheet.iter_rows(min_row=5, max_col=25, values_only=True), start=5))
        finally:
            workbook.close()
    require(REFERENCE in note and "scl_P_stn.csv" in note,
            "Historical workbook does not identify the predeclared export")
    original_rows = sorted((row[0], index, row) for index, row in physical_rows if row[0] is not None)
    rows = [row for _, _, row in original_rows]
    require(len(rows) == len(raw) == 2 * len(pairs) and raw.columns[-1] == "UPDRS3",
            "Historical score export and row audit have inconsistent lengths or score fields")
    require({r[0] for r in rows} == set(range(2, len(raw) + 2)),
            "Historical workbook source rows are missing or duplicated")
    if "研究编号" in raw:
        require(raw["研究编号"].astype(str).str.casefold().tolist() == [str(r[1]).casefold() for r in rows],
                "Score CSV study codes differ from the archived scan crosswalk")
    numerical = raw.drop(columns=["研究编号"], errors="ignore")
    require(numerical.shape[1] == 18 and numerical.columns[-2:].tolist() == ["tremor", "UPDRS3"],
            "Unexpected numeric form of the reference motor export")
    check_same(numerical.iloc[:, :-1].to_numpy(float), np.array([r[8:25] for r in rows], float),
               "Historical workbook differs from the original score CSV outside the total")
    check_same(raw.UPDRS3, np.array([r[4] for r in rows], float),
               "Historical workbook and raw total scores differ")
    long = pd.DataFrame({"source_row": [r[0] for r in rows],
                         "mapped_study_code": [r[1] for r in rows],
                         "condition": [r[2] for r in rows],
                         "scan": [r[3] for r in rows], "score": raw.UPDRS3,
                         "subtotal_12": numerical.iloc[:, 4:16].sum(axis=1, min_count=12),
                         "tremor_subtotal": numerical.tremor,
                         "tremor_components": numerical.iloc[:, 13:16].sum(axis=1, min_count=3),
                         "workbook_row": [j for _, j, _ in original_rows]})
    require(np.isfinite(long.score).all() and long.score.ge(0).all(),
            "Reference archive has missing or invalid motor scores")
    mapped, evidence = archive_pairs(long, pairs)
    evidence["source"] = REFERENCE
    evidence["source_file"] = str(csv_file.resolve())
    evidence["workbook_file"] = str(workbook_file.resolve())
    evidence["workbook_sheet"] = "原表与访视"
    mapped["visit_evidence"] = "archived_unverified_visit"
    return mapped, evidence


def original_versions(archive_project, pairs):
    """Use the original joint P/LT fingerprint loader for authorized server archives."""
    sys.path.insert(0, str(REPO / "experiments/bold_microstates_v1"))
    from clinical_archive_rerun import ENGLISH_DOMAINS, VERSIONS, clinical_visits, scan_key

    files, evidence, archived = {}, [], {}
    salt = secrets.token_hex(32)
    scans = scan_lookup(pairs)[["scan", "condition"]]
    keys = scans.scan.map(lambda scan: scan_key(scan, salt))
    for name, series in VERSIONS:
        require(name in ARCHIVES, "Historical source does not match the locked source list")
        path = archive_project / "result" / name
        leaves = (ARCHIVE_FILE, "scl_LT_stn.csv", "P2emp.csv", "LT2emp.csv", "P3emp.csv", "LT3emp.csv")
        if not all((path / leaf).is_file() for leaf in leaves) or not (archive_project / "data" / series).is_dir():
            continue
        mapped = clinical_visits(archive_project, name, series, salt)
        require(set(keys) == set(mapped.index), "Historical source series differs from the frozen 50 scans")
        raw = mapped.loc[keys].reset_index(drop=True)
        long = scans.assign(score=raw.score, subtotal_12=raw[list(ENGLISH_DOMAINS)].sum(axis=1))
        d, link = archive_pairs(long, pairs)
        d["visit_evidence"] = "archived_unverified_visit"
        archived[name] = d
        evidence.append(link.assign(source=name, source_file=str(path / ARCHIVE_FILE),
                                    source_basis="joint P/LT and named source-series fingerprint"))
        for leaf in leaves:
            files[f"{name}/{leaf}"] = path / leaf
    return archived, evidence, files


def verified_scores(score_path, evidence_path, pairs):
    needed = PAIR_KEYS + SCORE_KEYS + ["visit_evidence"]
    if not score_path.is_file() or not evidence_path.is_file():
        return pd.DataFrame(columns=needed), {}, VERIFY_MISSING
    scores = read_table(score_path)
    known = pairs.reset_index()[PAIR_KEYS + ["on_scan_time", "off_scan_time"]]
    require(scores[PAIR_KEYS].notna().all().all() and scores.participant_id.is_unique,
            "Verified clinical input contains null or duplicate paired keys")
    aligned = scores[needed].merge(known, on=PAIR_KEYS, how="left", indicator=True, validate="one_to_one")
    require(aligned._merge.eq("both").all(), "A verified record failed the exact frozen ON/OFF key check")
    require(aligned.visit_evidence.isin(["direct_visit_key", "confirmed_assessment_date"]).all(),
            "Verified visit input has an unsupported evidence category")
    for key in SCORE_KEYS:
        aligned[key] = pd.to_numeric(aligned[key], errors="raise")
    require((aligned[SCORE_KEYS].isna() | (np.isfinite(aligned[SCORE_KEYS]) & aligned[SCORE_KEYS].ge(0))).all().all(),
            "Verified clinical scores must be nonnegative and finite when present")
    evidence = read_table(evidence_path)
    key_cols = ["participant_id", "condition", "scan_id"]
    evidence_cols = [*key_cols, "source_path", "record_locator", "visit_basis", "assessment_date"]
    require(set(evidence_cols).issubset(evidence.columns) and not evidence.duplicated(key_cols).any(),
            "Verified visit evidence lacks required fields or has duplicate scan links")
    paired = aligned[aligned[SCORE_KEYS].notna().all(axis=1)].copy()
    crosswalk = scan_lookup(paired.set_index("participant_id"))
    crosswalk = crosswalk.rename(columns={"scan": "scan_id"}).merge(
        evidence[evidence_cols], on=key_cols, how="left", validate="one_to_one"
    ).merge(paired[["participant_id", "visit_evidence", "on_scan_time", "off_scan_time"]],
            on="participant_id", validate="many_to_one")
    require(crosswalk[["source_path", "record_locator", "visit_basis"]].notna().all().all(),
            "A complete verified pair has no source record or evidence locator")
    dates = pd.to_datetime(crosswalk.assessment_date, errors="coerce").dt.normalize()
    scans = pd.to_datetime(crosswalk.on_scan_time.where(crosswalk.condition.eq("ON"), crosswalk.off_scan_time),
                           errors="coerce").dt.normalize()
    date_required = crosswalk.visit_evidence.eq("confirmed_assessment_date")
    require(dates[date_required].eq(scans[date_required]).all(),
            "Date-based evidence does not link each assessment to its matched scan date")
    files = {"verified_scores": score_path, "verified_visit_evidence": evidence_path}
    for i, value in enumerate(crosswalk.source_path.drop_duplicates()):
        record = evidence_path.parent / Path(str(value))
        require(record.is_file(), "Verified visit evidence points to an unavailable source file")
        files[f"verified_source_{i+1}"] = record
    return paired[needed], files, ""


def imaging_visit_evidence(workbook_path, pairs, reference_link):
    if not workbook_path.is_file():
        return {"workbook_available": False}
    with workbook_path.open("rb") as source:
        book = load_workbook(source, read_only=True, data_only=True)
        try:
            rows = list(book["48次访视"].iter_rows(min_row=5, values_only=True))
        finally:
            book.close()
    rows = [row for row in rows if row[0] is not None and row[1] in ("ON", "OFF")]
    scans = [re.search(r"(?i)sub-STN(?:on|off)\d+", str(row[11])) for row in rows]
    require(all(scans), "Imaging workbook contains a visit without its original scan key")
    view = pd.DataFrame({
        "scan_key": [scan[0].casefold() for scan in scans], "condition": [r[1] for r in rows],
        "scan_date": pd.to_datetime([r[2] for r in rows], errors="coerce"),
        "assessment_date": pd.to_datetime([r[3] for r in rows], format="mixed", errors="coerce"),
        "medication_clock": pd.to_datetime([r[5] for r in rows], format="mixed", errors="coerce"),
        "score_source_row": pd.to_numeric(pd.Series([r[9] for r in rows]), errors="coerce"),
    })
    match = view.merge(scan_lookup(pairs)[["scan_key", "condition", "participant_id"]],
                       on=["scan_key", "condition"], validate="one_to_one")
    require(len(match) == len(view), "An imaging visit has an incompatible frozen scan or condition")
    row_matches = None
    if reference_link is not None:
        check = match.merge(reference_link[["scan_key", "source_row"]],
                            on="scan_key", validate="one_to_one")
        row_matches = int(check.score_source_row.eq(check.source_row).sum())
    return {"workbook_available": True, "scans_linked": len(match),
            "scan_dates_available": int(match.scan_date.notna().sum()),
            "assessment_dates_available": int(match.assessment_date.notna().sum()),
            "medication_clocks_available": int(match.medication_clock.notna().sum()),
            "reference_source_rows_matching": row_matches}


def archived_summaries(archive_trace, conflicts, origin, archived, links, pairs):
    previous = archive_trace["runs"]
    prior_conflicts = conflicts["version_checks"]["archived"]
    prior_arithmetic = origin["numerical"]["source_totals_equal_component_sum"]
    n = len(pairs)
    require(all(previous[name]["confirmed_source_pairs"] == n for name in ARCHIVES),
            "Historical score audit and frozen cohort have different pair counts")
    require(all(group["same_named_scans"] == 2 * n for group in archive_trace["cross_version"].values()),
            "Historical audit does not establish the same named scans across versions")
    out = {}
    for name, which in zip(ARCHIVES, ("main", "replica", "alternate"), strict=True):
        summary = {"historical_audited_pairs": previous[name]["confirmed_source_pairs"],
                   "historical_audited_scans": previous[name]["raw_rows_reassigned_from_joint_P_LT"],
                   "historical_totals_equal_subtotal": prior_arithmetic[which],
                   "historical_has_assessment_date": previous[name]["age_or_medication_or_date_in_raw_table"]["date"],
                   "historical_has_visit_key": previous[name]["age_or_medication_or_date_in_raw_table"]["visit"],
                   "historical_has_medication_clock": previous[name]["age_or_medication_or_date_in_raw_table"]["med"],
                   "readable_pairs": 0, "readable_scans": 0, "totals_equal_subtotal": None}
        if name in archived:
            link = next(x for x in links if x.source.iloc[0] == name)
            summary.update(readable_pairs=int(archived[name][SCORE_KEYS].notna().all(axis=1).sum()),
                           readable_scans=len(link),
                           totals_equal_subtotal=int(link.score.eq(link.subtotal_12).sum()),
                           tremor_subtotal_equal_components=(int(link.tremor_subtotal.eq(link.tremor_components).sum())
                                                            if "tremor_components" in link else None))
        out[name] = summary
    recovered_common = pd.Index([], dtype=str, name="participant_id")
    reread_conflicts = None
    if len(archived) >= 2:
        table = pd.concat({s: table.set_index("participant_id")[SCORE_KEYS] for s, table in archived.items()}, axis=1)
        table = table.dropna().sort_index()
        both_equal = pd.Series(True, index=table.index)
        counts = {}
        for j, name in enumerate(archived):
            for other in list(archived)[j + 1:]:
                diff = table[other] - table[name]
                paired_diff = diff.updrs_iii_off - diff.updrs_iii_on
                mismatch = diff.ne(0)
                both_equal &= ~mismatch.any(axis=1)
                counts[f"{name} vs {other}"] = {
                    "on_total_conflicts": int(mismatch.updrs_iii_on.sum()),
                    "off_total_conflicts": int(mismatch.updrs_iii_off.sum()),
                    "paired_improvement_conflicts": int(paired_diff.ne(0).sum()),
                    "pairs_differ_either_total": int(mismatch.any(axis=1).sum()),
                }
        reread_conflicts = {"pairs_compared": len(table), "source_count": len(archived),
                            "pairs_agree_both_conditions": int(both_equal.sum()), "comparisons": counts}
        # A cross-version consistency set refers to the versions actually reread.
        recovered_common = table.index[both_equal]
    prior_counts = {"pairs_compared": prior_conflicts["pairs"],
                    "pairs_agree_both_conditions": prior_conflicts["pairs_same_both_totals_all_three"],
                    "pairs_differ_either_total": prior_conflicts["pairs_differ_either_total"],
                    "pairs_with_different_improvement": prior_conflicts["pairs_different_raw_ON_OFF_change"],
                    "pairs_conflict_by_condition": {
                        c: prior_conflicts["by_condition"][c]["n"] - prior_conflicts["by_condition"][c]["same_total_4_1_vs_4_3"]
                        for c in ("off", "on")
                    },
                    "equal_12_component_values": origin["numerical"]["n_domains_equal_main_alternate"]}
    audit = {"historical_archive_counts": prior_counts, "rowwise_current_comparison": reread_conflicts}
    return out, recovered_common, audit


def load_sources(args):
    pairs, image_meta, files = load_frozen(args.v6_private)
    history_files = {
        "historical_named_scan_audit": args.audit_dir / SCORE_AUDIT_NAME,
        "historical_score_conflicts": args.audit_dir / "source105_score_conflicts_20261123.json",
        "historical_score_components": args.audit_dir / "score_origin_20261124/aggregate.json",
    }
    archive_trace, conflicts, origin = (read_json(v) for v in history_files.values())
    files.update(history_files)
    files.update({"v6_protocol": REPO / V6_NAME / "configs/protocol.json", "behavior_protocol": PROTOCOL_PATH})
    archived = {}
    links = []
    if args.archive_project:
        archived, links, original_files = original_versions(args.archive_project, pairs)
        files.update(original_files)
    if REFERENCE not in archived and args.reference_csv.is_file() and args.crosswalk.is_file():
        archived[REFERENCE], link = rowwise_archive(args.reference_csv, args.crosswalk, pairs)
        links.append(link)
        files.update({"reference_score_export": args.reference_csv, "reference_score_row_crosswalk": args.crosswalk})
    ref_link = next((link for link in links if link.source.iloc[0] == REFERENCE and "source_row" in link), None)
    image_visit = imaging_visit_evidence(args.scan_visit_workbook, pairs, ref_link)
    if image_visit["workbook_available"]:
        files["imaging_visit_workbook"] = args.scan_visit_workbook
    verified, verified_files, reason = verified_scores(args.verified_visits, args.verified_evidence, pairs)
    files.update(verified_files)
    version_stats, consistency, score_audit = archived_summaries(archive_trace, conflicts, origin,
                                                                archived, links, pairs)
    require(all(V6_CONFIG[k] == PROTOCOL[k] for k in ("seed", "permutations", "bootstrap")),
            "V6 resampling settings changed")
    baseline = {"method": "A3", "atlas": "AAL78", "k": 4, "tr": 2.1, "band_hz": [0.01, 0.08],
                "band": "F1", "normalization": "N0", "learner": "G1", "decoder": "S1", "lambda": 0.04}
    require(V6_CONFIG["baseline"] == baseline, "The frozen V6 imaging configuration differs")
    audit = {**image_meta, **score_audit, "imaging_visit_workbook": image_visit, "sources": version_stats,
             "reference": REFERENCE, "verified_pairs": len(verified), "verified_reason": reason,
             "historical_consistent_pairs_computable": len(consistency),
             "n_sources_reread": len(archived)}
    if links:
        write_csv(pd.concat(links, ignore_index=True), PRIVATE / "evidence/Archive_Scan_Source_Index.csv")
    for name, frame in archived.items():
        write_csv(frame, PRIVATE / "archived_unverified" / name / "Archived_Score_Input.csv")
    if len(verified):
        write_csv(verified, PRIVATE / "verified/Verified_Clinical_Pairs.csv")
    return Sources(pairs, archived, verified, version_stats, consistency, audit, files)
