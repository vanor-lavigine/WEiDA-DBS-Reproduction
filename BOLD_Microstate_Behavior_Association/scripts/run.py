"""Run the private score audit and four frozen-feature associations only."""

import argparse
import json
from pathlib import Path

import pandas as pd

from io_sources import (ARCHIVES, AUDIT_DIR, BEHAVIOR_DIR, ENDPOINTS, NONEXPORTED, ORIGINAL_FILES,
                        PRIVATE, PROJECT, PROTOCOL, REFERENCE, SCORE_KEYS, V6_NAME, WORKSPACE,
                        environment, fingerprint, load_sources, require, write_csv, write_json, write_result)


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-private", type=Path, default=WORKSPACE / "WEiDA-DBS-V6" / V6_NAME / "private")
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--reference-csv", type=Path, default=BEHAVIOR_DIR / "scl_P_stn.csv")
    parser.add_argument("--crosswalk", type=Path, default=BEHAVIOR_DIR / "STN患者评分与微状态对应.xlsx")
    scan_visits = sorted(WORKSPACE.glob("V100*/结论/**/105原始访视逐人核查.xlsx"))
    parser.add_argument("--scan-visit-workbook", type=Path,
                        default=scan_visits[0] if len(scan_visits) == 1 else PRIVATE / "inputs/Scan_Visits.xlsx")
    parser.add_argument("--archive-project", type=Path, help="Authorized WEiDA project root with the result and data directories")
    parser.add_argument("--verified-visits", type=Path, default=PRIVATE / "inputs/Verified_Visit_UPDRS.csv")
    parser.add_argument("--verified-evidence", type=Path, default=PRIVATE / "inputs/Verified_Visit_Evidence.csv")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def family(data, scores, path, source, reason="", historical_audited_pairs=0):
    from association import association, correct

    if scores is None:
        combined = pd.DataFrame()
    else:
        combined = data.join(scores.set_index("participant_id")[SCORE_KEYS], how="inner", validate="one_to_one")
        combined = combined.dropna(subset=SCORE_KEYS)
        combined["improvement"] = combined.updrs_iii_off - combined.updrs_iii_on
    n_score_pairs = len(combined)
    rows = []
    for item in ENDPOINTS:
        field = "delta_" + item["field"]
        result = association(combined[field].to_numpy(float), combined.improvement.to_numpy(float),
                             combined.delta_fd.to_numpy(float)) if scores is not None else association([], [], [])
        if reason and n_score_pairs == 0:
            result.update(status="BLOCKED", reason=reason, fd_status="BLOCKED", fd_reason=reason)
        rows.append(dict(analysis_path=path, score_source=source, endpoint=field,
                         computation_unit=item["unit"], display_unit=item["display_unit"],
                         display_multiplier=item["display_multiplier"], n_score_pairs=n_score_pairs,
                         historical_audited_pairs=historical_audited_pairs, **result))
    return correct(pd.DataFrame(rows)), combined


def run_associations(sources):
    audit = sources.audit
    scores = sources.verified if len(sources.verified) else None
    verified, verified_data = family(sources.pairs, scores, "verified_visit", "adjudicated_visit_records",
                                     audit["verified_reason"])
    if len(verified_data):
        write_csv(verified_data.reset_index(), PRIVATE / "verified/Analysis_Pairs.csv")
    archived_tables = []
    matches = {}
    for version in ARCHIVES:
        rowwise_scores = sources.archived.get(version)
        table, joined = family(sources.pairs, rowwise_scores, "archived_unverified_visit", version,
                               NONEXPORTED, sources.archive_stats[version]["historical_audited_pairs"])
        archived_tables.append(table)
        if len(joined):
            matches[version] = joined
            write_csv(joined.reset_index(), PRIVATE / "archived_unverified" / version / "Analysis_Pairs.csv")
    archived = pd.concat(archived_tables, ignore_index=True)
    sens = [archived.assign(sensitivity_set="all_evaluable_for_source")]
    if len(sources.archived) >= 2:
        for name, table in sources.archived.items():
            same = table.loc[table.participant_id.isin(sources.consistency)]
            test, _ = family(sources.pairs, same, "archived_unverified_visit", name,
                             "No current paired totals agree across all readable sources")
            sens.append(test.assign(sensitivity_set="same_both_totals_across_readable_sources"))
    else:
        test, _ = family(sources.pairs, None, "archived_unverified_visit", "cross_version_consistency",
                         "The consistent subset needs at least two directly matched score exports")
        sens.append(test.assign(sensitivity_set="same_both_totals_across_readable_sources"))
    sensitivity = pd.concat(sens, ignore_index=True)
    return verified, archived, sensitivity, matches, verified_data


def main():
    args = options()
    frozen_files = {f"v6_{k}": args.v6_private / rel for k, rel in ORIGINAL_FILES.items()}
    frozen_before = fingerprint(frozen_files)
    sources = load_sources(args)
    start = fingerprint(sources.files)
    require(fingerprint(frozen_files) == frozen_before, "Frozen imaging files changed while validating input")
    metadata = {"imaging_commit": PROTOCOL["imaging_commit"], "software": environment(),
                "seed": PROTOCOL["seed"], "permutations_requested": PROTOCOL["permutations"],
                "bootstrap_requested": PROTOCOL["bootstrap"], "alpha": PROTOCOL["alpha"],
                "score_and_imaging_inputs": sources.audit}
    write_json({"frozen_before": frozen_before, "inputs": start, "software": metadata["software"],
                "imaging_commit": PROTOCOL["imaging_commit"], "seed": PROTOCOL["seed"]},
               PRIVATE / "manifests/Input_Hashes.json")
    write_json(metadata, PRIVATE / "manifests/Audit_Before_Inference.json")
    if args.audit_only:
        print(json.dumps(sources.audit, ensure_ascii=False, indent=2))
        return

    from figures import draw_figures
    verified, archived, sensitivity, joined, verified_data = run_associations(sources)
    check = fingerprint(sources.files) == start
    require(check, "Input file digest changed during the association run")
    metadata["inputs_unchanged_during_run"] = check
    results = PROJECT / "results"
    write_result(verified, results / "Verified_Visit_Associations.csv")
    write_result(archived, results / "Archived_Score_Associations.csv")
    write_result(sensitivity, results / "Score_Source_Sensitivity.csv")
    draw_figures(verified, archived, sensitivity, joined, verified_data)
    write_json(metadata, results / "Run_Metadata.json")
    print(f"Verified visit pairs: {len(sources.verified)}")
    print(f"Rowwise historical exports: {len(joined)}")
    print(f"Four-endpoint family rows: {len(verified)} verified, {len(archived)} archive, {len(sensitivity)} sensitivity")


if __name__ == "__main__":
    main()
