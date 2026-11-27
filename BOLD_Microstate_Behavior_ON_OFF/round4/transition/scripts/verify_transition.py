"""Check the published tables and figures against the frozen private measurements."""

import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import false_discovery_control, spearmanr

from run_transition import ARCHIVED_TRANSITIONS, PUBLIC
from transition_io import (
    BASE, CFG, CONFIG, PRIVATE, REGISTRY, REPO, ROOT, collection_digest, digest, expected_registry,
    read_json, verify_base, verify_inputs, verify_registry,
)
from transition_report import report_text
from transition_summaries import describe_quality, hc_reference, summarize_m4, v6_parity


EDGES = ["from_state", "to_state"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(folder, name):
    return pd.read_csv(folder / (name + ".csv"), float_precision="round_trip")


def same_frame(left, right):
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)


def verify_measurements():
    folder = PRIVATE / "results"
    scans = table(folder, "Transition_Counts_All_Scans")
    matrix = table(folder, "Transition_Matrix_All_Scans")
    require(len(scans) == 65 and scans.scan_id.is_unique, "Wrong scan inventory")
    require(scans.condition.value_counts().to_dict() == {k: CFG["cohort"][k] for k in ("HC", "OFF", "ON")},
            "Wrong frozen group counts")
    patients = scans.loc[scans.condition.ne("HC")]
    pair_size = patients.groupby(["participant_id", "condition"]).size().unstack()
    require(len(pair_size) == 25 and pair_size.eq(1).all().all(), "Non-unique patient-condition pairs")
    require(patients.groupby("participant_id").fold.nunique().eq(1).all(), "Pair fold mismatch")
    require(scans.groupby(["condition", "fold"]).size().to_dict() ==
            {(condition, fold): CFG["cohort"][condition] // 5
             for condition in ("HC", "OFF", "ON") for fold in range(5)}, "Frozen fold distribution changed")
    require(len(matrix) == 1040 and set(matrix.scan_id) == set(scans.scan_id), "Missing matrix rows")
    require(not matrix.duplicated(["scan_id", *EDGES]).any(), "Duplicated matrix rows")
    require(matrix.groupby("scan_id").size().eq(16).all(), "Incomplete scan matrices")
    keys = ["scan_id", "participant_id", "condition", "fold"]
    require(set(matrix[keys].itertuples(index=False, name=None)) ==
            set(scans[keys].itertuples(index=False, name=None)), "Scan metadata mismatch")
    require(matrix["count"].ge(0).all() and matrix["count"].mod(1).eq(0).all(), "Invalid transition counts")

    totals = matrix.groupby(["scan_id", "from_state"])["count"].transform("sum")
    diagonal = matrix.from_state.eq(matrix.to_state)
    exits = matrix["count"].where(~diagonal, 0).groupby(
        [matrix.scan_id, matrix.from_state],
    ).transform("sum")
    for column, numerators, denominator in (
        ("p_weida", matrix["count"], totals), ("p_exit", matrix["count"].where(~diagonal), exits),
    ):
        expected = numerators / denominator.where(denominator > 0)
        np.testing.assert_allclose(matrix[column], expected, atol=1e-13, rtol=0, equal_nan=True)
        defined = matrix.groupby(["scan_id", "from_state"])[column].sum(min_count=1).dropna()
        np.testing.assert_allclose(defined, 1, atol=1e-13, rtol=0)
        filename = "Transition_Probability_WEiDA" if column == "p_weida" else "Transition_Probability_ExitConditional"
        same_frame(table(folder, filename), matrix[[*keys, *EDGES, column]])

    indexed = scans.set_index("scan_id")
    actual_sums = matrix.groupby("scan_id")["count"].sum().reindex(indexed.index)
    require(actual_sums.eq(indexed.n_adjacent).all(), "Counts disagree with valid adjacency")
    require(indexed.n_adjacent.eq(indexed.valid_trs - indexed.n_blocks).all(), "A transition crossed a block edge")
    for scan in scans.itertuples(index=False):
        starts = json.loads(scan.block_start_tr_zero_based)
        ends = json.loads(scan.block_end_tr_inclusive_zero_based)
        require(len(starts) == len(ends) == scan.n_blocks, "Incomplete block boundary list")
        require(sum(end - start + 1 for start, end in zip(starts, ends, strict=True)) == scan.valid_trs,
                "Block limits disagree with valid TR count")
    for state in CFG["states"]:
        frames = indexed[f"frames_{state}"]
        runs = indexed[f"episodes_{state}"]
        require(runs.eq(indexed[f"starts_{state}"] + indexed[f"enters_{state}"]).all(), "Invalid segment entries")
        require(runs.eq(indexed[f"ends_{state}"] + indexed[f"exits_{state}"]).all(), "Invalid segment exits")
        require(indexed[f"N_{state}_{state}"].eq(frames-runs).all(), "Invalid self-loop/segment identity")
        np.testing.assert_allclose(indexed[f"coverage_{state}"], frames / indexed.valid_trs, atol=1e-13)
        np.testing.assert_allclose(indexed[f"duration_seconds_{state}"],
                                   CFG["imaging"]["TR_seconds"] * frames / runs.where(runs > 0), atol=1e-13)
        np.testing.assert_allclose(indexed[f"occurrence_per_second_{state}"],
                                   runs / (CFG["imaging"]["TR_seconds"] * indexed.valid_trs), atol=1e-13)
        for target in CFG["states"]:
            indicated = matrix.loc[matrix.from_state.eq(state) & matrix.to_state.eq(target)].set_index("scan_id")
            require(indexed[f"N_{state}_{target}"].eq(indicated["count"]).all(), "Wide and long count tables differ")
    return scans, matrix


def verify_pvalues(comparisons, corrections, clinical):
    expected = set(expected_registry())
    seen = {(r.method, r.contrast, r.from_state, r.to_state, "") for r in corrections.itertuples(index=False)}
    seen |= {("Clinical_WEiDA", "ON-OFF", r.from_state, r.to_state, r.clinical_endpoint)
             for r in clinical.itertuples(index=False)}
    require(seen == expected and len(corrections) + len(clinical) == len(expected), "Correction registry mismatch")
    comparison_slots = comparisons.loc[comparisons.scope.eq("prespecified")]
    same_frame(comparison_slots[corrections.columns], corrections)
    require(comparisons.loc[comparisons.scope.eq("descriptive_only"), ["raw_p", "fdr_q", "holm_p"]].isna().all().all(),
            "A self-loop entered the main test family")
    for family, fragment in [*corrections.groupby("method"), ("Clinical_WEiDA", clinical)]:
        n = CFG["families"][family]
        require(len(fragment) == n and fragment.family_size.eq(n).all(), "Family size mismatch")
        p = fragment.raw_p.fillna(1).to_numpy()
        missing = fragment.raw_p.isna()
        np.testing.assert_allclose(fragment.fdr_q, np.where(missing, np.nan, false_discovery_control(p)),
                                   atol=1e-13, rtol=0, equal_nan=True)
        if family != "Clinical_WEiDA":
            order = np.argsort(p)
            ordered_holm = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(n, 0, -1)))
            holm = np.empty(n)
            holm[order] = ordered_holm
            np.testing.assert_allclose(fragment.holm_p, np.where(missing, np.nan, holm),
                                       atol=1e-13, rtol=0, equal_nan=True)


def verify_clinical(matrix, clinical):
    scores = table(PRIVATE / "results", "Frozen_Clinical_Pairs").set_index("patient_id")
    patient_edges = matrix.loc[matrix.condition.ne("HC")]
    require(scores.index.is_unique and set(scores.index) == set(patient_edges.participant_id),
            "Clinical patient keys differ from imaging")
    for condition in ("ON", "OFF"):
        keys = patient_edges.loc[patient_edges.condition.eq(condition), ["participant_id", "scan_id", "fold"]]
        keys = keys.drop_duplicates().set_index("participant_id")
        require(scores[f"{condition.lower()}_scan_id"].eq(keys.scan_id).all() and scores.fold.eq(keys.fold).all(),
                "Clinical scan or fold keys differ from imaging")
    for row in clinical.itertuples(index=False):
        selection = patient_edges.loc[patient_edges.from_state.eq(row.from_state) & patient_edges.to_state.eq(row.to_state)]
        wide = selection.pivot(index="participant_id", columns="condition", values="p_weida")
        absolute = scores[f"{row.clinical_endpoint}_improvement"].reindex(wide.index)
        delta = (wide.ON - wide.OFF).round(CFG["rank_decimals"])
        use = delta.notna() & absolute.notna()
        require(row.n_pairs == use.sum(), "Clinical effective sample count changed")
        if use.sum() >= CFG["minimum_n"] and delta[use].nunique() > 1 and absolute[use].nunique() > 1:
            rho, p = spearmanr(delta[use], absolute[use], alternative="two-sided")
            np.testing.assert_allclose([row.rho, row.raw_p], [rho, p], atol=1e-13, rtol=0)
        else:
            require(pd.isna(row.rho) and pd.isna(row.raw_p) and isinstance(row.reason, str), "Undefined clinical test not retained")


def verify_group_means(matrix, descriptions, comparisons):
    for method, probability in (("WEiDA", "p_weida"), ("ExitConditional", "p_exit")):
        descriptives = descriptions.loc[descriptions.method.eq(method)]
        for row in descriptives.itertuples(index=False):
            edge = matrix.loc[matrix.from_state.eq(row.from_state) & matrix.to_state.eq(row.to_state)
                              & matrix.condition.eq(row.condition), probability]
            require(edge.count() == row.n, "Group mean denominator disagrees")
            np.testing.assert_allclose(edge.mean(), row.mean, atol=1e-13, equal_nan=True)
        for row in comparisons.loc[comparisons.method.eq(method)].itertuples(index=False):
            cell = matrix.loc[matrix.from_state.eq(row.from_state) & matrix.to_state.eq(row.to_state)]
            a = cell.loc[cell.condition.eq(row.group_a)].set_index("participant_id")[probability]
            b = cell.loc[cell.condition.eq(row.group_b)].set_index("participant_id")[probability]
            if row.paired:
                matched = pd.concat([a, b], axis=1).dropna()
                a, b = matched.iloc[:, 0], matched.iloc[:, 1]
            require(a.count() == row.n_a and b.count() == row.n_b, "Contrast mean denominator disagrees")
            np.testing.assert_allclose([a.mean(), b.mean(), a.mean() - b.mean()],
                                       [row.mean_a, row.mean_b, row.mean_diff], atol=1e-13, equal_nan=True)


def verify_figures(files, metadata_key, descriptions, comparisons):
    def check(entries, frame, keys):
        table_lookup = frame.set_index(keys)
        for entry in entries:
            expected = table_lookup.loc[tuple(entry[k] for k in keys)]
            for name, value in entry.items():
                if name not in keys:
                    np.testing.assert_allclose(np.nan if value is None else value, expected[name],
                                               rtol=0, atol=1e-13, equal_nan=True)

    main = comparisons.loc[comparisons.method.eq("WEiDA")]
    group = descriptions.loc[descriptions.method.eq("WEiDA")]
    for name in files:
        path = ROOT / name
        if path.suffix != ".png":
            continue
        with Image.open(path) as figure:
            entries = json.loads(figure.info[metadata_key])
            require(figure.width >= 900 and figure.height >= 650, "Figure resolution below plotted layout")
            figure.verify()
        if path.stem == "M4_Exit_Directions":
            check(entries["group"], descriptions.loc[descriptions.method.eq("ExitConditional")], [*EDGES, "condition"])
            check(entries["exit_conditional_BH36"], comparisons.loc[comparisons.method.eq("ExitConditional")],
                  [*EDGES, "contrast"])
            require(len(entries["group"]) == 9 and len(entries["exit_conditional_BH36"]) == 3, "Incomplete exit plot")
        elif path.stem == "M4_SelfLoop_Paired_Summary":
            expected = main.loc[main.from_state.eq("M4") & main.to_state.eq("M4") & main.contrast.eq("ON-OFF")].iloc[0]
            for k, col in {"mean_on": "mean_a", "mean_off": "mean_b", "mean_diff": "mean_diff",
                           "higher": "paired_higher", "lower": "paired_lower", "same": "paired_same",
                           "n_pairs": "n_pairs", "ci95_low": "ci95_low", "ci95_high": "ci95_high"}.items():
                np.testing.assert_allclose(entries[k], expected[col], rtol=0, atol=1e-13)
        elif "condition" in entries[0]:
            require(len(entries) == 16, "Incomplete group matrix plot")
            check(entries, group, [*EDGES, "condition"])
        else:
            n = 36 if path.stem.endswith("Forest") else 16
            require(len(entries) == n, "Incomplete contrast plot")
            check(entries, main, [*EDGES, "contrast"])


def verify_publication(manifest, files, matrix):
    identifiers = set(matrix.scan_id) | set(matrix.participant_id)
    names = re.compile("|".join(rf"(?<![\w-]){re.escape(x)}(?![\w-])" for x in sorted(identifiers)), re.I)
    for file in [*files, "ANALYSIS_PLAN.md", "results/Transition_Run_Manifest.json"]:
        path = ROOT / file
        if path.suffix not in (".csv", ".md", ".json"):
            continue
        text = path.read_text(encoding="utf-8")
        require(not names.search(text) and not re.search(r"[A-Za-z]:[/\\]", text),
                f"Person-level identifier or absolute path in {path.name}")
        if path.suffix == ".csv":
            require(not set(pd.read_csv(path, nrows=0).columns).intersection(
                {"scan_id", "participant_id", "patient_id", "study_code"}), "A private table entered public outputs")
        if path.suffix == ".md":
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                require("://" in link or (path.parent / link).is_file(), f"A report link is broken: {link}")
    require(all((PRIVATE / "figures" / filename).is_file() for filename in manifest["private_outputs"]["figures"]),
            "Missing private paired figure")


def main():
    verify_base()
    verify_registry()
    manifest = read_json(PUBLIC / "Transition_Run_Manifest.json")
    require(manifest["source_commit"] == BASE and manifest["protocol"] == CFG, "Run protocol differs")
    for label, path in (("protocol", CONFIG), ("registry", REGISTRY), ("plan", ROOT / "ANALYSIS_PLAN.md")):
        require(digest(path, normalize=True) == manifest[f"{label}_sha256"], f"{label} receipt differs")
    for category, parent in (("previous_inputs", REPO), ("scripts", ROOT / "scripts"), ("public_outputs", ROOT)):
        for relative, expected in manifest[category + "_sha256"].items():
            path = parent / relative
            require(digest(path, normalize=path.suffix != ".png") == expected, f"Changed receipt: {path.name}")

    hashes = verify_inputs(read_json(PRIVATE / "Transition_Input_Hashes.json")["file_sha256"])
    require(len(hashes) == manifest["verified_input_files"] and collection_digest(hashes) == manifest["input_collection_sha256"],
            "Frozen input receipt changed")
    scans, matrix = verify_measurements()
    group = pd.concat([table(PUBLIC, "Transition_Probability_WEiDA"),
                       table(PUBLIC, "Transition_Probability_ExitConditional")], ignore_index=True)
    comparisons = table(PUBLIC, "Transition_Group_Comparisons")
    corrections = table(PUBLIC, "Transition_Statistical_Corrections")
    clinical = table(PUBLIC, "Transition_Clinical_Associations")
    m4, reference, quality = (table(PUBLIC, name) for name in (
        "M4_Transition_Summary", "Transition_HC_Reference", "Transition_QC_Summary",
    ))
    verify_group_means(matrix, group, comparisons)
    verify_pvalues(comparisons, corrections, clinical)
    verify_clinical(matrix, clinical)
    same_frame(quality, describe_quality(scans))
    same_frame(reference, hc_reference(group, comparisons))
    same_frame(m4, summarize_m4(group, comparisons, scans, matrix))
    v6_parity(matrix, pd.read_csv(ARCHIVED_TRANSITIONS, float_precision="round_trip"))
    text = report_text(group, comparisons, clinical, m4, reference, quality, manifest["archived_v6_probability_parity"])
    require(text == (ROOT / "reports/A3_Transition_Dynamics_Report.md").read_text(encoding="utf-8"),
            "Report differs from the public result tables")
    files = manifest["public_outputs_sha256"]
    verify_figures(files, manifest["plot_metadata_key"], group, comparisons)
    verify_publication(manifest, files, matrix)
    print(f"Verified {len(scans)} scans, {int(scans.n_adjacent.sum())} transitions, 36+36+80 test slots, and all figure and report values.")


if __name__ == "__main__":
    main()
