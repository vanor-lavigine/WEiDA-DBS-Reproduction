"""Replay fixed computations and verify data, correction scopes and publication boundaries."""

import json
from io import StringIO
import re

import numpy as np
from PIL import Image
from scipy.special import softmax
from statsmodels.stats.multitest import multipletests

from m4_clinical import run_clinical
from m4_dynamics import run_m4_dynamics
from motion_sensitivity import motion_sensitivity
from omnibus import bootstrap_indices, design, p0, paired_cells, signs
from round5_io import (
    BASE, CFG, CONFIG, MODULE, NONDIAGONAL, PRIVATE, RESULTS, ROOT, WORKSPACE, assert_tables,
    collection_digest, confirm_receipts, digest, load_frozen_inputs, read_csv, read_json, verify_base,
)
from round5_reports import reports
from row_ilr import BASIS, p2, transform


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integrity(manifest):
    require(manifest["source_commit"] == BASE and manifest["protocol"] == CFG,
            "Release manifest changed its frozen baseline or protocol")
    require(manifest["protocol_sha256"] == digest(CONFIG, text=True), "Protocol fingerprint changed")
    require(manifest["plan_sha256"] == digest(ROOT / "ANALYSIS_PLAN.md", text=True), "Plan fingerprint changed")
    for name, expected in manifest["scripts_sha256"].items():
        require(digest(ROOT / "scripts" / name, text=True) == expected, f"Source changed: {name}")
    require(set(manifest["scripts_sha256"]) == {path.name for path in (ROOT / "scripts").glob("*.py")},
            "A script is missing its receipt")
    for name, expected in manifest["public_outputs_sha256"].items():
        path = ROOT / name
        require(digest(path, text=path.suffix != ".png") == expected, f"Generated file changed: {name}")


def check_block_resampling(frozen, vector):
    on, off = paired_cells(frozen.matrices, NONDIAGONAL, "p_weida")
    differences = (on-off).to_numpy()
    require(differences.shape == (25, 12), "Whole-vector analysis requires 25 rows and 12 columns")
    require(frozen.pairs.participant_id.is_unique and len(frozen.scans) == 65, "Frozen cohort changed")
    keys = ["participant_id", "on_scan_id", "off_scan_id", "fold"]
    lookup = frozen.pairs.set_index("participant_id")
    private = read_csv(PRIVATE / "results/Omnibus_Patient_Vectors.csv").set_index("participant_id")
    require(private.index.equals(on.index) and set(on.index) == set(lookup.index) and
            frozen.pairs[keys].notna().all().all(), "Private vector keys do not match frozen pairs")
    np.testing.assert_allclose(private, differences, atol=1e-13, rtol=0)
    np.testing.assert_allclose(vector.mean_difference, differences.mean(axis=0), atol=1e-13, rtol=0)
    pattern = signs(len(on), CFG["seed"], CFG["permutations"])
    require(pattern.shape == (10000, 25) and np.unique(pattern, axis=0).shape[0] == 10000 and
            not (pattern == 1).all(axis=1).any(), "Whole-vector permutation schedule changed")
    require(set(np.unique(pattern)) == {-1, 1}, "Permutation flips are invalid")
    raw_draws = pattern[:5, :, None] * differences[None, :, :]
    _, (null_means, null_covs), (bs_means, _) = design(differences, 105)
    np.testing.assert_allclose(raw_draws.mean(axis=1), null_means[:5], atol=1e-13, rtol=0)
    np.testing.assert_allclose(np.stack([np.cov(row, rowvar=False) for row in raw_draws]),
                               null_covs[:5], atol=1e-13, rtol=0)
    indices = bootstrap_indices(len(on), CFG["seed"], CFG["bootstrap_draws"], 105)
    paired_resample = on.to_numpy()[indices[:5]] - off.to_numpy()[indices[:5]]
    np.testing.assert_allclose(paired_resample.mean(axis=1), bs_means[:5], atol=1e-13, rtol=0)


def check_measurement_identities(frozen, m4):
    private = m4["private_pairs"]
    require(len(private) == 25, "M4 private table must retain every patient pair")
    stored = read_csv(PRIVATE / "results/M4_Dynamics_Patient_Pairs.csv")
    assert_tables(stored, private)
    for condition in ("ON", "OFF"):
        np.testing.assert_allclose(private[f"P44_{condition}"] + private[f"exit_rate_{condition}"], 1,
                                   atol=1e-13, rtol=0)
        np.testing.assert_allclose(
            private[f"duration_seconds_{condition}"] * private[f"occurrence_per_minute_{condition}"] / 60,
            private[f"coverage_{condition}"], atol=1e-13, rtol=0,
        )
    np.testing.assert_allclose(private.coverage_duration_component_change + private.coverage_occurrence_component_change,
                               private.coverage_change, atol=1e-13, rtol=0)
    inherited = m4["M4_Entry_Exit_Analysis"].loc[lambda x: x.round4_raw_p.notna()]
    stored_p = frozen.comparisons.set_index(["method", "contrast", "from_state", "to_state"])
    for row in inherited.itertuples(index=False):
        path = row.metric.removesuffix("_exit")
        expected = stored_p.loc[(row.round4_family.removesuffix("_36"),
                                 row.contrast, "M"+path[1], "M"+path[2])]
        np.testing.assert_allclose([row.round4_raw_p, row.round4_fdr_q, row.round4_holm_p],
                                   expected[["raw_p", "fdr_q", "holm_p"]].to_numpy(float), atol=1e-13, rtol=0)


def check_ilr(frozen):
    require(CFG["jeffreys_count"] == .5, "Jeffreys pseudocount changed")
    np.testing.assert_allclose(BASIS @ BASIS.T, np.eye(3), atol=1e-13, rtol=0)
    np.testing.assert_allclose(BASIS.sum(axis=1), 0, atol=1e-13, rtol=0)
    counts = frozen.matrices.pivot(index=["scan_id", "from_state"], columns="to_state", values="count")
    counts = counts.loc[:, CFG["states"]].to_numpy()
    raw, smoothed, z = transform(counts)
    expected = (counts + .5) / (counts.sum(axis=1, keepdims=True) + 2)
    np.testing.assert_allclose(smoothed, expected, atol=1e-13, rtol=0)
    np.testing.assert_allclose(raw.sum(axis=1), 1, atol=1e-13, rtol=0)
    np.testing.assert_allclose(softmax(z @ BASIS, axis=1), smoothed, atol=1e-13, rtol=0)
    np.testing.assert_allclose(z[:, 0], np.log(smoothed[:, 0] / smoothed[:, 1]) / np.sqrt(2),
                               atol=1e-13, rtol=0)
    a, b, c = transform(np.array([[3, 0, 1, 0], [0, 0, 0, 0]]))
    require(np.isfinite(c[0]).all() and np.isnan(a[1]).all() and np.isnan(b[1]).all() and np.isnan(c[1]).all(),
            "Zero count cells and completely unobserved rows were treated alike")


def corrections(tables, manifest):
    motion = tables["Transition_Motion_Sensitivity"]
    base = tables["M4_Baseline_Adjusted_Models"]
    families = {
        "P0_primary": (tables["Omnibus_Transition_Results"], "fdr_q", "holm_p"),
        "P2_ILR": (tables["Transition_ILR_Omnibus"], "fdr_q", "holm_p"),
        "P3_motion": (motion.loc[motion.correction_slot.eq(True)], "bh_q", "holm_p"),
        "P4_historical_spearman": (tables["M4_Clinical_Associations"], "fdr_q", None),
        "P4_joint": (tables["M4_Joint_Response_Analysis"], "fdr_q", "holm_p"),
        "P4_baseline": (base.loc[base.hypothesis.eq(True)], "fdr_q", "holm_p"),
    }
    for name, (table, q, holm) in families.items():
        require(len(table) == CFG["families"][name] and table.family_size.eq(CFG["families"][name]).all(),
                f"Incorrect fixed family size: {name}")
        p = table.raw_p.to_numpy()
        for column, procedure in ((q, "fdr_bh"), (holm, "holm")):
            if column:
                expected = multipletests(np.where(np.isnan(p), 1, p), method=procedure)[1]
                np.testing.assert_allclose(table[column], np.where(np.isnan(p), np.nan, expected),
                                           atol=1e-13, rtol=0)
        expected_counts = {"slots": len(table), "estimable": int(table.raw_p.notna().sum()),
                           "bh_pass": int(table[q].lt(CFG["alpha"]).sum()),
                           "holm_pass": int(table[holm].lt(CFG["alpha"]).sum()) if holm else None}
        require(manifest["families"][name] == expected_counts, f"Manifest correction counts differ: {name}")


def check_release(tables, frozen, manifest, acquisition):
    published = manifest["public_outputs_sha256"]
    reports_fresh = reports(tables, frozen.v6_omnibus, acquisition, frozen.audit)
    for name, value in reports_fresh.items():
        path = ROOT / "reports" / name
        require(path.read_text(encoding="utf-8") == value, f"Report values changed: {name}")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", value):
            require("://" in link or (path.parent / link).is_file(), f"A report link is broken: {link}")
    clinical = reports_fresh["M4_Clinical_Interpretation.md"]
    require("历史评分探索" in clinical and "0/25" in clinical, "Report lost timing and interpretation limits")
    ids = sorted(set(frozen.scans.scan_id) | set(frozen.scans.participant_id))
    private_id = re.compile("|".join(rf"(?<![\w-]){re.escape(x)}(?![\w-])" for x in ids), re.I)
    absolute = re.compile(r"[A-Za-z]:[/\\]|\\\\[^\s\\]+\\")
    for name in [*published, "results/Round5_Run_Manifest.json", "ANALYSIS_PLAN.md", "configs/protocol.json"]:
        path = ROOT / name
        if path.suffix == ".png":
            with Image.open(path) as image:
                meta = image.info[manifest["figure_metadata_key"]]
                image.verify()
            for collection in json.loads(meta):
                data = tables[collection["table"].removesuffix(".csv")]
                for rec in collection["rows"]:
                    keys = [k for k in ("metric", "condition", "contrast") if k in rec]
                    chosen = data
                    for key in keys:
                        chosen = chosen.loc[chosen[key].eq(rec[key])]
                    require(len(chosen) == 1, f"Plot row is missing from the table: {name}")
                    row = chosen.iloc[0]
                    for col, observed in rec.items():
                        if col not in keys:
                            np.testing.assert_allclose(np.nan if observed is None else observed, row[col],
                                                       atol=1e-13, rtol=0, equal_nan=True)
            text = meta
        else:
            text = path.read_text(encoding="utf-8")
            if path.suffix == ".csv":
                forbidden = {"patient_id", "participant_id", "scan_id", "study_code", "scan_id_ON", "scan_id_OFF"}
                require(not forbidden.intersection(text.splitlines()[0].split(",")), "Private CSV schema in release")
        require(not private_id.search(text) and not absolute.search(text), f"Private key or absolute path in {name}")
    require((PRIVATE / "figures/M4_Individual_Paired_Changes.png").is_file(), "Private pairing image is missing")


def main():
    verify_base()
    manifest = read_json(RESULTS / "Round5_Run_Manifest.json")
    integrity(manifest)
    frozen = load_frozen_inputs()
    first, _ = p0(frozen.matrices, frozen.v6_omnibus)
    m4 = run_m4_dynamics(frozen.scans, frozen.matrices, frozen.comparisons,
                         seed=CFG["seed"], bootstrap=CFG["bootstrap_draws"])
    ilr = p2(frozen.matrices)
    motion, order, sensitive, motion_hashes, acquisition = motion_sensitivity(
        frozen.matrices, frozen.pairs, MODULE, WORKSPACE, CFG,
    )
    clinical = run_clinical(frozen.matrices, frozen.scores, frozen.associations, frozen.round4_protocol, CFG)
    tables = first | {k: v for k, v in m4.items() if k != "private_pairs"} | ilr | {
        "Transition_Motion_Sensitivity": motion, "Transition_Order_Sensitivity": order,
    } | {name.removesuffix(".csv"): value for name, value in clinical.items()}
    for name, expected in tables.items():
        # A fresh serialization roundtrip normalizes intentionally empty cells to NA.
        expected = read_csv(StringIO(expected.to_csv(index=False, na_rep="NA", float_format="%.17g")))
        assert_tables(read_csv(RESULTS / (name + ".csv")), expected)
    require(len(sensitive["Transition_Motion_Order_Pairs"]) == 25, "Motion model lost a patient")
    require(acquisition["on_first"] == 24 and acquisition["order_unknown"] == 1 and
            not acquisition["order_effect_identifiable"], "Scan order conclusion differs from archive")
    receipt = confirm_receipts(frozen.hashes, motion_hashes)
    require(receipt == read_json(PRIVATE / "Round5_Input_Hashes.json")["file_sha256"] and
            len(receipt) == manifest["verified_inputs"] and
            collection_digest(receipt) == manifest["input_collection_sha256"], "Input receipts changed")
    require(manifest["motion_order_audit"] == acquisition, "Motion and order audit receipt changed")
    check_block_resampling(frozen, tables["Omnibus_Transition_Vector"])
    check_measurement_identities(frozen, m4)
    check_ilr(frozen)
    corrections(tables, manifest)
    check_release(tables, frozen, manifest, acquisition)
    print(f"Verified {len(frozen.scans)} scans, 25 whole patient vectors, all family slots, {len(receipt)} source hashes, reports and figure values.")


if __name__ == "__main__":
    main()
