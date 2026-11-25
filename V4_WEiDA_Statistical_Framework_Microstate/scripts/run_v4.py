"""Run V4 against the read-only V3 checkout and local 65-scan WEiDA archive."""

import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import scipy
import statsmodels

from features import ARCHIVES, cross_method_matching, microstate_features, read_v3_archive, weida_features
from figures import write_figures
from outputs import comparison_table, report_comparison, report_micro
from statistics_v4 import Resampling, infer, measures, toward_hc


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
V3 = ROOT / "WEiDA-DBS-V3" / "V3_BOLD_Microstate_Method_Optimization"


def save_csv(data, directory, name):
    path = directory / f"{name}.csv"
    scratch = path.with_suffix(".csv.tmp")
    data.to_csv(scratch, index=False, encoding="utf-8-sig", lineterminator="\n")
    scratch.replace(path)


def group_rows(data, method):
    rows = []
    for metric in measures(method):
        for condition, block in data.groupby("condition", sort=False):
            x = block[metric.metric].dropna()
            rows.append(dict(method=method, **vars(metric), condition=condition, n=len(x), missing=len(block) - len(x),
                             mean=x.mean(), sd=x.std(), median=x.median()))
    return pd.DataFrame(rows)


def verify_group_archive(archive, calculated, cfg, states_qualified):
    primary = archive.experiment.eq(cfg["v3_state_archive"]) & states_qualified
    overall = archive.experiment.eq(cfg["v3_method"]) & archive.metric.isin(
        ["global_pooled_duration", "global_occurrence", "switching_rate"])
    old = archive.loc[archive.fit.eq(cfg["v3_fit"]) & (primary | overall)]
    keys = ["metric", "condition"]
    paired = calculated.merge(old, on=keys, suffixes=("", "_v3"), validate="one_to_one")
    for col in ["n", "missing", "mean", "sd", "median"]:
        np.testing.assert_allclose(paired[col], paired[f"{col}_v3"], atol=1e-12, rtol=1e-9, equal_nan=True)
    return len(paired)


def main():
    sys.path.insert(0, str(V3 / "scripts"))
    os.environ["WEIDA_105_ROOT"] = str(ROOT)
    from input_data import FROZEN, TR, digest, load_scans

    cfg_path = HERE / "configs" / "protocol.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if cfg["K"] != 4 or cfg["TR_seconds"] != TR:
        raise ValueError("Frozen V4 expects K=4 and TR=2.1 s")
    gate_file = V3 / "manifests" / "provenance_gate.json"
    gate = json.loads(gate_file.read_text(encoding="utf-8-sig"))
    if not gate["pd_comparable"] or not gate["hc_comparable"]:
        raise ValueError("V3 input evidence has not passed the provenance gate")
    scans = load_scans()
    frozen = pd.read_csv(FROZEN)
    archives = read_v3_archive(V3)
    ms, banks, folds, compatible = microstate_features(V3, scans, cfg)
    print(f"V3 state labels and features verified on {len(ms)} scans. Cross-fold alignment: {compatible}", flush=True)
    source_error = None
    weida_paths = []
    match = pd.DataFrame(columns=["fold", "microstate", "weida_state", "spatial_r", "abs_spatial_r", "hungarian_selected",
                                  "bidirectional_margin", "pair_pass", "same_order_in_all_folds", "state_match_pass"])
    mapping = {}
    try:
        weida, bank, scan_map, centroid_error, weida_paths = weida_features(ROOT, scans, frozen, cfg)
    except (AssertionError, ValueError, OSError, StopIteration) as err:
        source_error = f"{type(err).__name__}: archived WEiDA scan/feature verification failed"
        print(f"{source_error}: {err}", flush=True)
        weida, scan_map, centroid_error = pd.DataFrame(), pd.DataFrame(), None
    if source_error is None:
        match, mapping = cross_method_matching(banks, bank, compatible, cfg)
        print(f"WEiDA scan and frame mapping verified. Cross-method state correspondence: {bool(mapping)}", flush=True)

    resampling = Resampling(cfg["seed"], cfg["permutations"], cfg["bootstrap_draws"])
    all_v3 = archives["All_Methods_Comparison"]
    v3_matched = all_v3.loc[all_v3.experiment.isin([cfg["v3_method"], cfg["v3_state_archive"]]) & all_v3.fit.eq(cfg["v3_fit"])]
    v3_scalar = v3_matched.loc[v3_matched.model.eq("CR1")]
    micro = infer(ms, "Microstate", cfg, resampling, v3_scalar)
    print("Microstate participant-level analyses complete.", flush=True)
    w = infer(weida, "WEiDA", cfg, resampling, available=source_error is None)
    close = toward_hc(ms, micro, cfg, resampling)
    cross = comparison_table(micro, w, mapping, source_error is None)
    micro_group = group_rows(ms, "Microstate")
    n_v3_group = verify_group_archive(archives["Group_Level_Summary"], micro_group, cfg, compatible)
    group = pd.concat([micro_group, group_rows(weida, "WEiDA")], ignore_index=True) if source_error is None else micro_group

    dirs = [HERE / name for name in ("results", "figures", "private")]
    for directory in dirs:
        directory.mkdir(exist_ok=True)
    results, figures, private = dirs
    core = micro[micro.scope.eq("primary")]
    audits = {
        "config_sha256": digest(cfg_path), "protocol_sha256": digest(HERE / "reports" / "WEIDA_STATISTICAL_PROTOCOL.md"),
        "cohort": {"participants": ms.subject.nunique(), "HC": 15, "PD": 25, "ON": 25, "OFF": 25, "scans": len(ms),
                   "all_frozen_roi_file_hashes_verified": True, "effective_frames": int(frozen.frames.sum()),
                   "TR_seconds": TR, "atlas": "AAL78"},
        "v3_state_correspondence": bool(compatible), "v3_group_summary_reproduced_rows": n_v3_group,
        "v3_fd_reproduced_rows": int(micro.v3_fd_archive_status.eq("ok").sum()),
        "source_table_rows": {key: len(df) for key, df in archives.items()} | {"Subject_Level_Features": len(pd.read_csv(
            V3 / "private" / "Subject_Level_Features.csv", usecols=["experiment"]))},
        "weida_available": source_error is None, "weida_unavailable_reason": source_error,
        "weida_exact_scan_mapping_rows": len(scan_map), "weida_centroid_reconstruction_max_error": centroid_error,
        "cross_method_state_match": bool(mapping), "permutations": resampling.permutations,
        "bootstrap_draws": resampling.bootstrap, "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                                               "pandas": pd.__version__, "scipy": scipy.__version__,
                                                               "statsmodels": statsmodels.__version__},
    }
    checksums = [FROZEN, gate_file, V3 / "manifests" / "ROI_Axis.csv", V3 / "private" / "Subject_Level_Features.csv",
                 V3 / "results" / "Group_Templates.csv", *[V3 / "results" / f"{name}.csv" for name in ARCHIVES], *weida_paths]
    papers = list(ROOT.glob("*/s41531-026-01354-3.pdf"))
    if len(papers) == 1:
        checksums.extend(papers)
    audits["input_sha256"] = {path.relative_to(ROOT).as_posix(): digest(path) for path in checksums}
    backfits = V3 / "private" / "backfits" / cfg["v3_method"] / cfg["v3_fit"]
    save_csv(pd.DataFrame([{"scan": s, "sha256": digest(backfits / f"{s}.npz")} for s in sorted(ms.scan)]), private,
             "V3_Backfit_Hashes")

    for contrast, file in zip(cfg["contrasts"], ["HC_vs_OFF", "HC_vs_ON", "ON_vs_OFF"]):
        save_csv(core.loc[core.contrast.eq(contrast)], results, file)
    base = ["method", "scope", "family", "state", "metric", "contrast", "status", "family_size", "n_a", "n_b", "n_pairs",
            "effect_size", "ci95_low", "ci95_high", "raw_p", "fdr_p", "holm_p", "permutation_p"]
    save_csv(core.loc[core.family.eq("Transition Probability")], results, "Transition_Results")
    for correction in ("FDR", "Holm"):
        chosen = correction.lower() + "_p"
        extras = [col for col in core if col.endswith(chosen) and col not in base]
        save_csv(core[base + extras], results, f"{correction}_Results")
    columns = [col for col in core if ("permutation" in col or col.startswith("fd_cr1")) and col not in base]
    save_csv(core[base + columns], results, "Permutation_Results")
    for name, frame in (
        ("Microstate_Results", micro), ("WEiDA_Results", w), ("Cross_Method_Comparison", cross),
        ("State_Matching", match), ("V3_Fold_Matching", folds), ("Towards_HC", close),
        ("Group_Level_Summary", group), ("V3_FD_Adjusted_Archive", v3_matched),
    ):
        save_csv(frame, results, name)
    save_csv(ms, private, "Microstate_Subject_Features")
    if source_error is None:
        save_csv(scan_map, private, "Participant_Scan_Mapping")
        save_csv(weida, private, "WEiDA_Subject_Features")
    (results / "Input_Verification.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_figures(figures, ms, micro, cross, match)
    report_micro(HERE, micro, close, audits, cfg["alpha"])
    report_comparison(HERE, micro, w, cross, match, mapping, audits)
    print(f"Complete: {len(core)} primary microstate contrasts, {len(w)} WEiDA rows, {len(cross)} cross-method rows.", flush=True)


if __name__ == "__main__":
    main()
