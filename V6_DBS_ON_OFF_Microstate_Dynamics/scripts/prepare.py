"""Verify V4/V5 and construct the restricted V6 cohort from frozen scan records."""

from datetime import time
from hashlib import sha256
import subprocess
import sys

import numpy as np
import openpyxl
import pandas as pd
from scipy.linalg import helmert

from v6_common import (CFG, HERE, ILR, LOCAL_V4, PRIVATE, RELIABILITY, REPO, RESULT,
                       STATE, TR, V4, V4_NAME, V5, V5_NAME, WORKSPACE, digest, git, read_json, write_csv, write_json)
from v5_data import counts_episodes


def verify_releases(export=True):
    tag = git("rev-parse", f"{CFG['v4_tag']}^{{commit}}").decode().strip()
    assert tag == CFG["v4_commit"]
    assert git("rev-parse", f"{CFG['v5_commit']}^").decode().strip() == tag
    assert not git("diff", "--name-only", CFG["v5_commit"], "--", V4_NAME, V5_NAME)
    signed_v4 = {x["path"]: x["sha256"] for x in read_json(V4 / "workflow/Freeze_Manifest.json")["public_files"]}
    signed_v5 = read_json(V5 / "logs/Run_Verification.json")["v5_sha256"]
    v4_runtime = read_json(V4 / "results/Final_Verification.json")["hashes"]
    entries = []
    for name, prefix, signed in [(V4_NAME, "v4", signed_v4), (V5_NAME, "v5", signed_v5)]:
        commit = CFG[f"{prefix}_commit"]
        files = git("ls-tree", "-r", "-z", "--name-only", commit, "--", name).decode().strip("\0").split("\0")
        for repo_path in files:
            relative = repo_path.removeprefix(name + "/")
            public = (REPO / repo_path).read_bytes()
            canonical = public if (REPO / repo_path).suffix in (".png", ".pdf") else public.replace(b"\r\n", b"\n")
            expected = sha256(git("show", f"{commit}:{repo_path}")).hexdigest()
            assert sha256(canonical).hexdigest() == expected, repo_path
            if relative in signed:
                assert signed[relative] == expected, repo_path
            if prefix == "v4" and relative in v4_runtime:
                assert digest(LOCAL_V4 / relative) == v4_runtime[relative], relative
            entries.append(dict(release=prefix.upper(), commit=commit, path=relative, canonical_sha256=expected,
                                also_listed_in_release_receipt=relative in signed))
    manifest = pd.DataFrame(entries)
    assert manifest.release.value_counts().to_dict() == {"V4": 72, "V5": 67}
    assert manifest.also_listed_in_release_receipt.sum() == len(signed_v4) + len(signed_v5) and len(v4_runtime) == 41
    if export:
        subprocess.run([sys.executable, str(V4 / "scripts/verify_freeze.py"), "--workspace", str(WORKSPACE)], check=True)
        write_csv(manifest, HERE / "manifests/V4_V5_SHA256.csv")
        write_json(dict(v4_commit=tag, v5_commit=CFG["v5_commit"], v4_tag=CFG["v4_tag"],
                        v6_branch=CFG["branch"], v4_git_files=72, v4_original_run_hashes=41,
                        v5_git_files=67, v5_receipt_hashes=len(signed_v5), history_bytes_verified=True),
                   HERE / "V4_V5_FROZEN_REFERENCE.json")
    return manifest


def summarize_labels(labels, corr, gfp, mapping):
    """Decode already matched numbers and verify the original metric definitions."""
    keep = labels >= 0
    frozen = np.where(keep, mapping[labels], -1)
    transition, episodes, edge = counts_episodes(frozen)
    counts = np.bincount(frozen[keep], minlength=4)
    time_seconds = TR * keep.sum()
    starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
    ends = np.flatnonzero(keep & ~np.r_[keep[1:], False])
    result = dict(valid_frames=int(keep.sum()), valid_blocks=len(starts),
                  leading_unlabeled=int(starts[0]), trailing_unlabeled=int(len(keep)-1-ends[-1]),
                  interior_unlabeled=int(ends[-1]-starts[0]+1-keep.sum()),
                  switching_rate=(transition.sum()-np.trace(transition))/time_seconds)
    weights = gfp[keep]**2
    fit = np.clip(corr[keep, labels[keep]], 0, 1)
    mapped = frozen[keep]
    for i, state in enumerate(STATE):
        which = mapped == i
        result[f"coverage_{state}"] = counts[i] / keep.sum()
        result[f"duration_{state}"] = TR * counts[i] / len(episodes[i]) if episodes[i] else np.nan
        result[f"occurrence_{state}"] = len(episodes[i]) / time_seconds
        result[f"gev_{state}"] = np.sum(weights[which] * fit[which]**2) / weights.sum()
        result[f"dispersion_{state}"] = np.mean(np.sqrt(2*(1-fit[which]))) if which.any() else np.nan
        source = transition[i].sum()
        result[f"source_{state}"] = source
        for j, destination in enumerate(STATE):
            result[f"count_{state}_{destination}"] = transition[i,j]
            result[f"transition_{state}_{destination}"] = transition[i,j]/source if source else np.nan
        for name, values in [("window", episodes[i]), ("edge", edge[i])]:
            for stat, quantile in [("median", .5), ("p75", .75)]:
                result[f"dwell_{name}_{stat}_{state}"] = np.quantile(values, quantile) if values else np.nan
        if state == 4:
            result["dwell_edge_mean_4"] = np.mean(edge[i]) if edge[i] else np.nan
    result.update(dict(zip(ILR, helmert(4) @ np.log(counts+.5))))
    return result


def frozen_table():
    original_path = LOCAL_V4 / "private/Subject_Level_Microstate_Features.csv"
    original = pd.read_csv(original_path)
    assert original.scan.nunique() == len(original) == CFG["cohort"]["scans"]
    assert original.subject.nunique() == 40 and original.groupby("subject").fold.nunique().eq(1).all()
    assert original.condition.value_counts().to_dict() == {"ON":25,"OFF":25,"HC":15}
    clock = pd.read_csv(WORKSPACE / "WEiDA_STN25_HC15_K4_20261117/input_manifest.csv")
    assert clock.scan.nunique() == len(clock) == 65 and set(clock.scan) == set(original.scan)
    clock = clock.set_index("scan")
    np.testing.assert_array_equal(original.condition, clock.loc[original.scan, "condition"].map({1:"HC",2:"OFF",3:"ON"}))
    pd_mask = original.condition.ne("HC")
    assert np.array_equal(original.loc[pd_mask,"subject"],clock.loc[original.loc[pd_mask,"scan"],"pair_id"])
    template = pd.read_csv(V4 / "results/Fold_Templates.csv")
    assert template.fold.nunique() == 5 and len(template)==20 and template.stable.all()
    frozen_mapping = {}
    for fold, rows in template.groupby("fold"):
        np.testing.assert_array_equal(np.sort(rows.original_state),list(STATE))
        np.testing.assert_array_equal(np.sort(rows.state),list(STATE))
        frozen_mapping[fold] = rows.sort_values("original_state").state.to_numpy()-1
    np.testing.assert_array_equal(frozen_mapping[0],np.arange(4))
    evidence = []
    for row in original.itertuples(index=False):
        with np.load(RELIABILITY / "private/frames/A3" / f"{row.scan}.npz") as arr:
            assert arr["labels"].shape == (CFG["cohort"]["acquired_frames"],)
            assert np.isin(arr["labels"],[-1,0,1,2,3]).all()
            q = summarize_labels(arr["labels"], arr["corr"], arr["gfp"], frozen_mapping[row.fold])
        for key in q.keys() & set(original.columns):
            np.testing.assert_allclose(q[key],getattr(row,key),rtol=0,atol=1e-12,equal_nan=True)
        timing = clock.loc[row.scan]
        assert timing.frames == CFG["cohort"]["acquired_frames"]-timing.source_index_zero_based_start
        evidence.append(dict(scan=row.scan, input_frames=int(timing.frames),
                             input_missing_initial_frames=int(timing.source_index_zero_based_start),
                             **{k:v for k,v in q.items() if k not in original.columns}))
    quality = pd.read_csv(RELIABILITY / "private/scan_quality.csv").rename(columns={"fd":"qc_fd"})
    cohort = (original.merge(pd.DataFrame(evidence), on="scan", validate="one_to_one")
              .merge(quality,on=["scan","condition"],validate="one_to_one"))
    assert len(cohort)==65
    np.testing.assert_allclose(cohort.fd,cohort.qc_fd,atol=1e-14,rtol=0)
    assert (cohort.input_frames>=cohort.valid_frames).all()
    for state in STATE:
        assert cohort[f"source_{state}"].gt(0).all()
        np.testing.assert_allclose(cohort[[f"transition_{state}_{j}" for j in STATE]].sum(axis=1),1,atol=1e-12)
    return cohort.sort_values(["subject","condition"]).reset_index(drop=True), digest(original_path)


def read_visits():
    workbook_path, = WORKSPACE.glob("V100*/结论/**/105原始访视逐人核查.xlsx")
    with workbook_path.open("rb") as file:
        workbook = openpyxl.load_workbook(file, read_only=True, data_only=True)
        visits = pd.DataFrame(workbook["48次访视"].iter_rows(min_row=5, values_only=True)).dropna(subset=[0,1])
        workbook.close()
    keys = visits[11].astype(str).str.extract(r"(?i)(sub-STN(?:on|off)\d+)")[0].str.casefold()
    info = pd.DataFrame(dict(scan_key=keys, condition=visits[1], scan_time=pd.to_datetime(visits[2],errors="coerce"),
                             scoring_time=pd.to_datetime(visits[3],format="mixed",errors="coerce"),
                             medication_hours=pd.to_numeric(visits[6],errors="coerce"), record_key=visits[0]))
    assert info.scan_key.notna().all() and info.scan_key.is_unique
    return info, digest(workbook_path)


def hhmmss(value):
    try:
        t = time.fromisoformat(str(value))
        return t.hour*3600+t.minute*60+t.second+t.microsecond/1e6
    except ValueError:
        return np.nan


def audit_pairs(cohort):
    visits, workbook_hash = read_visits()
    pd_scans = cohort.loc[cohort.condition.ne("HC")].copy()
    pd_scans["scan_key"] = pd_scans.scan.str.casefold()
    joined = pd_scans.merge(visits, on=["scan_key","condition"], how="left",validate="one_to_one")
    assert joined.groupby("subject").condition.nunique().eq(2).all()
    assert joined.groupby("subject").record_key.nunique().le(1).all()
    assert len(joined)==50 and joined.subject.nunique()==25
    source_path = WORKSPACE / "WEiDA-raw-audit-private-20261116/source105_inventory.json"
    source = read_json(source_path)
    keys = set(joined.scan_key.str.removeprefix("sub-"))
    sidecars = {r["subject_label"].lower().removeprefix("sub-"): r["json_metadata"] for r in source["scan_records"]
                if r["modality"]=="bold" and r["subject_label"].lower().removeprefix("sub-") in keys}
    assert len(sidecars)==50
    sidecar_t = joined.scan_key.str.removeprefix("sub-").map(lambda x: hhmmss(sidecars[x].get("AcquisitionTime")))
    dc = joined.scan_time.dt
    delta = sidecar_t-(dc.hour*3600+dc.minute*60+dc.second+dc.microsecond/1e6)
    fields = {"scan":"scan_id", "valid_frames":"valid_frames", "input_frames":"input_frames",
              "input_missing_initial_frames":"input_missing_initial_frames", "fd":"mean_fd", "mean_raw_dvars":"dvars",
              "mean_std_dvars":"std_dvars", "scan_time":"scan_time", "medication_hours":"medication_hours"}
    pieces = []
    for condition in ["ON","OFF"]:
        rows = joined.loc[joined.condition.eq(condition)].set_index("subject")[list(fields)].sort_index()
        pieces.append(rows.rename(columns={key:f"{condition.lower()}_{value}" for key,value in fields.items()}))
    pairs = pd.concat(pieces, axis=1).rename_axis("participant_id").reset_index()
    assert len(pairs)==25 and pairs[["on_scan_id","off_scan_id"]].notna().all().all()
    interval = (pairs.off_scan_time-pairs.on_scan_time).dt.total_seconds()/60
    pairs["scan_order"] = np.select([interval.gt(0), interval.lt(0), interval.eq(0)],
                                    ["ON before OFF","OFF before ON","time tie"], default="unverified")
    pairs["scan_interval"] = interval.abs()
    meds = pairs[["on_medication_hours","off_medication_hours"]].notna().all(axis=1)
    pairs["medication_status"] = np.where(meds,"both elapsed intervals recorded","unverified")
    numeric = lambda col: {"n":int(col.notna().sum()),"mean":float(col.mean()),"median":float(col.median()),
                           "min":float(col.min()),"max":float(col.max())}
    metrics = {x:numeric(pairs[f"on_{x}"]-pairs[f"off_{x}"]) for x in ("valid_frames","mean_fd","dvars","std_dvars")}
    counts = dict(total_pairs=len(pairs), verified_pairs=int(interval.notna().sum()), on_first=int(interval.gt(0).sum()),
                  off_first=int(interval.lt(0).sum()), same_day=int(pairs.on_scan_time.dt.normalize().eq(pairs.off_scan_time.dt.normalize()).sum()),
                  unknown_order=int(interval.isna().sum()), verified_medication_pairs=int(meds.sum()),
                  documented_scoring_timestamps=int(joined.scoring_time.notna().sum()), interval_minutes=numeric(interval.abs()),
                  interval_quartiles_minutes=[float(x) for x in interval.abs().quantile([.25,.75])],
                  matching_basis="Exact original archived scan key and condition, checked against both frozen scans in each participant",
                  sidecar_clock_comparisons=int(delta.notna().sum()), sidecar_within_one_second=int(delta.abs().le(1).sum()),
                  scoring_workbook_sha256=workbook_hash, source_inventory_sha256=digest(source_path),
                  stimulus_order_separable_in_verified_pairs=bool(interval.gt(0).any() and interval.lt(0).any()),
                  on_minus_off_quality=metrics)
    write_csv(pairs,PRIVATE / "manifests/ON_OFF_Pair_Manifest.csv")
    write_json(counts,RESULT / "Scan_Order_Audit.json")
    return pairs,counts


def audit_clinical(pairs):
    source_path = WORKSPACE / "WEiDA-raw-audit-private-20261116/source105_score_verify_join_20261123.json"
    source = read_json(source_path)
    export = list(source["runs"].values())
    fields = ("age","med","date","timepoint","visit")
    assert all(not any(x["age_or_medication_or_date_in_raw_table"][k] for k in fields) for x in export)
    eligible_path = PRIVATE / "inputs/Verified_Visit_UPDRS.csv"
    n = 0
    if eligible_path.is_file():
        raw = pd.read_csv(eligible_path)
        paired = raw.merge(pairs, on=["participant_id","on_scan_id","off_scan_id"],validate="one_to_one")
        assert len(paired)==len(raw), "A clinical row does not match the frozen ON/OFF pair"
        paired = paired.loc[paired["visit_evidence"].isin(["direct_visit_key","confirmed_assessment_date"])
                            & paired[["updrs_iii_on","updrs_iii_off"]].notna().all(axis=1)]
        n = len(paired)
        write_csv(paired[["participant_id","updrs_iii_on","updrs_iii_off"]],PRIVATE / "results/Verified_Clinical_Pairs.csv")
    status = "READY" if n >= CFG["clinical"]["minimum_confirmed_pairs"] else "BLOCKED"
    write_json(dict(status=status, archived_export_versions=len(export),
                    archive_matched_scan_rows_per_version=[x["raw_rows_reassigned_from_joint_P_LT"] for x in export],
                    archive_has_assessment_dates=False, archive_has_medication_clock=False, n_eligible_pairs=n,
                    n_frozen_pairs=len(pairs), verified_visit_gate=CFG["clinical"]["minimum_confirmed_pairs"],
                    evidence_sha256=digest(source_path),
                    reason="Archived scores link to participant and DBS condition but lack verified original assessment-to-scan visit evidence" if status=="BLOCKED" else "Verified source links supplied"),
               RESULT / "UPDRS_Input_Audit.json")


def main():
    verify_releases()
    d, prior_hash = frozen_table()
    pairs,order = audit_pairs(d)
    audit_clinical(pairs)
    write_csv(d,PRIVATE / "results/Frozen_Cohort_Features.csv")
    write_csv(d.loc[d.condition.ne("HC")],PRIVATE / "results/ON_OFF_Subject_Features.csv")
    q = []
    for condition, rows in d.groupby("condition"):
        q.append(dict(condition=condition, scans=len(rows), participants=rows.subject.nunique(),
                      min_input_frames=rows.input_frames.min(), max_input_frames=rows.input_frames.max(),
                      min_valid_frames=rows.valid_frames.min(), median_valid_frames=rows.valid_frames.median(),
                      max_valid_frames=rows.valid_frames.max(), total_valid_frames=rows.valid_frames.sum(),
                      scans_with_internal_gaps=rows.interior_unlabeled.gt(0).sum(),
                      input_initial_frames_absent=rows.input_missing_initial_frames.sum(),
                      mean_fd=rows.fd.mean(), mean_dvars=rows.mean_raw_dvars.mean(), mean_std_dvars=rows.mean_std_dvars.mean()))
    write_csv(pd.DataFrame(q),RESULT / "Paired_Data_Quality_Summary.csv")
    n_duplicate = d.scan.duplicated().sum()
    match = read_json(V4 / "workflow/Freeze_Manifest.json")
    archives = {x["role"]: {"count":x["count"],"sha256":x["sha256"]} for x in match["local_archive_collections"]}
    write_json(dict(original_feature_table_sha256=prior_hash, template_sha256=digest(V4 / "results/Fold_Templates.csv"),
                    labels=archives["A3 archived leave-out backfits"], frozen_mapping_reused=True, original_metric_parity=True,
                    scans=len(d), duplicated_scan_ids=int(n_duplicate), patients=len(pairs),
                    valid_trs=int(d.valid_frames.sum()), verified_order_pairs=order["verified_pairs"],
                    protocol_sha256=digest(HERE / "configs/protocol.json"), plan_sha256=digest(HERE / "ANALYSIS_PLAN.md")),
               HERE / "manifests/Input_Audit.json")
    print(f"Audited {len(pairs)} complete PD pairs and 15 controls. Original label/feature parity passed.")


if __name__ == "__main__":
    main()
