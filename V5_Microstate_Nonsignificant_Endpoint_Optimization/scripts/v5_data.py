"""Verify V4, register its evidence, and read its participant features and clocks."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.linalg import helmert


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
WORKSPACE = REPO.parent
V4_NAME = "V4_Microstate_Statistical_Significance"
V4 = REPO / V4_NAME
LOCAL_V4 = WORKSPACE / "WEiDA-DBS-V4" / V4_NAME
RELIABILITY = LOCAL_V4.parent / "V4_BOLD_Microstate_Reliability"
CFG = json.loads((HERE / "configs/protocol.json").read_text(encoding="utf-8"))
RESULT = HERE / "results"
STATE = range(1, 5)
COORD = helmert(4)
TR = CFG["cohort"]["tr"]
sys.path[:0] = [str(LOCAL_V4 / "scripts")]
os.environ["WEIDA_105_ROOT"] = str(WORKSPACE)
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / "private/numba"))


def dump(obj, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, **kwargs) + "\n", encoding="utf-8", newline="\n")


def csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n", encoding="utf-8-sig")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args])


def frozen_bytes(path):
    body = path.read_bytes()
    return body if path.suffix.lower() == ".png" else body.replace(b"\r\n", b"\n")


def verify_v4(export=True):
    base = CFG["v4_commit"]
    assert git("rev-parse", f"{CFG['v4_tag']}^{{commit}}").decode().strip() == base
    assert not git("status", "--porcelain", "--", V4_NAME)
    manifest = json.loads((V4 / "workflow/Freeze_Manifest.json").read_text(encoding="utf-8"))
    found = git("ls-tree", "--name-only", "-r", "-z", base, "--", V4_NAME).decode().strip("\0").split("\0")
    by_name = {row["path"]: row["sha256"] for row in manifest["public_files"]}
    assert len(found) == 72 and len(by_name) == 71
    receipt = json.loads((V4 / "results/Final_Verification.json").read_text(encoding="utf-8"))["hashes"]
    assert len(receipt) == 41
    hashes = []
    for relative in found:
        name = relative.removeprefix(V4_NAME + "/")
        committed = sha256(frozen_bytes(REPO / relative)).hexdigest()
        if name in by_name:
            assert committed == by_name[name], name
        history = receipt.get(name)
        if history:
            local = (LOCAL_V4 / name).read_bytes()
            assert sha256(local).hexdigest() == history, name
            assert sha256(frozen_bytes(LOCAL_V4 / name)).hexdigest() == committed, name
        hashes.append(dict(path=name, canonical_sha256=committed, original_run_sha256=history or ""))
    if not export:
        return hashes
    subprocess.run([sys.executable, str(V4 / "scripts/verify_freeze.py"), "--workspace", str(WORKSPACE)], check=True)
    csv(pd.DataFrame(hashes), HERE / "manifests/V4_SHA256.csv")
    dump(dict(commit=base, tag=CFG["v4_tag"], branch="experiment/v4-microstate-statistical-significance",
              original_run_hashes_verified=41, committed_file_hashes_verified=len(hashes),
              report_hashes={d["path"]: d["canonical_sha256"] for d in hashes if d["path"].startswith("reports/")},
              result_hashes={d["path"]: d["canonical_sha256"] for d in hashes if d["path"].startswith("results/")}),
         HERE / "V4_FROZEN_REFERENCE.json")
    register_v4()
    return hashes


def register_v4():
    paths = ("All_Scalar_Results", "Coverage_ILR_Results", "Shared_PD_Direction_Results")
    data = pd.concat([pd.read_csv(V4 / "results" / f"{name}.csv").assign(v4_source=f"{name}.csv") for name in paths], ignore_index=True)
    data["v4_primary_composition_holm"] = data.metric.eq("ilr") & data.permutation_holm_p.lt(.05)
    data["v4_99_scalar_dual_holm"] = data.all_scalar_holm_p.lt(.05) & data.all_scalar_permutation_holm_p.lt(.05)
    data["v4_family_dual_holm_scalar"] = data.joint_holm_support.eq(True)
    data["v4_passed_family_bh"] = data.fdr_p.lt(.05)
    assert len(data) == 136
    assert data.v4_primary_composition_holm.sum() == 1 and data.v4_99_scalar_dual_holm.sum() == 4
    assert data.v4_family_dual_holm_scalar.sum() == 12
    assert (data.loc[data.family.eq("Transition Probability"), "v4_family_dual_holm_scalar"] == False).all()
    data["v4_commit"] = CFG["v4_commit"]
    csv(data, HERE / "V4_FROZEN_SIGNIFICANCE_REGISTRY.csv")


def counts_episodes(labels):
    keep = labels >= 0
    adj = keep[:-1] & keep[1:]
    transition = np.bincount(4 * labels[:-1][adj] + labels[1:][adj], minlength=16).reshape(4, 4)
    starts = np.r_[0, np.flatnonzero(np.diff(labels)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    episodes = [[] for _ in STATE]
    internal = [[] for _ in STATE]
    for a, b in zip(starts, ends):
        if labels[a] < 0:
            continue
        episodes[labels[a]].append((b-a) * TR)
        if a > 0 and b < len(labels) and keep[a-1] and keep[b]:
            internal[labels[a]].append((b-a) * TR)
    return transition, episodes, internal


def features(labels):
    x, episodes, internal = counts_episodes(labels)
    frames = np.bincount(labels[labels >= 0], minlength=4)
    exp = frames.sum() * TR
    cov = frames / frames.sum()
    count = x.sum(axis=1)
    pr = np.divide(x, count[:, None], out=np.full((4, 4), np.nan), where=count[:, None] > 0)
    row = dict(exposure=exp, valid_frames=frames.sum())
    for i in STATE:
        row.update({f"coverage_{i}": cov[i-1], f"count_{i}": len(episodes[i-1]), f"source_count_{i}": count[i-1]})
        for name, values in [("window", episodes[i-1]), ("edge", internal[i-1])]:
            for q, p in [("median", .50), ("p75", .75)]:
                row[f"dwell_{name}_{q}_{i}"] = np.quantile(values, p) if values else np.nan
    for i, j in np.ndindex(4, 4):
        row[f"count_{i+1}_{j+1}"] = x[i, j]
        row[f"transition_{i+1}_{j+1}"] = pr[i, j]
    row.update({f"ilr_{i+1}": value for i, value in enumerate(COORD @ np.log(frames + CFG["jeffreys"]))})
    row.update({f"trans_ilr_{i+1}": value for i, value in enumerate((np.log(x + CFG["jeffreys"]) @ COORD.T).ravel())})
    if not np.all(count > 0):
        for i in range(12):
            row[f"trans_ilr_{i+1}"] = np.nan
    return row


def load_data(penalty=None, save_private=True):
    # Archived V4 seed labels, templates and feature values are opened read-only.
    from micro_inputs import match_banks, temporal_labels

    prior = pd.read_csv(LOCAL_V4 / "private/Subject_Level_Microstate_Features.csv").sort_values("scan")
    with np.load(RELIABILITY / "private/V4_Maps.npz") as bank:
        match = match_banks(bank["G1"])
    assert match["stable"].all() and len(prior) == prior.scan.nunique() == 65
    assert prior.subject.nunique() == 40
    assert prior.groupby(["subject", "condition"]).size().max() == 1 and prior.groupby("subject").fold.nunique().max() == 1
    assert prior.condition.value_counts().to_dict() == {"OFF":25, "ON":25, "HC":15}
    assert set(prior.loc[prior.condition.eq("ON"), "subject"]) == set(prior.loc[prior.condition.eq("OFF"), "subject"])
    rows, mismatches = [], 0
    for r in prior.itertuples():
        with np.load(RELIABILITY / "private/frames/A3" / f"{r.scan}.npz") as saved:
            lab = saved["labels"]
            order = np.argsort(match["order"][r.fold])
            if penalty is not None:
                gfp = saved["gfp"]
                center = np.mean(gfp[gfp>0]**2)
                revised = temporal_labels(saved["corr"], gfp, lab >= 0, penalty, normalization=center)
                mismatches += int(np.sum(revised != lab))
                lab = revised
            mapped = np.where(lab < 0, -1, order[lab])
        result = features(mapped)
        assert result["valid_frames"] == r.valid_frames
        if penalty is None:
            keys = [f"coverage_{i}" for i in STATE] + [f"ilr_{i}" for i in range(1, 4)]
            keys += [f"transition_{i}_{j}" for i in STATE for j in STATE]
            np.testing.assert_allclose([result[k] for k in keys], [getattr(r, k) for k in keys], atol=1e-12, equal_nan=True)
            np.testing.assert_allclose([result[f"count_{i}"] / result["exposure"] for i in STATE],
                                       [getattr(r, f"occurrence_{i}") for i in STATE], atol=1e-12)
        rows.append({**{k: getattr(r, k) for k in ("scan", "subject", "condition", "fd", "fold")}, **result})
    data = pd.DataFrame(rows)
    assert (data.filter(regex=r"^coverage_\d$").sum(axis=1) - 1.).abs().max() < 1e-12
    if penalty == CFG["baseline"]["lambda"]:
        assert mismatches == 0
    if save_private:
        filename = "frozen_participant_features.csv" if penalty is None else f"lambda_{penalty:g}_participant_features.csv"
        csv(data, HERE / "private" / filename)
    return data


if __name__ == "__main__":
    verify_v4()
    d = load_data()
    dump(dict(participants=int(d.subject.nunique()), scans=len(d), paired_patients=25,
              recorded_valid_frames=int(d.valid_frames.sum()), missing_transition_scan_rows=int(d.trans_ilr_1.isna().sum()),
              frozen_coverage_transition_occurrence_parity=True), HERE / "manifests/Analysis_Input_Audit.json")
