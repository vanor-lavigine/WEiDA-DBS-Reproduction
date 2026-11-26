"""The authorized input index and the frozen V6/V7 feature bridge."""

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent
PRIVATE = PROJECT / "private"
RESULTS = PROJECT / "results"
PROTOCOL = PROJECT / "configs/protocol.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


CFG = read_json(PROTOCOL)
BASE = CFG["v7_commit"]
TOL = CFG["tolerance"]
PAIR_KEYS = ["participant_id", "on_scan_id", "off_scan_id"]
ARCHIVE = "WEiDA4-1_atlas1_group0"
OLD_COLUMNS = ("gev_1", "duration_4", "occurrence_1", "switching_rate", "fd", "mean_std_dvars")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fingerprint(paths):
    return {key: {"path": str(path.resolve()), "sha256": digest(path)} for key, path in paths.items()}


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n", na_rep="NA")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
                                   encoding="utf-8")


def verify_frozen_worktree():
    changes = git("diff", "--name-only", BASE).splitlines()
    require(all(path.startswith(PROJECT.name + "/") for path in changes), "A historical tracked file differs from V7")
    ancestry = git("merge-base", BASE, "HEAD").strip()
    require(ancestry == BASE, "The working branch does not descend from the requested V7 baseline")
    v6 = REPO / "V6_DBS_ON_OFF_Microstate_Dynamics/configs/protocol.json"
    v7 = REPO / "BOLD_Microstate_Mechanistic_Decomposition/configs/protocol.json"
    require(read_json(v6)["baseline"] == read_json(v7)["baseline"] == CFG["baseline"], "Frozen imaging definition differs")
    require(read_json(v7)["frozen_v6"] == CFG["v6_commit"], "Frozen V6 release differs")


def unchanged_index(entry, role):
    source = Path(entry["file"]).resolve(strict=True)
    require(digest(source) == entry["sha256"], f"An input differs from its frozen index ({role})")
    return source


def discover_sources(workspace, score_dir, v7_root, prior_index):
    historical = read_json(prior_index)["inputs"]
    paths = {f"score_{k}": (score_dir / leaf).resolve(strict=True) for k, leaf in CFG["source_names"].items()}
    for key in ("v6_manifest", "v6_cohort", "v6_scans", "v6_differences", "archived_score_links", "archived_score_pairs",
                "archived_score_domains", "previous_behavior_hashes"):
        paths[key] = unchanged_index(historical[key], key)
    behavior = read_json(paths["previous_behavior_hashes"])["inputs"]
    paths["acquisition_workbook"] = unchanged_index(behavior["imaging_visit_workbook"], "acquisition workbook")
    archive_dir = workspace / "WEiDA-raw-audit-private-20261116"
    paths["original_acquisition_capture"] = archive_dir / "source105_evidence.json"
    paths["original_acquisition_capture_check"] = archive_dir / "source105_dbs_roster_check.json"
    paths["frozen_aal_transfer"] = workspace / "STN-DBS_105_AAL78_20261117/_transfer/source-sha256.json"
    paths["v7_paired"] = (v7_root / "private/results/Paired_Decompositions.csv").resolve(strict=True)
    paths["prior_index"] = prior_index
    for path in paths.values():
        require(path.is_file(), f"Missing input {path.name}")
    return paths


@dataclass
class Imaging:
    roster: pd.DataFrame
    paired: pd.DataFrame
    old_differences: pd.DataFrame
    scans: pd.DataFrame


def load_imaging(paths):
    expected = CFG["cohort"]["original_pd_pairs"]
    roster = pd.read_csv(paths["v6_manifest"], usecols=PAIR_KEYS).set_index("participant_id", verify_integrity=True).sort_index()
    require(len(roster) == expected, "Unexpected number of frozen PD pairs")
    require(roster[["on_scan_id", "off_scan_id"]].stack().is_unique, "Frozen scan keys are reused")
    deltas = pd.read_csv(paths["v6_differences"], usecols=["participant_id", *OLD_COLUMNS])
    deltas = deltas.set_index("participant_id", verify_integrity=True).sort_index()
    v7 = pd.read_csv(paths["v7_paired"], usecols=["participant_id", "m1_gev_delta_pp", "delta_duration_4",
                      "m1_shapley_w_pp", "m1_shapley_f_pp", "fd", "mean_std_dvars"])
    v7 = v7.set_index("participant_id", verify_integrity=True).sort_index()
    require(v7.index.equals(roster.index) and deltas.index.equals(roster.index), "V6/V7 pair rosters do not agree")
    scans = pd.read_csv(paths["v6_scans"], usecols=["scan", "subject", "condition", "fold", *OLD_COLUMNS])
    cohort = pd.read_csv(paths["v6_cohort"], usecols=["scan", "subject", "condition", "fold", *OLD_COLUMNS])
    require(cohort.condition.value_counts().to_dict() == {"ON": expected, "OFF": expected, "HC": CFG["cohort"]["hc"]},
            "Incorrect frozen scan inventory")
    require(len(cohort) == CFG["cohort"]["original_scans"] and cohort.scan.is_unique, "Invalid V6 cohort scan keys")
    subset = cohort.loc[cohort.condition.isin(("ON", "OFF"))].sort_values("scan").reset_index(drop=True)
    pd.testing.assert_frame_equal(scans.sort_values("scan").reset_index(drop=True), subset, check_exact=True)
    side = {}
    for cond in ("ON", "OFF"):
        v = scans.loc[scans.condition.eq(cond)].set_index("subject", verify_integrity=True).sort_index()
        require(v.index.equals(roster.index), "V6 scan and pair participants disagree")
        np.testing.assert_array_equal(v.scan, roster[f"{cond.lower()}_scan_id"])
        side[cond] = v
    np.testing.assert_array_equal(side["ON"].fold, side["OFF"].fold)
    cols = list(OLD_COLUMNS)
    np.testing.assert_allclose(side["ON"][cols] - side["OFF"][cols], deltas[cols], rtol=0, atol=TOL, equal_nan=True)
    for new, old, scale in (("m1_gev_delta_pp", "gev_1", 100), ("delta_duration_4", "duration_4", 1),
                             ("fd", "fd", 1), ("mean_std_dvars", "mean_std_dvars", 1)):
        np.testing.assert_allclose(v7[new], deltas[old] * scale, rtol=0, atol=TOL, equal_nan=True)
    np.testing.assert_allclose(v7.m1_shapley_w_pp + v7.m1_shapley_f_pp, v7.m1_gev_delta_pp, rtol=0, atol=TOL,
                               equal_nan=True)
    return Imaging(roster, v7, deltas, scans)


def prior_association_module(workspace):
    """Import the historical implementation under its original protocol without writing to it."""
    os.environ["WEIDA_V6_DATA_ROOT"] = str(workspace)
    os.environ["NUMBA_CACHE_DIR"] = str(PRIVATE / "numba")
    scripts = REPO / "BOLD_Microstate_Behavior_Association/scripts"
    sys.path.insert(0, str(scripts))
    import association
    from io_sources import ENDPOINTS

    require(association.SEED == CFG["seed"] and association.PERMS == CFG["permutations"], "Historical resampling differs")
    return association, ENDPOINTS
