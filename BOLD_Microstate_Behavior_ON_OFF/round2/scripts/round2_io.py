"""Use verified frozen inputs without invoking the first-round runner."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT.parent
REPO = MODULE.parent
PRIVATE = MODULE / "private/round2"
RESULTS = ROOT / "results"
CONFIG_PATH = ROOT / "configs/protocol.json"
CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
BASE = CFG["reference_commit"]
PREVIOUS_RESULTS = MODULE / "results"
PAIR_FILE = MODULE / "private/results/Main_Pairs.csv"
ITEM_FILE = MODULE / "private/clinical/clinical_identity/results/STN_scores_long.csv"
OLD_INPUT_LOCK = MODULE / "private/Input_Hashes.json"


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
                          encoding="utf-8", capture_output=True, check=True).stdout.strip()


def digest(path, text=False):
    raw = Path(path).read_bytes()
    return sha256(raw.replace(b"\r\n", b"\n") if text else raw).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(data, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".pending")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(file)


def write_csv(frame, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".pending")
    frame.to_csv(tmp, index=False, na_rep="NA", float_format="%.16g", lineterminator="\n")
    tmp.replace(file)


def frozen_receipt():
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("diff", "--exit-code", BASE, "--", ".", f":(exclude){ROOT.relative_to(REPO).as_posix()}/**")
    git("diff", "--exit-code", "HEAD", "--", CONFIG_PATH.relative_to(REPO).as_posix(),
        (ROOT / "ANALYSIS_PLAN.md").relative_to(REPO).as_posix())
    old = read_json(PREVIOUS_RESULTS / "Run_Manifest.json")
    selected = (
        "results/Clinical_Paired_Results.csv", "results/Duration_Paired_Results.csv",
        "results/Score_Item_Dictionary.csv", "reports/Behavior_ON_OFF_Report.md", "reports/Data_and_Score_Audit.md",
    )
    locked_public = {name: old["public_outputs_sha256"][name] for name in selected}
    locked_public["configs/protocol.json"] = old["protocol_sha256"]
    for name, sha in locked_public.items():
        if digest(MODULE / name, text=True) != sha:
            raise ValueError("A required first-round release file differs from its recorded digest")
    return old, locked_public


def read_pairs():
    pairs = pd.read_csv(PAIR_FILE, float_precision="round_trip")
    n = CFG["clinical"]["n_pairs"]
    if (len(pairs) != n or pairs[["patient_id", "study_code", "on_scan_id", "off_scan_id", "fold"]].isna().any().any()
            or any(not pairs[k].is_unique for k in ("patient_id", "study_code", "on_scan_id", "off_scan_id"))
            or set(pairs.on_scan_id) & set(pairs.off_scan_id)):
        raise ValueError("The saved patients do not have unique complete ON/OFF scanning pairs")
    return pairs.sort_values("patient_id", kind="stable", ignore_index=True)


def public_artifact_digests(files):
    return {path.relative_to(ROOT).as_posix(): digest(path, text=path.suffix in {".csv", ".md", ".json"})
            for path in sorted(files)}
