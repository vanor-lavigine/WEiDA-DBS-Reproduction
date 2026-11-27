"""Path, provenance and serialization helpers for the independent round-three run."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT.parent
REPO = MODULE.parent
PRIVATE = MODULE / "private/round3"
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "configs/protocol.json"
REGISTRY = ROOT / "configs/test_registry.csv"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


CFG = read_json(PROTOCOL)
BASE = CFG["reference_commit"]
STATES = tuple(CFG["states"])
ENDPOINTS = tuple(CFG["clinical_endpoints"])
SLOTS = tuple((state, endpoint) for state in STATES for endpoint in ENDPOINTS)


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
                          capture_output=True, check=True, encoding="utf-8").stdout.strip()


def digest(path, text=False):
    raw = Path(path).read_bytes()
    return sha256(raw.replace(b"\r\n", b"\n") if text else raw).hexdigest()


def write_json(data, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".pending")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(file)


def write_csv(frame, file):
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(".pending")
    frame.to_csv(tmp, index=False, na_rep="NA", float_format="%.17g", lineterminator="\n")
    tmp.replace(file)


def verify_registry():
    rows = list(pd.read_csv(REGISTRY).itertuples(index=False, name=None))
    expected = [(state, endpoint, "rate", "absolute_same_cases") for state, endpoint in SLOTS]
    if rows != expected or len(rows) != CFG["statistics"]["bh_family_size"]:
        raise ValueError("The planned test registry differs from the two fixed BH20 families")


def verify_prior():
    """Check the exact published ancestor and earlier tracked analysis artifacts."""
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    if git("rev-parse", BASE + "^{tree}") != CFG["reference_tree"]:
        raise ValueError("The reference commit does not have its verified public tree")
    anchor = MODULE.relative_to(REPO).as_posix()
    git("diff", "--exit-code", BASE, "--", f"{anchor}/ANALYSIS_PLAN.md", f"{anchor}/configs",
        f"{anchor}/reports", f"{anchor}/results", f"{anchor}/scripts", f"{anchor}/tests",
        f"{anchor}/figures", f"{anchor}/round2")
    git("diff", "--exit-code", "HEAD", "--",
        PROTOCOL.relative_to(REPO).as_posix(), REGISTRY.relative_to(REPO).as_posix(),
        (ROOT / "ANALYSIS_PLAN.md").relative_to(REPO).as_posix())
    first = read_json(MODULE / "results/Run_Manifest.json")
    second = read_json(MODULE / "round2/results/Round2_Run_Manifest.json")
    claims = {
        "configs/protocol.json": first["protocol_sha256"],
        "reports/Behavior_ON_OFF_Report.md": first["public_outputs_sha256"]["reports/Behavior_ON_OFF_Report.md"],
        "results/Score_Item_Dictionary.csv": first["public_outputs_sha256"]["results/Score_Item_Dictionary.csv"],
        "round2/reports/ON_OFF_Round2_Report.md": second["public_outputs_sha256"]["reports/ON_OFF_Round2_Report.md"],
    }
    for name, expected in claims.items():
        if digest(MODULE / name, text=True) != expected:
            raise ValueError(f"Prior public source digest differs: {name}")
    for name in ("ANALYSIS_PLAN.md", "results/Run_Manifest.json", "round2/results/Round2_Run_Manifest.json",
                 "scripts/behavior_stats.py", "round2/scripts/round2_io.py", "round2/scripts/clinical.py",
                 "round2/scripts/brain.py"):
        claims[name] = digest(MODULE / name, text=True)
    return claims
