"""Use the frozen worktree and write deterministic public/private artifacts."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT.parents[1]
REPO = MODULE.parent
PRIVATE = MODULE / "private/round4/transition"
CONFIG = ROOT / "configs/protocol.json"
REGISTRY = ROOT / "configs/test_registry.csv"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


CFG = read_json(CONFIG)
BASE = CFG["source_commit"]


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
        check=True, capture_output=True, encoding="utf-8",
    ).stdout.strip()


def verify_base():
    git("merge-base", "--is-ancestor", BASE, "HEAD")
    git("diff", "--exit-code", BASE, "--", ".", f":(exclude){ROOT.relative_to(REPO).as_posix()}/**")
    git("check-ignore", (PRIVATE / "results/patient.csv").relative_to(REPO).as_posix())
    if git("branch", "--show-current") != CFG["branch"]:
        raise ValueError("Use the dedicated analysis branch for this run")


def digest(path, *, normalize=False):
    data = Path(path).read_bytes()
    return sha256(data.replace(b"\r\n", b"\n") if normalize else data).hexdigest()


def collection_digest(paths):
    return sha256("".join(value + "\n" for value in sorted(paths.values())).encode("ascii")).hexdigest()


def verify_inputs(*locks):
    verified = {}
    for lock in locks:
        for name, expected in lock.items():
            path = Path(name).resolve()
            if not path.is_relative_to(REPO.parent):
                raise ValueError("Frozen evidence must be in the authorized workspace")
            if str(path) in verified:
                if verified[str(path)] != expected:
                    raise ValueError("Frozen input digests disagree across evidence receipts")
            elif digest(path) != expected:
                raise ValueError("A frozen input hash changed")
            verified[str(path)] = expected
    return dict(sorted(verified.items()))


def atomic_text(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(body, encoding="utf-8", newline="\n")
    temporary.replace(path)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=json_value) + "\n")


def write_csv(path, table):
    atomic_text(path, table.to_csv(index=False, na_rep="NA", float_format="%.15g", lineterminator="\n"))


def expected_registry():
    states = CFG["states"]
    group_rows = [
        (family, contrast, source, target, "")
        for family in ("WEiDA", "ExitConditional") for contrast in CFG["contrasts"]
        for source in states for target in states if source != target
    ]
    clinical_rows = [
        ("Clinical_WEiDA", "ON-OFF", source, target, endpoint)
        for source in states for target in states for endpoint in CFG["clinical_endpoints"]
    ]
    return group_rows + clinical_rows


def verify_registry():
    registry = pd.read_csv(REGISTRY, keep_default_na=False)
    if list(registry.itertuples(index=False, name=None)) != expected_registry():
        raise ValueError("Fixed contrast or clinical registry differs from the protocol")
    if registry.family.value_counts().to_dict() != CFG["families"]:
        raise ValueError("One or more fixed family sizes differ from the protocol")
