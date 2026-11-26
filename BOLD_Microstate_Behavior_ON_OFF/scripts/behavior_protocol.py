"""Run configuration and output helpers."""

from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PRIVATE = ROOT / "private"
RESULTS = ROOT / "results"
PROTOCOL = ROOT / "configs/protocol.json"
REGISTRY = ROOT / "configs/test_registry.csv"
CFG = json.loads(PROTOCOL.read_text(encoding="utf-8-sig"))
STATS = CFG["statistics"]
BEHAVIORS = tuple(CFG["behaviors"])
STATES = tuple(CFG["states"])
KEYS = ["patient_id", "on_scan_id", "off_scan_id"]


def git(*args):
    return subprocess.run(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args],
                          encoding="utf-8", capture_output=True, check=True).stdout.strip()


def digest(path, text=False):
    data = Path(path).read_bytes()
    return sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    frame.to_csv(tmp, index=False, na_rep="NA", lineterminator="\n", float_format="%.16g")
    tmp.replace(path)


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pending")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8", newline="\n")
    tmp.replace(path)
