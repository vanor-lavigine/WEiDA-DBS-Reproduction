"""Paths, output helpers and predeclared endpoint families."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
WORKSPACE = Path(os.environ.get("WEIDA_V6_DATA_ROOT", REPO.parent)).resolve()
V4_NAME = "V4_Microstate_Statistical_Significance"
V5_NAME = "V5_Microstate_Nonsignificant_Endpoint_Optimization"
V4 = REPO / V4_NAME
V5 = REPO / V5_NAME
LOCAL_V4 = WORKSPACE / "WEiDA-DBS-V4" / V4_NAME
RELIABILITY = LOCAL_V4.parent / "V4_BOLD_Microstate_Reliability"
STATE = range(1, 5)
EDGES = [(i, j) for i in STATE for j in STATE if i != j]
ALL_TRANSITIONS = [f"transition_{i}_{j}" for i in STATE for j in STATE]
EDGE_COLUMNS = [f"transition_{i}_{j}" for i, j in EDGES]
ILR = [f"ilr_{i}" for i in range(1, 4)]
RESULT = HERE / "results"
PRIVATE = HERE / "private"
CONFIG_PATH = HERE / "configs/protocol.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


CFG = read_json(CONFIG_PATH)
TR = CFG["baseline"]["tr"]
os.environ.setdefault("NUMBA_CACHE_DIR", str(PRIVATE / "numba"))
sys.path[:0] = [str(LOCAL_V4 / "scripts"), str(V5 / "scripts")]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_csv(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, lineterminator="\n", encoding="utf-8-sig")


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={REPO.as_posix()}", "-C", str(REPO), *args])


def scalar_families():
    full = [f"dwell_window_{kind}_{i}" for kind in ("median", "p75") for i in STATE]
    edge = [f"dwell_edge_{kind}_{i}" for kind in ("median", "p75") for i in STATE]
    return {
        "Primary": CFG["primary"],
        "Secondary": CFG["secondary"],
        "Dwell exploration": [*full, *edge, "dwell_edge_mean_4"],
        "Spatial dispersion exploration": [f"dispersion_{i}" for i in STATE],
        "Transition localization": EDGE_COLUMNS,
    }
