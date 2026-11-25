"""Refit A3 across 2,000 participant resamples with the original fold strata."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import time

for var in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
import pandas as pd

from micro_inputs import FEATURES, HERE, RELIABILITY, SETTINGS, feature_mask, load_inputs, scheme, segment_a3
from v4_bootstrap import draw_weights, strata
from v4_models import fit_pair, match_banks


def setup(data):
    frame = data.features
    condition = frame.condition.to_numpy()
    subjects = frame.subject.to_numpy()
    groups = {c: np.flatnonzero(condition == c) for c in ("HC", "ON", "OFF")}
    on = groups["ON"][np.argsort(subjects[groups["ON"]])]
    off = groups["OFF"][np.argsort(subjects[groups["OFF"]])]
    np.testing.assert_array_equal(subjects[on], subjects[off])
    return condition, groups, on, off, strata(data.split, condition, subjects)


def weighted_mean(values, multiplicity):
    defined = np.isfinite(values) & (multiplicity > 0)[:, None]
    weights = defined * multiplicity[:, None]
    d = np.sum(weights, axis=0)
    numerator = np.sum(np.where(defined, values, 0) * weights, axis=0)
    return np.divide(numerator, d, out=np.full(values.shape[1], np.nan), where=d > 0)


def contrasts(values, w, groups, on, off):
    assert np.array_equal(w[on], w[off])
    hc, pd_off, pd_on = (weighted_mean(values[groups[g]], w[groups[g]]) for g in ("HC", "OFF", "ON"))
    return np.stack((pd_off - hc, pd_on - hc, weighted_mean(values[on] - values[off], w[on])))


def refit(data, condition, w):
    values = np.full((len(condition), len(FEATURES)), np.nan)
    fitted = {}
    for h in range(5):
        train = (data.split != h) & (w > 0)
        fitted[h] = fit_pair(data.prepared, condition, train, weights=w)
    bank = {-1: data.banks["G1"][0], **{h: fit["G1"].maps for h, fit in fitted.items()}}
    match = match_banks(bank)
    for h, fit in fitted.items():
        for i in np.flatnonzero((data.split == h) & (w > 0)):
            frame = segment_a3(data.prepared[i], fit["G0"].maps, fit["G1"].maps, SETTINGS["lambda"])
            values[i] = scheme(frame, match["order"][h])
    return values, match["stable"]


def init(data):
    global DATA, META
    DATA, META = data, setup(data)


def batch(path, lo, hi):
    path = Path(path)
    if path.exists():
        return path
    condition, groups, on, off, pools = META
    n = hi - lo
    fixed = np.full((n, 3, len(FEATURES)), np.nan)
    full = fixed.copy()
    matches = np.zeros((n, 4), bool)
    status = []
    seen = DATA.features[list(FEATURES)].to_numpy()
    for i, draw in enumerate(range(lo, hi)):
        w = draw_weights(pools, len(condition), SETTINGS["full_refit_seed_base"] + draw)
        fixed[i] = contrasts(seen, w, groups, on, off)
        try:
            changed, aligned = refit(DATA, condition, w)
        except (RuntimeError, np.linalg.LinAlgError) as exc:
            status.append(type(exc).__name__)
            continue
        matches[i] = aligned
        changed[:, ~feature_mask(aligned & DATA.matches["stable"])] = np.nan
        full[i] = contrasts(changed, w, groups, on, off)
        status.append("ok")
    scratch = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(scratch, fixed=fixed, full=full, matches=matches, status=np.array(status))
    scratch.replace(path)
    return path


def summarize(paths, data, meta):
    parts = []
    for path in sorted(paths):
        with np.load(path) as d:
            parts.append(dict(d))
    result = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = SETTINGS["full_refit_draws"]
    assert all(len(x) == n for x in result.values())
    condition, groups, on, off, _ = meta
    estimates = contrasts(data.features[list(FEATURES)].to_numpy(), np.ones(len(condition), int), groups, on, off)
    rows = []
    for kind in ("fixed", "full"):
        for i, contrast in enumerate(SETTINGS["contrasts"]):
            for j, metric in enumerate(FEATURES):
                values = result[kind][:, i, j]
                valid = np.isfinite(values)
                defined = int(valid.sum())
                ci = np.quantile(values[valid], [.025, .975]) if defined == n else (np.nan, np.nan)
                conditional = np.quantile(values[valid], [.025, .975]) if 100 <= defined < n else (np.nan, np.nan)
                robust = bool(defined == n and (ci[0] > 0 or ci[1] < 0))
                rows.append(dict(method="A3", scheme=kind, contrast=contrast, metric=metric,
                                 mean_difference=estimates[i, j], requested=n, defined=defined, defined_fraction=defined/n,
                                 ci95_low=ci[0], ci95_high=ci[1], full_interval_excludes_zero=robust if kind == "full" else False,
                                 matched_only_ci95_low=conditional[0], matched_only_ci95_high=conditional[1]))
    out = HERE / "results"
    pd.DataFrame(rows).to_csv(out / "Bootstrap_Results.csv", index=False)
    old_file = RELIABILITY / "results/V4_Bootstrap_Identifiability_formal.csv"
    historical = pd.read_csv(old_file).query("method == 'A3'").n_retrained_state_matches.to_numpy()
    matches = result["matches"].sum(axis=0)
    receipt = dict(draws=n, fully_refitted=int(np.sum(result["status"] == "ok")),
                   failures=pd.Series(result["status"]).value_counts().to_dict(),
                   full_composition_matches=int(np.sum(result["matches"].all(axis=1))),
                   matches_by_state=matches.tolist(), historical_a3_state_matches=historical.tolist(),
                   historical_state_counts_equal=bool(np.array_equal(matches, historical)))
    (out / "Bootstrap_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=10)
    args = cli.parse_args()
    start = time.monotonic()
    data = load_inputs()
    meta = setup(data)
    observed = data.features[list(FEATURES)].to_numpy()
    replay, _ = refit(data, meta[0], np.ones(len(observed), int))
    np.testing.assert_allclose(replay[:, feature_mask(data.matches["stable"])],
                               observed[:, feature_mask(data.matches["stable"])], atol=1e-11, rtol=0, equal_nan=True)
    print(f"Point replay matched all 65 A3 scans after {time.monotonic()-start:.1f}s", flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    cfg = [Path(__file__), Path(__file__).with_name("micro_inputs.py"), HERE / "configs/protocol.json",
           RELIABILITY / "scripts/v4_models.py", RELIABILITY / "private/V4_Maps.npz"]
    signature = sha256(b"".join(path.read_bytes() for path in cfg)).hexdigest()[:16]
    cache = HERE / "private" / "refits" / signature
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(cache / f"{i:04}.npz", i, min(i + 10, SETTINGS["full_refit_draws"]))
            for i in range(0, SETTINGS["full_refit_draws"], 10)]
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(data,)) as pool:
        waiting = [pool.submit(batch, *job) for job in jobs]
        paths = []
        for future in as_completed(waiting):
            paths.append(future.result())
            if len(paths) % 10 == 0:
                print(f"Full bootstrap: {len(paths) * 10}/{SETTINGS['full_refit_draws']} completed in {time.monotonic()-start:.0f}s", flush=True)
    summarize(paths, data, meta)


if __name__ == "__main__":
    main()
