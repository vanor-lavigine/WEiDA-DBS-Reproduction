"""Blinded synthetic calibration of six decoders on fixed V4 maps."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import pandas as pd
from scipy import optimize, signal, stats
from statsmodels.stats.proportion import proportion_confint

from v5_data import CFG, COORD, HERE, RELIABILITY, RESULT, TR, counts_episodes, csv, dump, load_data
from micro_inputs import cohort, match_banks, temporal_labels
from models_v3 import prepare
from input_data import transform
from inference_core import adjust, composition, scalar
from sim_v4_generator import mapped_fd


S = CFG["simulation"]
SOS = signal.butter(5, (.01, .08), fs=1 / TR, btype="bandpass", output="sos")
SIM_OUTPUT = RESULT / "simulation"
METRICS = ("coverage_1", "duration_1", "transition_2_1", "switching_rate", "global_gev", "mean_margin")


def phantom(seed, scenario, context):
    """V3-style paired semi-Markov truth with fixed A3 axes and pooled noise."""
    rng = np.random.default_rng(seed)
    reference, noise_root = context["banks"][0], context["noise_root"]
    basis = np.linalg.qr(reference.T)[0]
    twist = rng.normal(size=len(reference[0]))
    twist -= basis @ (basis.T @ twist)
    twist /= np.linalg.norm(twist)
    signals, noise, metadata, times = [], [], [], []
    for i in range(40):
        affected = i >= 15
        conditions = ("ON", "OFF") if affected else ("HC",)
        maps = reference + rng.normal(0, .017, reference.shape)
        if scenario == "S3" and affected:
            maps[0] = .77 * maps[0] + np.sqrt(1 - .77**2) * twist
        odds = np.exp(rng.normal(0, .20, 4))
        if scenario == "S1" and affected:
            odds[0] *= 2.3
        dur = max(2.01, 5 * np.exp(rng.normal(0, .17)) * (1.6 if scenario == "S2" and affected else 1))
        fd_base = rng.uniform(.10, .25)
        for condition in conditions:
            target = np.empty(210, int)
            sign = np.empty(210)
            start, previous = 0, -1
            while start < 210:
                before = odds.copy()
                if previous >= 0:
                    before[previous] = 0
                previous = rng.choice(4, p=before / before.sum())
                stop = min(210, start + 1 + rng.geometric(1 / (dur - 1)))
                target[start:stop] = previous
                sign[start:stop] = -1 if rng.random() < .5 else 1
                start = stop
            raw_fd = fd_base + rng.uniform(-.035, .035)
            level = "high" if scenario == "S5_high" and affected else "moderate" if scenario == "S5_moderate" and affected else "baseline"
            fd = float(mapped_fd(raw_fd, {"baseline": .5, "moderate": .75, "high": .9}[level],
                                 context["fd_quantiles"], context["fd_mm"]))
            signals.append(maps[target] * sign[:, None])
            noise.append(rng.normal(size=(210, 78)) @ noise_root)
            metadata.append((i, condition, fd, i % 5, level, rng.integers(0, len(context["windows"])), raw_fd))
            times.append(target)
    return signal.sosfiltfilt(SOS, np.stack(signals), axis=1), signal.sosfiltfilt(SOS, np.stack(noise), axis=1), metadata, np.stack(times)


def n0(x):
    x = signal.detrend(x, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True, ddof=1)
    x -= x.mean(axis=2, keepdims=True)
    return x


def residual_fraction(x):
    values = np.linalg.eigvalsh(x.swapaxes(-2, -1) @ x)
    return 1 - values[..., -4:].sum(axis=-1) / values.sum(axis=-1)


def lag_gfp(values):
    a, b = values[:, 1:], values[:, :-1]
    lag = (a*b).sum(axis=(1,2)) / np.sqrt((a*a).sum(axis=(1,2))*(b*b).sum(axis=(1,2)))
    return np.quantile(lag, [.25,.5,.75]).tolist(), np.quantile(np.sqrt((values*values).mean(axis=2)), [.25,.5,.75]).tolist()


def setup():
    loaded = load_data(save_private=False)
    original, _ = cohort()
    arrays = [prepare(transform(s["x"], "F1")) for s in original]
    valid = np.vstack([p.vectors[p.valid] for p in arrays])
    motion = loaded.fd.to_numpy()
    with np.load(RELIABILITY / "private/V4_Maps.npz") as archive:
        raw = archive["G1"]
    mapping = match_banks(raw)
    bank = np.stack([x[mapping["order"][i]] for i, x in enumerate(raw)])
    for x in bank:
        x *= np.sign((x*bank[0]).sum(axis=1, keepdims=True))
    proj = np.linalg.qr(bank[0].T)[0]
    innovation = valid - (valid @ proj) @ proj.T
    cov = innovation.T @ innovation / len(innovation)
    ia = np.eye(78) * np.trace(cov) / 78
    cov = .85 * cov + .15 * ia
    root = np.linalg.cholesky(cov * 78 / np.trace(cov)).T
    masks = []
    for name in loaded.scan:
        with np.load(RELIABILITY / "private/frames/A3" / f"{name}.npz") as row:
            masks.append(row["labels"] >= 0)
    probability = np.array([0,.05,.1,.25,.5,.75,.9,.95,1.])
    context = dict(banks=bank, noise_root=root, windows=np.stack(masks), fd_quantiles=probability,
                   fd_mm=np.quantile(motion, probability))
    raws = np.stack([p.vectors for p in arrays])
    targets = np.quantile(residual_fraction(raws), [.50,.75,.90])
    components = [phantom(S["calibration_seed"]+i, "S0", context) for i in range(4)]
    source, noise = (np.concatenate([k[i] for k in components]) for i in (0,1))
    meta = [value for k in components for value in k[2]]
    baseline_fd = np.array([m[2] for m in meta])
    raw_fd = np.array([m[6] for m in meta])
    context["scales"] = {}
    calibration = []
    for level, q, target in zip(("baseline","moderate","high"), (.5,.75,.9), targets):
        fd = mapped_fd(raw_fd, q, probability, context["fd_mm"])
        base_noise = (.057 + .12*fd[:,None,None])*noise
        multiplier = optimize.brentq(lambda z: np.median(residual_fraction(n0(source + z*base_noise)))-target, .001, 40.)
        context["scales"][level] = multiplier
        simulated = n0(source+multiplier*base_noise)
        lag, gfp = lag_gfp(simulated)
        calibration.append(dict(level=level, empirical_rank4_residual=target, fitted_rank4_residual=np.median(residual_fraction(simulated)),
                                scale=multiplier, lag1_quartiles=lag, gfp_quartiles=gfp))
    lag, gfp = lag_gfp(raws)
    relation = np.corrcoef(np.cov(raws.reshape(-1,78).T)[np.triu_indices(78,1)],
                            np.cov(n0(source+context["scales"]["baseline"]*(.057+.12*baseline_fd[:,None,None])*noise).reshape(-1,78).T)[np.triu_indices(78,1)])[0,1]
    dump(dict(empirical_lag1_quartiles=lag, empirical_gfp_quartiles=gfp, calibration=calibration,
              baseline_covariance_edge_correlation=relation, fixed_map_count=20, calibration_cohorts=4,
              empirical_window_frames_quartiles=np.quantile(context["windows"].sum(axis=1), [.25,.5,.75]).tolist()),
         SIM_OUTPUT / "Simulation_Matching.json")
    a = phantom(42, "S0", context)
    b = phantom(42, "S5_high", context)
    np.testing.assert_array_equal(a[3], b[3])
    np.testing.assert_array_equal(a[0], b[0])
    return context


def ari(truth, labels):
    counts = np.bincount(4*truth+labels, minlength=16).reshape((4,4))
    def pairs(x):
        return (x*(x-1)).sum() / 2
    total = len(truth)*(len(truth)-1)/2
    real, observed = pairs(counts.sum(0)), pairs(counts.sum(1))
    expect = real*observed/total
    upper = (real+observed)/2
    return (pairs(counts)-expect)/(upper-expect) if upper>expect else 1.


def readouts(y, gfp=None, corr=None):
    count, episodes, _ = counts_episodes(y)
    good = y >= 0
    n = int(good.sum())
    cov = np.bincount(y[good], minlength=4)/n
    row = dict(coverage_1=cov[0], duration_1=np.mean(episodes[0]) if episodes[0] else np.nan,
               transition_2_1=count[1,0]/count[1].sum() if count[1].sum() else np.nan,
               switching_rate=(count.sum()-count.trace())/(n*TR))
    row.update({f"c{i+1}": cov[i] for i in range(4)})
    if corr is not None:
        row["duration_s"] = n*TR
        row["global_gev"] = np.average(corr[good, y[good]]**2, weights=gfp[good]**2)
        top = np.sort(corr[good], axis=1)[:, -2:]
        row["mean_margin"] = np.mean(top[:,1]-top[:,0])
    return row


def quality(truth, y, keep):
    cuts = np.r_[0, np.flatnonzero(np.diff(truth)) + 1, len(truth)]
    short = [(a,b) for a,b in zip(cuts[:-1], cuts[1:]) if 2 <= b-a <= 4 and a and b<len(keep) and keep[a-1:b+1].all()]
    return ari(truth[keep], y[keep]), sum(np.mean(y[a:b] == truth[a]) >= .5 for a,b in short), len(short)


def init(context):
    global DATA
    DATA = context


def model_values(data, key):
    try:
        return scalar(data.loc[data[key].notna()], key)
    except (ValueError, np.linalg.LinAlgError):
        return {c: {"Effect_Size": np.nan, "P_raw": np.nan} for c in ("C1", "C2")}


def one(setting, index):
    scenario = S["settings"][setting]
    seed = S["formal_seed"] + S["stride"]*setting + index
    signal_, noise, rows, truth = phantom(seed, scenario, DATA)
    fd = np.array([r[2] for r in rows])
    scale = np.array([DATA["scales"][r[4]] for r in rows])
    x = n0(signal_ + (scale*(.057+.12*fd))[:,None,None]*noise)
    gfp = np.linalg.norm(x, axis=2) / np.sqrt(78)
    units = x / (gfp[:,:,None]*np.sqrt(78))
    banks = DATA["banks"][[r[3] for r in rows]]
    scores = np.abs(units @ banks.swapaxes(1,2))
    windows = DATA["windows"][[r[5] for r in rows]]
    meta = [dict(subject=r[0], condition=r[1], fd=r[2]) for r in rows]
    t = pd.DataFrame([{**d, **readouts(np.where(w, y, -1))} for d,w,y in zip(meta, windows, truth)])
    weight = np.where(t.condition.eq("HC"),1,.5)
    metrics = METRICS[:4]
    target = {k: model_values(t, k) for k in metrics}
    results = []
    for penalty in S["candidates"]:
        labels = [temporal_labels(r, g, w, penalty, normalization=np.mean(g*g)) for r,g,w in zip(scores,gfp,windows)]
        observed = pd.DataFrame([{**d, **readouts(y,g,r)} for d,y,g,r in zip(meta, labels, gfp, scores)])
        stats_by_metric = {k: model_values(observed, k) for k in METRICS}
        try:
            comp = composition(observed, ["c1","c2","c3","c4"], draws=S["wild_draws"], seed=CFG["seed"]+index)
        except (ValueError, np.linalg.LinAlgError):
            comp = {c: {"P_raw": np.nan} for c in ("C1", "C2")}
        core = np.array([a[c]["P_raw"] for a in (comp, stats_by_metric["global_gev"], stats_by_metric["mean_margin"])
                         for c in ("C1","C2")])
        quality_values = np.array([quality(y, l, w) for y,l,w in zip(truth, labels, windows)])
        result = dict(scenario=scenario, repeat=index, switch_penalty=penalty,
                      core_reject=bool(np.any(adjust(core,"holm") < .05)), ari=np.average(quality_values[:,0], weights=weight),
                      short_recall=quality_values[:,1].sum()/max(1,quality_values[:,2].sum()),
                      min_coverage=np.average(observed[["c1","c2","c3","c4"]], axis=0, weights=weight).min(),
                      mean_gev=np.average(observed.global_gev, weights=weight), mean_fd_hc=observed.loc[observed.condition.eq("HC"),"fd"].mean(),
                      mean_fd_pd=observed.loc[observed.condition.ne("HC"),"fd"].mean())
        for i, p in enumerate(core):
            result[f"core_p_{i+1}"] = p
        for key in metrics:
            for code, label in [("C2","OFF_HC"),("C1","ON_HC")]:
                rec, actual = stats_by_metric[key][code], target[key][code]
                result[f"{key}_{label}_effect"] = rec["Effect_Size"]
                result[f"{key}_{label}_p"] = rec["P_raw"]
                result[f"{key}_{label}_truth"] = actual["Effect_Size"]
        results.append(result)
    return results


def chunk(cache, setting, first):
    path = Path(cache) / f"{setting}_{first:04}.csv"
    if not path.exists():
        rows = [row for i in range(first, min(first+5, S["draws_per_setting"])) for row in one(setting, i)]
        tmp = path.with_suffix(".tmp")
        csv(pd.DataFrame(rows), tmp)
        tmp.replace(path)
    return path


def summarize(d):
    rows = []
    recovery = []
    for (scenario, penalty), block in d.groupby(["scenario","switch_penalty"], sort=False):
        n = len(block)
        assert n == block.repeat.nunique() == S["draws_per_setting"]
        hits = block.core_reject.sum()
        low, high = proportion_confint(hits,n,method="wilson")
        threshold = S["s0_max_wilson"] if scenario=="S0" else S["s5_max_wilson"] if scenario.startswith("S5") else 1.
        estimability = block.filter(regex=r"^core_p_").notna().mean().min()
        small = block.min_coverage.lt(S["small_state_minimum"]).mean()
        rows.append(dict(scenario=scenario, penalty=penalty, n=n, rejected=hits, rate=hits/n, wilson_low=low,
                         wilson_upper=high, threshold=threshold, estimability=estimability, small_state_rate=small,
                         passed=(high<=threshold and estimability>=S["min_core_estimability"] and small<=S["max_small_state_rate"]),
                         ari=block.ari.mean(), gev=block.mean_gev.mean(), short_recall=block.short_recall.mean()))
        for measure in METRICS[:4]:
            for contrast in ("OFF_HC","ON_HC"):
                key = f"{measure}_{contrast}"
                actual, inferred = block[f"{key}_truth"], block[f"{key}_effect"]
                detect = block[f"{key}_p"].lt(.05)
                recovery.append(dict(scenario=scenario, penalty=penalty, outcome=measure, contrast=contrast,
                                     truth_mean=actual.mean(), estimate_mean=inferred.mean(), bias=(inferred-actual).mean(),
                                     rmse=np.sqrt(((inferred-actual)**2).mean()), rejection_rate=detect.mean(),
                                     sign_and_rejection=(detect & (actual*inferred>0)).mean()))
    summary = pd.DataFrame(rows)
    base = summary.loc[summary.penalty.eq(0)].set_index("scenario")
    gates = []
    for penalty, g in summary.groupby("penalty"):
        gev = (g.set_index("scenario").gev/base.gev).min()
        recall_loss = base.loc["S0","short_recall"] - g.loc[g.scenario.eq("S0"),"short_recall"].item()
        q = d.loc[d.switch_penalty.eq(penalty) & d.scenario.str.startswith("S5")]
        relative_error = np.mean([(q[f"switching_rate_{c}_effect"]-q[f"switching_rate_{c}_truth"]).abs().mean()/.09
                                  for c in ("ON_HC","OFF_HC")])
        eligible = bool(g.passed.all() and gev>=S["min_gev_retention"] and recall_loss<=S["max_short_recall_loss"])
        score = g.loc[g.scenario.isin(["S0","S1","S2","S3"]),"ari"].mean() - .25*relative_error
        gates.append(dict(penalty=penalty, eligible=eligible, core_gate=g.passed.all(), min_gev_retention=gev,
                          s0_short_recall_loss=recall_loss, simulation_score=score))
    chooser = pd.DataFrame(gates).sort_values(["eligible","simulation_score","penalty"], ascending=[False,False,True])
    best = float(chooser.iloc[0].penalty) if chooser.eligible.any() else None
    csv(summary, SIM_OUTPUT / "Calibration_by_Scenario.csv")
    csv(pd.DataFrame(recovery), SIM_OUTPUT / "Planted_Effect_Recovery.csv")
    csv(chooser, SIM_OUTPUT / "Penalty_Selection_Table.csv")
    dump(dict(selected_lambda=best, candidate_count=len(gates), cohorts_per_candidate=len(S["settings"])*S["draws_per_setting"],
              baseline_lambda=.04, uses_real_group_p=False, fixed_templates=True), SIM_OUTPUT / "Penalty_Selection.json")
    return best


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workers", type=int, default=10)
    args = cli.parse_args()
    context = setup()
    init(context)
    sample = one(0, 0)
    assert len(sample) == 6 and all(np.isfinite(r["ari"]) for r in sample)
    key = sha256(Path(__file__).read_bytes()+(HERE/"configs/protocol.json").read_bytes()+context["banks"].tobytes()).hexdigest()[:12]
    cache = HERE / "private" / "simulation" / key
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(cache,s,i) for s in range(len(S["settings"])) for i in range(0,S["draws_per_setting"],5)]
    print(f"Formal calibration started: {len(jobs)*5} cohorts per candidate, {args.workers} workers.", flush=True)
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(context,)) as pool:
        done = []
        for job in as_completed(pool.submit(chunk,*task) for task in jobs):
            done.append(job.result())
            if len(done)%20 == 0:
                print(f"Synthetic cohorts complete per candidate: {len(done)*5}/{len(jobs)*5}", flush=True)
    data = pd.concat([pd.read_csv(p) for p in sorted(done)], ignore_index=True)
    csv(data, SIM_OUTPUT / "Formal_Synthetic_Cohorts.csv")
    print(f"Synthetic calibration complete. Selected penalty = {summarize(data)}", flush=True)


if __name__ == "__main__":
    main()
