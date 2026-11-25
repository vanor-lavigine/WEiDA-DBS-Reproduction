"""Fold-stratified participant bootstrap for fixed maps and entire V4 refits."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time

for variable in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from threadpoolctl import threadpool_limits

from v4_inference import BANK, CONFIG, CONTRASTS, METHODS, ROOT, STATE_COLUMNS, read_calibration
from v4_models import cohort, fit_pair, match_banks, prepare_scans, segment_scan, TR


SETTINGS = CONFIG["bootstrap"]
POINT_METRICS = (
    "global_gev","mean_margin","low_margin_fraction","switching_rate",*STATE_COLUMNS,
    "coverage_ilr_1","coverage_ilr_2","coverage_ilr_3",
)
ILR = helmert(4)
COEFFICIENT_CONTRASTS = np.array([[0,1,0,0],[0,0,1,0],[0,1,-1,0]])


@dataclass
class BootstrapData:
    ids: np.ndarray
    condition: np.ndarray
    fold: np.ndarray
    design: np.ndarray
    pools: list
    prepared: list
    banks: dict
    observed: dict
    fixed_metrics: np.ndarray
    methods: tuple
    penalty: float


def strata(fold, condition, ids):
    """One index list for each participant; 5 PD / 3 HC pools within each fold."""
    id_index = {v: np.flatnonzero(ids == v) for v in np.unique(ids)}
    pools = []
    for h in range(5):
        group = (fold == h)
        for name in ("ON","HC"):
            people = sorted(np.unique(ids[group & (condition == name)]))
            indexes = [id_index[v] for v in people]
            assert len(indexes) == (5 if name == "ON" else 3)
            assert all(np.all(fold[idx] == h) and len(idx) == (2 if name == "ON" else 1) for idx in indexes)
            pools.append(indexes)
    return pools


def draw_weights(pools, size, seed):
    rng = np.random.default_rng(seed)
    multiplicity = np.zeros(size, dtype=np.int32)
    for pool in pools:
        picked = rng.integers(len(pool), size=len(pool))
        for j, repeats in enumerate(np.bincount(picked, minlength=len(pool))):
            multiplicity[pool[j]] = repeats
    return multiplicity


def readout(labels, similarity, gfp, perm):
    """Recreate exactly the target V4 metrics on a labeled clock that can contain gaps."""
    good = labels >= 0
    n = good.sum()
    if not n:
        return np.full(len(POINT_METRICS), np.nan)
    starts = good & np.r_[True, labels[1:] != labels[:-1]]
    adjacent = good[:-1] & good[1:]
    switches = np.sum(adjacent & (labels[1:] != labels[:-1]))
    by_frame = labels[good]
    count = np.bincount(labels[starts], minlength=4)[perm]
    sums = np.bincount(by_frame, minlength=4)[perm]
    duration = np.divide(TR*sums, count, out=np.full(4,np.nan), where=count>0)
    power = gfp[good]**2
    assigned = np.bincount(by_frame, weights=power*similarity[good,by_frame]**2, minlength=4)[perm]
    state_gev = assigned/power.sum()
    top_two = np.partition(similarity[good], -2, axis=1)[:, -2:]
    margin = top_two[:, 1]-top_two[:, 0]
    return np.r_[state_gev.sum(), margin.mean(), (margin<.05).mean(), switches/(TR*n), sums/n, duration,
                 count/(TR*n), state_gev, ILR @ np.log(sums+.5)]


def observed_metrics(features, names, bank_matches, methods):
    result = []
    for method in methods:
        part = features[(features.method == method) & (features.window == "common")].set_index("scan").loc[names]
        values = np.full((len(names),len(POINT_METRICS)), np.nan)
        for h, order in bank_matches[BANK[method]]["order"].items():
            use = (part.fold == h) & (part.duration_s > 0)
            metric_cols = list(POINT_METRICS[:4])+[f"{state}_{q+1}" for state in CONFIG["state_metrics"] for q in order]
            values[use, :-3] = part.loc[use,metric_cols].to_numpy(float)
            count = part.loc[use,[f"coverage_{q+1}" for q in order]].to_numpy(float)*part.loc[use,["duration_s"]].to_numpy()/TR
            values[use, -3:] = np.log(count+.5) @ ILR.T
        result.append(values)
    return np.stack(result)


def assemble(args, methods, penalty):
    scans, fold = cohort()
    features = pd.read_csv(args.features, dtype={"scan":str,"subject":str},low_memory=False)
    names = np.array([row["scan"] for row in scans])
    ids = np.array([str(row["subject"]) for row in scans])
    conditions = np.array([row["condition"] for row in scans])
    fd = np.array([row["fd"] for row in scans])
    design = np.c_[np.ones(len(scans)), conditions == "ON", conditions == "OFF", fd-fd.mean()]
    with np.load(args.maps) as archive:
        banks = {g: archive[g] for g in ("G0","G1")}
    observed = {g: match_banks(maps) for g,maps in banks.items()}
    fixed = observed_metrics(features, names, observed, methods)
    return BootstrapData(ids,conditions,fold,design,strata(fold,conditions,ids),prepare_scans(scans),
                         banks,observed,fixed,methods,penalty)


def stable_metrics(stable, values):
    """Missing alignment stays missing across the entire draw, keyed to fold-zero naming."""
    out = values.copy()
    out[:,4:20] = np.where(np.tile(stable, len(CONFIG["state_metrics"]))[None,:],out[:,4:20],np.nan)
    if not np.all(stable):
        out[:,-3:] = np.nan
    return out


def refit(data, multiplicity):
    n = len(data.ids)
    result = np.full((len(data.methods),n,len(POINT_METRICS)), np.nan)
    banks = {g: {-1: arr[0]} for g,arr in data.banks.items()}
    converged = np.full((5,), np.nan)
    train_unique = np.empty(5,int)
    for holdout in range(5):
        included = multiplicity>0
        test = (data.fold==holdout) & included
        train = (data.fold!=holdout) & included
        assert not (set(data.ids[train]) & set(data.ids[test]))
        train_unique[holdout] = len(set(data.ids[train]))
        try:
            fits = fit_pair(data.prepared, data.condition, train, weights=multiplicity)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            return result, np.zeros((len(data.methods),4),bool), type(error).__name__, converged, train_unique[:holdout+1]
        converged[holdout] = fits["G1"].n_converged
        for name, fit in fits.items():
            banks[name][holdout] = fit.maps
    correspondence = {g: match_banks(b) for g,b in banks.items()}
    for h in range(5):
        selected = np.flatnonzero((data.fold==h) & (multiplicity>0))
        maps = {g: b[h] for g,b in banks.items()}
        for i in selected:
            segments = segment_scan(data.prepared[i],maps,data.penalty,window="common",summarize=False)
            for j,method in enumerate(data.methods):
                values = segments[method]
                perm = correspondence[BANK[method]]["order"][h]
                result[j,i] = readout(values["labels"], values["corr"], values["gfp"],perm)
    qualified = np.stack([correspondence[BANK[m]]["stable"] for m in data.methods])
    return result, qualified, "ok", converged, train_unique


def weighted_contrasts(data, values, multiplicity):
    """The OLS effect on the repeated sample; CR1 does not change the point estimate."""
    out = np.full((len(CONTRASTS), values.shape[1]), np.nan)
    take = (multiplicity>0)[:,None] & np.isfinite(values)
    masks, idx = np.unique(take.T, axis=0, return_inverse=True)
    groups = data.condition
    for j, mask in enumerate(masks):
        items = idx==j
        w = np.sqrt(multiplicity[mask,None])
        design = data.design[mask,:]*w
        if all(np.count_nonzero(mask & (groups==g))>=3 for g in ("HC","ON","OFF")) and np.linalg.matrix_rank(design)==4:
            fit = np.linalg.lstsq(design, values[np.ix_(mask,items)]*w, rcond=None)[0]
            out[:,items] = COEFFICIENT_CONTRASTS @ fit
        valid_patients = groups != "HC"
        unique, counts = np.unique(data.ids[mask & valid_patients],return_counts=True)
        pairs = mask & (np.isin(data.ids,unique[counts==2]) | ~valid_patients)
        if np.array_equal(mask,pairs):
            continue
        out[2,items] = np.nan
        w_pair = np.sqrt(multiplicity[pairs,None])
        paired_design = data.design[pairs]*w_pair
        if all(np.count_nonzero(pairs & (groups==g))>=3 for g in ("HC","ON","OFF")) and np.linalg.matrix_rank(paired_design)==4:
            beta = np.linalg.lstsq(paired_design,values[np.ix_(pairs,items)]*w_pair,rcond=None)[0]
            out[2,items] = COEFFICIENT_CONTRASTS[2] @ beta
    return out


def initialize(args, methods, penalty):
    global DATA
    DATA = assemble(args, methods, penalty)


def resample(data, seed):
    w = draw_weights(data.pools, len(data.ids), seed)
    refitted, qualified, status, converged, train_sizes = refit(data,w)
    stable = np.stack([data.observed[BANK[method]]["stable"] for method in data.methods])
    fixed_betas = [weighted_contrasts(data,stable_metrics(good,values),w)
                   for good, values in zip(stable,data.fixed_metrics)]
    refit_betas = [weighted_contrasts(data,stable_metrics(good,values),w)
                   for good, values in zip(stable & qualified,refitted)]
    return np.stack([fixed_betas,refit_betas]), qualified, status, converged, train_sizes.min()


def verify_unity(data):
    """Use multiplicity one to compare the replay with original blinded features."""
    multiplicity = np.ones(len(data.ids),int)
    rerun, _, status, _, _ = refit(data,multiplicity)
    assert status == "ok"
    np.testing.assert_allclose(rerun,data.fixed_metrics,atol=1e-10,rtol=0,equal_nan=True)
    return float(np.nanmax(np.abs(rerun-data.fixed_metrics)))


def write_chunk(job):
    folder, stage, lo, hi, seed_base = job
    path = folder / f"{stage}_{lo:04}_{hi:04}.npz"
    if path.is_file():
        return path
    arrays = [resample(DATA,seed_base+i) for i in range(lo,hi)]
    beta,stable,status,converged,counts = map(np.stack,zip(*arrays))
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp,beta=beta,stable=stable,status=status,converged=converged,counts=counts)
    tmp.replace(path)
    return path


def combined_chunks(paths, n):
    blocks = []
    for path in sorted(paths):
        with np.load(path) as d:
            blocks.append(dict(d))
    out = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    assert all(len(array)==n for array in out.values())
    return out


def report_rounds(data, arrays, stage, n, preferred, published, output):
    stable = np.stack([data.observed[BANK[m]]["stable"] for m in data.methods])
    raw = arrays["beta"]
    success = arrays["status"] == "ok"
    point = np.stack([weighted_contrasts(data,stable_metrics(stable[i],data.fixed_metrics[i]),np.ones(len(data.ids)))
                      for i in range(len(data.methods))])
    rows, ids = [], []
    for i, method in enumerate(data.methods):
        if method not in published:
            continue
        qualified = arrays["stable"][:,i,:]
        role = "calibrated" if method==preferred else "fixed_descriptive_comparator"
        for j, state in enumerate(stable[i]):
            ids.append(dict(method=method,role=role,stage=stage,state=f"R{j+1}",expected_repetitions=n,
                            successful_fits=int(success.sum()),
                            original_state_identifiable=bool(state), n_retrained_state_matches=int(qualified[:,j].sum()),
                            n_full_composition_matches=int(qualified.all(axis=1).sum())))
        for approach, scheme in enumerate(("fixed_templates","complete_refit")):
            for j, contrast in enumerate(CONTRASTS):
                for k, outcome in enumerate(POINT_METRICS):
                    series = raw[:,approach,i,j,k]
                    good = np.isfinite(series)
                    valid = int(good.sum())
                    ci = np.quantile(series,[.025,.975]) if good.all() else (np.nan,np.nan)
                    rows.append(dict(method=method,role=role,stage=stage,contrast=contrast,outcome=outcome,
                                     scheme=scheme,estimate=point[i,j,k],bootstrap_draws=n,identified_draws=valid,
                                     ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(rows).to_csv(output/f"V4_Bootstrap_{stage}.csv",index=False)
    pd.DataFrame(ids).to_csv(output/f"V4_Bootstrap_Identifiability_{stage}.csv",index=False)
    failure_types, failure_counts = np.unique(arrays["status"][~success],return_counts=True)
    failures = dict(zip(failure_types.tolist(),failure_counts.tolist()))
    quality = pd.DataFrame([dict(stage=stage, bootstrap_draws=n, fully_fitted_draws=int(success.sum()),
                                 failed_fit_types=json.dumps(failures,sort_keys=True),
                                 minimum_unique_participants_in_training_fold=int(arrays["counts"].min()),
                                 minimum_converged_g1_starts=int(np.nanmin(arrays["converged"])),
                                 median_converged_g1_starts=float(np.nanmedian(arrays["converged"])))])
    quality.to_csv(output/f"V4_Bootstrap_Quality_{stage}.csv",index=False)


def run(args, published, penalty, preferred):
    data = assemble(args,METHODS,penalty)
    parity = verify_unity(data)
    print(f"V4 bootstrap point-estimate readout matches all blind feature rows: error <= {parity:.3g}",flush=True)
    if args.point_check_only:
        return
    signature = hashlib.sha256()
    signature.update(json.dumps([METHODS,penalty]).encode())
    for path in (args.features,args.maps,ROOT/"configs/inference.json",ROOT/"configs/protocol.json",
                 Path(__file__),Path(__file__).with_name("v4_models.py")):
        signature.update(path.read_bytes())
    folder = ROOT/"private/bootstrap"/signature.hexdigest()[:15]
    folder.mkdir(parents=True,exist_ok=True)
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(args.workers,initializer=initialize,initargs=(args,METHODS,penalty)) as pool:
        for stage in ("pilot","formal"):
            n = SETTINGS[stage+"_repetitions"]
            jobs = [(folder,stage,i,min(i+10,n),SETTINGS[stage+"_seed_base"]) for i in range(0,n,10)]
            paths = []
            start = time.perf_counter()
            for k,future in enumerate(as_completed([pool.submit(write_chunk,job) for job in jobs]),1):
                paths.append(future.result())
                if k%10==0 or k==len(jobs):
                    print(stage,"batches complete",k,"/",len(jobs),
                          "elapsed_s",round(time.perf_counter()-start,1),flush=True)
            blocks = combined_chunks(paths,n)
            n_succeeded = int(np.sum(blocks["status"]=="ok"))
            if stage=="pilot" and n_succeeded<.95*n:
                raise RuntimeError(f"Too many failed numerical fits in pilot: {n_succeeded}/{n}")
            report_rounds(data,blocks,stage,n,preferred,published,output)
            if args.pilot_only:
                return


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--features",type=Path,default=ROOT/"private/V4_Scan_Metrics.csv")
    cli.add_argument("--maps",type=Path,default=ROOT/"private/V4_Maps.npz")
    cli.add_argument("--calibration",type=Path,default=ROOT/"results/Calibration_Selection.json")
    cli.add_argument("--development",type=Path,default=ROOT/"results/Lambda_Selection.json")
    cli.add_argument("--output",type=Path,default=ROOT/"results")
    cli.add_argument("--bootstrap-fallback",action="store_true",help="Compute predeclared A0 and A3 without a formal decision")
    cli.add_argument("--point-check-only",action="store_true")
    cli.add_argument("--pilot-only",action="store_true")
    cli.add_argument("--workers",type=int,default=4)
    args = cli.parse_args()
    if args.bootstrap_fallback:
        gate = json.loads(args.development.read_text(encoding="utf-8"))
        if gate.get("stage")!="development":
            raise ValueError("Expected the locked blinded development penalty")
        preferred = None
        methods = tuple(SETTINGS["if_no_calibrated_method"])
    else:
        gate = read_calibration(args.calibration)
        preferred = gate["preferred_method"]
        methods = (preferred,) if preferred is not None else tuple(SETTINGS["if_no_calibrated_method"])
    penalty=float(gate["lambda"])
    if not np.isfinite(penalty) or penalty<0:
        raise ValueError("Invalid locked transition penalty")
    with threadpool_limits(1):
        run(args,methods,penalty,preferred)


if __name__=="__main__":
    main()
