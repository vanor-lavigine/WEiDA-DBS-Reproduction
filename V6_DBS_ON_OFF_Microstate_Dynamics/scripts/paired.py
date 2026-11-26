"""Paired scalar and multivariate tests with fixed-template resampling."""

from functools import lru_cache
from itertools import permutations
from math import factorial

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from v6_common import CFG
from micro_inference import UniqueResampling
from statistics_v4 import hc3_t, normality, tail_probability
from v5_stats import paired_wald, regression_wald


ALPHA = CFG["alpha"]
B = CFG["bootstrap"]
RESAMPLE = UniqueResampling(CFG["seed"],CFG["permutations"],B)
P_MODELS = {"raw_p":"", "permutation_p":"permutation_", "fd_p":"fd_", "dvars_p":"dvars_", "order_subset_p":"order_subset_"}


@lru_cache(maxsize=32)
def signs(n):
    total = (1 << n)-1
    if total >= CFG["permutations"]:
        return RESAMPLE.signs(n)
    masks = np.arange(1,total+1)
    return 1.-2.*((masks[:,None] >> np.arange(n)) & 1)


@lru_cache(maxsize=32)
def association_permutations(n):
    """Sample different nonidentity participant orders for the Spearman null."""
    if factorial(n)-1 <= CFG["permutations"]:
        return np.asarray(list(permutations(range(n)))[1:])
    rng = np.random.default_rng([CFG["seed"],203,n])
    seen = {tuple(range(n))}
    rows = []
    while len(rows)<CFG["permutations"]:
        for order in rng.random((CFG["permutations"]-len(rows),n)).argsort(axis=1):
            key = tuple(order)
            if key not in seen:
                rows.append(order)
                seen.add(key)
    return np.stack(rows)


def draws(n):
    return RESAMPLE.resample(n,n,True)[0]


def interval(x):
    observed = np.asarray(x)
    valid = observed[np.isfinite(observed)]
    return (*np.quantile(valid,[.025,.975]),len(valid)) if len(valid) else (np.nan,np.nan,0)


def standardized(delta, axis=-1):
    mean = delta.mean(axis=axis)
    sd = delta.std(axis=axis,ddof=1)
    return np.divide(mean,sd,out=np.full_like(mean,np.nan,dtype=float),where=sd>0)


def direction(delta):
    tolerance = CFG["tie_tolerance"]
    return np.where(delta>tolerance,1,np.where(delta < -tolerance,-1,0))


def bootstrap_record(mean_draws, estimate, vector_direction=None, prefix=""):
    low,high,n = interval(mean_draws)
    valid = np.isfinite(mean_draws)
    if n==0:
        agree = np.nan
    elif vector_direction is None:
        agree = np.mean(direction(mean_draws[valid]) == direction(estimate))
    else:
        agree = np.mean(vector_direction[valid])
    return {f"{prefix}{k}": v for k,v in dict(ci95_low=low, ci95_high=high, bootstrap_requested=B,
                                                bootstrap_valid=n, bootstrap_failed=B-n,
                                                bootstrap_direction_consistency=agree).items()}


def regression_bootstrap(y,cov,idx):
    xb,yb = cov[idx], y[idx]
    xmean,ymean = xb.mean(axis=1),yb.mean(axis=1)
    xc = xb-xmean[:,None]
    var = (xc*xc).sum(axis=1)
    coef = np.divide(np.einsum("ij,ij->i",xc,yb),var,out=np.full(len(var),np.nan),where=var>0.)
    return ymean-coef*xmean


def covariate_test(delta,cov,vector=False,greater=False,sign_set=None):
    """Freedman–Lane on one difference per independent participant."""
    valid = np.isfinite(cov) & (np.isfinite(delta).all(axis=1) if vector else np.isfinite(delta))
    y, x = delta[valid],cov[valid]
    if len(x)<3 or np.ptp(x)==0:
        return dict(effect=np.nan,p=np.nan,draws=0,n=len(x),status="covariate or paired data not estimable"), None
    design = np.c_[np.ones(len(x)),x]
    null = x[:,None]
    fitted = null @ np.linalg.lstsq(null,y,rcond=None)[0]
    residual = y-fitted
    flips = signs(len(x)) if sign_set is None else sign_set
    orientation = None
    if vector:
        observed = regression_wald(design,y,0,CFG["wald_ridge"])
        test = np.concatenate([regression_wald(design, fitted+s[...,None]*residual,0,CFG["wald_ridge"])
                               for s in np.array_split(flips,10)])
        v = np.linalg.lstsq(design,y,rcond=None)[0][0]
        effect = np.linalg.norm(v)
        idx = draws(len(x))
        dd = design[idx]
        ranks = np.linalg.matrix_rank(dd)==2
        changes = (np.linalg.pinv(dd)@y[idx])[:,0,:]
        boot = np.where(ranks,np.linalg.norm(changes,axis=1),np.nan)
        orientation = changes @ v > 0
    else:
        effect,observed = hc3_t(design,y,0)
        _,test = hc3_t(design,fitted+flips*residual,0)
        boot = regression_bootstrap(y,x,draws(len(x)))
    p = tail_probability(test,observed,two_sided=not (greater or vector))
    result = dict(effect=float(effect),p=float(p),draws=len(test),n=len(x),status="ok")
    result.update(bootstrap_record(boot,effect,vector_direction=orientation))
    return result,boot


def logical_p(delta,selected,supplied_signs=None):
    y = delta[selected]
    if not len(y):
        return np.nan,0
    z = signs(len(y)) if supplied_signs is None else supplied_signs
    return tail_probability(z@y/len(y),y.mean()),len(z)


def scalar(on,off,fd,dvars,verified):
    eligible = np.isfinite(on) & np.isfinite(off)
    u,v = on[eligible],off[eligible]
    delta = u-v
    n = len(delta)
    row = dict(n_pairs=n,missing_pairs=int(len(on)-n),contrast="ON-OFF")
    if n<2:
        return {**row, "status":"insufficient complete pairs"},None
    measured = delta.mean()
    med = np.median(delta)
    diag = normality(delta)
    ttest = stats.ttest_1samp(delta,0.)
    t_p = float(ttest.pvalue)
    rounded = delta.round(12)
    wilcoxon = stats.wilcoxon(rounded,method="approx").pvalue if np.any(rounded) else 1.
    chosen = "paired t" if diag>=CFG["normality_alpha"] and np.isfinite(t_p) else "Wilcoxon signed-rank"
    perm,n_perm = logical_p(delta,np.ones(n,dtype=bool))
    t_ci = ttest.confidence_interval(.95)
    index = draws(n)
    resamples = delta[index]
    avg = resamples.mean(axis=1)
    median_draws = np.median(resamples,axis=1)
    dz_draws = standardized(resamples,axis=1)
    medlow,medhigh,nmed = interval(median_draws)
    dzlow,dzhigh,ndz = interval(dz_draws)
    row.update(status="ok", mean_on=u.mean(),mean_off=v.mean(),mean_difference=measured,
               median_difference=med,sd_difference=delta.std(ddof=1),dz=float(standardized(delta)),
               dz_ci95_low=dzlow,dz_ci95_high=dzhigh,dz_bootstrap_valid=ndz,median_ci95_low=medlow,median_ci95_high=medhigh,
               median_bootstrap_valid=nmed,t_ci95_low=t_ci.low,t_ci95_high=t_ci.high,
               raw_test=chosen, shapiro_p=diag,raw_p=t_p if chosen=="paired t" else wilcoxon,
               paired_t_p=t_p,wilcoxon_p=wilcoxon,permutation_p=perm,permutation_draws=n_perm,
               **bootstrap_record(avg,measured))
    adjusted_draws = []
    for tag, cov in (("fd",fd),("dvars",dvars)):
        adjusted, boot = covariate_test(delta,cov[eligible])
        row.update({f"{tag}_{k}":v for k,v in adjusted.items()})
        adjusted_draws.append(boot if boot is not None else np.full(B,np.nan))
    subset = verified[eligible]
    q,nq = logical_p(delta,subset)
    subset_delta = delta[subset]
    row.update(order_subset_p=q,order_subset_draws=nq,order_subset_n=len(subset_delta),
               order_subset_effect=subset_delta.mean() if len(subset_delta) else np.nan)
    codes = direction(delta)
    for label, level in [("positive",1),("negative",-1),("tied",0)]:
        count = int(np.sum(codes==level))
        a,b = proportion_confint(count,n,method="wilson")
        row.update({f"n_{label}":count,f"proportion_{label}":count/n,f"{label}_wilson_low":a,f"{label}_wilson_high":b})
    return row,np.column_stack([avg,median_draws,dz_draws,*adjusted_draws])


def omnibus(on,off,fd,dvars,verified):
    keep = np.isfinite(on).all(axis=1) & np.isfinite(off).all(axis=1)
    delta = (on-off)[keep]
    n,dimensions = delta.shape
    point = delta.mean(axis=0)
    ridge = CFG["wald_ridge"]
    statistic = paired_wald(delta,ridge)
    fake = signs(n)[...,None]*delta
    p = tail_probability(paired_wald(fake,ridge),statistic,two_sided=False)
    resampled = delta[draws(n)].mean(axis=1)
    values = np.linalg.norm(resampled,axis=1)
    length = np.linalg.norm(point)
    row = dict(n_pairs=n,dimensions=dimensions,contrast="ON-OFF",statistic=statistic,
               mean_difference_norm=length, raw_test="paired shrinkage Wald with sign-flip reference",raw_p=p,
               permutation_p=p, permutation_draws=len(fake), status="ok",
               **bootstrap_record(values,length,vector_direction=(resampled @ point)>0.))
    adjusted_draws = []
    for tag, cov in (("fd",fd),("dvars",dvars)):
        adjusted, boot = covariate_test(delta,cov[keep],vector=True)
        row.update({f"{tag}_{k}":v for k,v in adjusted.items()})
        adjusted_draws.append(boot if boot is not None else np.full(B,np.nan))
    if verified[keep].sum()>=3:
        subset = delta[verified[keep]]
        sims = paired_wald(signs(len(subset))[...,None]*subset,ridge)
        q = tail_probability(sims,paired_wald(subset,ridge),two_sided=False)
        row.update(order_subset_p=q,order_subset_effect=np.linalg.norm(subset.mean(axis=0)),order_subset_n=len(subset))
    return row,np.column_stack([values,*adjusted_draws]),resampled


def influence(delta):
    data = delta[np.isfinite(delta)]
    n = len(data)
    sample = np.repeat(data[None,:],n,axis=0)[~np.eye(n,dtype=bool)].reshape(n,n-1)
    loo = sample.mean(axis=1)
    dz = standardized(sample,axis=1)
    nominal = stats.ttest_1samp(sample,0.,axis=1).pvalue
    codes = direction(data)
    n_nonzero = np.count_nonzero(codes)
    sign_p = stats.binomtest(np.sum(codes>0),n_nonzero).pvalue if n_nonzero else 1.
    saved = dict(n_pairs=n,trimmed_mean=stats.trim_mean(data,CFG["trim_fraction"]),exact_sign_p=sign_p,
                 leave_one_out_mean_low=loo.min(),leave_one_out_mean_high=loo.max(),
                 leave_one_out_dz_low=np.nanmin(dz),leave_one_out_dz_high=np.nanmax(dz),
                 leave_one_out_t_p_low=np.nanmin(nominal),leave_one_out_t_p_high=np.nanmax(nominal),
                 max_absolute_leave_one_out_shift=np.max(np.abs(loo-data.mean())),
                 leave_one_out_direction_consistency=np.mean(direction(loo)==direction(data.mean())))
    return saved, np.c_[loo,dz,nominal]


def spearman_rows(a,b):
    ra,rb = stats.rankdata(a,axis=-1),stats.rankdata(b,axis=-1)
    ra -= ra.mean(axis=-1,keepdims=True)
    rb -= rb.mean(axis=-1,keepdims=True)
    numerator = (ra*rb).sum(axis=-1)
    denominator = np.linalg.norm(ra,axis=-1)*np.linalg.norm(rb,axis=-1)
    return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>0)


def association(x,y,fd):
    valid = np.isfinite(x)&np.isfinite(y)&np.isfinite(fd)
    x,y,fd = x[valid],y[valid],fd[valid]
    if len(x)<3 or min(np.ptp(x),np.ptp(y))==0:
        return dict(n_pairs=len(x),status="rank association undefined",raw_p=np.nan,permutation_p=np.nan,fd_p=np.nan)
    n = len(x)
    rank = stats.spearmanr(x,y)
    ix = draws(n)
    boot = spearman_rows(x[ix],y[ix])
    order = association_permutations(n)
    null = spearman_rows(np.broadcast_to(x,order.shape),y[order])
    p = tail_probability(null,rank.statistic)
    loo = np.array([spearman_rows(np.delete(x,i),np.delete(y,i)) for i in range(n)])
    row = dict(n_pairs=n,status="ok",rho=rank.statistic,raw_p=rank.pvalue,permutation_p=p,permutation_draws=len(order),
               loo_rho_low=np.nanmin(loo) if np.isfinite(loo).any() else np.nan,
               loo_rho_high=np.nanmax(loo) if np.isfinite(loo).any() else np.nan,
               **bootstrap_record(boot,rank.statistic))
    if n<4 or np.ptp(fd)==0:
        return {**row,"fd_p":np.nan,"fd_status":"FD rank regression not estimable"}
    cov = np.c_[np.ones(n),stats.rankdata(fd)]
    a,b = stats.rankdata(x),stats.rankdata(y)
    fit_b = cov @ np.linalg.lstsq(cov,b,rcond=None)[0]
    residual_b = b-fit_b
    residual_a = a-cov @ np.linalg.lstsq(cov,a,rcond=None)[0]
    design = np.c_[cov,a]
    if np.linalg.matrix_rank(design)<3:
        return {**row,"fd_p":np.nan,"fd_status":"FD rank regression is singular"}
    _,t = hc3_t(design,b,2)
    _,fake = hc3_t(design,fit_b+residual_b[order],2)
    row.update(fd_rho=np.corrcoef(residual_a,residual_b)[0,1],fd_p=tail_probability(fake,t),fd_status="ok")
    return row


def adjust(frame, p_models=P_MODELS):
    d = frame.copy()
    d["family_size"] = d.groupby("family").endpoint.transform("size")
    for p,stem in p_models.items():
        if p not in d:
            d[p] = np.nan
        for procedure,prefix in (("holm","holm"),("fdr_bh","fdr")):
            name = f"{stem}{prefix}_p"
            d[name] = np.nan
            for _,group in d.groupby("family"):
                i = group.index[group[p].notna()]
                if len(i):
                    d.loc[i,name] = multipletests(d.loc[i,p],method=procedure)[1]
    return d


def classify(frame,omnibus_table):
    d = frame.copy()
    parent = omnibus_table.set_index("endpoint").loc["transition_omnibus"]
    localized = d.family.eq("Transition localization")
    for p, stem in P_MODELS.items():
        d[f"{stem}gate_open"] = ~localized | (parent[f"{stem}holm_p"]<ALPHA)
        holm = d[f"{stem}gate_open"] & d[f"{stem}holm_p"].lt(ALPHA)
        nominal = d[f"{stem}gate_open"] & (d[p].lt(ALPHA) | d[f"{stem}fdr_p"].lt(ALPHA))
        d[f"{stem}evidence_class"] = np.select([holm & d.family.eq("Primary"),holm,nominal],["A","B","C"],default="D")
    return d
