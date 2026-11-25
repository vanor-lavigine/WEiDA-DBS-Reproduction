"""Participant permutation, Freedman–Lane and fixed-template bootstrap helpers."""

import numpy as np
import pandas as pd
from scipy import stats

from v5_data import CFG, HERE
from micro_inference import UniqueResampling
from statistics_v4 import corrections, freedman_lane, hc3_t, outcome_test, studentize, tail_probability


RESAMPLE = UniqueResampling(CFG["seed"], CFG["permutations"], CFG["bootstrap"])
P_COLUMNS = {"raw_p": "", "permutation_p": "permutation_", "fd_p": "fd_"}


def groups(data, keys, contrast):
    a,b = contrast.split("-")
    fields = list(keys) + ["fd"]
    if contrast == "ON-OFF":
        d = data.loc[data.condition.isin((a,b))].pivot(index="subject", columns="condition", values=fields).dropna()
        columns = [d.xs(x, level="condition", axis=1) for x in (a,b)]
    else:
        columns = [data.loc[data.condition.eq(x), fields+["subject"]].dropna().sort_values("subject") for x in (a,b)]
    x,y = columns
    return x[list(keys)].to_numpy(), y[list(keys)].to_numpy(), x.fd.to_numpy(), y.fd.to_numpy()


def ci(x):
    valid = np.asarray(x)[np.isfinite(x)]
    if len(valid) == 0:
        return np.nan,np.nan,0
    low,high = np.quantile(valid,[.025,.975])
    return low,high,len(valid)


def describe(values, group):
    return {f"{key}_{group}": val for key,val in dict(mean=values.mean(), sd=values.std(ddof=1), median=np.median(values),
                                                       iqr=stats.iqr(values)).items()}


def scalar_tests(data, metrics, result_family, contrasts=None):
    rows = []
    for contrast in CFG["contrasts"] if contrasts is None else contrasts:
        for name, key in metrics:
            a,b,fd_a,fd_b = groups(data,[key],contrast)
            x,y = a[:,0], b[:,0]
            paired = contrast=="ON-OFF"
            row = dict(endpoint=name, feature=key, contrast=contrast, family=result_family, n_a=len(x), n_b=len(y),
                       paired=paired, defined=len(x)>=3 and len(y)>=3)
            if row["defined"]:
                raw = outcome_test(x,y,paired,CFG,RESAMPLE)
                fd = freedman_lane(x,y,fd_a,fd_b,paired,RESAMPLE)
                row.update(raw, **describe(x,"a"), **describe(y,"b"))
                row.update(fd_p=fd.get("fd_permutation_p",np.nan), fd_effect=fd.get("fd_permutation_effect",np.nan),
                           fd_status=fd["fd_permutation_status"], permutation_draws=CFG["permutations"],
                           bootstrap_requested=CFG["bootstrap"], bootstrap_valid=CFG["bootstrap"],
                           bootstrap_failure_reason="", effect_units="unstandardized mean difference")
            else:
                row.update(bootstrap_requested=CFG["bootstrap"], bootstrap_valid=0, bootstrap_failure_reason="insufficient complete cases")
            rows.append(row)
    return pd.DataFrame(rows)


def distance(x, y):
    a = np.sum(x*x,axis=-1)[..., :,None]
    b = np.sum(y*y,axis=-1)[..., None,:]
    return np.sqrt(np.maximum(a+b-2*x@y.swapaxes(-1,-2), 0.))


def energy(x,y):
    a,b = distance(x,x),distance(y,y)
    # The within-group distance is exactly zero on the diagonal.
    i,j = np.arange(a.shape[-1]),np.arange(b.shape[-1])
    a[...,i,i],b[...,j,j] = 0.,0.
    return 2*distance(x,y).mean(axis=(-1,-2))-a.mean(axis=(-1,-2))-b.mean(axis=(-1,-2))


def centroid_distance(x,y):
    return np.linalg.norm(x.mean(axis=-2)-y.mean(axis=-2),axis=-1)


def dispersion_distance(x):
    return np.linalg.norm(x-x.mean(axis=-2,keepdims=True),axis=-1)*np.sqrt(x.shape[-2]/(x.shape[-2]-1))


def dispersion_effect(x,y):
    return dispersion_distance(x).mean(axis=-1)-dispersion_distance(y).mean(axis=-1)


def dispersion_stat(x,y):
    a,b = dispersion_distance(x),dispersion_distance(y)
    se = np.sqrt(a.var(axis=-1,ddof=1)/x.shape[-2]+b.var(axis=-1,ddof=1)/y.shape[-2])
    return studentize(a.mean(axis=-1)-b.mean(axis=-1),se)


def covariance_mean(x):
    centered = x-x.mean(axis=-2,keepdims=True)
    n = x.shape[-2]
    return centered.swapaxes(-1,-2) @ centered/(n*(n-1))


def quadratic(delta,cov,ridge):
    d = delta.shape[-1]
    if ridge:
        trace = np.trace(cov,axis1=-2,axis2=-1)
        cov = (1-ridge)*cov+ridge*(trace/d)[...,None,None]*np.eye(d)
    z = np.linalg.solve(cov,delta[...,None])[...,0]
    return np.einsum("...i,...i->...",delta,z)


def location_wald(x,y,ridge=0.):
    return quadratic(x.mean(axis=-2)-y.mean(axis=-2), covariance_mean(x)+covariance_mean(y),ridge)


def paired_wald(d,ridge):
    return quadratic(d.mean(axis=-2),covariance_mean(d),ridge)


def regression_wald(x, y, contrast_index, ridge):
    inverse = np.linalg.pinv(x)
    h = np.sum(x*inverse.T,axis=1)
    b = inverse@y
    residual = y-x@b
    w = inverse[contrast_index]/(1-h)
    adjusted = residual*w[:,None]
    return quadratic(b[...,contrast_index,:],adjusted.swapaxes(-1,-2)@adjusted,ridge)


def fd_adjusted_values(design, y):
    coefficient = (np.linalg.pinv(design) @ y)[..., -1:, :]
    return y - design[:, -1, None] * coefficient


def batches(arrays,n=200):
    return np.array_split(arrays, max(1, int(np.ceil(len(arrays)/n))))


def pvalue(null, observed):
    if len(null) != CFG["permutations"] or not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("The fixed permutation family did not produce 10,000 finite statistics")
    return tail_probability(null, observed, two_sided=False)


def two_sample_mv(x,y,fd_x,fd_y,kind):
    nx,ny = len(x),len(y)
    pooled = np.vstack((x,y))
    fd = np.r_[fd_x,fd_y]
    z = np.c_[np.ones(nx+ny),fd-fd.mean()]
    x_full = np.c_[z[:,0],np.r_[np.ones(nx),np.zeros(ny)],z[:,1]]
    fitted = z @ np.linalg.lstsq(z, pooled,rcond=None)[0]
    errors = pooled-fitted
    order = RESAMPLE.independent(nx,ny)
    ridge = CFG["transition_ridge"] if kind == "transition_wald" else 0.
    if kind == "coverage_energy":
        stat, effect, units = energy, energy, "Aitchison energy distance statistic"
    elif kind == "coverage_dispersion":
        stat = lambda a,b: abs(dispersion_stat(a,b))
        effect,units = dispersion_effect,"Aitchison mean dispersion difference"
    else:
        stat = lambda a,b: location_wald(a,b,ridge)
        effect,units = centroid_distance,"ILR centroid distance"
    observed = stat(x,y)
    perm = []
    residual = []
    location = kind in ("coverage_center","transition_wald")
    for shuffle in batches(order):
        perm.append(stat(pooled[shuffle[:,:nx]],pooled[shuffle[:,nx:]]))
        star = fitted+errors[shuffle]
        if location:
            residual.append(regression_wald(x_full,star,1,ridge))
        else:
            corrected = fd_adjusted_values(x_full,star)
            residual.append(stat(corrected[...,:nx,:],corrected[...,nx:,:]))
    corrected = fd_adjusted_values(x_full,pooled)
    fd_observed = regression_wald(x_full,pooled,1,ridge) if location else stat(corrected[:nx],corrected[nx:])
    a,b = RESAMPLE.resample(nx,ny,False)
    values = np.concatenate([effect(x[i],y[j]) for i,j in zip(batches(a),batches(b))])
    low,high,n = ci(values)
    raw = pvalue(np.concatenate(perm),observed)
    fd_p = pvalue(np.concatenate(residual),fd_observed)
    summary = dict(raw_p=raw, permutation_p=raw, fd_p=fd_p, statistic=observed, fd_statistic=fd_observed,
                   effect_size=effect(x,y), ci95_low=low, ci95_high=high, effect_units=units,
                   fd_effect=effect(corrected[:nx],corrected[nx:]),
                   centroid_distance=centroid_distance(x,y), group_dispersion_a=dispersion_distance(x).mean(),
                   group_dispersion_b=dispersion_distance(y).mean(), bootstrap_valid=n)
    return summary


def paired_mv(x,y,fd_x,fd_y):
    change = x-y
    motion = fd_x-fd_y
    ridge = CFG["transition_ridge"]
    design = np.c_[np.ones(len(x)),motion]
    fitted = motion[:,None] @ np.linalg.lstsq(motion[:,None],change,rcond=None)[0]
    err = change-fitted
    signs = RESAMPLE.signs(len(x))
    obs = paired_wald(change,ridge)
    obs_fd = regression_wald(design,change,0,ridge)
    p = pvalue(paired_wald(signs[...,None]*change,ridge),obs)
    fd_p = pvalue(regression_wald(design,fitted+signs[...,None]*err,0,ridge),obs_fd)
    draws,_ = RESAMPLE.resample(len(x),len(x),True)
    low,high,n = ci(np.linalg.norm(change[draws].mean(axis=1),axis=-1))
    return dict(raw_p=p, permutation_p=p, fd_p=fd_p, statistic=obs, fd_statistic=obs_fd,
                effect_size=np.linalg.norm(change.mean(axis=0)), ci95_low=low, ci95_high=high,
                fd_effect=np.linalg.norm(np.linalg.lstsq(design,change,rcond=None)[0][0]),
                effect_units="norm of paired mean 12D ILR change", bootstrap_valid=n)


def omnibus(data,keys,contrast,kind):
    x,y,fda,fdb = groups(data,keys,contrast)
    paired = contrast == "ON-OFF"
    row = dict(endpoint=kind, feature=kind, contrast=contrast, family=kind, n_a=len(x), n_b=len(y), paired=paired)
    if min(len(x),len(y)) < 5:
        return {**row,"defined":False,"bootstrap_requested":CFG["bootstrap"],"bootstrap_valid":0,
                "bootstrap_failure_reason":"insufficient participants with all rows defined"}
    result = paired_mv(x,y,fda,fdb) if paired else two_sample_mv(x,y,fda,fdb,kind)
    row.update(result, defined=True, permutation_draws=CFG["permutations"], raw_p_basis="Monte Carlo permutation",
               bootstrap_requested=CFG["bootstrap"], bootstrap_failure_reason="")
    return row


def restoration(data):
    keys = [f"ilr_{i}" for i in range(1,4)]
    h = data.loc[data.condition.eq("HC")].sort_values("subject")[keys].to_numpy()
    x,y,fx,fy = groups(data,keys,"ON-OFF")
    center = h.mean(axis=0)
    d_on = np.linalg.norm(x-center,axis=1)
    d_off = np.linalg.norm(y-center,axis=1)
    improve = d_off-d_on
    Δfd = fy-fx
    signs = RESAMPLE.signs(len(x))
    observed = improve.mean()
    p = pvalue((signs@improve)/len(improve),observed)
    p_two = tail_probability((signs@improve)/len(improve),observed)
    fd_fit = Δfd*np.linalg.lstsq(Δfd[:,None],improve,rcond=None)[0][0]
    design = np.c_[np.ones(len(x)),Δfd]
    fd_eff, fd_stat = hc3_t(design, improve,0)
    _, fiducial = hc3_t(design, fd_fit+signs*(improve-fd_fit),0)
    fdp = pvalue(fiducial,fd_stat)
    patients,hc = RESAMPLE.resample(len(x),len(h),False)
    anchors = h[hc].mean(axis=1,keepdims=True)
    boot = np.linalg.norm(y[patients]-anchors,axis=-1)-np.linalg.norm(x[patients]-anchors,axis=-1)
    low,high,n = ci(boot.mean(axis=1))
    a,b,_ = ci(np.median(boot,axis=1))
    param = stats.ttest_1samp(improve,0.,alternative="greater").pvalue
    row = dict(endpoint="restoration", feature="restoration", family="restoration", contrast="ON-OFF", paired=True,
               defined=True,n_a=len(x),n_b=len(y),n_hc_reference=len(h), effect_size=observed, ci95_low=low,ci95_high=high,
               effect_units="Aitchison distance OFF minus ON", median_improvement=np.median(improve),
               median_ci95_low=a,median_ci95_high=b,mean_on=d_on.mean(),mean_off=d_off.mean(),
               dz=observed/improve.std(ddof=1),raw_p=param,permutation_p=p,fd_p=fdp,
               fd_effect=float(fd_eff),two_sided_permutation_p=p_two,permutation_draws=CFG["permutations"],
               bootstrap_requested=CFG["bootstrap"],bootstrap_valid=n,bootstrap_failure_reason="",raw_p_basis="one-sided paired t")
    return row,np.c_[d_off,d_on,improve]


def adjustment(table):
    d = table.copy()
    corrections(d,P_COLUMNS,["family"])
    d["family_size"] = d.groupby("family").endpoint.transform("size")
    return d
