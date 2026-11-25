"""Poisson/NB2 episode counts with exposure offsets and patient clustering."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from v5_data import CFG, STATE
from v5_stats import RESAMPLE, ci, scalar_tests


WEIGHTS = np.array([[0,1,0],[0,0,1],[0,-1,1]])


def bootstrap_weights(data):
    hc = np.flatnonzero(data.condition.eq("HC"))
    on = np.flatnonzero(data.condition.eq("ON"))
    off = np.flatnonzero(data.condition.eq("OFF"))
    sid = data.subject.to_numpy()
    on = on[np.argsort(sid[on])]
    off = off[np.argsort(sid[off])]
    np.testing.assert_array_equal(sid[on],sid[off])
    a,b = RESAMPLE.resample(len(hc),len(on),False)
    indices = np.c_[hc[a],on[b],off[b]]
    return np.stack([np.bincount(row,minlength=len(data)) for row in indices])


def fit_counts(y,x,offset,groups,family):
    fitted = sm.GLM(y,x,family=family,offset=offset).fit(cov_type="cluster",cov_kwds={"groups":groups,"use_correction":True})
    if not fitted.converged or not np.isfinite(fitted.params).all() or not np.isfinite(fitted.cov_params()).all():
        raise ValueError("episode-count GLM failed to converge")
    return fitted


def refit_weighted(y,x,offset,weights,alpha,start):
    """Vectorized IRLS point estimates on the repeated participant samples."""
    output = []
    for w in np.array_split(weights, max(1,len(weights)//100)):
        b = np.tile(start,(len(w),1))
        converged = np.zeros(len(b),bool)
        for _ in range(60):
            m = np.exp(np.clip(b@x.T+offset,-35,35))
            d = 1+alpha*m
            score = (w*(y-m)/d)@x
            information = x.T @ ((w*m/d)[...,None]*x)
            step = np.linalg.pinv(information) @ score[...,None]
            b += np.clip(step[...,0],-2.,2.)
            converged = np.max(np.abs(step[...,0]),axis=1) < 1e-9
            if converged.all():
                break
        output.append(np.where(converged[:,None],b,np.nan))
    return np.vstack(output)


def diagnostic_p0(mu,alpha):
    return np.exp(-mu) if alpha==0 else (1+alpha*mu)**(-1/alpha)


def models(data):
    table = data.sort_values(["condition","subject"]).reset_index(drop=True).copy()
    for s in STATE:
        table[f"rate_{s}"] = table[f"count_{s}"]/table.exposure
    # Count-model and rate-permutation estimands are separately labeled.
    sensitivity = scalar_tests(table,[(f"M{s}",f"rate_{s}") for s in STATE],"Occurrence rate permutation sensitivity")
    rates = sensitivity.set_index(["endpoint","contrast"])
    design = np.c_[np.ones(len(table)),table.condition.eq("OFF"),table.condition.eq("ON")]
    motion = table.fd.to_numpy()-table.fd.mean()
    fd_design = np.c_[design,motion]
    group = pd.factorize(table.subject)[0]
    offset = np.log(table.exposure.to_numpy())
    w = bootstrap_weights(table)
    diagnostics, output = [],[]
    for state in STATE:
        y = table[f"count_{state}"].to_numpy()
        primary, adjusted = None,None
        why = ""
        alpha = 0.
        phi = np.nan
        expected_zeros, fitted_zeros = np.nan, np.nan
        model = "Poisson"
        try:
            base = fit_counts(y,design,offset,group,sm.families.Poisson())
            mu = base.fittedvalues
            phi = base.pearson_chi2/base.df_resid
            expected_zeros = diagnostic_p0(mu,0).sum()
            if phi>CFG["count_phi_nb2"]:
                alpha = max(1e-8,(((y-mu)**2-mu).sum())/(mu**2).sum())
                family = sm.families.NegativeBinomial(alpha=alpha)
                model = "NB2"
                primary = fit_counts(y,design,offset,group,family)
            else:
                family = sm.families.Poisson()
                primary = base
            fitted_zeros = diagnostic_p0(primary.fittedvalues,alpha).sum()
            adjusted = fit_counts(y,fd_design,offset,group,family)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            why = f"{type(error).__name__}: count GLM not estimable"
        diagnostics.append(dict(state=f"M{state}", model=model, alpha=alpha,poisson_pearson_over_df=phi,
                                observed_zero_scans=int(np.sum(y==0)), expected_poisson_zero_scans=expected_zeros,
                                expected_selected_model_zero_scans=fitted_zeros, n_scans=len(y),status=why or "ok"))
        if primary is not None:
            b0 = refit_weighted(y,design,offset,w,alpha,primary.params)
        if adjusted is not None:
            b1 = refit_weighted(y,fd_design,offset,w,alpha,adjusted.params)
        for k,contrast in enumerate(CFG["contrasts"]):
            s = rates.loc[(f"M{state}",contrast)]
            base_row = dict(endpoint=f"M{state}", feature=f"count_{state}", contrast=contrast,family="Count Occurrence",
                            defined=True,n_a=s.n_a,n_b=s.n_b,paired=s.paired,permutation_p=s.permutation_p,
                            fd_rate_permutation_p=s.fd_p,rate_effect=s.effect_size,permutation_draws=CFG["permutations"],
                            bootstrap_requested=CFG["bootstrap"],model=model,alpha=alpha,bootstrap_failure_reason="")
            if primary is None:
                output.append(dict(**base_row, effect_size=s.effect_size,ci95_low=s.ci95_low,ci95_high=s.ci95_high,
                                   bootstrap_valid=s.bootstrap_valid,raw_p=s.permutation_p,fd_p=s.fd_p,fd_effect=s.fd_effect,
                                   effect_units="episodes/s",fallback=why))
                continue
            contrast0 = WEIGHTS[k]
            delta = contrast0@primary.params
            se = np.sqrt(contrast0 @ primary.cov_params() @ contrast0)
            p = 2*stats.t.sf(abs(delta/se),len(np.unique(group))-1)
            low,high,n = ci(b0 @ contrast0)
            if adjusted is not None:
                contrast1 = np.r_[contrast0,0.]
                d1 = contrast1@adjusted.params
                se1 = np.sqrt(contrast1 @ adjusted.cov_params() @ contrast1)
                fd_p = 2*stats.t.sf(abs(d1/se1),len(np.unique(group))-1)
                lo1,hi1,n1 = ci(b1 @ contrast1)
                fd_row = dict(fd_p=fd_p, fd_effect=d1, fd_irr=np.exp(d1),fd_irr_ci95_low=np.exp(lo1),fd_irr_ci95_high=np.exp(hi1),
                              fd_bootstrap_valid=n1, fd_bootstrap_failure_reason="IRLS did not converge" if n1<CFG["bootstrap"] else "")
            else:
                fd_row = dict(fd_p=s.fd_p,fd_effect=np.nan,fd_fallback=why)
            base_row["bootstrap_failure_reason"] = "IRLS did not converge" if n<CFG["bootstrap"] else ""
            output.append(dict(**base_row,**fd_row,effect_size=delta,ci95_low=low,ci95_high=high,bootstrap_valid=n,raw_p=p,
                               irr=np.exp(delta),irr_ci95_low=np.exp(low),irr_ci95_high=np.exp(high),effect_units="log episode rate ratio",fallback=""))
    return pd.DataFrame(output),pd.DataFrame(diagnostics),sensitivity
