"""Matched primary-band 4-by-4 state co-occurrences and scan-wise circular-shift tests."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from compare_legacy_105 import REPORTS
from matched24_analysis import shared_interior
from matched24_data import load_study
from peak_states import backfit
from run import digest, new_output, write_json
from same_band_analysis import SOURCES
from same_band_stats import SEED, SHIFT_DRAWS, group_shifts, joint_counts_by_shift, shift_profile, zero_motion_weights
from stats import holm_adjust
from verify_peak_analysis import holm
from verify_same_band import amplitude_labels


PARENT = REPORTS / "matched24_primary/results"
TEMPLATE_SHA = "17db70b146e1b431007a5589cc10be33ddb0c1e605471ecf4eba4a510225a408"
BOOT_DRAWS = 19999


def rebuild_tables(root):
    archive = json.loads((PARENT/"audit.json").read_text(encoding="utf-8"))
    matched = root/"V100脑深部刺激WEiDA复现_20261118/代码与辅助/同人群精确度对照"
    for src in ("legacy105","v100_syn9"):
        for name in ("inputs.mat","author_matched_k4.mat"):
            assert digest(matched/src/name) == archive["source_sha256"][f"{src}/{name}"]
    study = load_study(root)
    file = PARENT/"shared_templates.tsv"
    assert hashlib.sha256(file.read_bytes().replace(b"\r\n",b"\n")).hexdigest() == TEMPLATE_SHA
    templates = pd.read_csv(file,sep="\t",index_col=[0,1],float_precision="round_trip")
    np.testing.assert_array_equal(templates.columns,study.axis)
    for method in ("WEiDA","micro"):
        np.testing.assert_array_equal(templates.loc[method].index,[1,2,3,4])
    phase,amp = (templates.loc[m].to_numpy() for m in ("WEiDA","micro"))
    tables, profiles, fds = [[],[]], [{},{}], [{},{}]
    for p in range(24):
        for v,visit in enumerate(("OFF","ON")):
            pair = [study.visits[si,p,v] for si in range(2)]
            micro = [backfit(item.peaks,amp)[0] for item in pair]
            keep = shared_interior(micro)
            for si,item in enumerate(pair):
                np.testing.assert_array_equal(micro[si],amplitude_labels(item.wave,amp))
                s = cdist(item.features[keep],phase,"sqeuclidean").argmin(1)
                distances = [((item.features[keep]-c)**2).sum(1) for c in phase]
                np.testing.assert_array_equal(s,np.argmin(distances,axis=0))
                m = micro[si][keep]
                # Public matrices have WEiDA on rows and microstates on columns.
                counts = joint_counts_by_shift(m,s).transpose(0,2,1)
                direct = np.array([np.bincount(4*np.roll(s,k)+m,minlength=16).reshape(4,4)
                                   for k in range(len(s))])
                np.testing.assert_array_equal(counts,direct)
                tables[si].append(counts)
                profiles[si][p,visit] = shift_profile(m,s)
                fds[si][p,visit] = item.peaks.fd
    lengths = np.array([len(t) for t in tables[0]])
    np.testing.assert_array_equal(lengths,[len(t) for t in tables[1]])
    previous = pd.read_csv(PARENT/"fold_quality.csv").set_index("fold").loc["all"]
    np.testing.assert_array_equal([len(lengths),lengths.sum(),lengths.min(),np.median(lengths),lengths.max()],
        [2*previous.people,previous.common_frames_per_source,previous.common_min,previous.common_median,previous.common_max])
    return tables,profiles,fds,study.audit


def check_null_regressions(probabilities,lag,fd,null,observed):
    """Fit OLS to separate patient means for every sampled displacement and cell."""
    design = np.column_stack([np.ones(len(fd)),fd])
    tail = np.zeros((4,4),dtype=int)
    for start in range(0,len(null),1000):
        block = slice(start,min(start+1000,len(null)))
        patient = np.stack([.5*(probabilities[2*p][lag[2*p,block]]+
                                probabilities[2*p+1][lag[2*p+1,block]]) for p in range(len(fd))])
        intercept = np.linalg.lstsq(design,patient.reshape(len(fd),-1),rcond=None)[0][0]
        check = intercept.reshape(-1,4,4)
        np.testing.assert_allclose(check,null[block],atol=1e-12,rtol=0)
        tail += (check>=observed-1e-12).sum(axis=0)
    return tail


def cell_tests(tables,fd,seed,draws=SHIFT_DRAWS):
    n = len(fd)
    assert len(tables)==2*n
    weights = zero_motion_weights(fd)
    probabilities = [t/len(t) for t in tables]
    obs_visits = np.stack([a[0] for a in probabilities])
    # Product of the two margins is the exact expectation over all within-scan shifts.
    expected_visits = obs_visits.sum(2)[:,:,None] * obs_visits.sum(1)[:,None,:]
    np.testing.assert_allclose(expected_visits,[a.mean(axis=0) for a in probabilities],atol=1e-12)
    obs_patients = obs_visits.reshape(n,2,4,4).mean(1)
    exp_patients = expected_visits.reshape(n,2,4,4).mean(1)
    observed,expected = (np.einsum("p,pij->ij",weights,a) for a in (obs_patients,exp_patients))
    design = np.column_stack([np.ones(n),fd])
    for check,values in ((observed,obs_patients),(expected,exp_patients)):
        np.testing.assert_allclose(check,np.linalg.lstsq(design,values.reshape(n,16),rcond=None)[0][0].reshape(4,4),
                                   rtol=0,atol=1e-12)
    excess = observed-expected
    np.testing.assert_allclose(observed.sum(),1)
    for axis in (0,1):
        np.testing.assert_allclose(excess.sum(axis),0,atol=1e-12)

    rng = np.random.default_rng(seed)
    lag = np.stack([rng.integers(len(a),size=draws).astype(np.int32) for a in tables])
    null = np.zeros((draws,4,4))
    for i,(a,w) in enumerate(zip(probabilities,np.repeat(weights/2,2))):
        null += w*a[lag[i]]
    extreme = (null>=observed-1e-12).sum(0)
    np.testing.assert_array_equal(extreme,check_null_regressions(probabilities,lag,fd,null,observed))
    raw_p = (extreme+1)/(draws+1)
    adjusted = holm_adjust(raw_p.ravel()).reshape(4,4)
    np.testing.assert_allclose(adjusted.ravel(),holm(raw_p.ravel()))
    lower,upper = np.quantile(null,[.025,.975],axis=0)
    raw_counts = np.stack([t[0] for t in tables]).sum(0)
    rows = []
    for i,j in np.ndindex(4,4):
        rows.append(dict(weida=i+1,micro=j+1,raw_frames=int(raw_counts[i,j]),
                         observed_pct=100*observed[i,j],shift_expected_pct=100*expected[i,j],
                         excess_pp=100*excess[i,j],null_q025_pct=100*lower[i,j],null_q975_pct=100*upper[i,j],
                         patient_equal_unadjusted_pct=100*obs_patients[:,i,j].mean(),
                         patient_equal_unadjusted_expected_pct=100*exp_patients[:,i,j].mean(),
                         extreme=int(extreme[i,j]),p_upper=raw_p[i,j],holm_16=adjusted[i,j],
                         enriched_holm=bool(adjusted[i,j]<.05 and excess[i,j]>0)))
    return rows,(obs_patients-exp_patients)[:,1,:]*100


def winner_weights(excess):
    tied = np.isclose(excess,excess.max(axis=-1,keepdims=True),rtol=0,atol=1e-12)
    return tied/tied.sum(axis=-1,keepdims=True)


def s2_bootstrap(effects,motion,by_cell):
    n = effects[0].shape[0]
    draw = np.random.default_rng([SEED,24,16,2]).integers(n,size=(BOOT_DRAWS,n))
    boot = []
    est = []
    for values,fd in zip(effects,motion):
        est.append(zero_motion_weights(fd)@values)
        w = zero_motion_weights(fd[draw])
        b = np.einsum("bp,bpm->bm",w,values[draw])
        # Verify paired sampling and re-estimation of the nuisance intercept.
        for index in (0,len(draw)//2,len(draw)-1):
            ix = draw[index]
            design = np.column_stack([np.ones(n),fd[ix]])
            np.testing.assert_allclose(b[index],np.linalg.lstsq(design,values[ix],rcond=None)[0][0],atol=1e-11)
        boot.append(b)
    diff = est[1]-est[0]
    boot_diff = boot[1]-boot[0]
    radius = np.quantile(np.abs(boot_diff-diff).max(axis=1),.95)
    wins = [winner_weights(x) for x in boot]
    ranks = []
    for si,src in enumerate(SOURCES):
        s2 = by_cell[(by_cell.source==src)&(by_cell.weida==2)].sort_values("micro")
        np.testing.assert_allclose(s2.excess_pp,est[si],atol=1e-12)
        s2_total = s2.observed_pct.sum()
        np.testing.assert_allclose(s2_total,s2.shift_expected_pct.sum())
        for j,r in enumerate(s2.itertuples()):
            ranks.append(dict(source=src,micro=r.micro,rank=int(1+(est[si]>est[si][j]+1e-12).sum()),
                              excess_pp=r.excess_pp,p_upper=r.p_upper,holm_16=r.holm_16,
                              micro_within_s2_pct=100*r.observed_pct/s2_total,
                              shift_expected_within_s2_pct=100*r.shift_expected_pct/s2_total,
                              bootstrap_first_fraction=wins[si][:,j].mean()))
    differences = [dict(micro=j+1,syn9_minus_105_excess_pp=d,simultaneous_ci_low_pp=d-radius,
                         simultaneous_ci_high_pp=d+radius) for j,d in enumerate(diff)]
    same_top = float((wins[0]*wins[1]).sum(1).mean())
    return ranks,differences,same_top


def plot_cells(cells,path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    available = {font.name for font in fm.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if f in available),None)
    title = "同帧共现率与随机移位期望之差" if font else "Same-frame co-occurrence above the circular-shift expectation"
    legend = ("实测减移位期望（百分点）  ·  * 表示 16 项 Holm p < 0.05" if font else
              "Observed minus shift expectation (percentage points)  ·  * Holm p < 0.05 across 16 cells")
    limit = max(1,np.ceil(abs(cells.excess_pp).max()))
    norm = matplotlib.colors.Normalize(-limit,limit)
    cmap = matplotlib.colormaps["RdBu_r"]
    with plt.rc_context({"font.family":font or "DejaVu Sans"}):
        fig,axes = plt.subplots(1,2,figsize=(8.3,4.7),layout="constrained")
        fig.suptitle(title,fontsize=15,fontweight="bold")
        for ax,src in zip(axes,SOURCES):
            part = cells[cells.source==src].sort_values(["weida","micro"])
            a = part.excess_pp.to_numpy().reshape(4,4)
            sig = part.enriched_holm.to_numpy().reshape(4,4)
            heat = ax.imshow(a,cmap=cmap,norm=norm)
            ax.set_title(src,fontsize=13,pad=10)
            ax.set_xticks(range(4),[f"M{i}" for i in range(1,5)])
            ax.set_yticks(range(4),[f"S{i}" for i in range(1,5)])
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(.5,3.6),minor=True)
            ax.set_yticks(np.arange(.5,3.6),minor=True)
            ax.tick_params(which="minor",length=0)
            ax.grid(which="minor",color="white",linewidth=1)
            for i,j in np.ndindex(4,4):
                red,green,blue,_ = cmap(norm(a[i,j]))
                ink = "white" if .2126*red+.7152*green+.0722*blue<.48 else "#17202A"
                ax.text(j,i,f"{a[i,j]:+.2f}"+("*" if sig[i,j] else ""),ha="center",va="center",
                        fontsize=11,color=ink,fontweight="bold" if sig[i,j] else "normal")
        bar = fig.colorbar(heat,ax=axes,orientation="horizontal",fraction=.06,pad=.05,aspect=45,shrink=.84)
        bar.set_label(legend,fontsize=10)
        fig.savefig(path,dpi=175,bbox_inches="tight",facecolor="white")
        plt.close(fig)


def run(root,destination):
    tables,profiles,fds,audit = rebuild_tables(root)
    reference = pd.read_csv(PARENT/"cooccurrence.csv").set_index("source")
    rows, effects, motions, ami_checks = [],[],[],[]
    for si,src in enumerate(SOURCES):
        seed = [SEED,si,0,99]
        ami,_ = group_shifts(profiles[si],fds[si],np.random.default_rng(seed))
        for name,value in ami.items():
            np.testing.assert_allclose(value,reference.loc[src,name],rtol=0,atol=1e-12)
        ami_checks.append({**ami,"source":src})
        delta_fd = np.array([fds[si][p,"ON"]-fds[si][p,"OFF"] for p in range(24)])
        group,individual_s2 = cell_tests(tables[si],delta_fd,seed)
        rows.extend(dict(source=src,**r) for r in group)
        effects.append(individual_s2)
        motions.append(delta_fd)
    cells = pd.DataFrame(rows)
    ranks,diff,same_top = s2_bootstrap(effects,motions,cells)
    out = new_output(destination)
    cells.to_csv(out/"cells.csv",index=False)
    pd.DataFrame(ranks).to_csv(out/"s2_ranking.csv",index=False)
    pd.DataFrame(diff).to_csv(out/"s2_pipeline_differences.csv",index=False)
    plot_cells(cells,out/"excess_cooccurrence.png")
    write_json(out/"audit.json",dict(data_kind="real",tr_s=2.1,frequency_hz=[.04,.07],**audit,
               common_frames_per_source=sum(len(a) for a in tables[0]),templates_sha256_lf=TEMPLATE_SHA,
               original_result_commit="40b33428640ff7a555f6e83a956cfaa2cfbef4a5",shift_draws=SHIFT_DRAWS,
               shift_seeds=[[SEED,si,0,99] for si in range(2)],holm_family_per_pipeline=16,
               bootstrap_draws=BOOT_DRAWS,bootstrap_seed=[SEED,24,16,2],bootstrap_same_top_fraction=same_top,
               motion_zero_within_data={src:bool(x.min()<=0<=x.max()) for src,x in zip(SOURCES,motions)},
               reproduced_ami=ami_checks,verified_cell_tests=32,verified_all_scan_lags=True,
               python=platform.python_version(),platform=platform.platform(),numpy=np.__version__,
               pandas=pd.__version__,scipy=scipy.__version__,sklearn=version("scikit-learn"),
               matplotlib=version("matplotlib")))
    print(f"Verified {len(cells)} cell tests and the archived omnibus results; exported group statistics.")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data_root,args.out)
