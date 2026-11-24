"""Export de-identified fit results and render figures on a public AAL atlas."""
import argparse
import gzip
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform
from scipy.stats import t


NAMES = {"weida_band":"WEiDA 主频段 0.04–0.07 Hz"}
COLORS = {"weida_band":"#116c74"}
METRICS = [("duration_s", "持续时间（秒）", 1), ("occurrence_s", "出现率（次/分）", 60),
           ("coverage", "时间占比（百分点）", 100)]


def read_reference(path):
    """Read the integer, axis-aligned NIfTI-1 public reference images for this figure."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    dims = struct.unpack_from("<8h", data, 40)
    if struct.unpack_from("<i",data)[0]!=348 or dims[0]!=3 or struct.unpack_from("<h",data,254)[0]<=0:
        raise ValueError("Expected a three-dimensional NIfTI-1 with an explicit sform")
    dtype = {2:"u1",4:"<i2",512:"<u2"}[struct.unpack_from("<h",data,70)[0]]
    start, slope, offset = struct.unpack_from("<3f",data,108)
    v = np.frombuffer(data, dtype=dtype, offset=int(start), count=int(np.prod(dims[1:4])))
    a = np.eye(4)
    a[:3] = np.array(struct.unpack_from("<12f",data,280)).reshape(3,4)
    return v.reshape(dims[1:4], order="F")*(slope or 1)+offset, a


def anatomical_figures(result, out, atlas_path, lut_path, template_path, mask_path):
    regions, affine = read_reference(atlas_path)
    if not np.allclose(affine[:3,:3], np.diag(np.diag(affine)[:3])):
        raise ValueError("Atlas has an oblique affine")
    t1, grid = read_reference(template_path)
    mask, mask_grid = read_reference(mask_path)
    np.testing.assert_allclose(grid, mask_grid)
    transform = np.linalg.solve(grid, affine)
    t1 = affine_transform(t1*(mask>0), transform[:3,:3], transform[:3,3], output_shape=regions.shape, order=1)
    lut = pd.read_csv(lut_path).set_index("Name")
    assert set(lut.index) == set(result["axis"]) and lut.index.is_unique
    labels = set(np.unique(regions))-{0}
    id_col = next(c for c in ("Index","Intensity") if set(lut[c])==labels)
    ids = lut.loc[result["axis"],id_col].astype(int).to_numpy()
    z_levels = (8, 42)
    slices = [int(round((z-affine[2,3])/affine[2,2])) for z in z_levels]
    dx, dy = [int(np.sign(affine[i,i])) for i in (0,1)]
    def plane(arr, z):
        return arr[::dx,::dy,z].T
    vmax = max(np.max(np.abs(m)) for data in result["bands"].values() for m in data["maps"].values())
    top = np.percentile(t1[t1>0], 98)
    for band, data in result["bands"].items():
        fig, axes = plt.subplots(4,4,figsize=(8.15,8.2),facecolor="white")
        fig.subplots_adjust(top=.88,bottom=.12,left=.11,right=.94,wspace=.035,hspace=.06)
        fig.suptitle(f"{NAMES[band]} 的四个 BOLD 幅度模板", fontsize=16, x=.5,y=.98)
        for i, row in enumerate(axes):
            row[0].set_ylabel(f"M{i+1}", rotation=0, labelpad=18, va="center", fontsize=13)
            for j, ax in enumerate(row):
                condition = ("OFF","ON")[j//2]
                z = slices[j%2]
                look = np.full(int(max(ids))+1, np.nan)
                look[ids] = data["maps"][condition][i]
                ax.imshow(plane(t1,z), cmap="gray", origin="lower", vmax=top)
                im = ax.imshow(look[plane(regions,z).astype(int)], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               interpolation="nearest", origin="lower", alpha=.92)
                ax.set(xticks=[],yticks=[])
                if i==0:
                    ax.set_title(f"{condition}   z={z_levels[j%2]} mm", fontsize=10)
        cb = fig.colorbar(im, cax=fig.add_axes([.28,.075,.48,.014]), orientation="horizontal")
        cb.set_label("空间模板权重（整张模板可同时翻转正负）", fontsize=9)
        fig.text(.5,.012,"轴位切面，图左为解剖左侧；底图为公共 MNI 模板。",ha="center", fontsize=9)
        fig.savefig(out/f"maps_{band}.png", dpi=155)
        plt.close(fig)


def paired_plot(paired, out):
    fig, axes = plt.subplots(1,3,figsize=(10.8,3.7),layout="constrained", squeeze=False)
    fig.suptitle("DBS 开关机配对差异 · 0.04–0.07 Hz", fontsize=16)
    for row, band in zip(axes, NAMES):
        for ax, (metric, label, scale) in zip(row, METRICS):
            sub = paired[(paired.band==band)&(paired.metric==metric)&(paired.state!="all")].sort_values("state")
            delta = sub.delta.to_numpy()*scale
            ax.axvline(0,color="#8b9497",lw=.9,ls="--")
            ax.errorbar(delta,np.arange(4),xerr=[delta-sub.ci_low*scale,sub.ci_high*scale-delta],
                        fmt="o",color=COLORS[band],capsize=3)
            ax.set(yticks=range(4),yticklabels=[f"{s}  n={n}" for s,n in zip(sub.state,sub.n)],
                   ylim=(3.6,-.65),xlabel=f"ON − OFF  {label}")
            ax.set_title(label.split("（")[0],fontsize=12,loc="left")
            ax.spines[["right","top","left"]].set_visible(False)
            ax.tick_params(axis="y",length=0)
    fig.savefig(out/"paired_effects.png",dpi=160)
    plt.close(fig)


def clinical_plot(clinical, out):
    fig, axes = plt.subplots(1,2,figsize=(9.3,3.35),sharey=True)
    fig.subplots_adjust(left=.19,right=.98,bottom=.22,top=.76,wspace=.12)
    fig.suptitle("关机时的运动评分与总体微状态指标",fontsize=16)
    for ax, metric, title, eeg in zip(axes,("duration_s","occurrence_s"),("平均持续时间","总出现率"),(.68,-.76)):
        ax.axvline(0,color="#949b9d",ls="--",lw=1)
        ax.scatter(eeg,0,color="#555555",marker="D",s=40)
        ax.annotate(f"{eeg:+.2f}",(eeg,0),xytext=(0,9),textcoords="offset points",ha="center")
        for i,band in enumerate(NAMES, start=1):
            s = clinical[(clinical.band==band)&(clinical.state=="all")&(clinical.metric==metric)
                         &(clinical.contrast=="off_score")].iloc[0]
            ax.errorbar(s.r,i,xerr=[[s.r-s.ci_low],[s.ci_high-s.r]],fmt="o",color=COLORS[band],capsize=3)
            ax.annotate(f"{s.r:+.2f}  (n={s.n})",(s.r,i),xytext=(0,8),textcoords="offset points",ha="center",fontsize=9)
        ax.set(title=title, xlim=(-1.02,1.02), ylim=(1.48,-.55), yticks=[0,1],
               yticklabels=["EEG 论文（20 人）","BOLD 0.04–0.07 Hz"], xlabel="Pearson r")
        ax.spines[["right","top","left"]].set_visible(False)
        ax.tick_params(axis="y",length=0)
    fig.text(.5,.035,"BOLD 横线为逐项 95% 置信区间；EEG 点取自 Li 等论文图 4。",ha="center",fontsize=9)
    fig.savefig(out/"clinical_off.png", dpi=165)
    plt.close(fig)


def export(path, atlas, lut, template, mask):
    result = json.loads(path.read_text(encoding="utf8"))
    out = path.parent
    tables = {}
    for key in ("paired","clinical","transitions","pooled_duration_sensitivity"):
        table = pd.concat([pd.DataFrame(v[key]).assign(band=b) for b,v in result["bands"].items()],ignore_index=True)
        table.to_csv(out/f"{key}.csv",index=False)
        tables[key] = table
    templates = []
    for b,v in result["bands"].items():
        for cond,m in v["maps"].items():
            tmap=pd.DataFrame(m,columns=result["axis"])
            templates.append(tmap.assign(band=b,condition=cond,state=["M1","M2","M3","M4"]))
    pd.concat(templates).to_csv(out/"templates.tsv",sep="\t",index=False)
    c = tables["clinical"]
    p = tables["paired"]
    assert result["visits"] == 50 and set(result["bands"])=={"weida_band"} and len(c)==42 and len(p)==14
    np.testing.assert_allclose(c.p_raw,2*t.sf(abs(c.r)*np.sqrt((c.n-2)/(1-c.r**2)),c.n-2),atol=1e-12)
    np.testing.assert_allclose(p.on-p.off,p.delta,atol=1e-12)
    trans=tables["transitions"]
    # Source rows can be absent in a scan, but all outgoing columns use the same patients.
    np.testing.assert_allclose(trans.groupby(["band","condition","source"])["mean"].sum(),1,atol=1e-12)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    cjk=next((f for f in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if f in fonts),None)
    if not cjk:
        raise RuntimeError("Rendering Chinese figures requires a Chinese font")
    plt.rcParams["font.family"]=[cjk]
    paired_plot(p,out)
    clinical_plot(c,out)
    anatomical_figures(result,out,atlas,lut,template,mask)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("result","atlas","lut","template","mask"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    export(args.result,args.atlas,args.lut,args.template,args.mask)
