"""Map the unchanged AAL78 templates to Yeo-7 and audit five-fold agreement."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


MODULE = Path(__file__).resolve().parents[1]
V4 = MODULE.parent / "V4_Microstate_Statistical_Significance"
NETWORKS = ("Visual", "Somatomotor", "Dorsal Attention", "Ventral Attention",
            "Limbic", "Frontoparietal", "Default")
CATEGORIES = (*NETWORKS, "Unmapped Cortex", "Subcortical")
NETWORK_COLORS = ("#781286", "#4682b4", "#00760e", "#c43afa",
                  "#aebd55", "#e69422", "#cd3e4e")
AAL_SOURCE = "https://github.com/chenfei-ye/BIDS-fMRIpost/tree/c345a4a82685b67e5b578197927fe0dc90171efd/atlases"
YEO_SOURCE = "https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011"
YEO_FILE = "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz"
YEO_HASH = "a11cfeba3c88e25b6cd9c6d4b542ef67d3587bc674023bb8661447f213329d3f"
AAL_HASHES = {
    "map": "a05fc7e698b13d71b1733a652562b3bbf84a6bb1f90fdd73967b9742053463af",
    "labels": "8fb91310eb5f301cc3965d36dbb4c7e3ea1d4c8bd77000a1f8fe20b7b10870dd",
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def atlas_overlap(aal_path, label_path, yeo_path):
    """Count labels on the native AAL grid, retaining the power outside Yeo."""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    if {name: digest(path) for name, path in
            {"map": aal_path, "labels": label_path}.items()} != AAL_HASHES:
        raise ValueError("AAL assets must match the pinned project atlas")
    if digest(yeo_path) != YEO_HASH:
        raise ValueError("Yeo volume must match the pinned seven-network tight mask")
    labels = pd.read_csv(label_path).sort_values("Index").reset_index(drop=True)
    if (labels.Index.tolist() != list(range(1, 79)) or
            not labels.Name.is_unique or not labels.Intensity.is_unique):
        raise ValueError("Expected a one-to-one, ordered set of 78 parcels")
    aal = nib.as_closest_canonical(nib.load(aal_path))
    yeo = nib.squeeze_image(nib.load(yeo_path))
    if aal.ndim != 3 or yeo.ndim != 3:
        raise ValueError("Atlas labels require three-dimensional volumes")
    aal_voxels = np.asarray(aal.dataobj)
    if set(np.unique(aal_voxels)) != {0, *labels.Intensity}:
        raise ValueError("ROI intensities do not identify the frozen AAL parcels")
    if not np.array_equal(np.unique(yeo.dataobj), np.arange(8)):
        raise ValueError("The Yeo source must contain exactly labels 0 through 7")
    yeo_voxels = np.asarray(
        resample_from_to(yeo, (aal.shape, aal.affine), order=0).dataobj
    ).astype(int)
    counts = np.zeros((len(labels), len(CATEGORIES)), dtype=int)
    for i, roi in enumerate(labels.itertuples(index=False)):
        parcel = aal_voxels == roi.Intensity
        if roi.Region.casefold() == "cortex":
            histogram = np.bincount(yeo_voxels[parcel], minlength=8)
            counts[i, :7] = histogram[1:]
            counts[i, 7] = histogram[0]
        else:
            counts[i, 8] = parcel.sum()
    parcels = counts.sum(axis=1)
    np.testing.assert_equal(parcels.sum(), np.count_nonzero(aal_voxels))
    fractions = counts / parcels[:, None]
    np.testing.assert_allclose(fractions.sum(axis=1), 1, rtol=0, atol=1e-14)
    crosswalk = pd.DataFrame({
        "roi": np.repeat(labels.Name.to_numpy(), len(CATEGORIES)),
        "roi_index": np.repeat(labels.Index.to_numpy(), len(CATEGORIES)),
        "aal_voxel_label": np.repeat(labels.Intensity.to_numpy(), len(CATEGORIES)),
        "hemisphere": np.repeat(labels.Hemi.to_numpy(), len(CATEGORIES)),
        "region": np.repeat(labels.Region.to_numpy(), len(CATEGORIES)),
        "category": np.tile(CATEGORIES, len(labels)),
        "parcel_voxels": np.repeat(parcels, len(CATEGORIES)),
        "overlap_voxels": counts.ravel(),
        "fraction_of_parcel": fractions.ravel(),
    })
    return labels, aal, yeo, aal_voxels, counts, fractions, crosswalk


def templates(labels):
    """Enforce the original spatial axis and use reference fold 0 verbatim."""
    reference = pd.read_csv(V4 / "results/Reference_Templates.csv")
    folds = pd.read_csv(V4 / "results/Fold_Templates.csv")
    metadata = ["fold", "state", "original_state", "stable"]
    roi_columns = labels.Name.tolist()
    if list(folds.columns) != metadata + roi_columns:
        raise ValueError("The AAL labels do not match the frozen template axis")
    if folds.groupby("fold").state.apply(sorted).tolist() != [list(range(1, 5))] * 5:
        raise ValueError("Expected all twenty frozen fold/state templates")
    if folds.fold.unique().tolist() != list(range(5)):
        raise ValueError("Expected frozen folds 0 through 4")
    pd.testing.assert_frame_equal(reference, folds.loc[folds.fold.eq(0)].reset_index(drop=True))
    np.testing.assert_allclose(
        np.linalg.norm(folds[roi_columns], axis=1), 1, rtol=0, atol=1e-12
    )
    return folds


def network_share(weights, fractions):
    """Apportion consistently using voxel fractions within each whole AAL parcel."""
    absolute = np.abs(weights)
    squared = weights ** 2
    return (absolute @ fractions / absolute.sum(), squared @ fractions / squared.sum())


def coverage_share(vector):
    subset = vector[:7]
    return subset / subset.sum()


def annotation(folds, labels, fractions):
    records = []
    roi_names = labels.Name.tolist()
    hemisphere = labels.Hemi.to_numpy()
    region = labels.Region.str.casefold().to_numpy()
    for _, template in folds.iterrows():
        keys = dict(fold=template.fold, state=template.state)
        weights = template[roi_names].to_numpy(dtype=float)
        absolute, squared = np.abs(weights), weights ** 2
        abs_by_network, sq_by_network = absolute @ fractions, squared @ fractions
        abs_total, sq_total = absolute.sum(), squared.sum()
        for i, roi in enumerate(roi_names):
            regional = dict(roi=roi, hemisphere=hemisphere[i], region=region[i])
            records.append({
                **keys, **regional, "scope": "roi", "category": roi,
                "signed_roi_weight": weights[i], "absolute_weight": absolute[i],
                "squared_weight": squared[i], "absolute_share_whole_template": absolute[i] / abs_total,
                "squared_share_whole_template": squared[i] / sq_total,
            })
            for j in np.flatnonzero(fractions[i]):
                multiplier = fractions[i, j]
                abs_part, sq_part = multiplier * absolute[i], multiplier * squared[i]
                records.append({
                    **keys, **regional, "scope": "roi_network", "category": CATEGORIES[j],
                    "effective_roi_count": multiplier, "absolute_weight": abs_part,
                    "squared_weight": sq_part, "absolute_share_whole_template": abs_part / abs_total,
                    "squared_share_whole_template": sq_part / sq_total,
                    "absolute_share_within_network": abs_part / abs_by_network[j] if abs_by_network[j] else np.nan,
                    "squared_share_within_network": sq_part / sq_by_network[j] if sq_by_network[j] else np.nan,
                })
        for j, category in enumerate(CATEGORIES):
            count = float(fractions[:, j].sum())
            records.append({
                **keys, "scope": "network", "category": category,
                "effective_roi_count": count, "absolute_weight": abs_by_network[j],
                "squared_weight": sq_by_network[j], "absolute_share_whole_template": abs_by_network[j] / abs_total,
                "squared_share_whole_template": sq_by_network[j] / sq_total,
                "absolute_share_among_yeo7": abs_by_network[j] / abs_by_network[:7].sum() if j < 7 else np.nan,
                "squared_share_among_yeo7": sq_by_network[j] / sq_by_network[:7].sum() if j < 7 else np.nan,
                "mean_absolute_weight_per_effective_roi": abs_by_network[j] / count if count else np.nan,
                "mean_squared_weight_per_effective_roi": sq_by_network[j] / count if count else np.nan,
            })
        group_masks = [
            ("hemisphere", "L", hemisphere == "L"), ("hemisphere", "R", hemisphere == "R"),
            ("region", "Cortex", region == "cortex"), ("region", "Subcortical", region != "cortex"),
        ]
        for scope, category, mask in group_masks:
            records.append({
                **keys, "scope": scope, "category": category, "effective_roi_count": int(mask.sum()),
                "absolute_weight": absolute[mask].sum(), "squared_weight": squared[mask].sum(),
                "absolute_share_whole_template": absolute[mask].sum() / abs_total,
                "squared_share_whole_template": squared[mask].sum() / sq_total,
            })
    result = pd.DataFrame(records)
    for scope in ("roi", "roi_network", "network", "hemisphere", "region"):
        shares = result.loc[result.scope.eq(scope)].groupby(["fold", "state"])[
            ["absolute_share_whole_template", "squared_share_whole_template"]
        ].sum()
        np.testing.assert_allclose(shares, 1, rtol=0, atol=1e-12)
    return result


def pearson(x, y):
    return float(np.corrcoef(x, y)[0, 1])


def top_ten_similarity(x, y):
    left = set(np.argsort(-np.abs(x), kind="stable")[:10])
    right = set(np.argsort(-np.abs(y), kind="stable")[:10])
    return len(left & right) / len(left | right)


def comparison(folds, roi_names, fractions, i, j, gate, comparison_type):
    source = folds.loc[folds.fold.eq(i)].sort_values("state")
    target = folds.loc[folds.fold.eq(j)].sort_values("state")
    vectors_i = source[roi_names].to_numpy()
    vectors_j = target[roi_names].to_numpy()
    # The V4 matcher gates the absolute cosine and both row/column separations.
    # Keep all the already frozen state labels even if this independent check fails.
    scores = np.abs(vectors_i @ vectors_j.T)
    source_index, assignment = linear_sum_assignment(-scores)
    np.testing.assert_array_equal(source_index, range(4))
    records = []
    for k, (x, y) in enumerate(zip(vectors_i, vectors_j, strict=True)):
        match = assignment[k]
        score = scores[k, match]
        margin = score - max(np.delete(scores[k], match).max(), np.delete(scores[:, match], k).max())
        abs_x, sq_x = network_share(x, fractions)
        abs_y, sq_y = network_share(y, fractions)
        signed_r = pearson(x, y)
        records.append({
            "comparison": comparison_type, "fold_i": i, "fold_j": j, "state": k + 1,
            "original_state_i": int(source.iloc[k].original_state),
            "original_state_j": int(target.iloc[k].original_state),
            "frozen_state_stable": bool(source.iloc[k].stable and target.iloc[k].stable),
            "signed_spatial_pearson": signed_r, "absolute_spatial_pearson": abs(signed_r),
            "absolute_roi_weight_pearson": pearson(np.abs(x), np.abs(y)),
            "yeo7_absolute_profile_pearson": pearson(coverage_share(abs_x), coverage_share(abs_y)),
            "yeo7_squared_profile_pearson": pearson(coverage_share(sq_x), coverage_share(sq_y)),
            "yeo7_squared_profile_total_variation": float(np.abs(coverage_share(sq_x) - coverage_share(sq_y)).sum() / 2),
            "roi_absolute_rank_spearman": float(spearmanr(np.abs(x), np.abs(y)).statistic),
            "top_10_absolute_roi_jaccard": top_ten_similarity(x, y),
            "hungarian_state_j": int(match + 1), "hungarian_abs_cosine": float(score),
            "hungarian_bidirectional_margin": float(margin), "hungarian_agrees_with_frozen": bool(match == k),
            "hungarian_original_gate_pass": bool(score >= gate["min_abs_cosine"] and
                                                  margin >= gate["min_row_and_column_gap"]),
        })
    return records


def stability(folds, roi_names, fractions):
    reliability = V4 / "workflow/source/V4_BOLD_Microstate_Reliability/configs/protocol.json"
    gate = json.loads(reliability.read_text(encoding="utf-8-sig"))["identification"]
    records = []
    for j in range(5):
        records.extend(comparison(folds, roi_names, fractions, 0, j, gate, "reference"))
    for i, j in combinations(range(5), 2):
        records.extend(comparison(folds, roi_names, fractions, i, j, gate, "all_distinct_pairs"))
    result = pd.DataFrame(records)
    assert result.comparison.value_counts().to_dict() == {"all_distinct_pairs": 40, "reference": 20}
    return result


def export_figure(fig, output, name):
    for suffix in ("png", "pdf"):
        metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else None
        fig.savefig(output / f"{name}.{suffix}", dpi=600, bbox_inches="tight",
                    facecolor="white", metadata=metadata)
    plt.close(fig)


def plot_networks(annotated, labels, aal, voxels, output):
    reference = annotated.loc[annotated.fold.eq(0)]
    reference_roi = reference.loc[reference.scope.eq("roi")]
    chosen = reference.loc[reference.scope.eq("network") & reference.category.isin(NETWORKS)]
    bound = reference_roi.signed_roi_weight.abs().max()
    norm = TwoSlopeNorm(vcenter=0, vmin=-bound, vmax=bound)
    y = np.r_[0:aal.shape[1]]
    x = np.r_[0:aal.shape[0]]
    levels = (-12, 16, 40)
    fig = plt.figure(figsize=(9.6, 7.6), layout="constrained")
    grid = fig.add_gridspec(4, 5, width_ratios=(1, 1, 1, 0.065, 2.30))
    for state in range(1, 5):
        field = np.full(voxels.shape, np.nan)
        values = reference_roi.loc[reference_roi.state.eq(state)].set_index("roi").signed_roi_weight
        for roi in labels.itertuples(index=False):
            field[voxels == roi.Intensity] = values[roi.Name]
        for j, cut in enumerate(levels):
            ax = fig.add_subplot(grid[state - 1, j])
            k = int(round((cut - aal.affine[2, 3]) / aal.affine[2, 2]))
            mapped = np.ma.masked_invalid(field[:, :, k].T)
            image = ax.pcolormesh(x, y, mapped, norm=norm, cmap="RdBu_r", shading="nearest", rasterized=False)
            occupied = np.isfinite(field[:, :, k])
            crop_i, crop_j = np.where(occupied)
            ax.set(xlim=(crop_i.min() - 2, crop_i.max() + 2), ylim=(crop_j.min() - 2, crop_j.max() + 2),
                   aspect="equal")
            ax.set_axis_off()
            if j == 0:
                ax.text(-0.09, 0.5, f"M{state}", transform=ax.transAxes, va="center", ha="right",
                        fontsize=11, fontweight="semibold")
            if state == 1:
                ax.set_title(f"z = {cut} mm", fontsize=9)
            if state == 4 and j == 0:
                ax.text(0.02, 0.01, "L", transform=ax.transAxes, ha="left", fontsize=8)
                ax.text(0.98, 0.01, "R", transform=ax.transAxes, ha="right", fontsize=8)
    cbax = fig.add_subplot(grid[1:3, 3])
    legend = fig.colorbar(image, cax=cbax, label="Signed weight")
    legend.solids.set_rasterized(False)
    for state in range(1, 5):
        ax = fig.add_subplot(grid[state - 1, 4])
        data = chosen.loc[chosen.state.eq(state)].set_index("category").reindex(NETWORKS)
        ax.bar(np.arange(7), data.squared_share_among_yeo7, color=NETWORK_COLORS, width=0.70)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, .5, 1])
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.tick_params(length=0, labelsize=8)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        if state == 1:
            ax.set_title("Squared weight within Yeo-covered cortex", fontsize=9.5)
        if state == 4:
            ax.set_xticks(range(7), ["VIS", "SM", "DA", "VA", "LIM", "FP", "DEF"])
        else:
            ax.set_xticks([])
        covered = data.squared_share_whole_template.sum()
        ax.text(.99, .91, f"{covered:.0%} of total weight mapped", transform=ax.transAxes,
                va="top", ha="right", fontsize=8, color="#415269")
    fig.suptitle("Frozen AAL78 microstate templates and cortical network profiles",
                 fontsize=14, fontweight="semibold")
    export_figure(fig, output, "Figure_05_Microstate_Network_Maps")


def plot_stability(rows, output):
    ref_rows = rows.loc[rows.comparison.eq("reference")]
    specs = [
        ("absolute_spatial_pearson", "Spatial |r|"),
        ("absolute_roi_weight_pearson", "ROI |weight| Pearson r"),
        ("yeo7_squared_profile_pearson", "Yeo-7 squared-profile r"),
        ("roi_absolute_rank_spearman", "ROI |weight| rank ρ"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.7, 5.9), layout="constrained", sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.ravel(), specs, strict=True):
        table = ref_rows.pivot(index="state", columns="fold_j", values=metric)
        matrix = table.to_numpy()
        mesh = ax.pcolormesh(matrix, cmap="viridis", vmin=-1, vmax=1, rasterized=False)
        ax.set_ylim(4, 0)
        ax.set_title(label, fontsize=10, loc="left", fontweight="semibold")
        ax.set_xticks(np.arange(5) + .5, ["Ref.", "1", "2", "3", "4"])
        ax.set_yticks(np.arange(4) + .5, [f"M{k}" for k in range(1, 5)])
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j + .5, i + .5, f"{value:.2f}", ha="center", va="center",
                    color="#122238" if value > .35 else "white", fontsize=8)
        ax.tick_params(length=0, labelsize=9)
        ax.spines[:].set_visible(False)
    legend = fig.colorbar(mesh, ax=axes.ravel(), label="Correlation", shrink=.80)
    legend.solids.set_rasterized(False)
    fig.suptitle("Five-fold agreement with the frozen reference templates",
                 fontsize=14, fontweight="semibold")
    fig.supxlabel("Template fold (Ref. is the fold-0 self comparison)", fontsize=9)
    export_figure(fig, output, "Figure_06_Five_Fold_Template_Stability")


def report_markdown(path, annotated, rows, counts, labels):
    reference = annotated.loc[annotated.fold.eq(0)]
    agreement = rows.loc[rows.comparison.eq("reference") & rows.fold_j.ne(0)]
    all_pairs = rows.loc[rows.comparison.eq("all_distinct_pairs")]
    covered_voxels = counts[:, :7].sum()
    mapped_rois = int((counts[:, :7].sum(axis=1) > 0).sum())
    region_counts = labels.Region.value_counts()
    matched = agreement.hungarian_agrees_with_frozen & agreement.hungarian_original_gate_pass
    pair_matched = all_pairs.hungarian_agrees_with_frozen & all_pairs.hungarian_original_gate_pass
    lines = [
        "# V7 四个冻结微状态的功能网络注释",
        "",
        "## 图谱重叠",
        "",
        "以本项目实际使用的 2 mm AAL1ctx MNI 图谱为共用网格。读取作者发布的 Yeo 2011 MNI152 7 网络紧掩膜，通过 NIfTI 仿射在模板坐标中最近邻重采样。重叠按每个 AAL ROI 的全部体素作分母，允许一个 ROI 跨多个网络。未标注的皮层体素单列。两套图谱使用各自的 MNI 模板，重叠是基于坐标对照的近似估计。",
        "",
        f"冻结标签表有 {region_counts.get('cortex', 0)} 个皮层 ROI，左右各 39 个。"
        f"本图谱没有收录皮层下 ROI，皮层下类别在表中保留为零。{mapped_rois}/78 个 ROI 与至少一个 Yeo 网络有体素重叠。"
        f"{covered_voxels:,}/{counts.sum():,} 个 AAL 体素落在七个网络内，占 {covered_voxels / counts.sum():.1%}。",
        "",
        "以每个 ROI 的绝对权重与平方权重分别乘上该 ROI 的体素重叠比例。表中同时给出 ROI 在整个模板内的权重占比、各 ROI 在所属网络份额中的权重占比、各网络在全模板及 Yeo 已覆盖皮层中的权重占比，以及每个等效 ROI 的平均权重。等效 ROI 数是各 ROI 体素比例之和。",
        "",
        "## 四个状态的空间分布",
        "",
        "下表用折 0 冻结参考模板。前三个脑区按绝对权重排序。网络排序用 Yeo 已覆盖皮层中的平方权重，覆盖率用整个模板作分母。"
        "切面图的正负沿用原模板的整体极性约定，网络权重取绝对值或平方。",
        "",
        "| 状态 | 绝对权重最高的三个 ROI | Yeo 网络占比前两位 | 被 Yeo 分配的平方权重 | 左 / 右平方权重 |",
        "|---|---|---|---:|---:|",
    ]
    for state in range(1, 5):
        s = reference.loc[reference.state.eq(state)]
        roi = s.loc[s.scope.eq("roi")].nlargest(3, "absolute_weight")
        networks = s.loc[s.scope.eq("network") & s.category.isin(NETWORKS)]
        top = networks.nlargest(2, "squared_share_among_yeo7")
        network_line = "、".join(f"{r.category} {r.squared_share_among_yeo7:.1%}" for r in top.itertuples())
        hemi = s.loc[s.scope.eq("hemisphere")].set_index("category").squared_share_whole_template
        rois = "、".join(roi.category)
        lines.append(f"| M{state} | {rois} | {network_line} | {networks.squared_share_whole_template.sum():.1%} | {hemi['L']:.1%} / {hemi['R']:.1%} |")
    lines.extend([
        "",
        "## 五折一致性",
        "",
        f"对参考折的 {len(agreement)} 项跨折匹配有 {matched.sum()}/{len(matched)} 项同时保留冻结编号并通过原 V4 的门槛。"
        f"五折两两对照有 {pair_matched.sum()}/{len(pair_matched)} 项通过相同条件。"
        "匈牙利匹配使用绝对余弦，得分至少为 0.60，行和列各自的次优差值至少为 0.05。参考折自身的四项记录单列。",
        "",
        "| 状态 | 空间 \\|r\\| 范围 | ROI 绝对权重 r 范围 | Yeo 平方组成 r 范围 | ROI 绝对权重排名 ρ 范围 |",
        "|---|---:|---:|---:|---:|",
    ])
    for state in range(1, 5):
        part = agreement.loc[agreement.state.eq(state)]
        ranges = []
        for metric in ("absolute_spatial_pearson", "absolute_roi_weight_pearson",
                       "yeo7_squared_profile_pearson", "roi_absolute_rank_spearman"):
            ranges.append(f"{part[metric].min():.3f} 至 {part[metric].max():.3f}")
        lines.append(f"| M{state} | {' | '.join(ranges)} |")
    lines += [
        "",
        "模板折之间的训练样本有重叠。上述数值描述冻结模板的空间一致性和图谱重叠，支持对模式分布作解剖描述。网络富集和疾病特异性没有在本轮接受推断检验。",
        "",
        f"图谱来源 [项目 AAL1ctx 原始文件]({AAL_SOURCE}) · [Yeo 2011 作者发布的图谱]({YEO_SOURCE})。",
        "",
        "[体素重叠表](../results/AAL78_Yeo7_Overlap.csv) · [逐折注释表](../results/Microstate_Network_Annotation.csv) · "
        "[逐折一致性](../results/Microstate_Network_Stability.csv) · [四态网络图](../figures/Figure_05_Microstate_Network_Maps.png) · "
        "[空间一致性图](../figures/Figure_06_Five_Fold_Template_Stability.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aal-map", type=Path, required=True)
    parser.add_argument("--aal-labels", type=Path, required=True)
    parser.add_argument("--yeo-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=MODULE)
    parser.add_argument("--skip-figures", action="store_true", help="Recheck numerical outputs only")
    args = parser.parse_args()
    base = args.output_dir
    for subdir in ("results", "figures", "reports", "logs"):
        (base / subdir).mkdir(parents=True, exist_ok=True)
    labels, aal, yeo, aal_voxels, counts, fractions, overlap = atlas_overlap(
        args.aal_map, args.aal_labels, args.yeo_map
    )
    folds = templates(labels)
    annotated = annotation(folds, labels, fractions)
    rows = stability(folds, labels.Name.tolist(), fractions)
    for table, name in (
        (overlap, "AAL78_Yeo7_Overlap"),
        (annotated, "Microstate_Network_Annotation"),
        (rows, "Microstate_Network_Stability"),
    ):
        table.to_csv(base / "results" / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    receipt = {
        "aal_source": AAL_SOURCE,
        "aal_geometry": {"shape": aal.shape, "affine": aal.affine.tolist()},
        "aal_sha256": AAL_HASHES,
        "yeo_source": YEO_SOURCE,
        "yeo_source_file": YEO_FILE,
        "yeo_sha256": YEO_HASH,
        "yeo_geometry": {"shape": yeo.shape, "affine": yeo.affine.tolist()},
        "mapping": "Nearest-neighbor from Yeo tight mask to AAL canonical native grid in standard coordinates",
        "aal_roi_count": int(len(labels)),
        "aal_cortical_roi_count": int(labels.Region.str.casefold().eq("cortex").sum()),
        "aal_subcortical_roi_count": int(labels.Region.str.casefold().ne("cortex").sum()),
        "aal_voxels": int(counts.sum()),
        "yeo_mapped_voxels": int(counts[:, :7].sum()),
        "roi_fraction_and_weight_conservation": True,
        "nibabel_version": version("nibabel"),
    }
    (base / "logs/Network_Atlas_Verification.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    if not args.skip_figures:
        plot_networks(annotated, labels, aal, aal_voxels, base / "figures")
        plot_stability(rows, base / "figures")
    report_markdown(base / "reports/V7_Functional_Network_Annotation.md", annotated, rows, counts, labels)
    print(f"Annotated {len(folds)} frozen templates and checked {len(rows)} template matches.")


if __name__ == "__main__":
    main()
