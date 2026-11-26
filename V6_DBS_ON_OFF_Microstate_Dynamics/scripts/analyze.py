"""Run V6 paired microstate inference and export private and aggregate products."""

import math

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.stats import ttest_1samp
from sklearn.metrics import adjusted_rand_score, silhouette_score
from statsmodels.stats.power import TTestPower

from v6_common import (ALL_TRANSITIONS, CFG, EDGE_COLUMNS, HERE, ILR, PRIVATE, RESULT, STATE, V4, V5, digest,
                       read_json, scalar_families, write_csv, write_json)
from paired import (ALPHA, B, P_MODELS, adjust, association, bootstrap_record, classify, covariate_test, direction,
                    draws, influence, interval, logical_p, omnibus, scalar, signs, standardized)
from v5_stats import RESAMPLE as V5_RESAMPLE, restoration
from statistics_v4 import tail_probability


PRIORITY_CONTEXT = [*CFG["primary"], *CFG["secondary"]]
HETEROGENEITY_KEYS = [*CFG["primary"], "occurrence_1", "switching_rate"]


def effect_units(key):
    if key.startswith(("duration_","dwell_")):
        return "seconds"
    if key.startswith("dispersion_"):
        return "distance to unit template"
    if key.startswith("occurrence_"):
        return "episodes per second"
    if key=="switching_rate":
        return "transitions per second"
    return "proportion"


def paired_inputs():
    cohort = pd.read_csv(PRIVATE / "results/Frozen_Cohort_Features.csv")
    manifest = pd.read_csv(PRIVATE / "manifests/ON_OFF_Pair_Manifest.csv").set_index("participant_id").sort_index()
    assert len(manifest)==25 and manifest.index.is_unique
    assert (read_json(HERE / "manifests/Input_Audit.json")["protocol_sha256"]==digest(HERE / "configs/protocol.json"))
    extra = ["mean_raw_dvars", "mean_std_dvars", "valid_frames", "fd"]
    columns = list(dict.fromkeys([*sum(scalar_families().values(),[]), *ALL_TRANSITIONS, *ILR, *extra]))
    d = cohort.loc[cohort.condition.ne("HC")]
    on,off = (d.loc[d.condition.eq(x)].set_index("subject").loc[manifest.index] for x in ("ON","OFF"))
    assert on.scan.eq(manifest.on_scan_id).all() and off.scan.eq(manifest.off_scan_id).all()
    delta = on[columns]-off[columns]
    assert len(delta)==25 and len(cohort)==65
    return cohort,on,off,delta,manifest


def inference(on,off,delta,verified):
    fd = delta.fd.to_numpy()
    dv = delta.mean_std_dvars.to_numpy()
    scalar_rows, many_draws = [],[]
    for family, endpoints in scalar_families().items():
        for key in endpoints:
            row,samples = scalar(on[key].to_numpy(),off[key].to_numpy(),fd,dv,verified)
            scalar_rows.append(dict(endpoint=key,family=family,units=effect_units(key),**row))
            if samples is not None:
                columns = ["mean_difference","median_difference","dz","fd_effect","dvars_effect"]
                many_draws.append(pd.DataFrame(samples,columns=columns)
                                  .assign(endpoint=key,replicate=np.arange(1,B+1)))
    omnibus_rows, coords = [],[]
    for key,columns,unit in [("transition_omnibus",EDGE_COLUMNS,"conditional probability vector"),
                             ("coverage_ilr_omnibus",ILR,"Aitchison ILR vector")]:
        row, values, samples = omnibus(on[columns].to_numpy(),off[columns].to_numpy(),fd,dv,verified)
        omnibus_rows.append(dict(endpoint=key,family="Exploratory omnibus",units=unit,**row))
        names = ["mean_difference_norm","fd_effect_norm","dvars_effect_norm"]
        many_draws.append(pd.DataFrame(values,columns=names).assign(endpoint=key,replicate=np.arange(1,B+1)))
        for j,feature in enumerate(columns):
            a,b,n = interval(samples[:,j])
            coords.append(dict(omnibus=key,coordinate=feature,n_pairs=row["n_pairs"],mean_difference=delta[feature].mean(),
                               ci95_low=a,ci95_high=b,bootstrap_valid=n))
    om = adjust(pd.DataFrame(omnibus_rows))
    stats = classify(adjust(pd.DataFrame(scalar_rows)),om)
    om = classify(om,om)
    write_csv(stats,RESULT / "ON_OFF_Feature_Summary.csv")
    write_csv(stats.loc[stats.family.eq("Transition localization")],RESULT / "Transition_Path_Results.csv")
    write_csv(om,RESULT / "ON_OFF_Omnibus_Results.csv")
    write_csv(pd.DataFrame(coords),RESULT / "Omnibus_Coordinate_Effects.csv")
    return stats,om,many_draws


def transition_matrices(on,off,delta):
    rows = []
    index = draws(len(delta))
    for i in STATE:
        for j in STATE:
            key = f"transition_{i}_{j}"
            d = delta[key].to_numpy()[index].mean(axis=1)
            a,b,n = interval(d)
            rows.append(dict(source=f"M{i}",destination=f"M{j}",n_pairs=len(delta),self_transition=i==j,
                             on_count=on[f"count_{i}_{j}"].sum(),off_count=off[f"count_{i}_{j}"].sum(),
                             on_source=on[f"source_{i}"].sum(),off_source=off[f"source_{i}"].sum(),
                             on_mean_probability=on[key].mean(),off_mean_probability=off[key].mean(),
                             mean_difference=delta[key].mean(),ci95_low=a,ci95_high=b,bootstrap_valid=n))
    write_csv(pd.DataFrame(rows),RESULT / "Transition_Matrices.csv")
    cols = [f"count_{i}_{j}" for i in STATE for j in STATE]+[f"source_{i}" for i in STATE]+ALL_TRANSITIONS
    private = pd.concat([on[cols].add_prefix("on_"),off[cols].add_prefix("off_"),delta[ALL_TRANSITIONS].add_prefix("delta_")],axis=1)
    write_csv(private.rename_axis("participant_id").reset_index(),PRIVATE / "results/ON_OFF_Transition_Matrices.csv")


def health_reference(cohort,on,off,delta,verified):
    # Repeat V5's single, fixed-template distance definition and random sequence.
    ref, dist = restoration(cohort)
    assert V5_RESAMPLE.seed == CFG["restoration_seed"]
    h = cohort.loc[cohort.condition.eq("HC")].sort_values("subject")[ILR].to_numpy()
    a,b = on[ILR].to_numpy(),off[ILR].to_numpy()
    patients,hc = V5_RESAMPLE.resample(len(a),len(h),False)
    anchors = h[hc].mean(axis=1,keepdims=True)
    boot_off = np.linalg.norm(b[patients]-anchors,axis=-1)
    boot_on = np.linalg.norm(a[patients]-anchors,axis=-1)
    nested = boot_off-boot_on
    avg = nested.mean(axis=1)
    med = np.median(nested,axis=1)
    dz = standardized(nested,axis=1)
    dzlo,dzhi,dzn = interval(dz)
    avg_off_low,avg_off_high,_ = interval(boot_off.mean(axis=1))
    avg_on_low,avg_on_high,_ = interval(boot_on.mean(axis=1))
    observed = dist[:,2]
    check = bootstrap_record(avg,ref["effect_size"])
    np.testing.assert_allclose([ref["ci95_low"],ref["ci95_high"]],[check["ci95_low"],check["ci95_high"]],atol=1e-12)
    ref.update(family="Health reference",contrast="OFF distance minus ON distance",**check,
               dz_ci95_low=dzlo,dz_ci95_high=dzhi,dz_bootstrap_valid=dzn,
               mean_off_ci95_low=avg_off_low,mean_off_ci95_high=avg_off_high,mean_on_ci95_low=avg_on_low,mean_on_ci95_high=avg_on_high,
               n_closer=int(np.sum(direction(observed)>0)),n_farther=int(np.sum(direction(observed)<0)),n_tied=int(np.sum(direction(observed)==0)))
    adjusted_draws = {}
    for tag,field in [("fd","fd"),("dvars","mean_std_dvars")]:
        cov = delta[field].to_numpy()
        adjusted, _ = covariate_test(observed,cov,greater=True,sign_set=V5_RESAMPLE.signs(len(observed)))
        if tag=="fd":
            np.testing.assert_allclose([adjusted["effect"],adjusted["p"]],[ref["fd_effect"],ref["fd_p"]],atol=1e-12)
        xv = cov[patients]
        xmean = xv.mean(axis=1)
        xc = xv-xmean[:,None]
        var = (xc*xc).sum(axis=1)
        slopes = np.divide((xc*nested).sum(axis=1),var,out=np.full(len(var),np.nan),where=var>0)
        boot = avg-slopes*xmean
        adjusted_draws[f"{tag}_effect"] = boot
        adjusted.update(bootstrap_record(boot,adjusted["effect"]))
        ref.update({f"{tag}_{k}":v for k,v in adjusted.items()})
    valid = observed[verified]
    ref.update(order_subset_p=tail_probability(signs(len(valid))@valid/len(valid),valid.mean(),two_sided=False),
               order_subset_effect=valid.mean(),order_subset_n=len(valid))
    reference = pd.read_csv(V5 / "results/coverage/Coverage_Energy.csv").set_index("contrast").loc["ON-HC"]
    fields = ["endpoint","raw_p_basis","effect_size","effect_units","ci95_low","ci95_high","fd_effect","raw_p","permutation_p","fd_p",
              "permutation_draws","n_a","n_b","bootstrap_requested","bootstrap_valid"]
    origin = {key:reference[key] for key in fields}
    origin.update(family="Health reference",contrast="ON-HC",v5_six_permutation_holm=reference.permutation_holm_p,
                  v5_six_fd_holm=reference.fd_holm_p,study_role="frozen V5 independent health comparison")
    ref["study_role"] = "V6 paired health reference using V5 definition"
    historical = pd.read_csv(V5 / "results/restoration/Restoration_Results.csv").iloc[0]
    ref.update(v5_six_permutation_holm=historical.permutation_holm_p,v5_six_fd_holm=historical.fd_holm_p)
    rows = adjust(pd.DataFrame([ref,origin]))
    rows["tost_status"] = "No externally supported equivalence margin"
    write_csv(rows,RESULT / "Health_Reference_Results.csv")
    write_csv(rows.loc[rows.endpoint.eq("restoration")],RESULT / "Restoration_Results.csv")
    health = pd.DataFrame(dist,index=delta.index,columns=["Distance_OFF_to_HC","Distance_ON_to_HC","Restoration_Score"])
    health["coverage_paired_aitchison_distance"] = np.linalg.norm(a-b,axis=1)
    write_csv(health.rename_axis("participant_id").reset_index(),PRIVATE / "results/HC_Reference_Distances.csv")
    magnitude = health.coverage_paired_aitchison_distance.to_numpy()
    low,high,n = interval(magnitude[draws(len(magnitude))].mean(axis=1))
    write_csv(pd.DataFrame([dict(summary="within-patient ON to OFF Aitchison distance",n_pairs=len(magnitude),
                                 mean=magnitude.mean(),median=np.median(magnitude),ci95_low=low,ci95_high=high,bootstrap_valid=n)]),
              RESULT / "Coverage_Paired_Distance_Summary.csv")
    boot_table = pd.DataFrame(dict(endpoint="restoration",replicate=np.arange(1,B+1),mean_difference=avg,median_difference=med,dz=dz,
                                  **adjusted_draws))
    return rows,health,boot_table


def heterogeneity(delta, tests):
    features = (*HETEROGENEITY_KEYS,*(f"coverage_{i}" for i in STATE))
    key = tests.set_index("endpoint")
    direction_rows = []
    for feature in features:
        row = key.loc[feature]
        d = delta[feature].dropna().to_numpy()
        iqr = np.quantile(d,[.25,.75])
        detail = {k:row[k] for k in ["n_pairs","mean_difference","median_difference","ci95_low","ci95_high",
                                     "median_ci95_low","median_ci95_high","positive_wilson_low","positive_wilson_high",
                                     "negative_wilson_low","negative_wilson_high","n_positive","n_negative","n_tied","sd_difference"]}
        direction_rows.append(dict(endpoint=feature,units=effect_units(feature),delta_q1=iqr[0],delta_q3=iqr[1],**detail))
    write_csv(pd.DataFrame(direction_rows),RESULT / "Paired_Change_Directions.csv")

    d = delta[HETEROGENEITY_KEYS].dropna()
    standardized_change = d.to_numpy()/d.std(ddof=1).to_numpy()
    # Leave the current person out of the comparison direction.
    reference = (standardized_change.sum(axis=0)-standardized_change)/(len(d)-1)
    norm_product = np.linalg.norm(reference,axis=1)*np.linalg.norm(standardized_change,axis=1)
    cosine = np.divide((standardized_change*reference).sum(axis=1),norm_product,
                       out=np.full(len(d),np.nan),where=norm_product>0)
    patterns = (d-d.mean()).to_numpy()/d.std(ddof=1).to_numpy()
    cluster = cut_tree(linkage(patterns,method="ward"),n_clusters=2).ravel()
    rng = np.random.default_rng([CFG["seed"],55])
    subsample_n = int(round(.8*len(d)))
    similarity = []
    for _ in range(CFG["heterogeneity_subsamples"]):
        idx = np.sort(rng.choice(len(d),subsample_n,replace=False))
        sub = cut_tree(linkage(patterns[idx],method="ward"),n_clusters=2).ravel()
        similarity.append(adjusted_rand_score(cluster[idx],sub))
    profiles = d.assign(group_direction_cosine=cosine,exploratory_ward_cluster=cluster+1)
    write_csv(profiles.rename_axis("participant_id").reset_index(),PRIVATE / "results/Paired_Response_Patterns.csv")
    report = dict(n_profiles=len(d),n_direction_defined=int(np.isfinite(cosine).sum()),
                  n_same_halfspace=int(np.sum(cosine>0)),n_opposite_halfspace=int(np.sum(cosine<0)),
                  median_leave_one_out_cosine=float(np.nanmedian(cosine)),
                  cosine_quartiles=np.nanquantile(cosine,[.25,.75]).tolist(),
                  ward_clusters=2,ward_cluster_sizes=sorted(np.bincount(cluster).tolist()),
                  ward_silhouette=float(silhouette_score(patterns,cluster)),
                  stability_median_ari=float(np.median(similarity)),stability_ari_quartiles=np.quantile(similarity,[.25,.75]).tolist(),
                  stability_subsamples=len(similarity),subsample_pairs=subsample_n)
    write_json(report,RESULT / "Paired_Response_Patterns.json")

    impact, private = [],[]
    for feature in features:
        values = delta[feature].dropna()
        result, loo = influence(values.to_numpy())
        impact.append(dict(endpoint=feature,units=effect_units(feature),**result))
        private.append(pd.DataFrame(loo,columns=["loo_mean","loo_dz","loo_paired_t_p"],index=values.index).assign(endpoint=feature))
    write_csv(pd.DataFrame(impact),RESULT / "Leave_One_Out_Influence_Summary.csv")
    write_csv(pd.concat(private).rename_axis("participant_id").reset_index(),PRIVATE / "results/Leave_One_Out_Paired_Effects.csv")

    dispersion = association(delta.gev_1.to_numpy(),delta.dispersion_1.to_numpy(),delta.fd.to_numpy())
    write_csv(adjust(pd.DataFrame([dict(endpoint="Delta M1 GEV vs Delta M1 dispersion",family="Template fit association",**dispersion)])),
              RESULT / "M1_GEV_Dispersion_Association.csv")


def quality_sensitivity(delta, manifest):
    rows = []
    for feature in ("fd","mean_raw_dvars","mean_std_dvars","valid_frames"):
        x = delta[feature].to_numpy()
        p,n = logical_p(x,np.isfinite(x))
        effect = x.mean()
        row = dict(endpoint=feature,family="Scan quality descriptive",n_pairs=len(x),mean_difference=effect,
                   median_difference=np.median(x),raw_p=ttest_1samp(x,0.).pvalue,permutation_p=p,permutation_draws=n,
                   **bootstrap_record(x[draws(len(x))].mean(axis=1),effect))
        rows.append(row)
    write_csv(adjust(pd.DataFrame(rows),{"raw_p":"","permutation_p":"permutation_"}),RESULT / "Paired_Scan_Quality_Sensitivity.csv")
    timing = manifest.scan_interval.to_numpy()
    keys = [*CFG["primary"],"occurrence_1","switching_rate"]
    assoc = [dict(endpoint=feature,family="Scan interval effect association",**association(timing,delta[feature].to_numpy(),delta.fd.to_numpy()))
             for feature in keys]
    write_csv(adjust(pd.DataFrame(assoc)),RESULT / "Scan_Interval_Associations.csv")


def clinical_associations(delta):
    audit = read_json(RESULT / "UPDRS_Input_Audit.json")
    if audit["status"]=="BLOCKED":
        rows = [dict(endpoint=key,family="UPDRS association",status=audit["status"],n_pairs=audit["n_eligible_pairs"],
                     reason=audit["reason"],rho=np.nan,ci95_low=np.nan,ci95_high=np.nan,loo_rho_low=np.nan,loo_rho_high=np.nan,
                     raw_p=np.nan,permutation_p=np.nan,fd_p=np.nan) for key in HETEROGENEITY_KEYS]
    else:
        clinical = pd.read_csv(PRIVATE / "results/Verified_Clinical_Pairs.csv").set_index("participant_id")
        combined = delta.join(clinical,how="inner",validate="one_to_one")
        improvement = combined.updrs_iii_off-combined.updrs_iii_on
        rows = [dict(endpoint=key,family="UPDRS association",**association(combined[key].to_numpy(),improvement.to_numpy(),combined.fd.to_numpy()))
                for key in HETEROGENEITY_KEYS]
        write_csv(combined[HETEROGENEITY_KEYS].assign(updrs_off_minus_on=improvement).rename_axis("participant_id").reset_index(),
                  PRIVATE / "results/UPDRS_Association_Pairs.csv")
    columns = {k: P_MODELS[k] for k in ("raw_p","permutation_p","fd_p")}
    write_csv(adjust(pd.DataFrame(rows),columns),RESULT / "UPDRS_Association_Results.csv")


def history():
    old = pd.read_csv(V4 / "results/All_Scalar_Results.csv")
    old = old.loc[old.contrast.eq("ON-OFF") & old.metric.isin(PRIORITY_CONTEXT+['dispersion_3','dispersion_4'])].copy()
    old["frozen_commit"] = CFG["v4_commit"]
    write_csv(old,RESULT / "V4_Frozen_ON_OFF_Results.csv")
    om = pd.read_csv(V5 / "results/transition/Transition_Omnibus.csv")
    keys = ["endpoint","contrast","effect_size","ci95_low","ci95_high","permutation_p","permutation_holm_p","fd_p","fd_holm_p"]
    rows = [om.loc[om.contrast.eq("ON-OFF"),keys],pd.read_csv(V5 / "results/restoration/Restoration_Results.csv")[keys]]
    write_csv(pd.concat(rows,ignore_index=True).assign(frozen_commit=CFG["v5_commit"]),RESULT / "V5_Frozen_Paired_Results.csv")


def planning(stats):
    spec = CFG["planning"]
    solver = TTestPower()
    alpha = ALPHA/spec["primary_tests"]
    values = [("conservative fixed target",d) for d in spec["effect_targets"]]
    primary = stats.loc[stats.family.eq("Primary")]
    for record in primary.itertuples(index=False):
        value = min(.40,.5*abs(record.dz))
        if value>1e-5:
            values.append((f"half observed {record.endpoint}, cap 0.40",value))
    rows = []
    for basis,d in values:
        for target in spec["power_targets"]:
            n = math.ceil(solver.solve_power(d,alpha=alpha,power=target))
            recruited = math.ceil(n/(1-spec["pair_attrition"]))
            rows.append(dict(basis=basis,dz=d,power_target=target,alpha=alpha,retained_pairs=n,planned_to_recruit=recruited,
                             planned_scans=2*recruited,assumed_pair_loss=spec["pair_attrition"],achieved_power=solver.power(d,n,alpha)))
    write_csv(pd.DataFrame(rows),RESULT / "Paired_Sample_Size_Planning.csv")


def main():
    cohort,on,off,delta,manifest = paired_inputs()
    verified = manifest.scan_order.ne("unverified").to_numpy()
    stats,om,boot = inference(on,off,delta,verified)
    transition_matrices(on,off,delta)
    health,distances,nested = health_reference(cohort,on,off,delta,verified)
    write_csv(pd.concat([delta,distances],axis=1).rename_axis("participant_id").reset_index(),PRIVATE / "results/ON_OFF_Paired_Differences.csv")
    write_csv(pd.concat([*boot,nested],ignore_index=True),PRIVATE / "results/Paired_Bootstrap_Draws.csv")
    keep = ["endpoint","family","n_pairs","ci95_low","ci95_high","bootstrap_requested","bootstrap_valid","bootstrap_failed",
            "bootstrap_direction_consistency","fd_bootstrap_valid","fd_bootstrap_failed","fd_ci95_low","fd_ci95_high",
            "fd_bootstrap_direction_consistency","dvars_bootstrap_valid","dvars_bootstrap_failed","dvars_ci95_low","dvars_ci95_high",
            "dvars_bootstrap_direction_consistency"]
    reg = health.loc[health.endpoint.eq("restoration")].assign(n_pairs=lambda d:d.n_a)
    write_csv(pd.concat([stats[keep],om[keep],reg[keep]],ignore_index=True),RESULT / "Paired_Bootstrap_Audit.csv")
    heterogeneity(delta,stats)
    quality_sensitivity(delta,manifest)
    clinical_associations(delta)
    history()
    planning(stats)
    print(f"V6 computed {len(stats)} paired scalar tests, {len(om)} paired omnibus tests, and {len(health)} health-reference estimates.")


if __name__ == "__main__":
    main()
