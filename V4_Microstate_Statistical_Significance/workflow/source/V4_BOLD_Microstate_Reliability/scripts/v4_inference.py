"""Frozen V4 condition contrasts, legacy correspondence and cohort planning."""

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent / "V3_BOLD_Microstate_Method_Optimization"
sys.path.insert(0, str(V3 / "scripts"))
from inference_core import adjust, composition, scalar  # noqa: E402

CONFIG = json.loads((ROOT / "configs/inference.json").read_text(encoding="utf-8"))
METHODS = ("A0", "A1", "A2", "A3")
BANK = {"A0": "G0", "A1": "G1", "A2": "G0", "A3": "G1"}
HC_CONTRASTS = tuple(CONFIG["primary_contrasts"])
CONTRASTS = HC_CONTRASTS + (CONFIG["paired_secondary_contrast"],)
STATE_METRICS = tuple(CONFIG["state_metrics"])
COVERAGE = [f"coverage_{state}" for state in range(1, 5)]
CORE = tuple(CONFIG["primary_endpoints"])
COMPANION = tuple(CONFIG["companion_endpoints"])
STATE_COLUMNS = tuple(f"{m}_{i}" for m in STATE_METRICS for i in range(1, 5))
MEASURES = CORE + COMPANION + STATE_COLUMNS
STATE_INDEX = {name: int(name[-1]) - 1 for name in STATE_COLUMNS}
EXPECTED = {"ON": 25, "OFF": 25, "HC": 15}
MAP_COLUMNS = [f"ROI_{i}" for i in range(1, 79)]


def read_calibration(path):
    """Group-outcome inference follows the completed formal calibration."""
    selected = json.loads(path.read_text(encoding="utf-8"))
    reps = selected.get("completed_repetitions")
    if (selected.get("stage") != "formal" or selected.get("preferred_method") not in (*METHODS, None)
            or "preferred_method" not in selected or not isinstance(reps, dict) or len(reps) < 6
            or not all(isinstance(n, int) and n > 0 for n in reps.values())
            or not np.isfinite(float(selected["lambda"])) or float(selected["lambda"]) < 0):
        raise ValueError("Complete and freeze the formal calibration before real condition contrasts")
    return selected


def verify_features(data):
    """Enforce the frozen scan and participant roster in every lane."""
    columns = {"method", "window", "scan", "subject", "condition", "fold", "fd", "duration_s"}
    columns.update(metric for metric in MEASURES if metric != "coverage_ilr")
    if not columns <= set(data):
        raise ValueError(f"Missing feature columns: {sorted(columns - set(data))}")
    if data.duplicated(["method", "window", "scan"]).any():
        raise ValueError("Duplicated scan in a method/window")
    grouped = data.groupby(["method", "window"])
    for method in METHODS:
        if (method, "common") not in grouped.groups:
            raise ValueError(f"Missing primary window for {method}")
    identity = ["scan", "subject", "condition", "fold", "fd"]
    roster = grouped.get_group(("A0", "common"))[identity].sort_values("scan").reset_index(drop=True)
    fold_sizes = {"HC": dict.fromkeys(range(5), 3), "ON": dict.fromkeys(range(5), 5),
                  "OFF": dict.fromkeys(range(5), 5)}
    if (len(roster) != 65 or roster.subject.nunique() != 40 or not np.isfinite(roster.fd).all()
            or roster.groupby("condition").size().to_dict() != EXPECTED
            or roster.groupby(["subject", "condition"]).size().max() != 1
            or roster.groupby("subject").fold.nunique().max() != 1
            or roster.groupby(["fold", "condition"]).size().unstack().to_dict() != fold_sizes):
        raise ValueError("Expected the frozen five-fold roster")
    patients = roster[roster.condition.isin(("ON","OFF"))]
    if (patients.subject.nunique() != 25 or (patients.groupby("subject").size() != 2).any()):
        raise ValueError("ON and OFF must be paired within the same patients")
    for key, block in grouped:
        if key[1] != "edge" and key[0] not in CONFIG["inference_windows"].get(key[1], []):
            raise ValueError(f"Unregistered method/window: {key}")
        pd.testing.assert_frame_equal(roster, block[identity].sort_values("scan").reset_index(drop=True))
        duration = block.duration_s.to_numpy(float)
        coverage = block.loc[np.isfinite(duration) & (duration > 0), COVERAGE].to_numpy(float)
        if not (np.isfinite(coverage).all() and np.all(coverage >= 0)
                and np.allclose(coverage.sum(axis=1), 1)):
            raise ValueError(f"Invalid coverage simplex for {key}")
    exposure = data[data.window == "common"].pivot(index="scan",columns="method",values="duration_s")
    if not (np.isfinite(exposure.to_numpy()).all() and np.allclose(exposure,exposure[["A0"]].to_numpy())):
        raise ValueError("Methods must share one exposure per scan in the main window")


def reorder_features(frame, order):
    """Give scan-level state features the map-matched fold-zero labels."""
    result = frame.copy()
    for fold, perm in order.items():
        take = result.fold == fold
        for measure in STATE_METRICS:
            names = [f"{measure}_{j + 1}" for j in range(4)]
            result.loc[take, names] = frame.loc[take, [names[j] for j in perm]].to_numpy()
    return result


def family_of(metric):
    if metric in CORE:
        return "core"
    return "state" if metric in STATE_INDEX else "supporting"


def summarize_group_means(data, stable):
    for condition, block in data.groupby("condition"):
        for metric in MEASURES:
            if metric == "coverage_ilr":
                continue
            mapped = metric not in STATE_INDEX or stable[STATE_INDEX[metric]]
            y = block.loc[(block.duration_s > 0) & np.isfinite(block[metric]), metric] if mapped else pd.Series(dtype=float)
            reason = "unmatched across folds" if not mapped else ("estimated" if len(y) >= 3 else "fewer than three")
            yield dict(condition=condition, outcome=metric, n=len(y),
                       mean=y.mean() if len(y) >= 3 else np.nan, sd=y.std() if len(y) >= 3 else np.nan,
                       status=reason)


def missing_row(method, window, metric, contrast, reason):
    return dict(method=method, window=window, outcome=metric, contrast=contrast,
                family=family_of(metric), status=reason)


def conditional_rows(frame, metric, base):
    """Make C3 use complete patient pairs if a state duration is unavailable."""
    usable = np.isfinite(frame.duration_s) & (frame.duration_s > 0)
    if metric != "coverage_ilr":
        usable &= np.isfinite(frame[metric])
    data = frame.loc[usable]
    pd_rows = data.condition.isin(("ON","OFF"))
    counts = data.loc[pd_rows, "subject"].value_counts()
    keep_pairs = ~pd_rows | data.subject.isin(counts.index[counts.eq(2)])
    if metric == "coverage_ilr":
        cfg = CONFIG["composition"]

        def fit(cases):
            return composition(cases, COVERAGE, draws=cfg["wild_bootstrap_draws"], seed=cfg["wild_bootstrap_seed"])
    else:
        def fit(cases):
            return scalar(cases, metric)

    reasons = {}
    summaries = {}
    try:
        summaries.update(fit(data))
    except (ValueError, np.linalg.LinAlgError) as exc:
        reasons.update(dict.fromkeys(CONTRASTS, str(exc)))
    if not keep_pairs.all():
        try:
            summaries["C3"] = fit(data.loc[keep_pairs])["C3"]
            reasons.pop("C3", None)
        except (ValueError, np.linalg.LinAlgError, KeyError) as exc:
            reasons["C3"] = str(exc)
            summaries.pop("C3", None)
    for contrast in CONTRASTS:
        cases = data.loc[keep_pairs] if contrast == "C3" else data
        test = summaries.get(contrast, {})
        status = reasons.get(contrast, "estimated" if test else "contrast unavailable")
        if test and not np.isfinite(test["P_raw"]):
            status = "undefined robust variance"
        yield dict(base, contrast=contrast, n_scans=len(cases), n_participants=cases.subject.nunique(),
                   n_hc=int((cases.condition == "HC").sum()), n_on=int((cases.condition == "ON").sum()),
                   n_off=int((cases.condition == "OFF").sum()), n_complete_patient_pairs=int(counts.eq(2).sum()),
                   status=status) | test


def multiple_testing(table, preferred):
    """Keep failed endpoints in the registered adjustment denominator."""
    table = table.copy()
    table["source"] = "conditional_on_held_out_maps"
    table["calibrated_core"] = (table.method.eq(preferred) & table.window.eq("common") &
                                table.family.eq("core") & table.contrast.isin(HC_CONTRASTS))
    table["scope"] = np.where(table.contrast.isin(HC_CONTRASTS), "HC", "paired_auxiliary")
    families = table.groupby(["method", "window", "family", "scope"], sort=False)["P_raw"]
    table["P_BH_family"] = families.transform(lambda p: adjust(p, "fdr_bh"))
    table["P_Holm_family"] = families.transform(lambda p: adjust(p, "holm"))
    table["P_Holm_search"] = table.groupby("scope", sort=False)["P_raw"].transform(lambda p: adjust(p, "holm"))
    return table


def conjunction(table):
    """Test the larger two-sided scalar P if the two effects have the same direction."""
    subset = table[table.contrast.isin(HC_CONTRASTS) & table.outcome.ne("coverage_ilr")]
    rows = []
    for (method, window, outcome, family), group in subset.groupby(["method", "window", "outcome", "family"]):
        tests = group.set_index("contrast")
        a, b = (tests.loc[c] for c in HC_CONTRASTS)
        evaluable = np.isfinite([a.P_raw, b.P_raw]).all()
        same = evaluable and a.Effect_Size * b.Effect_Size > 0
        rows.append(dict(method=method, window=window, outcome=outcome, family=family, same_direction=same,
                         effect_on_hc=a.Effect_Size, effect_off_hc=b.Effect_Size,
                         both_hc_search_pass=bool(same and max(a.P_Holm_search, b.P_Holm_search) < .05),
                         P_raw=(max(a.P_raw, b.P_raw) if same else 1.) if evaluable else np.nan))
    out = pd.DataFrame(rows)
    out["P_Holm_family"] = out.groupby(["method", "window", "family"])["P_raw"].transform(lambda p: adjust(p, "holm"))
    out["P_Holm_search"] = adjust(out.P_raw, "holm")
    return out


def fit_inference(data, alignments, preferred):
    tables, summaries = [], []
    verify_features(data)
    observed = data.groupby(["method", "window"])
    for window, methods in CONFIG["inference_windows"].items():
        for method in methods:
            key = (method, window)
            if key not in observed.groups:
                tables.extend(missing_row(method, window, metric, c, "registered sensitivity not available")
                              for metric in MEASURES for c in CONTRASTS)
                continue
            match = alignments[BANK[method]]
            block = reorder_features(observed.get_group(key), match["order"])
            summaries.extend(dict(method=method, window=window, **row)
                             for row in summarize_group_means(block, match["stable"]))
            for metric in MEASURES:
                base = dict(method=method, window=window, outcome=metric, family=family_of(metric))
                if metric in STATE_INDEX and not match["stable"][STATE_INDEX[metric]]:
                    tables.extend(missing_row(method, window, metric, c, "unmatched across folds") for c in CONTRASTS)
                elif metric == "coverage_ilr" and not match["composition"]:
                    tables.extend(missing_row(method, window, metric, c, "full composition not identifiable") for c in CONTRASTS)
                else:
                    tables.extend(conditional_rows(block, metric, base))
    table = pd.DataFrame(tables)
    for column in ("P_raw","Effect_Size"):
        if column not in table:
            table[column] = np.nan
    return multiple_testing(table, preferred), pd.DataFrame(summaries)


def archived_banks(v3_templates, v3_method):
    source = v3_templates.loc[(v3_templates.experiment == v3_method) & (v3_templates.fold >= 0)]
    maps = [frame.sort_values("state")[MAP_COLUMNS].to_numpy(float) for _, frame in source.groupby("fold")]
    if len(maps) != 5 or any(arr.shape != (4, 78) for arr in maps):
        raise ValueError(f"Expected five-fold four-map archive of {v3_method}")
    return maps


def legacy_links(bank_dict, alignments, v3_templates):
    """Compare every V4 fold with every V3 fold and keep legacy names separate."""
    from v4_models import match_banks

    rows = []
    for legacy_method in ("E09","E04"):
        legacy = archived_banks(v3_templates, legacy_method)
        legacy_reference = match_banks(legacy)
        for name, bank in bank_dict.items():
            maps = [bank[fold][alignments[name]["order"][fold]] for fold in range(5)]
            comparison = match_banks(np.stack(maps + legacy))
            for state in range(4):
                raw_orders = [int(comparison["order"][fold+5][state]) for fold in range(5)]
                r_orders = [int(np.argsort(legacy_reference["order"][f])[j])+1 for f,j in enumerate(raw_orders)]
                candidates = [j+1 for j in raw_orders] if legacy_method=="E09" else r_orders
                same_name = len(set(candidates)) == 1
                v4_stable = bool(alignments[name]["stable"][state])
                aligned = same_name and v4_stable and bool(comparison["stable"][state])
                rows.append(dict(bank=name, v4_state=state+1, v3_experiment=legacy_method,
                                 v3_naming="archived" if legacy_method=="E09" else "R",
                                 v3_candidate_state=candidates[0] if same_name else None,
                                 n_v3_folds_agreeing=max(candidates.count(s) for s in set(candidates)),
                                 v4_stable=v4_stable, combined_crossfold_gate=bool(comparison["stable"][state]),
                                 reliable_legacy_link=aligned))
    return pd.DataFrame(rows)


def clue_comparison(table, legacy, archived):
    """Display prespecified V3 OFF–HC clues next to comparable V4 estimates."""
    items = [("E09","mean_margin"),("E09","low_margin_fraction"),
             ("E09","coverage_3"),("E09","gev_3"),("E04_R","coverage_3"),("E04_R","gev_3")]
    rows = []
    for experiment, metric in items:
        chosen = archived[(archived.experiment==experiment)&(archived.metric==metric)&
                          (archived.contrast=="C2")&(archived.fit=="crossfit")&(archived.model=="CR1")]
        old = chosen.iloc[0]
        for method in METHODS:
            state_metric = metric in STATE_INDEX
            link = legacy[(legacy.bank==BANK[method]) & (legacy.v3_experiment==experiment.split("_")[0])
                          & legacy.reliable_legacy_link & (legacy.v3_candidate_state==3)]
            mapped = not state_metric or len(link)==1
            outcome = (f"{metric[:-1]}{int(link.v4_state.iloc[0])}" if state_metric else metric) if mapped else None
            for window in ("common","fd_censor"):
                row = dict(v3_experiment=experiment,v3_outcome=metric,v3_effect=old.Effect_Size,
                           v3_P_raw=old.P_raw,v3_P_Holm_family=old.P_Holm,v3_P_Holm_search=old.P_across_methods,
                           method=method,window=window,v4_outcome=outcome,v4_status="unmatched to legacy state")
                if mapped:
                    new = table[(table.method==method)&(table.window==window)&(table.outcome==outcome)&(table.contrast=="C2")].iloc[0]
                    row.update(v4_effect=new.Effect_Size,v4_ci_low=new.get("CI_low"),v4_ci_high=new.get("CI_high"),
                               v4_P_raw=new.P_raw,v4_P_Holm_family=new.P_Holm_family,
                               v4_P_Holm_search=new.P_Holm_search,v4_status=new.status)
                rows.append(row)
    return pd.DataFrame(rows)


def plan_power(options=None):
    """Balanced independent-sample planning using prespecified standardized shifts."""
    cfg = CONFIG["planning"] if options is None else options
    significance = cfg["alpha_family"] / cfg["tests"]
    rows = []
    for endpoint, effect, target in itertools.chain(
        (("scalar", effect,target) for effect,target in itertools.product(cfg["d"], cfg["power"])),
        (("coverage_hotelling_3d", effect,target)
         for effect,target in itertools.product(cfg["coverage_mahalanobis_d"],cfg["power"])),
    ):
        for n in range(4,10000):
            noncentrality = effect * np.sqrt(n/2)
            if endpoint == "scalar":
                df = 2*n-2
                cutoff = stats.t.isf(significance/2, df)
                power = stats.nct.sf(cutoff,df,noncentrality) + stats.nct.cdf(-cutoff,df,noncentrality)
            else:
                numerator_df = 3
                denominator_df = 2*n-1-numerator_df
                cutoff = stats.f.isf(significance,numerator_df,denominator_df)
                power = stats.ncf.sf(cutoff,numerator_df,denominator_df,noncentrality**2)
            if power >= target:
                break
        else:
            raise ValueError("Planning target outside declared n range")
        recruit = int(np.ceil(n/(1-cfg["attrition"])))
        rows.append(dict(endpoint=endpoint,target_standardized_effect=effect, per_test_alpha=significance,
                         target_marginal_power=target, achieved_power=power, analyzable_patient_n=n,
                         analyzable_HC_n=n, attrition=cfg["attrition"], recruit_patient_n=recruit,
                         recruit_HC_n=recruit, expected_scans_if_all_complete=3*recruit))
    return pd.DataFrame(rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--features", type=Path, default=ROOT/"private/V4_Scan_Metrics.csv")
    cli.add_argument("--maps", type=Path, default=ROOT/"private/V4_Maps.npz")
    cli.add_argument("--calibration",type=Path,default=ROOT/"results/Calibration_Selection.json")
    cli.add_argument("--output",type=Path,default=ROOT/"results")
    cli.add_argument("--power-only",action="store_true")
    args = cli.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plan_power().to_csv(args.output/"Independent_Cohort_Power.csv",index=False)
    if args.power_only:
        return
    from v4_models import match_banks

    gate = read_calibration(args.calibration)
    data = pd.read_csv(args.features, dtype={"scan":str,"subject":str}, low_memory=False)
    with np.load(args.maps) as archive:
        banks = {key: archive[key] for key in ("G0","G1")}
    alignments = {key: match_banks(maps) for key, maps in banks.items()}
    with threadpool_limits(1):
        table, groups = fit_inference(data, alignments, gate["preferred_method"])
    table.to_csv(args.output/"V4_Conditional_Inference.csv",index=False)
    groups.to_csv(args.output/"V4_Group_Summary.csv",index=False)
    conjunction(table).to_csv(args.output/"V4_HC_Conjunction.csv",index=False)
    archive_table = pd.read_csv(V3/"results/Group_Templates.csv",low_memory=False)
    legacy = legacy_links(banks, alignments, archive_table)
    legacy.to_csv(args.output/"V3_State_Correspondence.csv",index=False)
    archive_inference = pd.read_csv(V3/"results/All_Methods_Comparison.csv",low_memory=False)
    clue_comparison(table,legacy,archive_inference).to_csv(args.output/"V3_V4_Clue_Comparison.csv",index=False)
    print("Conditional real-data rows:", len(table), "estimated:", table.P_raw.notna().sum(),
          "search-Holm rejections:", (table.P_Holm_search<.05).sum(), "calibrated method:",gate["preferred_method"])


if __name__ == "__main__":
    main()
