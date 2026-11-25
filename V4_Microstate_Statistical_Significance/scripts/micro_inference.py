"""Complete scalar and compositional tests on participants, with fixed families."""

import json
from functools import lru_cache
from math import comb
import sys

import numpy as np
import pandas as pd

from micro_inputs import EDGE, FEATURES, HERE, PREVIOUS, SETTINGS, STATE, load_inputs, write_templates

sys.path.insert(0, str(PREVIOUS))
from statistics_v4 import Resampling, corrections, freedman_lane, interval, outcome_test, select_samples, tail_probability
from inference_core import adjust, scalar


CODES = {"OFF-HC": "C2", "ON-HC": "C1", "ON-OFF": "C3"}
CORE_FAMILIES = {"Coverage", "Duration", "Occurrence"}
EXPLORATORY_FAMILIES = {"GEV", "Spatial Dispersion", "Switching Rate", "Sorted Coverage ILR"}
P_VALUES = {"raw_p": "", "permutation_p": "permutation_", "fd_permutation_p": "fd_permutation_", "fd_cr1_p": "fd_cr1_"}


class UniqueResampling(Resampling):
    @lru_cache(maxsize=32)
    def independent(self, nx, ny):
        if comb(nx + ny, nx) - 1 < self.permutations:
            raise ValueError("Fewer than 10,000 unique group allocations")
        rng = self.rng(11, nx, ny)
        identities = {tuple(range(nx))}
        rows = []
        while len(rows) < self.permutations:
            batch = rng.random((self.permutations - len(rows), nx + ny)).argsort(axis=1)
            for order in batch:
                key = tuple(sorted(order[:nx]))
                if key not in identities:
                    identities.add(key)
                    rows.append(order)
        return np.stack(rows)

    @lru_cache(maxsize=32)
    def signs(self, n):
        if (1 << n) - 1 < self.permutations:
            raise ValueError("Fewer than 10,000 unique paired sign flips")
        # Use -1 for a set bit to reserve mask zero for the observed orientation.
        masks = self.rng(12, n, 0).choice((1 << n) - 1, size=self.permutations, replace=False) + 1
        return 1. - 2. * ((masks[:, None] >> np.arange(n)) & 1)


def endpoints():
    labels = [("coverage", "Coverage", "proportion"), ("duration", "Duration", "s"),
              ("occurrence", "Occurrence", "episodes/s"), ("gev", "GEV", "proportion"),
              ("dispersion", "Spatial Dispersion", "unit-vector distance")]
    rows = [dict(metric=f"{key}_{j}", state=f"M{j}", family=family, unit=unit) for key, family, unit in labels for j in STATE]
    rows += [dict(metric=f"transition_{i}_{j}", state=f"M{i}->M{j}", family="Transition Probability", unit="proportion")
             for i, j in EDGE]
    rows.append(dict(metric="switching_rate", state="", family="Switching Rate", unit="transitions/s"))
    return rows


def correction_family(family, contrast):
    if family in CORE_FAMILIES:
        block = "paired" if contrast == "ON-OFF" else "HC contrasts"
        return f"{family}: {block}"
    return family


def summary_stats(x, suffix):
    if not len(x):
        return {f"{k}_{suffix}": np.nan for k in ("mean", "sd", "median", "q1", "q3", "iqr")}
    a, b = np.quantile(x, [.25, .75])
    return {f"{k}_{suffix}": v for k, v in dict(mean=x.mean(), sd=np.std(x, ddof=1) if len(x)>1 else np.nan,
                                                  median=np.median(x), q1=a, q3=b, iqr=b-a).items()}


def scalar_tests(data, resampling):
    rows = []
    for endpoint in endpoints():
        key = endpoint["metric"]
        assess = data.loc[data[key].notna()]
        try:
            cr1 = scalar(assess, key)
        except (ValueError, np.linalg.LinAlgError):
            cr1 = {}
        for contrast in SETTINGS["contrasts"]:
            first, second = contrast.split("-")
            paired = contrast == "ON-OFF"
            x, y, fd_x, fd_y = select_samples(data, key, first, second, paired)
            status = "ok" if min(len(x), len(y)) >= 3 else "unidentified state or insufficient observations"
            d = dict(**endpoint, contrast=contrast, correction_family=correction_family(endpoint["family"], contrast),
                     group_a=first, group_b=second, n_a=len(x), n_b=len(y), n_pairs=len(x) if paired else np.nan, status=status)
            d.update(summary_stats(x, "a"), **summary_stats(y, "b"))
            if status == "ok":
                d.update(outcome_test(x, y, paired, SETTINGS, resampling))
                d.update(freedman_lane(x, y, fd_x, fd_y, paired, resampling))
                d["permutation_draws"] = resampling.permutations
                d["fd_permutation_draws"] = resampling.permutations if d.get("fd_permutation_status") == "ok" else 0
            if CODES[contrast] in cr1:
                record = cr1[CODES[contrast]]
                d.update(fd_cr1_effect=record["Effect_Size"], fd_cr1_p=record["P_raw"],
                         fd_cr1_ci95_low=record["CI_low"], fd_cr1_ci95_high=record["CI_high"])
            rows.append(d)
    result = pd.DataFrame(rows)
    corrections(result, P_VALUES, ["correction_family"])
    result["family_size"] = result.groupby("correction_family").metric.transform("size")
    broad = result.assign(all_scalar_family="all scalar contrasts")
    corrections(broad, {p: "all_scalar_" + s for p, s in P_VALUES.items()}, ["all_scalar_family"])
    for col in broad.filter(regex=r"^all_scalar_.*_p$"):
        result[col] = broad[col]
    return result


def wald(delta, covariance):
    solution = np.linalg.solve(covariance, delta[..., None])[..., 0]
    return np.sum(delta * solution, axis=-1)


def independent_wald(x, y):
    a = x - x.mean(axis=-2, keepdims=True)
    b = y - y.mean(axis=-2, keepdims=True)
    nx, ny = x.shape[-2], y.shape[-2]
    covariance = a.swapaxes(-1, -2) @ a / (nx * (nx - 1)) + b.swapaxes(-1, -2) @ b / (ny * (ny - 1))
    return wald(x.mean(axis=-2) - y.mean(axis=-2), covariance)


def hc3_wald(design, outcome):
    inv = np.linalg.pinv(design)
    beta = inv @ outcome
    u = outcome - design @ beta
    leverage = np.sum(design * inv.T, axis=1)
    if np.any(leverage >= 1 - 1e-12):
        raise ValueError("ILR HC3 undefined at leverage one")
    contributions = u * (inv[-1] / (1 - leverage))[:, None]
    return wald(beta[..., -1, :], contributions.swapaxes(-1, -2) @ contributions)


def multivariate_fd(pooled, motion, nx, order):
    nuisance = np.column_stack((np.ones(len(pooled)), motion - motion.mean()))
    group = np.r_[np.ones(nx), np.zeros(len(pooled) - nx)]
    design = np.column_stack((nuisance, group))
    fitted = nuisance @ np.linalg.lstsq(nuisance, pooled, rcond=None)[0]
    residual = pooled - fitted
    stat = hc3_wald(design, pooled)
    null = hc3_wald(design, fitted + residual[order])
    if not np.isfinite(null).all():
        raise ValueError("Nonfinite nuisance-adjusted ILR permutations")
    return float(stat), tail_probability(null, stat, two_sided=False)


def composition_tests(data, resampling):
    out = []
    for kind, family in [("ilr", "Coverage ILR"), ("sorted_ilr", "Sorted Coverage ILR")]:
        keys = [f"{kind}_{i}" for i in range(1, 4)]
        for target in ("OFF", "ON"):
            frames = [data.loc[data.condition.eq(group), ["subject", "fd", *keys]].dropna().sort_values("subject")
                      for group in (target, "HC")]
            x, y = [d[keys].to_numpy() for d in frames]
            row = dict(metric=kind, state="all four", family=family, correction_family=family,
                       contrast=f"{target}-HC", group_a=target, group_b="HC", n_a=len(x), n_b=len(y),
                       unit="ILR distance", test="participant-level Welch-type Wald permutation",
                       status="ok" if min(len(x), len(y)) >= 5 else "unidentified composition or insufficient observations")
            if row["status"] == "ok":
                ix, iy = resampling.resample(len(x), len(y), False)
                diff = x.mean(axis=0) - y.mean(axis=0)
                cd = x[ix].mean(axis=1) - y[iy].mean(axis=1)
                lower, upper = interval(np.linalg.norm(cd, axis=1))
                order = resampling.independent(len(x), len(y))
                pooled = np.vstack((x, y))
                obs = independent_wald(x, y)
                null = independent_wald(pooled[order[:, :len(x)]], pooled[order[:, len(x):]])
                if not np.isfinite(null).all():
                    raise ValueError("Nonfinite unadjusted ILR permutations")
                p = tail_probability(null, obs, two_sided=False)
                fd_stat, fdp = multivariate_fd(pooled, np.concatenate([d.fd.to_numpy() for d in frames]), len(x), order)
                row.update(effect_size=np.linalg.norm(diff), ci95_low=lower, ci95_high=upper, statistic=float(obs),
                           raw_p=p, permutation_p=p, fd_permutation_p=fdp, fd_permutation_wald=fd_stat,
                           permutation_draws=len(null), fd_permutation_draws=len(null))
                for j in range(3):
                    idx = j + 1
                    lo, hi = interval(cd[:, j])
                    row.update({f"ilr_delta_{idx}": diff[j], f"ilr_delta_ci95_low_{idx}": lo,
                                f"ilr_delta_ci95_high_{idx}": hi})
                    row.update(summary_stats(x[:, j], f"a_ilr_{idx}"), **summary_stats(y[:, j], f"b_ilr_{idx}"))
            out.append(row)
    result = pd.DataFrame(out)
    corrections(result, {p: s for p, s in P_VALUES.items() if p != "fd_cr1_p"}, ["correction_family"])
    result["family_size"] = 2
    return result


def joint_p(a, b, p, effect):
    if not np.isfinite([a[p], b[p]]).all():
        return np.nan
    return max(a[p], b[p]) if a[effect] * b[effect] > 0 else 1.


def shared_effects(result):
    point = result.set_index(["metric", "contrast"])
    rows = []
    for e in endpoints():
        a, b = [point.loc[(e["metric"], c)] for c in ("OFF-HC", "ON-HC")]
        rows.append(dict(**e, contrast="both-HC", correction_family="Shared PD: " + e["family"],
                         estimated_same_direction=a.effect_size * b.effect_size > 0,
                         off_minus_hc=a.effect_size, on_minus_hc=b.effect_size,
                         raw_p=joint_p(a, b, "raw_p", "effect_size"),
                         permutation_p=joint_p(a, b, "permutation_p", "effect_size"),
                         fd_permutation_p=joint_p(a, b, "fd_permutation_p", "fd_permutation_effect"),
                         fd_cr1_p=joint_p(a, b, "fd_cr1_p", "fd_cr1_effect")))
    table = pd.DataFrame(rows)
    corrections(table, P_VALUES, ["correction_family"])
    table["family_size"] = table.groupby("correction_family").metric.transform("size")
    return table


def exploratory_correction(results, composition, shared):
    table = pd.concat([results.loc[results.family.isin(EXPLORATORY_FAMILIES)],
                       composition.loc[composition.family.eq("Sorted Coverage ILR")], shared], ignore_index=True)
    table["scope"] = "exploratory"
    corrections(table, {p: "global_" + s for p, s in P_VALUES.items()}, ["scope"])
    eligible = table[list(P_VALUES)].copy()
    eligible.loc[table.family.eq("Sorted Coverage ILR"), "raw_p"] = np.nan  # The two fields are the same test.
    mask = np.ones(eligible.shape, bool)
    mask[table.family.eq("Sorted Coverage ILR"), 0] = False
    mask[table.family.eq("Sorted Coverage ILR"), list(P_VALUES).index("fd_cr1_p")] = False
    for name, how in (("fdr", "fdr_bh"), ("holm", "holm")):
        joint = np.full(eligible.shape, np.nan)
        joint[mask] = adjust(eligible.to_numpy()[mask], how)
        for j, (_, label) in enumerate(P_VALUES.items()):
            table[f"global_model_union_{label}{name}_p"] = joint[:, j]
    table["global_feature_slots"] = len(table)
    table["global_model_union_slots"] = int(mask.sum())
    return table


def gate_coverage(primary, composition):
    alpha = SETTINGS["alpha"]
    gate = composition.loc[composition.metric.eq("ilr")].set_index("contrast").permutation_holm_p.lt(alpha)
    primary["coverage_parent_holm_p"] = np.nan
    for contrast in gate.index:
        selected = primary.family.eq("Coverage") & primary.contrast.eq(contrast)
        p = composition.loc[composition.metric.eq("ilr") & composition.contrast.eq(contrast), "permutation_holm_p"].item()
        primary.loc[selected, "coverage_parent_holm_p"] = p
    selected = primary.family.eq("Coverage") & primary.contrast.isin(gate.index)
    primary["passes_coverage_gate"] = ~selected | primary.contrast.map(gate).eq(True)
    primary["interpret_fdr"] = primary.fdr_p.lt(alpha) & primary.passes_coverage_gate
    primary["interpret_holm"] = primary.holm_p.lt(alpha) & primary.passes_coverage_gate
    primary["permutation_support"] = primary.permutation_holm_p.lt(alpha) & primary.passes_coverage_gate
    primary["joint_holm_support"] = primary.interpret_holm & primary.permutation_support
    return primary


def transition_summary(data):
    rows = []
    for i in STATE:
        for j in STATE:
            metric = f"transition_{i}_{j}"
            for name, subset in data.groupby("condition"):
                x = subset[metric].dropna().to_numpy()
                rows.append(dict(metric=metric, source=i, destination=j, group=name, n=len(x),
                                 tested=i != j, **summary_stats(x, "group")))
    return pd.DataFrame(rows)


def link_global(result, exploratory):
    cols = [c for c in exploratory if c.startswith("global_")]
    return result.merge(exploratory[["metric", "contrast", *cols]], on=["metric", "contrast"], how="left", validate="one_to_one")


def save(table, filename):
    table.to_csv(HERE / "results" / f"{filename}.csv", encoding="utf-8-sig", index=False)


def main():
    inputs = load_inputs()
    data = inputs.features
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "private").mkdir(exist_ok=True)
    data.to_csv(HERE / "private/Subject_Level_Microstate_Features.csv", index=False)
    write_templates(inputs, HERE / "results")
    resampling = UniqueResampling(SETTINGS["seed"], SETTINGS["permutations"], SETTINGS["fixed_feature_ci_draws"])
    scalars = scalar_tests(data, resampling)
    comp = composition_tests(data, resampling)
    shared = shared_effects(scalars)
    exploratory = exploratory_correction(scalars, comp, shared)
    scalars = link_global(gate_coverage(scalars, comp), exploratory)
    comp, shared = link_global(comp, exploratory), link_global(shared, exploratory)
    save(scalars, "All_Scalar_Results")
    save(comp, "Coverage_ILR_Results")
    save(shared, "Shared_PD_Direction_Results")
    save(exploratory, "Exploratory_Global_Corrections")
    for label, file in zip(SETTINGS["contrasts"], ("HC_vs_OFF", "HC_vs_ON", "ON_vs_OFF")):
        save(scalars.loc[scalars.contrast.eq(label)], file)
    for family, file in {"Coverage": "Coverage", "Duration": "Duration", "Occurrence": "Occurrence",
                         "Transition Probability": "Transition", "GEV": "GEV", "Spatial Dispersion": "Spatial_Dispersion"}.items():
        save(scalars.loc[scalars.family.eq(family)], f"{file}_Results")
    save(transition_summary(data), "Transition_Matrix_Descriptive")
    ledger = pd.concat([scalars, comp, shared], ignore_index=True)
    common = ["metric", "state", "family", "contrast", "correction_family", "family_size", "status", "n_a", "n_b", "n_pairs",
              "effect_size", "ci95_low", "ci95_high", "raw_p", "passes_coverage_gate", "coverage_parent_holm_p"]
    save(ledger[common + [c for c in ledger if c.endswith("fdr_p")]], "FDR_Results")
    save(ledger[common + [c for c in ledger if c.endswith("holm_p") and c not in common]], "Holm_Results")
    save(ledger[common + [c for c in ledger if "permutation" in c]], "Permutation_Results")
    receipt = dict(cohort=dict(HC=15, ON=25, OFF=25, independent_participants=40, scans=65),
                   all_point_labels_replayed=True, frames_total=int(data.valid_frames.sum()),
                   stable_state_flags=inputs.matches["stable"].tolist(), scalar_rows=len(scalars),
                   composition_rows=len(comp), shared_rows=len(shared), permutation_draws=resampling.permutations)
    (HERE / "results/Analysis_Verification.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
