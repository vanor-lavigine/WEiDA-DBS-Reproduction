"""Frozen head-motion and order checks for paired WEiDA transition probabilities."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control


STATES = ("M1", "M2", "M3", "M4")
EDGES = [(a, b) for a in STATES for b in STATES if a != b]
PATHS = {"P41": ("M4", "M1"), "P34": ("M3", "M4"), "P44": ("M4", "M4")}
PAIR_KEYS = ["participant_id", "on_scan_id", "off_scan_id"]
SHRINKAGE = 0.10
MASK_MODE = "fd_censor_frozen_labels"


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _read(path, workspace, receipts, expected=None):
    path = Path(path).resolve()
    _require(path.is_relative_to(workspace), "Motion input is outside the authorized workspace")
    if not path.is_file():
        raise FileNotFoundError(f"Missing motion input: {path.name}")
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None:
        _require(digest == expected, f"Motion frozen digest differs: {path.name}")
    receipts[str(path)] = digest
    return raw


def _table(path, workspace, receipts, expected=None, **kwargs):
    return pd.read_csv(BytesIO(_read(path, workspace, receipts, expected)),
                       float_precision="round_trip", **kwargs)


def _locked(lock, name, workspace, receipts):
    found = [(Path(path), digest) for path, digest in lock.items() if Path(path).name == name]
    _require(bool(found) and len({digest for _, digest in found}) == 1,
             f"Expected one frozen motion source digest for {name}")
    path, expected = next(((path, digest) for path, digest in found if path.is_file()), found[0])
    return _read(path, workspace, receipts, expected)


def _match_pairs(frame, pairs, allow_subset=False):
    _require(frame.participant_id.is_unique, "Source contains duplicate motion or date pair keys")
    joined = pairs[PAIR_KEYS].merge(frame, on=PAIR_KEYS,
                                   validate="one_to_one", how="inner" if allow_subset else "left",
                                   indicator=True)
    expected = len(frame) if allow_subset else len(pairs)
    _require(len(joined) == expected == len(frame) and joined._merge.eq("both").all(),
             "Motion or date source differs from the frozen patient/scan pair keys")
    return joined.drop(columns="_merge").set_index("participant_id")


def _paired_changes(matrix, pairs):
    pairs = pairs.sort_values("participant_id", ignore_index=True)
    patient = matrix.loc[matrix.condition.isin(("ON", "OFF"))]
    _require(len(patient) == 50 * 16, "Expected complete 4-by-4 original patient matrices")
    conditions = {}
    columns = [*EDGES, PATHS["P44"]]
    for condition in ("ON", "OFF"):
        selected = patient.loc[patient.condition.eq(condition)]
        keys = selected[["participant_id", "scan_id", "fold"]].drop_duplicates().set_index("participant_id")
        expected = pairs.set_index("participant_id").loc[keys.index]
        _require(keys.scan_id.eq(expected[f"{condition.lower()}_scan_id"]).all() and
                 keys.fold.eq(expected.fold).all(), "Patient matrix scan or five-fold identity mismatch")
        wide = selected.pivot(index="participant_id", columns=["from_state", "to_state"], values="p_weida")
        conditions[condition] = wide.reindex(pairs.participant_id).loc[:, columns]
        _require(np.isfinite(conditions[condition]).all().all(), "Missing primary transition values")
    return conditions["ON"] - conditions["OFF"]


def _metadata(pairs, module, workspace, receipts, lock):
    frozen = pd.read_csv(BytesIO(_locked(lock, "ON_OFF_Pair_Manifest.csv", workspace, receipts)),
                         float_precision="round_trip")
    frozen = _match_pairs(frozen, pairs)
    cols = {f"{c}_scan_id": f"{c.lower()}_scan_id" for c in ("ON", "OFF")}
    cols["patient_id"] = "participant_id"
    directory = module / "private/round4/audit"
    order = _match_pairs(_table(directory / "Scan_Order_Audit.csv", workspace, receipts).rename(columns=cols), pairs)
    visit = _match_pairs(_table(directory / "Clinical_Scan_Visit_Matching.csv", workspace, receipts).rename(columns=cols),
                         pairs)
    legacy_path = module / "private/results/Date_Verified_Pairs.csv"
    legacy = _match_pairs(_table(legacy_path, workspace, receipts, usecols=[
        "patient_id", "on_scan_id", "off_scan_id", "date_verified",
    ]).rename(columns=cols), pairs, allow_subset=True)
    main = _match_pairs(_table(module / "private/results/Main_Pairs.csv", workspace, receipts, usecols=[
        "patient_id", "on_scan_id", "off_scan_id", "date_verified",
    ]).rename(columns=cols), pairs)
    _require(legacy.date_verified.eq(True).all() and set(legacy.index) == set(main.index[main.date_verified.eq(True)]),
             "Legacy confirmed-date subset disagrees with its original archived inclusion flags")
    off_time = pd.to_datetime(order.OFF_scan_datetime, format="ISO8601")
    on_time = pd.to_datetime(order.ON_scan_datetime, format="ISO8601")
    raw_delta = (off_time - on_time).dt.total_seconds() / 3600
    np.testing.assert_allclose(raw_delta, order.OFF_minus_ON_hours, atol=1e-10, rtol=0, equal_nan=True)
    flags = np.select([raw_delta.gt(0), raw_delta.lt(0), raw_delta.eq(0)],
                      ["ON first", "OFF first", "same timestamp"], default="unverified")
    _require(np.array_equal(flags, order.scan_order) and
             np.array_equal(raw_delta.gt(0), frozen.scan_order.eq("ON before OFF")) and
             np.array_equal(raw_delta.lt(0), frozen.scan_order.eq("OFF before ON")),
             "V6 and Round4 clock/order evidence disagree")
    metadata = pd.DataFrame(index=frozen.index)
    metadata["FD"] = frozen.on_mean_fd - frozen.off_mean_fd
    metadata["std_DVARS"] = frozen.on_std_dvars - frozen.off_std_dvars
    metadata["raw_DVARS"] = frozen.on_dvars - frozen.off_dvars
    metadata["order"] = order.scan_order
    metadata["interval_hours"] = raw_delta
    metadata["legacy_date_verified"] = metadata.index.isin(legacy.index)
    same_day = []
    for side in ("ON", "OFF"):
        verified = visit[f"{side}_visit_match_status"].eq("Confirmed same day")
        calendar_day = visit[f"{side}_scan_date"].eq(visit[f"{side}_score_date"])
        _require((~verified | calendar_day).all(), "Round4 same-day flags disagree with source dates")
        same_day.append(verified)
    metadata["round4_both_same_day"] = same_day[0] & same_day[1]
    metadata["round4_both_same_visit"] = (
        visit.ON_visit_match_status.eq("Confirmed same visit") & visit.OFF_visit_match_status.eq("Confirmed same visit")
    )
    _require(metadata.FD.notna().all() and metadata.std_DVARS.notna().all() and
             (metadata.legacy_date_verified.sum(), metadata.round4_both_same_day.sum(),
              metadata.round4_both_same_visit.sum()) == (22, 21, 0) and
             metadata.order.value_counts().to_dict() == {"ON first": 24, "unverified": 1},
             "Frozen motion, timing, or assessment-date groups differ")
    return metadata, frozen


def _retained(fd, labels):
    """V10 mask-only window. Removed labels remain gaps on the original clock."""
    result = labels.copy()
    for frame in np.flatnonzero(fd > .50):
        result[max(0, frame - 1):min(len(labels), frame + 3)] = -1
    return result


def _counts(labels):
    kept = labels >= 0
    paired = kept[:-1] & kept[1:]
    n_blocks = int(np.count_nonzero(kept & ~np.r_[False, kept[:-1]]))
    counts = np.bincount(4 * labels[:-1][paired] + labels[1:][paired], minlength=16).reshape(4, 4)
    _require(counts.sum() == kept.sum() - n_blocks, "A quality-window adjacency crossed a deleted TR")
    return counts, int(kept.sum()), n_blocks


def _quality(pairs, matrix, frozen, module, workspace, receipts, r4_lock):
    old = module / "private/imaging_v10/manifests/Input_Hashes.json"
    v10_lock = json.loads(_read(old, workspace, receipts))["file_sha256"]
    frozen_fd = json.loads(_locked(v10_lock, "frozen_dvars.json", workspace, receipts))
    old_metrics = pd.read_csv(BytesIO(_locked(v10_lock, "V4_Scan_Metrics.csv", workspace, receipts)),
                              float_precision="round_trip")
    old_metrics = old_metrics.loc[old_metrics.method.eq("A3") & old_metrics.window.eq("fd_censor")]
    old_metrics = old_metrics.set_index("scan", verify_integrity=True)
    maps = pd.read_csv(BytesIO(_locked(r4_lock, "Fold_Templates.csv", workspace, receipts)))
    mapping = {fold: part.sort_values("original_state").state.to_numpy(int) - 1
               for fold, part in maps.groupby("fold")}
    _require(set(mapping) == set(range(5)) and all(sorted(v) == list(range(4)) for v in mapping.values()),
             "Immutable A3 state mapping is invalid")
    v10 = module.parent / "BOLD_Microstate_M1_Robustness"
    rule = json.loads(_read(v10 / "configs/protocol.json", workspace, receipts))["fd"]
    _require((rule["strictly_greater_than"], rule["preceding_frames"], rule["following_frames"]) == (.50, 1, 2),
             "Frozen V10 quality rule changed")
    exposures = _table(v10 / "results/Scan_Window_Summary.csv", workspace, receipts)
    exposures = exposures.loc[exposures["mode"].eq(MASK_MODE)].set_index("condition")
    archive_files = {Path(path).stem: (path, expected) for path, expected in r4_lock.items()
                     if Path(path).parent.name == "A3" and Path(path).suffix == ".npz"}
    original = matrix.set_index(["scan_id", "from_state", "to_state"])
    quality_rows, quality_meta = [], []
    for pair in pairs.itertuples(index=False):
        for condition in ("ON", "OFF"):
            side = condition.lower()
            scan = getattr(pair, side + "_scan_id")
            path, expected = archive_files[scan]
            with np.load(BytesIO(_read(path, workspace, receipts, expected)), allow_pickle=False) as archive:
                raw_labels = archive["labels"]
            fd = np.asarray(frozen_fd[scan]["framewise_displacement"], float)
            dvars = np.asarray(frozen_fd[scan]["std_dvars"], float)
            _require(np.isin(raw_labels, [-1, 0, 1, 2, 3]).all() and fd.shape == dvars.shape == raw_labels.shape == (210,)
                     and np.isnan(fd[0]) and np.isfinite(fd[1:]).all() and (fd[1:] >= 0).all(),
                     "Locked V10 quality clock is invalid")
            original_row = frozen.loc[pair.participant_id]
            np.testing.assert_allclose([np.nanmean(fd), np.nanmean(dvars)],
                                       [original_row[f"{side}_mean_fd"], original_row[f"{side}_std_dvars"]],
                                       atol=1e-12, rtol=0)
            canonical = np.where(raw_labels >= 0, mapping[pair.fold][raw_labels], -1)
            pre, n_original, _ = _counts(canonical)
            wanted = original.loc[scan]["count"].reindex(pd.MultiIndex.from_product([STATES, STATES])).to_numpy()
            np.testing.assert_array_equal(pre.reshape(-1), wanted)
            count, n_retained, n_blocks = _counts(_retained(fd, canonical))
            old_scan = old_metrics.loc[scan]
            _require(old_scan.fold == pair.fold and old_scan.subject == pair.participant_id and
                     old_scan.condition == condition and old_scan.n_labeled_frames == n_retained,
                     "Fixed quality retained frames differ from the original V4/V10 archive")
            denom = count.sum(axis=1)
            probs = np.divide(count, denom[:, None], where=denom[:, None] > 0, out=np.full((4, 4), np.nan))
            quality_meta.append({"participant_id": pair.participant_id, "scan_id": scan, "condition": condition,
                                 "original_frames": n_original, "retained_frames": n_retained,
                                 "valid_blocks": n_blocks, "adjacency_pairs": int(count.sum())})
            quality_rows += [
                {"participant_id": pair.participant_id, "scan_id": scan, "condition": condition, "fold": pair.fold,
                 "from_state": source, "to_state": target, "count": int(count[i, j]), "p_weida": probs[i, j]}
                for i, source in enumerate(STATES) for j, target in enumerate(STATES)
            ]
    counts = pd.DataFrame(quality_meta)
    totals = counts.groupby("condition")[["original_frames", "retained_frames", "valid_blocks"]].sum()
    np.testing.assert_array_equal(totals, exposures.loc[totals.index, [
        "original_frames_total", "retained_frames_total", "contiguous_blocks_total",
    ]])
    return pd.DataFrame(quality_rows), counts


def _cosine(left, right):
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / scale) if scale > 0 else np.nan


def _hc3_wald(design, responses, coefficient):
    """V6 HC3 covariance, with fixed isotropic trace shrinkage for vectors."""
    inverse = np.linalg.pinv(design)
    leverage = np.einsum("ij,ji->i", design, inverse)
    beta = inverse @ responses
    residual = responses - design @ beta
    weights = inverse[coefficient] / (1 - leverage)
    weighted = residual * weights[:, None]
    cov = np.swapaxes(weighted, -1, -2) @ weighted
    dim = cov.shape[-1]
    trace = np.trace(cov, axis1=-2, axis2=-1) / dim
    cov = (1 - SHRINKAGE) * cov + SHRINKAGE * trace[..., None, None] * np.eye(dim)
    selected = beta[..., coefficient, :]
    standardized = np.linalg.solve(cov, selected[..., None])[..., 0]
    return np.einsum("...i,...i->...", selected, standardized)


def _build_design(covariate):
    sd = np.std(covariate, ddof=1)
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        raise ValueError("Motion adjustment is unestimable for a constant covariate")
    return np.column_stack([np.ones(len(covariate)), covariate / sd]), sd


def _boot_coefficients(design, response, indices):
    view = design[indices]
    outputs = np.full((len(indices), 2, response.shape[1]), np.nan)
    estimable = np.linalg.matrix_rank(view) == 2
    outputs[estimable] = np.linalg.pinv(view[estimable]) @ response[indices[estimable]]
    return outputs


def _ci(values, minimum_fraction=.95):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if len(finite) < max(1, minimum_fraction * len(values)):
        return np.nan, np.nan, len(finite)
    return *np.quantile(finite, [.025, .975]), len(finite)


def _wild_patterns(n, draws, seed):
    """Uniform random nonidentity sign masks without replacement."""
    masks = np.random.default_rng(seed).choice((1 << n) - 1, size=draws, replace=False) + 1
    return 1. - 2. * ((masks[:, None] >> np.arange(n)) & 1)


def _wald_test(design, response, kind, patterns):
    coefficient = 0 if kind == "adjusted_intercept" else 1
    observed = float(_hc3_wald(design, response, coefficient))
    # Exclude the coefficient of interest from the fitted null model.
    null_design = design[:, [1 - coefficient]]
    fitted = null_design @ (np.linalg.pinv(null_design) @ response)
    residual = response - fitted
    null = np.concatenate([_hc3_wald(design, fitted + signs[..., None] * residual, coefficient)
                           for signs in np.array_split(patterns, 20)])
    if not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("A motion sensitivity null statistic was not finite")
    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (1 + len(null))
    return observed, float(p), len(null)


def _models(delta, metadata, seed, permutations, bootstraps):
    covariates = {"FD": metadata.FD.to_numpy(float), "std_DVARS": metadata.std_DVARS.to_numpy(float)}
    n = len(delta)
    indices = np.random.default_rng([seed, 521]).integers(n, size=(bootstraps, n))
    patterns = _wild_patterns(n, permutations, [seed, 522])
    results, components = [], []
    for covariate, x in covariates.items():
        design, sd = _build_design(x)
        condition = float(np.linalg.cond(design))
        xlow, xhigh = float(np.min(x)), float(np.max(x))
        models = [("omnibus_12", EDGES, "adjusted_intercept"), ("omnibus_12", EDGES, "covariate_association"),
                  ("P41", [PATHS["P41"]], "adjusted_intercept"), ("P44", [PATHS["P44"]], "adjusted_intercept"),
                  ("P34", [PATHS["P34"]], "adjusted_intercept")]
        for endpoint, edges, kind in models:
            # The P34 intercept is a directional diagnostic only.
            inferential = endpoint != "P34"
            y = delta.loc[:, edges].to_numpy(float)
            coefficient = 0 if kind == "adjusted_intercept" else 1
            beta = (np.linalg.pinv(design) @ y)[coefficient]
            boot = _boot_coefficients(design, y, indices)[:, coefficient, :]
            vector = y.shape[1] > 1
            effect = float(np.linalg.norm(beta)) if vector else float(beta[0])
            boot_effect = np.linalg.norm(boot, axis=1) if vector else boot[:, 0]
            low, high, finite = _ci(boot_effect)
            baseline = y.mean(axis=0)
            observed, p, draws = np.nan, np.nan, 0
            status = "unstable design; motion terms not separable" if condition > 30 else "ok"
            if inferential and status == "ok":
                try:
                    observed, p, draws = _wald_test(design, y, kind, patterns)
                except np.linalg.LinAlgError:
                    status = "singular motion statistic"
            results.append(dict(
                analysis=kind, covariate=covariate, endpoint=endpoint, dimensions=y.shape[1],
                n=len(y), effect=effect, ci95_low=low, ci95_high=high, bootstrap_valid=finite,
                units=("probability_vector_norm" if vector else "probability_difference")
                + ("_per_covariate_SD" if coefficient else ""),
                statistic=observed, raw_p=p, permutation_draws=draws, correction_slot=inferential,
                unadjusted_effect=float(np.linalg.norm(baseline)) if vector else float(baseline[0]),
                cosine_to_unadjusted=_cosine(beta, baseline) if not coefficient else np.nan,
                same_component_direction=int(np.count_nonzero(np.sign(beta) == np.sign(baseline)))
                if not coefficient else np.nan,
                covariance_shrinkage=SHRINKAGE, covariate_mean=float(np.mean(x)), covariate_sd=sd,
                covariate_min=xlow, covariate_max=xhigh, zero_supported=xlow <= 0 <= xhigh,
                design_condition_number=condition, severe_collinearity_flag=condition > 30,
                status=status, note=("whole-pair residual sign flips under slope-only null" if not coefficient else
                                    "whole-pair residual sign flips under intercept-only null")
                if inferential else "descriptive no additional edge test",
            ))
            if vector:
                for j, (source, target) in enumerate(edges):
                    lo, hi, _ = _ci(boot[:, j])
                    components.append(dict(analysis=kind + "_coordinate", covariate=covariate,
                                           endpoint=f"P{source[-1]}{target[-1]}", dimensions=1, n=len(y),
                                           effect=beta[j], ci95_low=lo, ci95_high=hi,
                                           unadjusted_effect=baseline[j],
                                           units="probability_difference" + ("_per_covariate_SD" if coefficient else ""),
                                           correction_slot=False, status="ok", note="coordinate of vector model"))
    table = pd.DataFrame(results)
    eligible = table.correction_slot
    _require(eligible.sum() == 8, "P3 adjustment/association family must retain eight slots")
    p = table.loc[eligible, "raw_p"].fillna(1).to_numpy()
    q = false_discovery_control(p)
    rank = np.argsort(p)
    holm = np.empty(len(p))
    holm[rank] = np.minimum(1., np.maximum.accumulate(p[rank] * np.arange(len(p), 0, -1)))
    known = table.loc[eligible, "raw_p"].notna().to_numpy()
    table.loc[eligible, "bh_q"] = np.where(known, q, np.nan)
    table.loc[eligible, "holm_p"] = np.where(known, holm, np.nan)
    table.loc[eligible, "family_size"] = 8
    return pd.concat([table, pd.DataFrame(components)], ignore_index=True)


def _descriptives(delta, original, mask, scenario, seed, bootstraps):
    """Directional paired sensitivity with no new pathway tests."""
    subset = delta.loc[mask]
    baseline = original.loc[mask]
    rows = []
    for endpoint, columns in (("omnibus_12", EDGES), *[(name, [edge]) for name, edge in PATHS.items()]):
        group = subset.loc[:, columns].to_numpy(float)
        base = baseline.loc[:, columns].to_numpy(float)
        take = np.isfinite(group).all(axis=1)
        group, base = group[take], base[take]
        n = len(group)
        if n == 0:
            rows.append(dict(analysis=scenario, endpoint=endpoint, n=0, status="no defined patient pairs"))
            continue
        vector = group.shape[1] > 1
        rng = np.random.default_rng([seed, 526])
        boot = rng.integers(n, size=(bootstraps, n))
        estimate = group.mean(axis=0)
        old = base.mean(axis=0)
        means = group[boot].mean(axis=1)
        draws = np.linalg.norm(means, axis=1) if vector else means[:, 0]
        low, high, finite = _ci(draws)
        rows.append(dict(analysis=scenario, endpoint=endpoint, n=n, n_expected=len(subset), dimensions=group.shape[1],
                         effect=np.linalg.norm(estimate) if vector else estimate[0], ci95_low=low, ci95_high=high,
                         bootstrap_valid=finite, units="probability_vector_norm" if vector else "probability_difference",
                         unadjusted_effect=np.linalg.norm(old) if vector else old[0],
                         cosine_to_unadjusted=_cosine(estimate, old), correction_slot=False,
                         status="ok", note="descriptive whole-pair bootstrap, no new pathway test"))
    return rows


def motion_sensitivity(matrix, pairs, module, workspace, config):
    """Read locked ancillary inputs and return P3 public tables plus private receipts."""
    workspace, module = Path(workspace).resolve(), Path(module).resolve()
    seed, permutations, bootstraps = (config[k] for k in ("seed", "permutations", "bootstrap_draws"))
    receipts = {}
    lock_path = module / "private/round4/transition/Transition_Input_Hashes.json"
    lock = json.loads(_read(lock_path, workspace, receipts))["file_sha256"]
    pairs = pairs.sort_values("participant_id", ignore_index=True)
    original = _paired_changes(matrix, pairs)
    metadata, frozen = _metadata(pairs, module, workspace, receipts, lock)
    _require(metadata.index.equals(original.index), "Transition and motion patient order differs")
    models = _models(original, metadata, seed, permutations, bootstraps)
    private_tables = {"Transition_Motion_Order_Pairs": metadata.reset_index()}
    try:
        quality, counts = _quality(pairs, matrix, frozen, module, workspace, receipts, lock)
    except FileNotFoundError:
        quality_rows = [{"analysis": MASK_MODE, "endpoint": endpoint, "n": 0, "status": "locked quality source missing"}
                        for endpoint in ("omnibus_12", *PATHS)]
        quality_audit = {"status": "locked quality source missing"}
    else:
        # Keep undefined quality-window rows rather than treating them as zeros.
        quality_delta = _quality_changes(quality, pairs)
        quality_rows = _descriptives(quality_delta, original, metadata.index, MASK_MODE, seed, bootstraps)
        private_tables["Transition_Quality_Exposure"] = counts
        private_tables["Transition_Quality_Frozen_Labels_Matrix"] = quality
        terms = ["original_frames", "retained_frames", "valid_blocks", "adjacency_pairs"]
        quality_audit = {"status": "matched original V4/V10 exposure archives",
                         "mode": MASK_MODE, "scans": len(counts), "counts": counts.groupby("condition")[terms].sum().to_dict("index")}
    motion = pd.concat([models, pd.DataFrame(quality_rows)], ignore_index=True)

    all_pairs = metadata.index
    verified = metadata.index[metadata.order.ne("unverified")]
    subsets = [("all_frozen_pairs", all_pairs), ("verified_scan_order", verified),
               ("legacy_date_verified", metadata.index[metadata.legacy_date_verified]),
               ("round4_both_conditions_same_day", metadata.index[metadata.round4_both_same_day])]
    rows = []
    for scenario, group in subsets:
        joined = _descriptives(original, original, group, scenario, seed, bootstraps)
        for row in joined:
            row["on_first"] = int(metadata.loc[group, "order"].eq("ON first").sum())
            row["order_unverified"] = int(metadata.loc[group, "order"].eq("unverified").sum())
            row["off_first"] = int(metadata.loc[group, "order"].eq("OFF first").sum())
        rows.extend(joined)
    times = metadata.interval_hours.dropna()
    rows.extend([
        dict(analysis="scan_interval", endpoint="OFF_minus_ON_interval_hours", n=len(times), status="ok",
             units="hours", effect=float(times.median()), interval_min=float(times.min()),
             interval_max=float(times.max()), note="median and observed range, temporal order consistent"),
        dict(analysis="order_causal_effect", endpoint="omnibus_12", n=0, status="not_identifiable",
             note="all 24 verified orders ON first; one unknown; no opposite-order contrast"),
    ])
    order = pd.DataFrame(rows)
    audit = {
        "n_pairs": len(pairs), "on_first": len(verified), "off_first": 0,
        "order_unknown": int(metadata.order.eq("unverified").sum()), "order_effect_identifiable": False,
        "legacy_date_verified": int(metadata.legacy_date_verified.sum()),
        "round4_both_same_day": int(metadata.round4_both_same_day.sum()), "round4_both_same_visit": 0,
        "head_motion_available_pairs": int(metadata[["FD", "std_DVARS"]].notna().all(axis=1).sum()),
        "std_dvars_source": "mean standardized DVARS (ON minus OFF), fixed V6 variable",
        "quality": quality_audit, "motion_family_slots": 8,
        "exposure_order_source": "Round4 original-clock audit cross-checked against V6 paired acquisition times",
    }
    return motion, order, private_tables, receipts, audit


def _quality_changes(matrix, pairs):
    """Return all pairs and keep undefined estimates for each quality-window edge."""
    keys = [*EDGES, PATHS["P44"]]
    group = {
        condition: part.pivot(index="participant_id", columns=["from_state", "to_state"], values="p_weida")
        .reindex(index=pairs.participant_id, columns=pd.MultiIndex.from_tuples(keys))
        for condition, part in matrix.groupby("condition")
    }
    return group["ON"] - group["OFF"]
