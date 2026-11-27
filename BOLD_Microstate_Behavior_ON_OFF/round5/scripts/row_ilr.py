"""Fixed Jeffreys-smoothed row compositions from the archived transition counts."""

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.special import softmax

from omnibus import adjust, bootstrap_indices, design, omnibus_result, paired_cells
from round5_io import CELLS, CFG, STATES


BASIS = helmert(4, full=False)


def transform(counts):
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("ILR requires archived nonnegative integer counts")
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, totals, where=totals > 0, out=np.full(counts.shape, np.nan))
    smoothed = (counts + CFG["jeffreys_count"]) / (totals + 4 * CFG["jeffreys_count"])
    smoothed = np.where(totals > 0, smoothed, np.nan)
    ilr = np.log(smoothed) @ BASIS.T
    return probabilities, smoothed, ilr


def p2(matrices):
    on, off = paired_cells(matrices, CELLS, "count")
    n = len(on)
    x, y = [frame.to_numpy().reshape(n, 4, 4) for frame in (on, off)]
    p_on, s_on, z_on = transform(x)
    p_off, s_off, z_off = transform(y)
    raw = np.nanmean(p_on-p_off, axis=0)
    added = np.nanmean(s_on-s_off, axis=0)
    center_on = softmax(np.nanmean(z_on, axis=0) @ BASIS, axis=-1)
    center_off = softmax(np.nanmean(z_off, axis=0) @ BASIS, axis=-1)
    center = center_on - center_off
    ix = bootstrap_indices(n, CFG["seed"], CFG["bootstrap_draws"], 206)
    raw_ci = np.quantile((p_on-p_off)[ix].mean(axis=1), (.025, .975), axis=0)
    smoothed_ci = np.quantile((s_on-s_off)[ix].mean(axis=1), (.025, .975), axis=0)
    centers = [softmax(values[ix].mean(axis=1) @ BASIS, axis=-1) for values in (z_on, z_off)]
    center_ci = np.quantile(centers[0]-centers[1], (.025, .975), axis=0)
    rows, summaries = [], []
    delta = z_on - z_off
    cases = [("all_rows", delta.reshape(n, 12))] + [
        (state, delta[:, index]) for index, state in enumerate(STATES)
    ]
    for label, profile in cases:
        good = np.isfinite(profile).all(axis=1)
        if not good.all():
            summaries.append(dict(scope=label, status="undefined_row_among_frozen_pairs",
                                  raw_p=np.nan, n_pairs=int(good.sum()), dimensions=profile.shape[1]))
            continue
        observed, randomized, bootstrap = design(profile, 206)
        estimate = omnibus_result(observed, randomized, bootstrap, CFG["ridge"], n)
        summaries.append(dict(scope=label, status="ok", **estimate))
        if label == "all_rows":
            continue
        low, high = np.quantile(bootstrap[0], (.025, .975), axis=0)
        idx = STATES.index(label)
        for i in range(3):
            rows.append({
                "from_state": label, "coordinate": f"ILR{i+1}", "n_pairs": n, "mean_on": z_on[:, idx, i].mean(),
                "mean_off": z_off[:, idx, i].mean(), "mean_difference": observed[0][i],
                "ci95_low": low[i], "ci95_high": high[i], "estimand": "coordinate description",
            })
    tests = adjust(pd.DataFrame(summaries))
    checks = []
    for i, source in enumerate(STATES):
        for j, target in enumerate(STATES):
            checks.append({
                "from_state": source, "to_state": target, "n_pairs": int(np.isfinite(delta[:, i]).all(axis=1).sum()),
                "jeffreys_per_cell": CFG["jeffreys_count"], "zero_counts_ON": int((x[:, i, j] == 0).sum()),
                "zero_counts_OFF": int((y[:, i, j] == 0).sum()),
                "undefined_rows_ON": int((x[:, i].sum(axis=1) == 0).sum()),
                "undefined_rows_OFF": int((y[:, i].sum(axis=1) == 0).sum()),
                "original_mean_ON": np.nanmean(p_on[:, i, j]), "original_mean_OFF": np.nanmean(p_off[:, i, j]),
                "original_difference": raw[i, j], "original_ci95_low": raw_ci[0, i, j],
                "original_ci95_high": raw_ci[1, i, j],
                "smoothed_mean_ON": np.nanmean(s_on[:, i, j]), "smoothed_mean_OFF": np.nanmean(s_off[:, i, j]),
                "smoothed_difference": added[i, j], "aitchison_center_ON": center_on[i, j],
                "smoothed_ci95_low": smoothed_ci[0, i, j], "smoothed_ci95_high": smoothed_ci[1, i, j],
                "aitchison_center_OFF": center_off[i, j], "aitchison_center_difference": center[i, j],
                "aitchison_ci95_low": center_ci[0, i, j], "aitchison_ci95_high": center_ci[1, i, j],
                "smoothed_direction_agrees": np.sign(raw[i, j]) == np.sign(added[i, j]),
                "aitchison_direction_agrees": np.sign(raw[i, j]) == np.sign(center[i, j]),
            })
    return {
        "Transition_ILR_Results": pd.DataFrame(rows),
        "Transition_ILR_Omnibus": tests,
        "Transition_ILR_Sensitivity": pd.DataFrame(checks),
    }
