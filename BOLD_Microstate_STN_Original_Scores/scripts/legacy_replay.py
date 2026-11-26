"""Re-run the original four-feature association on matched scores and samples."""

import numpy as np
import pandas as pd

from pipeline_io import ARCHIVE, REPO, RESULTS, TOL, prior_association_module, require, write_csv

FEATURES = ["rho", "ci95_low", "ci95_high", "permutation_p", "permutation_holm_p", "fd_rho", "fd_p",
            "fd_ci95_low", "fd_ci95_high", "fd_permutation_holm_p"]
COMPARABLE = ["n_pairs", *FEATURES]


def replay_legacy(workspace, imaging, historical, analysis):
    code, endpoints = prior_association_module(workspace)
    old = historical.set_index("participant_id").legacy_motor_improvement.sort_index()
    current = analysis.set_index("participant_id").motor_total_improvement.sort_index()
    prior_in_eligible = old.loc[current.index]
    groups = [
        ("legacy_full_original_scores", old),
        ("v8_eligible_original_legacy_scores", prior_in_eligible),
        ("v8_eligible_recomputed_scores", current),
    ]
    tables = []
    for label, scores in groups:
        data = imaging.old_differences.loc[scores.index]
        rows = []
        for feature in endpoints:
            key = feature["field"]
            result = code.association(data[key].to_numpy(float), scores.to_numpy(float), data.fd.to_numpy(float))
            rows.append(dict(endpoint="delta_" + key, computation_unit=feature["unit"], **result))
        tables.append(code.correct(pd.DataFrame(rows)).assign(comparison=label))
    previous = REPO / "BOLD_Microstate_Behavior_Association/results/Archived_Score_Associations.csv"
    old_log = pd.read_csv(previous)
    old_log = old_log.loc[old_log.score_source.eq(ARCHIVE)].set_index("endpoint")
    original = tables[0].set_index("endpoint")
    np.testing.assert_allclose(original[COMPARABLE], old_log.loc[original.index, COMPARABLE],
                               rtol=0, atol=TOL, equal_nan=True)
    if np.array_equal(current.to_numpy(), prior_in_eligible.to_numpy(), equal_nan=True):
        pd.testing.assert_frame_equal(tables[1].drop(columns="comparison"), tables[2].drop(columns="comparison"),
                                      check_exact=True)
    output = pd.concat(tables, ignore_index=True)
    write_csv(output, RESULTS / "Legacy_Score_Replacement_Associations.csv")
    return output
