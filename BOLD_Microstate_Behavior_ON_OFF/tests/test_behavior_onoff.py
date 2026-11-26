from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from behavior_protocol import BEHAVIORS, STATES
from behavior_stats import adjust, association, matched_pairs, paired_test, registry_check


def test_registry_and_test_choice():
    registry_check()
    assert paired_test(np.zeros(25))[:3] == ("all_zero", 0., 1.)
    assert paired_test(norm.ppf(np.linspace(.03, .97, 25))+.5)[0] == "paired_t"
    assert paired_test(np.r_[np.arange(20), 500])[0] == "Wilcoxon_signed_rank"


def test_unavailable_spearman_preserved_in_bh():
    frame = pd.DataFrame({f"duration_{i}_change": np.ones(5) for i, _ in enumerate(STATES, 1)}
                         | {f"{name}_improvement": np.arange(5) for name in BEHAVIORS})
    results = association(frame, "test")
    assert len(results) == 20 and results.raw_p.isna().all()
    assert results.reason.eq("constant_duration").all() and results.fdr_q.isna().all()
    output = adjust(pd.DataFrame({"raw_p": [.01, .02, np.nan, .3]}), "duration")
    np.testing.assert_allclose(output.fdr_q, [.04, .04, np.nan, .4], equal_nan=True)


def test_pairing_uses_all_three_keys_and_never_imputes_zero_rate():
    keys = pd.DataFrame({"patient_id": ["P1", "P2"], "on_scan_id": ["a", "b"], "off_scan_id": ["x", "y"]})
    scores = keys.assign(**{f"{name}_{c}": [2, 0] if c == "off" else [1, 1]
                            for name in BEHAVIORS for c in ("off", "on")})
    scans = keys.assign(**{f"duration_{i}_{c}": [3, 4] if c == "off" else [2, 6]
                           for i, _ in enumerate(STATES, 1) for c in ("off", "on")})
    joined = matched_pairs(scores, scans.iloc[::-1])
    assert joined.total_improvement.tolist() == [1, -1]
    assert joined.duration_1_change.tolist() == [-1, 2]
    np.testing.assert_allclose(joined.total_improvement_rate, [.5, np.nan], equal_nan=True)
    broken = scans.copy()
    broken.loc[1, "off_scan_id"] = "unmatched"
    assert len(matched_pairs(scores, broken)) == 1
