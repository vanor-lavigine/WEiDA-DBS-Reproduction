"""Synthetic regression checks for visit pairing, relative improvement and absent states."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import sequence_metrics
from li_clinical_stats import analyze


@pytest.fixture(scope="module", params=[24, 25])
def fixture(request):
    rng = np.random.default_rng(187)
    n = request.param
    visits, measures, off_scores, on_scores, on_durations = [], [], [], [], []
    for person in range(n):
        for condition in ("OFF", "ON"):
            rid = f"synthetic-{person}-{condition}"
            score = 30 + person if condition=="OFF" else 12 + (person*7) % 21
            visits.append(dict(person=person, run_id=rid, condition=condition, score=score))
            labels = rng.integers(0, 3 if (person,condition)==(0,"OFF") else 4, size=210)
            measured = sequence_metrics(labels, 2.1, 4)
            if condition=="ON":
                on_scores.append(score)
                on_durations.append(441 / (1 + np.count_nonzero(np.diff(labels))))
            else:
                off_scores.append(score)
            for variant in ("direct", "peak"):
                for state in measured["states"]:
                    measures.append(dict(variant=variant, run_id=rid, n_valid_frames=210,
                                         mean_fd=.05+.005*(person*3 % 19), **state))
    result = analyze(pd.DataFrame(measures).sample(frac=1, random_state=42),
                     pd.DataFrame(visits).sample(frac=1, random_state=24), expected_patients=n)
    return pd.DataFrame(result["tests"]), np.array(off_scores), np.array(on_scores), on_durations


def test_improvement_is_paired_to_on_level_and_normalised_per_person(fixture):
    rows, off, on, durations = fixture
    found = rows.query("variant=='peak' and contrast=='on_improvement_pct' and state=='all' and metric=='duration_sec'").iloc[0]
    assert found["n"] == len(off)
    assert found.r == pytest.approx(pearsonr(durations, (off-on)/off).statistic, abs=1e-12)


def test_absence_and_multiplicity_apply_to_the_correct_endpoints(fixture):
    rows, off, *_ = fixture
    values = rows.query("contrast=='off_severity' and state=='A4' and variant=='peak'").set_index("metric")
    assert values.loc["duration_sec","n"] == len(off)-1 and values.loc["coverage","n"] == len(off)
    for _, family in rows.groupby("variant"):
        ordered = family.sort_values("p_raw")
        assert len(ordered)==42
        scaled = 42 * ordered.p_raw.to_numpy() / np.arange(1, 43)
        expected = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
        np.testing.assert_allclose(ordered.q_bh42, expected, atol=1e-12)
