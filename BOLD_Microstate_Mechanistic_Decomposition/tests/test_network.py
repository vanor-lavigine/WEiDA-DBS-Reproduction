from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from network import CATEGORIES, comparison, network_share


def test_fractional_mapping_preserves_unmapped_weight_and_polarity():
    fractions = np.zeros((2, len(CATEGORIES)))
    fractions[0, [0, 1, 7]] = (.2, .3, .5)
    fractions[1, [0, 6, 7]] = (.25, .5, .25)
    absolute, squared = network_share(np.array([-1., 2.]), fractions)
    np.testing.assert_allclose(np.array([absolute.sum(), squared.sum()]), [1, 1])
    np.testing.assert_allclose(absolute[[0, 1, 6, 7, 8]], np.array([.7, .3, 1, 1, 0]) / 3)
    np.testing.assert_allclose(squared[[0, 1, 6, 7, 8]], np.array([1.2, .3, 2, 1.5, 0]) / 5)
    np.testing.assert_allclose(network_share(np.array([1., -2.]), fractions), [absolute, squared])


def test_hungarian_relabel_is_detected_without_overwriting_frozen_names():
    roi = [f"r{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    vectors = np.linalg.qr(rng.normal(size=(10, 4)))[0].T
    relabeled = vectors[[1, 0, 2, 3]] * np.array([-1, 1, 1, -1])[:, None]
    rows = [
        dict(fold=fold, state=i + 1, original_state=i + 1, stable=True, **dict(zip(roi, weights)))
        for fold, matrix in enumerate([vectors, relabeled]) for i, weights in enumerate(matrix)
    ]
    fractions = rng.dirichlet(np.ones(len(CATEGORIES)), size=10)
    gate = dict(min_abs_cosine=.6, min_row_and_column_gap=.05)
    result = comparison(pd.DataFrame(rows), roi, fractions, 0, 1, gate, "reference")
    assert [r["state"] for r in result] == [1, 2, 3, 4]
    assert [r["hungarian_state_j"] for r in result] == [2, 1, 3, 4]
    assert [r["hungarian_agrees_with_frozen"] for r in result] == [False, False, True, True]
    assert all(r["hungarian_original_gate_pass"] for r in result)
