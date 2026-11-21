"""Tests of scientific conventions and hand-computable state sequences."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


_CORE_PATH = Path(__file__).resolve().parents[1] / "core.py"
_SPEC = importlib.util.spec_from_file_location("bold_microstates_core", str(_CORE_PATH))
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)


class PrepareBoldTests(unittest.TestCase):
    def test_temporal_standardization_zero_frame_and_no_spatial_demeaning(self):
        data = np.array([[1.0, 0.0], [2.0, 2.0], [3.0, 4.0]])
        original = data.copy()
        result = core.prepare_bold(data)
        np.testing.assert_array_equal(data, original)
        np.testing.assert_allclose(result["temporal_mean"], [2.0, 2.0])
        np.testing.assert_allclose(result["temporal_std"], [1.0, 2.0])
        np.testing.assert_allclose(result["frame_norm"], [np.sqrt(2), 0, np.sqrt(2)])
        np.testing.assert_array_equal(result["valid"], [True, False, True])
        expected = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float) / np.sqrt(2)
        np.testing.assert_allclose(result["unit_vectors"], expected)

    def test_bad_roi_is_rejected_instead_of_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Zero-variance"):
            core.prepare_bold([[1, 1], [2, 1], [3, 1]])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            core.prepare_bold([[1, np.nan], [2, 1], [3, 2]])


class ClusteringTests(unittest.TestCase):
    def test_weighted_signed_centroid_matches_replicated_samples(self):
        vectors = np.array([[100.0, 0.0], [0.0, 1.0]])
        original = vectors.copy()
        weights = np.array([3.0, 1.0])
        fit = core.spherical_kmeans(vectors, k=1, n_init=4, weights=weights)
        expected = np.array([3.0, 1.0]) / np.sqrt(10.0)
        np.testing.assert_allclose(fit["centers"][0], expected)
        self.assertAlmostEqual(fit["objective"], 1.0 - np.sqrt(10.0) / 4.0)
        repeated = core.spherical_kmeans([[1, 0], [1, 0], [1, 0], [0, 1]], k=1, n_init=4)
        np.testing.assert_allclose(repeated["centers"], fit["centers"])
        np.testing.assert_allclose(vectors, original)
        np.testing.assert_allclose(weights, [3, 1])
        np.testing.assert_allclose(np.linalg.norm(fit["centers"], axis=1), 1.0)
        self.assertEqual(len(fit["run_objectives"]), 4)

    def test_opposite_patterns_are_distinct_signed_states_but_one_unsigned_axis(self):
        vectors = np.array([[1, 0], [1, 0], [-1, 0], [-1, 0]], dtype=float)
        signed = core.spherical_kmeans(vectors, k=2, n_init=5, seed=42)
        self.assertEqual(signed["labels"][0], signed["labels"][1])
        self.assertEqual(signed["labels"][2], signed["labels"][3])
        self.assertNotEqual(signed["labels"][0], signed["labels"][2])
        self.assertAlmostEqual(signed["objective"], 0.0)
        unsigned = core.spherical_kmeans(vectors, k=1, n_init=5, seed=42, polarity="unsigned")
        np.testing.assert_array_equal(unsigned["labels"], 0)
        np.testing.assert_allclose(unsigned["centers"], [[1, 0]])
        self.assertAlmostEqual(unsigned["objective"], 0.0)
        assigned = core.assign_states(vectors, unsigned["centers"], polarity="unsigned")
        np.testing.assert_allclose(assigned["similarity"], 1.0)
        np.testing.assert_array_equal(assigned["polarity_sign"], [1, 1, -1, -1])
        one_signed = core.spherical_kmeans(vectors, k=1, n_init=2)
        self.assertAlmostEqual(one_signed["objective"], 1.0)

    def test_unsigned_weighted_axis_uses_scatter_not_signed_mean(self):
        fit = core.spherical_kmeans([[1, 0], [0, 1]], k=1, n_init=3,
                                   weights=[3, 1], polarity="unsigned")
        np.testing.assert_allclose(fit["centers"], [[1, 0]])
        self.assertAlmostEqual(fit["objective"], 0.25)

    def test_zero_weight_does_not_change_fit_and_invalid_training_rows_fail(self):
        fit = core.spherical_kmeans([[1, 0], [0, 1]], k=1, n_init=2, weights=[1, 0])
        np.testing.assert_allclose(fit["centers"], [[1, 0]])
        self.assertAlmostEqual(fit["objective"], 0.0)
        with self.assertRaisesRegex(ValueError, "zero or invalid rows"):
            core.spherical_kmeans([[1, 0], [0, 0]], k=1, n_init=1)

    def test_invalid_assignments_do_not_get_fake_labels_or_similarities(self):
        frames = np.array([[1, 0], [0, 0], [-1, 0], [0, 1], [np.nan, 0]])
        result = core.assign_states(frames, [[1, 0], [-1, 0]],
                                    valid=np.array([True, True, True, False, True]))
        np.testing.assert_array_equal(result["labels"], [0, -1, 1, -1, -1])
        np.testing.assert_array_equal(result["polarity_sign"], [1, 0, 1, 0, 0])
        self.assertTrue(np.isnan(result["similarity"][[1, 3, 4]]).all())


class SequenceMetricsTests(unittest.TestCase):
    def test_hand_calculated_metrics_transitions_and_coverage_identity(self):
        result = core.sequence_metrics([0, 0, 1, 1, 1, 0], tr=2.0, k=3)
        first, second, absent = result["states"]
        self.assertEqual(first["state"], 1)
        self.assertEqual(first["n_episodes"], 2)
        self.assertEqual(first["n_state_frames"], 3)
        self.assertAlmostEqual(first["coverage"], 0.5)
        self.assertAlmostEqual(first["duration_sec"], 3.0)
        self.assertAlmostEqual(first["occurrence_per_min"], 10.0)
        self.assertAlmostEqual(second["duration_sec"], 6.0)
        self.assertAlmostEqual(second["occurrence_per_min"], 5.0)
        for state in (first, second):
            self.assertAlmostEqual(state["coverage"],
                                   state["duration_sec"] * state["occurrence_per_min"] / 60.0)
        self.assertEqual(absent["coverage"], 0.0)
        self.assertEqual(absent["occurrence_per_min"], 0.0)
        self.assertTrue(np.isnan(absent["duration_sec"]))
        np.testing.assert_array_equal(result["frame_transition_counts"],
                                      [[1, 1, 0], [1, 2, 0], [0, 0, 0]])
        np.testing.assert_allclose(result["frame_transition_probabilities"][:2],
                                   [[0.5, 0.5, 0], [1 / 3, 2 / 3, 0]])
        np.testing.assert_array_equal(result["episode_transition_counts"],
                                      [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(result["episode_transition_probabilities"][:2],
                                   [[0, 1, 0], [1, 0, 0]])
        self.assertTrue(np.isnan(result["frame_transition_probabilities"][2]).all())
        self.assertTrue(np.isnan(result["episode_transition_probabilities"][2]).all())

    def test_gaps_break_episodes_and_never_create_cross_gap_transitions(self):
        result = core.sequence_metrics([0, 0, -1, 1, 1, -1, 0, 1, 1], tr=2, k=2)
        self.assertEqual(result["n_valid_frames"], 7)
        self.assertEqual(result["valid_duration_sec"], 14)
        self.assertEqual(result["states"][0]["n_episodes"], 2)
        self.assertEqual(result["states"][1]["n_episodes"], 2)
        np.testing.assert_array_equal(result["frame_transition_counts"], [[1, 1], [0, 2]])
        np.testing.assert_array_equal(result["episode_transition_counts"], [[0, 1], [0, 0]])
        self.assertTrue(np.isnan(result["episode_transition_probabilities"][1]).all())
        same_state = core.sequence_metrics([0, -1, 0], tr=2, k=1)
        self.assertEqual(same_state["states"][0]["n_episodes"], 2)
        self.assertAlmostEqual(same_state["states"][0]["duration_sec"], 2.0)
        self.assertEqual(same_state["frame_transition_counts"][0, 0], 0)

    def test_no_valid_exposure_is_undefined_not_zero(self):
        for labels in ([-1, -1], []):
            result = core.sequence_metrics(labels, tr=2.1, k=2)
            self.assertEqual(result["n_valid_frames"], 0)
            for state in result["states"]:
                self.assertTrue(np.isnan(state["coverage"]))
                self.assertTrue(np.isnan(state["duration_sec"]))
                self.assertTrue(np.isnan(state["occurrence_per_min"]))
            self.assertTrue(np.isnan(result["frame_transition_probabilities"]).all())
        with self.assertRaisesRegex(ValueError, "between -1 and k-1"):
            core.sequence_metrics([0, 2], tr=2.1, k=2)


if __name__ == "__main__":
    unittest.main()
