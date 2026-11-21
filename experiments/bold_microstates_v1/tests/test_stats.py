"""Tests of inferential units, fixed families, and clinically relevant signs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tempfile
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "bold_microstates_stats", Path(__file__).resolve().parents[1] / "stats.py"
)
stats = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(stats)


def example_data(n=6, k=4):
    metrics = []
    metadata = []
    improvements = [2, 5, 1, 4, 3, 6, 7, 8][:n]
    for subject_index in range(n):
        subject_id = f"person-{subject_index}"
        improvement = improvements[subject_index]
        off_score = 40 + subject_index
        for condition in ("ON", "OFF"):
            run_id = f"{subject_id}-{condition}"
            metadata.append({
                "run_id": run_id, "subject_id": subject_id,
                "cohort": "DBS", "condition": condition,
                "mapping_source": "verified_condition_manifest.tsv",
                "clinical_scale": "UPDRS-III",
                "clinical_score": off_score - improvement if condition == "ON" else off_score,
            })
            for state in range(1, k + 1):
                delta = improvement if condition == "ON" else 0
                metrics.append({
                    "run_id": run_id, "state": state,
                    "coverage": 0.15 + state * 0.01 + subject_index * 0.001 + delta * 0.002,
                    "occurrence_per_min": 2 + state * 0.1 + subject_index * 0.1 + delta * 0.02,
                    "duration_sec": 2 + state + subject_index * 0.1 + delta * 0.1,
                })
    return pd.DataFrame(metrics), pd.DataFrame(metadata)


def run_analysis(tmp_path, metrics, metadata, **kwargs):
    summary = stats.analyze(
        metrics, metadata, tmp_path,
        n_permutations=399, n_bootstrap=99, seed=7, **kwargs,
    )
    return summary, pd.read_csv(summary["stats_path"]), pd.read_csv(summary["pairing_audit_path"], sep="\t")


def endpoint(results, family, state=1, metric="duration_sec"):
    return results.loc[
        results.family.eq(family) & results.state.eq(state) & results.metric.eq(metric)
    ].iloc[0]


class StatsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp_path = Path(directory.name)

    def test_known_pairing_uses_people_and_preserves_change_direction(self):
        metrics, metadata = example_data()
        summary, results, audit = run_analysis(self.tmp_path, metrics, metadata)
        assert summary["n_eligible_pairs"] == 6
        assert summary["n_clinically_eligible_pairs"] == 6
        assert len(results) == 12 + 12 + 4
        assert audit.pair_eligible.all()
        on_off = endpoint(results, stats.FAMILY_ON_OFF)
        assert on_off.n_subjects == 6
        assert np.isclose(on_off.estimate, 0.35)
        assert 0 <= on_off.p_raw <= 1
        assert on_off.ci_low <= on_off.estimate <= on_off.ci_high
        change = endpoint(results, stats.FAMILY_CLINICAL_CHANGE)
        baseline = endpoint(results, stats.FAMILY_CLINICAL_OFF)
        # OFF-ON clinical improvement, not the accidentally reversed ON-OFF score.
        assert np.isclose(change.estimate, 1.0)
        assert np.isclose(baseline.estimate, 1.0)


    def test_duplicate_metadata_run_excludes_subject_instead_of_expanding_n(self):
        metrics, metadata = example_data()
        metadata = pd.concat([metadata, metadata.iloc[[0]]], ignore_index=True)
        summary, results, audit = run_analysis(self.tmp_path, metrics, metadata)
        assert summary["n_eligible_pairs"] == 5
        assert endpoint(results, stats.FAMILY_ON_OFF).n_subjects == 5
        affected = audit.loc[audit.subject_id.eq("person-0")]
        assert not affected.pair_eligible.any()
        assert affected.pair_reason.str.contains("duplicate_run_id").all()


    def test_duplicate_metric_state_is_not_silently_averaged(self):
        metrics, metadata = example_data()
        metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)
        summary, results, audit = run_analysis(self.tmp_path, metrics, metadata)
        assert summary["n_eligible_pairs"] == 5
        assert endpoint(results, stats.FAMILY_ON_OFF).n_subjects == 5
        assert audit.loc[audit.subject_id.eq("person-0"), "pair_reason"].str.contains("duplicate_metric_state").all()


    def test_holm_keeps_missing_slots_and_is_monotone(self):
        p = np.array([0.01] + [np.nan] * 11)
        adjusted = stats.holm_adjust(p)
        assert np.isclose(adjusted[0], 0.12)
        assert np.isnan(adjusted[1:]).all()
        p = np.array([0.04, 0.001, np.nan, 0.02, 0.5])
        adjusted = stats.holm_adjust(p)
        ordered = np.argsort(p[np.isfinite(p)])
        assert np.all(np.diff(adjusted[np.isfinite(p)][ordered]) >= 0)
        assert np.all(adjusted[np.isfinite(p)] >= p[np.isfinite(p)])


    def test_missing_endpoint_keeps_family_and_subject_specific_n(self):
        metrics, metadata = example_data()
        metrics.loc[metrics.state.eq(4), "duration_sec"] = np.nan
        metrics.loc[metrics.run_id.eq("person-0-ON") & metrics.state.eq(1), "duration_sec"] = np.nan
        _, results, _ = run_analysis(self.tmp_path, metrics, metadata)
        absent = endpoint(results, stats.FAMILY_ON_OFF, state=4)
        assert absent.n_subjects == 0
        assert absent.family_size == 12
        assert pd.isna(absent.p_raw) and pd.isna(absent.p_holm)
        assert endpoint(results, stats.FAMILY_ON_OFF, metric="coverage").n_subjects == 6
        assert endpoint(results, stats.FAMILY_CLINICAL_CHANGE).n_subjects == 5
        # Baseline association does not select on whether the ON state appeared.
        assert endpoint(results, stats.FAMILY_CLINICAL_OFF).n_subjects == 6
        for family in (stats.FAMILY_ON_OFF, stats.FAMILY_CLINICAL_CHANGE):
            group = results.loc[results.family.eq(family)]
            assert len(group) == 12
            np.testing.assert_allclose(group.p_holm, stats.holm_adjust(group.p_raw), equal_nan=True)


    def test_zero_difference_has_p_one_and_no_infinite_effect(self):
        assert stats.signflip_pvalue([0, 0, 0, 0], 99, seed=1) == 1.0
        # For three identical positive differences, the exact two-sided null is 2/8.
        assert abs(stats.signflip_pvalue([1, 1, 1], 19999, seed=1) - 0.25) <= 0.015
        metrics, metadata = example_data()
        off = metrics.loc[metrics.run_id.str.endswith("-OFF")].copy()
        for _, row in off.iterrows():
            mask = metrics.run_id.eq(row.run_id.replace("-OFF", "-ON")) & metrics.state.eq(row.state)
            metrics.loc[mask, list(stats.METRICS)] = row[list(stats.METRICS)].to_numpy()
        _, results, _ = run_analysis(self.tmp_path, metrics, metadata)
        row = endpoint(results, stats.FAMILY_ON_OFF)
        assert row.estimate == 0
        assert row.p_raw == row.p_holm == 1
        assert row.ci_low == row.ci_high == 0
        assert pd.isna(row.effect_dz)
        assert row.reason == "all_zero_difference"
        association = endpoint(results, stats.FAMILY_CLINICAL_CHANGE)
        assert association.reason == "constant_variable"
        assert pd.isna(association.p_raw)


    def test_absent_clinical_score_does_not_block_condition_effect(self):
        metrics, metadata = example_data()
        metadata.loc[metadata.run_id.eq("person-0-ON"), "clinical_score"] = np.nan
        summary, results, audit = run_analysis(self.tmp_path, metrics, metadata)
        assert summary["n_eligible_pairs"] == 6
        assert summary["n_clinically_eligible_pairs"] == 5
        assert endpoint(results, stats.FAMILY_ON_OFF).n_subjects == 6
        assert endpoint(results, stats.FAMILY_CLINICAL_CHANGE).n_subjects == 5
        assert endpoint(results, stats.FAMILY_CLINICAL_OFF).n_subjects == 5
        assert audit.loc[audit.subject_id.eq("person-0"), "clinical_reason"].str.contains("clinical_score_missing").all()


    def test_scale_names_are_not_implicitly_merged(self):
        metrics, metadata = example_data()
        metadata.loc[metadata.subject_id.eq("person-0"), "clinical_scale"] = "UPDRS"
        summary, results, _ = run_analysis(self.tmp_path / "ambiguous", metrics, metadata)
        assert summary["clinical_scale_status"] == "ambiguous_clinical_scale"
        assert results.loc[results.family.ne(stats.FAMILY_ON_OFF), "p_raw"].isna().all()
        assert endpoint(results, stats.FAMILY_ON_OFF).n_subjects == 6
        summary, results, _ = run_analysis(
            self.tmp_path / "chosen", metrics, metadata, clinical_scale="UPDRS-III"
        )
        assert summary["n_clinically_eligible_pairs"] == 5
        assert endpoint(results, stats.FAMILY_CLINICAL_CHANGE).n_subjects == 5


    def test_evidence_is_required_and_orphan_runs_are_audited(self):
        metrics, metadata = example_data()
        metadata.loc[metadata.run_id.eq("person-0-ON"), "mapping_source"] = "  "
        metadata = metadata.loc[metadata.subject_id.ne("person-1")]
        summary, _, audit = run_analysis(self.tmp_path, metrics, metadata)
        assert summary["n_eligible_pairs"] == 4
        assert audit.loc[audit.subject_id.eq("person-0"), "pair_reason"].str.contains("missing_mapping_source").all()
        orphans = audit.loc[audit.run_id.isin(["person-1-ON", "person-1-OFF"])]
        assert len(orphans) == 2
        assert orphans.pair_reason.eq("metadata_missing").all()


    def test_fewer_than_three_pairs_produces_na_inference(self):
        metrics, metadata = example_data(n=2)
        _, results, _ = run_analysis(self.tmp_path, metrics, metadata)
        assert results.p_raw.isna().all()
        assert results.p_holm.isna().all()
        assert endpoint(results, stats.FAMILY_ON_OFF).n_subjects == 2
        assert endpoint(results, stats.FAMILY_ON_OFF).reason == "insufficient_pairs"


    def test_empty_metadata_does_not_invent_pairs(self):
        metrics, _ = example_data()
        summary, results, audit = run_analysis(self.tmp_path, metrics, pd.DataFrame())
        assert summary["status"] == "no_eligible_dbs_pairs"
        assert len(results) == 28
        assert len(audit) == 12
        assert results.p_raw.isna().all()
        assert audit.pair_reason.eq("metadata_missing").all()


    def test_bootstrap_count_does_not_change_permutation_pvalues(self):
        metrics, metadata = example_data()
        first = stats.analyze(metrics, metadata, self.tmp_path / "a", n_permutations=199, n_bootstrap=0, seed=8)
        second = stats.analyze(metrics, metadata, self.tmp_path / "b", n_permutations=199, n_bootstrap=39, seed=8)
        a = pd.read_csv(first["stats_path"])
        b = pd.read_csv(second["stats_path"])
        np.testing.assert_allclose(a.p_raw, b.p_raw, equal_nan=True)

    def test_constant_change_roundoff_does_not_create_clinical_ranks(self):
        metrics, metadata = example_data()
        for _, row in metrics.loc[metrics.run_id.str.endswith("-OFF")].iterrows():
            mask = metrics.run_id.eq(row.run_id.replace("-OFF", "-ON")) & metrics.state.eq(row.state)
            metrics.loc[mask, "duration_sec"] = row.duration_sec + 0.1
        _, results, _ = run_analysis(self.tmp_path, metrics, metadata)
        association = endpoint(results, stats.FAMILY_CLINICAL_CHANGE)
        assert association.reason == "constant_variable"
        assert pd.isna(association.estimate)
        assert pd.isna(association.p_raw)
        assert pd.isna(endpoint(results, stats.FAMILY_ON_OFF).effect_dz)


if __name__ == "__main__":
    unittest.main()
