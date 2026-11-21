"""Meaningful I/O and end-to-end checks; all data here are synthetic."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from run import load_roi, verified_metadata


class PipelineTests(unittest.TestCase):
    def test_duplicate_roi_header_is_not_silently_mangled(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "duplicate.tsv"
            path.write_text("time\tROI_A\tROI_A\n0\t1\t2\n1\t2\t3\n")
            with self.assertRaisesRegex(ValueError, "duplicate ROI"):
                load_roi(path)

    def test_duplicate_identity_with_whitespace_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.tsv"
            pd.DataFrame({"run_id": ["r1", " r1 "], "subject_id": ["s1", "s1"],
                          "cohort": ["DBS", "DBS"], "condition": ["ON", "OFF"],
                          "mapping_source": ["synthetic", "synthetic"]}).to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "duplicate run_id"):
                verified_metadata(path)

    def test_synthetic_index_fit_then_clinical_scores_added(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rng = np.random.default_rng(721)
            patterns = rng.normal(size=(4, 8))
            index_rows, truth = [], {}
            for patient in range(6):
                for condition in ("OFF", "ON"):
                    labels = rng.integers(0, 4, size=48)
                    x = patterns[labels] + rng.normal(scale=.35, size=(48, 8))
                    path = root / ("synthetic_%d_%s.tsv" % (patient, condition))
                    table = pd.DataFrame(x, columns=["ROI_%d" % i for i in range(8)])
                    table.insert(0, "time", np.arange(48) * 2.1)
                    table.to_csv(path, sep="\t", index=False)
                    index_rows.append({"roi_time_series": str(path), "atlas": "TEST_ATLAS",
                                       "format": "cifti", "label": "COMPUTATIONAL_CHECKS_PASS",
                                       "n_frames": 48, "n_regions": 8})
                    truth[str(path.resolve())] = {"subject_id": "S%d" % patient, "cohort": "DBS",
                                        "condition": condition, "mapping_source": "synthetic_fixture",
                                        "clinical_scale": "MDS-UPDRS-III",
                                        "clinical_score": (20 + patient * 2 if condition == "OFF" else 14 + patient)}
            index = root / "results.tsv"
            pd.DataFrame(index_rows).to_csv(index, sep="\t", index=False)

            def execute(*arguments, expected=0):
                completed = subprocess.run([sys.executable, str(MODULE / "run.py"), *map(str, arguments)],
                                           capture_output=True, text=True, timeout=60)
                self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
                return completed

            execute("index", "--index", index, "--out", root / "input", "--atlas", "TEST_ATLAS", "--format", "cifti")
            runs = pd.read_csv(root / "input/runs.tsv", sep="\t")
            meta_rows = []
            for row in runs.to_dict("records"):
                meta_rows.append(dict(truth[row["roi_time_series"]], run_id=row["run_id"]))
            meta = pd.DataFrame(meta_rows)
            clinical_path = root / "metadata.tsv"
            no_clinical = meta.copy()
            no_clinical["clinical_score"] = ""
            no_clinical.to_csv(clinical_path, sep="\t", index=False)
            execute("fit", "--runs", root / "input/runs.tsv", "--metadata", clinical_path,
                    "--out", root / "fit", "--mode", "analysis", "--data-kind", "synthetic",
                    "--k", 4, "--n-init", 4, "--stability-repeats", 1)
            summary = json.loads((root / "fit/summary.json").read_text())
            self.assertEqual(summary["verified_subjects"], 6)
            self.assertEqual(summary["included_runs"], 12)
            self.assertEqual(summary["data_kind"], "synthetic")
            metrics = pd.read_csv(root / "fit/metrics.tsv", sep="\t")
            self.assertEqual(len(metrics), 48)
            np.testing.assert_allclose(metrics.groupby("run_id")["coverage"].sum(), 1)
            np.testing.assert_allclose(metrics["coverage"],
                                       metrics["occurrence_per_min"] * metrics["duration_sec"] / 60)
            self.assertTrue((root / "fit/overview.png").is_file())
            # Adding clinical scores after model fitting is legitimate and must not require refitting.
            meta.to_csv(clinical_path, sep="\t", index=False)
            execute("stats", "--fit-dir", root / "fit", "--metadata", clinical_path,
                    "--out", root / "stats", "--permutations", 199, "--bootstrap", 50,
                    "--clinical-scale", "MDS-UPDRS-III")
            statistics = pd.read_csv(root / "stats/stats.csv")
            self.assertEqual(len(statistics), 28)
            self.assertEqual(statistics.groupby("family").size().to_dict(),
                             {"on_off": 12, "clinical_change": 12, "clinical_off_duration": 4})
            self.assertEqual(json.loads((root / "stats/analysis_summary.json").read_text())["data_kind"], "synthetic")
            # Replacing participant identity after template weighting must be detected.
            meta.loc[0, "subject_id"] = "CHANGED_IDENTITY"
            meta.to_csv(clinical_path, sep="\t", index=False)
            failed = execute("stats", "--fit-dir", root / "fit", "--metadata", clinical_path,
                             "--out", root / "changed", "--permutations", 99, "--bootstrap", 20, expected=2)
            self.assertIn("identity/condition mapping differs", failed.stderr)


if __name__ == "__main__":
    unittest.main()
