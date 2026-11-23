"""Synthetic fixtures for the historical P/LT score-row linkage."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clinical_archive_rerun import clinical_visits, ENGLISH_DOMAINS, scan_key


SYNTHETIC_START = 901


class ArchiveJoinTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.project = Path(tmp.name)
        data = self.project/"data"/"input"
        self.archive = self.project/"result"/"output"
        data.mkdir(parents=True)
        self.archive.mkdir(parents=True)
        suffix = "_BOLD_AAL1+PD25_MNI_basic36_sm6.csv"
        (data/("sub-HC01"+suffix)).touch()
        for condition in ("off", "on"):
            for subject in range(SYNTHETIC_START, SYNTHETIC_START + 25):
                (data/(f"sub-STN{condition}{subject:02d}"+suffix)).touch()
        sequence = np.arange(50)
        shuffle = np.random.default_rng(7).permutation(50)
        # Every occupancy row repeats; the accompanying lifetime resolves identity.
        for metric, values in (("P", np.ones((50,4))), ("LT", np.column_stack([sequence+i for i in range(4)]))):
            columns = {f"LPMS{i+1}":values[shuffle,i] for i in range(4)}
            columns["UPDRS3"] = sequence[shuffle]
            columns.update({name:sequence[shuffle] for name in ENGLISH_DOMAINS})
            pd.DataFrame(columns).to_csv(self.archive/f"scl_{metric}_stn.csv", index=False)
            for g,positions in ((2,range(25)), (3,range(25,50))):
                mat = np.full((51,4), np.nan)
                for i in positions:
                    mat[i+1] = values[i]
                np.savetxt(self.archive/f"{metric}{g}emp.csv",mat,delimiter=",")

    def test_shuffled_rows_recovered_from_both_metrics(self):
        result = clinical_visits(self.project,"output","input","ab"*32)
        self.assertEqual(len(result), 50)
        self.assertEqual(result.loc[scan_key(f"sub-STNoff{SYNTHETIC_START}","ab"*32),"score"], 0)
        self.assertEqual(result.loc[scan_key(f"sub-STNon{SYNTHETIC_START}","ab"*32),"score"], 25)

    def test_ambiguous_fingerprint_fails(self):
        path = self.archive/"LT2emp.csv"
        mat = np.loadtxt(path, delimiter=",")
        mat[2] = mat[1]
        np.savetxt(path, mat, delimiter=",")
        with self.assertRaisesRegex(ValueError, "Nonunique"):
            clinical_visits(self.project,"output","input","ab"*32)
