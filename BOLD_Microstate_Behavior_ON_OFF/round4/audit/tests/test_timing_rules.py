"""Synthetic regression cases for the evidence hierarchy."""

from pathlib import Path
import sys
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from timing_rules import classify, link_visits, scan_order, time_difference


CORE = {
    "identity_confirmed": True, "condition_confirmed": True, "score_source_confirmed": True,
    "scan_identity_confirmed": True, "scan_condition_confirmed": True,
}


class TimingRulesTest(unittest.TestCase):
    def test_date_only_never_becomes_midnight(self):
        result = classify(dict(CORE, score_date="2020-02-04", scan_date="2020-02-04",
                               score_datetime="2020-02-04", scan_datetime="2020-02-04T13:48:50"))
        self.assertEqual(result["visit_match_status"], "Confirmed same day")
        self.assertEqual(result["calendar_date_difference_days"], 0)
        self.assertIsNone(result["date_difference_hours"])
        self.assertEqual(result["assessment_relative_to_scan"], "unknown")

    def test_direct_same_visit_requires_identity_condition_and_score(self):
        result = classify(dict(CORE, same_visit_documented=True))
        self.assertEqual(result["visit_match_status"], "Confirmed same visit")
        result = classify(dict(CORE, same_visit_documented=True, condition_confirmed=False))
        self.assertEqual(result["visit_match_status"], "Identity uncertain")

    def test_direct_and_calendar_evidence(self):
        self.assertEqual(classify(dict(CORE, different_visit_documented=True))["visit_match_status"], "Different visit")
        result = classify(dict(CORE, score_date="2020-02-06", scan_date="2020-02-04"))
        self.assertEqual(result["visit_match_status"], "Different visit")
        self.assertEqual(result["calendar_date_difference_days"], 2)
        self.assertEqual(result["assessment_relative_to_scan"], "after")
        self.assertEqual(result["assessment_order_evidence"], "distinct_calendar_days")
        self.assertIsNone(result["date_difference_hours"])
        result = classify(dict(CORE, score_date="2020-02-06", scan_date="2020-02-04", scan_date_unresolved=True))
        self.assertEqual(result["visit_match_status"], "Date unavailable")
        self.assertIsNone(result["calendar_date_difference_days"])

    def test_clock_bases(self):
        self.assertIsNone(time_difference("2020-02-04T12:00", "2020-02-04T10:30", shared_clock=False))
        self.assertEqual(time_difference("2020-02-04T12:00", "2020-02-04T10:30", shared_clock=True), 1.5)
        self.assertEqual(time_difference("2020-02-04T12:00+08:00", "2020-02-04T03:00+00:00", shared_clock=False), 1)
        self.assertIsNone(time_difference("2020-02-04T12:00+08:00", "2020-02-04T03:00", shared_clock=True))

    def test_groups_require_both_conditions(self):
        scans, scores = [], []
        for patient in ("a", "b", "c"):
            for condition in ("ON", "OFF"):
                keys = dict(patient_id=patient, condition=condition, scan_id=patient + condition)
                scans.append(dict(keys, scan_identity_confirmed=True, scan_condition_confirmed=True))
                scores.append(dict(keys, study_code=patient, identity_confirmed=True,
                                   condition_confirmed=(patient, condition) != ("c", "OFF"),
                                   score_source_confirmed=True,
                                   same_visit_documented=patient == "a" or condition == "ON"))
        visits, groups = link_visits(pd.DataFrame(scores), pd.DataFrame(scans), 3)
        self.assertEqual(dict(zip(groups.patient_id, groups.group)), {"a": "A", "b": "B", "c": "C"})
        self.assertEqual(len(visits), 6)

    def test_scan_order_uses_acquisition_clocks(self):
        on = dict(patient_id="a", condition="ON", scan_id="on", scan_date="2021-12-04",
                  scan_datetime="2021-12-04T23:30", scan_clock_basis="scanner")
        off = dict(patient_id="a", condition="OFF", scan_id="off", scan_date="2021-12-05",
                   scan_datetime="2021-12-05T01:00", scan_clock_basis="scanner")
        result = scan_order(pd.DataFrame([off, on])).iloc[0]
        self.assertEqual(result.scan_order, "ON first")
        self.assertEqual(result.OFF_minus_ON_hours, 1.5)
        off["scan_clock_basis"] = "other"
        self.assertEqual(scan_order(pd.DataFrame([on, off])).iloc[0].scan_order, "unverified")


if __name__ == "__main__":
    unittest.main()
