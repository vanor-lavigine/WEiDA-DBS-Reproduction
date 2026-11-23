"""Check the published group outputs against the frozen design, including multiplicity."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check_holm(frame, column, expected_count):
    p = frame.sort_values("p_raw", kind="stable")
    if len(p) != expected_count:
        raise ValueError("Incomplete Holm family")
    expected = np.minimum(1, np.maximum.accumulate(p.p_raw.to_numpy() * np.arange(len(p), 0, -1)))
    np.testing.assert_allclose(p[column], expected, rtol=0, atol=3e-14)


def verify(folder):
    rows = pd.read_csv(folder / "stats.csv")
    maps = pd.read_csv(folder / "templates.tsv", sep="\t")
    qa = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
    assert len(rows) == 120 and not rows.duplicated(["scenario", "method", "family", "state", "metric"]).any()
    assert maps.shape == (8, 80) and not maps.duplicated(["method", "state"]).any()
    assert set(maps.method) == set(rows.method) == {"AAHC", "signed"}
    assert qa["fit"]["assigned_frames"] == {"AAHC": 10080, "signed": 10080}
    assert (qa["fit"]["n_hc"], qa["fit"]["runs"], qa["fit"]["patients"]) == (0, 48, 24)
    assert qa["sequence_metric_check"]["scan_state_metric_rows"] == 384
    assert qa["sequence_metric_check"]["sequences"] == 96
    assert qa["sequence_metric_check"]["max_abs_error"] < 1e-12
    assert sum(s["effects_checked"] for s in qa["checked_effects"]) == 120
    assert max(s["max_abs_error"] for s in qa["checked_effects"]) < 1e-12
    np.testing.assert_allclose(np.linalg.norm(maps.iloc[:, 2:].values, axis=1), 1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(maps[maps.method.eq("AAHC")].iloc[:, 2:].mean(axis=1), 0, atol=1e-12)
    groups = {"archive_main": 24, "archive_alternate": 24, "concordant_pairs": 17}
    families = {"clinical_off_duration": 4, "clinical_change": 12, "on_off": 12}
    for scenario, n in groups.items():
        group = rows[rows.scenario.eq(scenario)]
        wanted = set(families) if scenario == "archive_main" else set(families) - {"on_off"}
        assert set(group.family) == wanted
        assert group.n_subjects.eq(n).all() and group.n_pool.eq(n).all() and group.n_missing_endpoint.eq(0).all()
        for family_name, family in group.groupby("family"):
            size = families[family_name]
            check_holm(family, "p_holm_both_methods", 2 * size)
            for _, method in family.groupby("method"):
                check_holm(method, "p_holm", size)
    conditions = rows[rows.family.eq("on_off")]
    np.testing.assert_allclose(conditions.mean_on - conditions.mean_off, conditions.estimate, atol=1e-12)
    occupancy = conditions[conditions.metric.eq("coverage")]
    np.testing.assert_allclose(occupancy.groupby("method")[["mean_on", "mean_off"]].sum(), 1, atol=1e-12)
    assert rows.n_bootstrap_valid.eq(2000).all()
    print("Verified 8 templates, 120 aggregate estimates and 21 Holm families (14 within-method, 7 across-method).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    verify(parser.parse_args().folder)
