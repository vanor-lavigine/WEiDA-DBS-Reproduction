"""Independently check output checksums, pairing signs and FDR families."""

import json

import numpy as np
import pandas as pd
from scipy import stats

from behavior_protocol import BEHAVIORS, PRIVATE, RESULTS, ROOT, STATES, digest, git
from behavior_stats import registry_check


def verify_q(table, size):
    assert len(table) == size and table.bh_family_size.eq(size).all()
    p = table.raw_p.to_numpy(float)
    finite = np.isfinite(p)
    expected = stats.false_discovery_control(np.where(finite, p, 1.), method="bh")
    np.testing.assert_allclose(table.fdr_q, np.where(finite, expected, np.nan), rtol=0, atol=1e-13, equal_nan=True)


def verify_c(table, private, label):
    verify_q(table, 20)
    assert list(zip(table.state, table.behavior, strict=True)) == [(s, b) for s in STATES for b in BEHAVIORS]
    assert table["sample"].eq(label).all()
    for row in table.itertuples():
        i = STATES.index(row.state)+1
        x = private[f"duration_{i}_on"] - private[f"duration_{i}_off"]
        y = private[f"{row.behavior}_off"] - private[f"{row.behavior}_on"]
        keep = x.notna() & y.notna()
        assert row.n == keep.sum()
        if row.n >= 3 and x[keep].nunique() > 1 and y[keep].nunique() > 1:
            actual = stats.spearmanr(x[keep], y[keep])
            np.testing.assert_allclose([row.rho, row.raw_p], [actual.statistic, actual.pvalue], rtol=0, atol=1e-13)
        else:
            assert np.isnan(row.raw_p) and isinstance(row.reason, str) and len(row.reason)


def main():
    registry_check()
    manifest = json.loads((RESULTS / "Run_Manifest.json").read_text(encoding="utf-8"))
    for file, checksum in manifest["public_outputs_sha256"].items():
        path = ROOT / file
        assert digest(path, text=path.suffix in {".md", ".csv"}) == checksum, file
    for file, checksum in manifest["scripts_sha256"].items():
        assert digest(ROOT / "scripts" / file, text=True) == checksum, file
    confidential = {"patient_id", "participant_id", "on_scan_id", "off_scan_id", "scan_id", "person_key",
                    "study_code", "name", "assessment_date", "on_assessment_date", "off_assessment_date"}
    for path in RESULTS.glob("*.csv"):
        assert not confidential.intersection(pd.read_csv(path, nrows=0).columns), path.name
    private = {}
    for label, file in (("main", "Main_Pairs.csv"), ("date_verified", "Date_Verified_Pairs.csv"),
                        ("fd_censor_s1", "Quality_Pairs.csv")):
        path = PRIVATE / "results" / file
        git("check-ignore", "--", path.relative_to(ROOT.parent).as_posix())
        private[label] = pd.read_csv(path)
        assert private[label].patient_id.is_unique
    full = private["main"]
    date = private["date_verified"]
    assert set(date.patient_id) == set(full.loc[full.date_verified.eq(True), "patient_id"])
    for behavior in BEHAVIORS:
        delta = full[f"{behavior}_off"] - full[f"{behavior}_on"]
        np.testing.assert_allclose(full[f"{behavior}_improvement"], delta)
        rate = delta.div(full[f"{behavior}_off"].where(full[f"{behavior}_off"].gt(0)))
        np.testing.assert_allclose(full[f"{behavior}_improvement_rate"], rate, equal_nan=True)
    for file, endpoints, prefix, sign in (("Clinical_Paired_Results.csv", BEHAVIORS, "", -1),
                                          ("Duration_Paired_Results.csv", STATES, "duration_", 1)):
        rows = pd.read_csv(RESULTS / file)
        verify_q(rows, len(endpoints))
        assert tuple(rows.endpoint) == endpoints
        for i, row in enumerate(rows.itertuples(), 1):
            key = f"{prefix}{i}" if prefix else row.endpoint
            delta = sign*(full[key+"_on"]-full[key+"_off"])
            assert delta.notna().sum() == row.n
            np.testing.assert_allclose(row.mean_change, delta.mean(), rtol=0, atol=1e-13)
    verify_c(pd.read_csv(RESULTS / "Duration_Motor_Associations.csv"), full, "main")
    sensitivity = pd.read_csv(RESULTS / "Association_Sensitivities.csv")
    for label in ("date_verified", "fd_censor_s1"):
        verify_c(sensitivity.loc[sensitivity["sample"].eq(label)], private[label], label)
    item_dictionary = pd.read_csv(RESULTS / "Score_Item_Dictionary.csv")
    assert len(item_dictionary) == 33 and set(item_dictionary.loc[item_dictionary.domains.str.contains(
        "gait_balance"), "column"]) == {"AD", "AF", "AG"}
    assert item_dictionary.set_index("column").loc["AI", "item"] == "body_bradykinesia"
    assert not set(item_dictionary.column).intersection({"Q", "R", "AA", "AB", "AE"})
    print("Verified source-aligned outcomes, 5/4/20 BH families, 40 sensitivity slots, and the release hashes.")


if __name__ == "__main__":
    main()
