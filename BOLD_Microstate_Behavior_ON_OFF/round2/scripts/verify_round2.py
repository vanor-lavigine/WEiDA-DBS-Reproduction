"""Verify the second-round outputs against their frozen inputs and saved pair calculations."""

from pathlib import Path

import numpy as np
import pandas as pd

from round2_io import (BASE, CFG, PRIVATE, REPO, RESULTS, ROOT, digest, frozen_receipt, git, read_json, read_pairs)


def table(name, folder=RESULTS):
    return pd.read_csv(folder / name, float_precision="round_trip")


def same(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=True)


def verify_clinical(pairs, has_composition):
    n = len(pairs)
    profile = table("Clinical_Response_Profile.csv").set_index("endpoint", verify_integrity=True)
    private = table("Clinical_Changes_Paired.csv", PRIVATE / "results").set_index("patient_id", verify_integrity=True)
    private = private.loc[pairs.patient_id]
    for endpoint, row in profile.iterrows():
        off, on = (pairs[f"{endpoint}_{s}"].to_numpy(float) for s in ("off", "on"))
        improvement = off-on
        has_rate = off > 0
        individual_rate = np.divide(improvement, off, out=np.full(n, np.nan), where=has_rate)
        same(private[f"{endpoint}_improvement"], improvement)
        same(private[f"{endpoint}_improvement_rate"], individual_rate)
        assert (row.n_improved, row.n_unchanged, row.n_worse) == (
            int((improvement > 0).sum()), int((improvement == 0).sum()), int((improvement < 0).sum()),
        )
        assert row.rate_n == row.off_positive_n == int(has_rate.sum())
        assert row.off_zero_n == (~has_rate).sum()
        assert row.off_positive_n_improved == (has_rate & (improvement > 0)).sum()
        for prefix, values in (("off", off), ("on", on), ("improvement", improvement),
                               ("rate", individual_rate[has_rate])):
            if len(values):
                same(row[[prefix+"_mean", prefix+"_median", prefix+"_q25", prefix+"_q75"]].to_numpy(float),
                     [values.mean(), np.median(values), *np.percentile(values, [25, 75])])
    if not has_composition:
        return
    composition = table("Clinical_Improvement_Composition.csv").set_index("component", verify_integrity=True)
    parts = list(composition.index.drop("total"))
    for side in ("off", "on", "improvement"):
        same(private[[f"{p}_{side}" for p in parts]].sum(axis=1), private[f"total_{side}"])
    same(composition.loc[parts, "improvement_mean"].sum(), composition.at["total", "improvement_mean"])
    same(composition.fraction_of_total_mean, composition.improvement_mean/composition.at["total", "improvement_mean"])


def verify_brain(pairs, has_hc_reference):
    tolerance = CFG["duration"]["zero_tolerance_s"]
    shifts = table("M2_M4_Duration_Change_Summary.csv").set_index("state")
    patterns = table("M2_M4_Response_Patterns.csv").set_index("pattern")
    private = table("M2_M4_Changes_Paired.csv", PRIVATE / "results").set_index("patient_id", verify_integrity=True)
    private = private.loc[pairs.patient_id]
    n = len(pairs)
    directions = []
    for k in (2, 4):
        delta = pairs[f"duration_{k}_on"].to_numpy(float) - pairs[f"duration_{k}_off"].to_numpy(float)
        same(delta, private[f"delta_m{k}_s"])
        short, longer = delta < -tolerance, delta > tolerance
        s = shifts.loc[f"M{k}"]
        assert (short.sum(), longer.sum(), (~(short | longer)).sum()) == (s.shorter, s.longer, s.unchanged)
        same(np.percentile(delta, [25, 50, 75]), [s.q1_s, s.median_s, s.q3_s])
        directions.append(np.where(short, -1, np.where(longer, 1, 0)))
    a, b = directions
    masks = dict(both_shorter=(a<0)&(b<0), only_m2_shorter=(a<0)&(b>0), only_m4_shorter=(a>0)&(b<0),
                 both_longer=(a>0)&(b>0), zero_in_either=(a==0)|(b==0))
    assert int(patterns["count"].sum()) == n
    for label, mask in masks.items():
        assert patterns.at[label, "count"] == mask.sum()
        same(patterns.at[label, "proportion"], mask.mean())

    if not has_hc_reference:
        return
    hc = table("HC_M4_Values.csv", PRIVATE / "results").m4_duration_s.to_numpy()
    reference, = table("M4_HC_Reference_Summary.csv").itertuples(index=False)
    assert len(hc) == reference.n_hc == CFG["m4_reference"]["n_hc"]
    center = hc.mean()
    same(center, reference.hc_center_s)
    off, on = (pairs[f"duration_4_{side}"].to_numpy(float) for side in ("off", "on"))
    d0, d1 = abs(off-center), abs(on-center)
    change = d0-d1
    same(d0, private.off_distance_s)
    same(d1, private.on_distance_s)
    same(change, private.distance_change_s)
    same([d0.mean(), d1.mean(), change.mean()],
         [reference.off_distance_mean_s, reference.on_distance_mean_s, reference.distance_change_mean_s])
    assert reference.n_closer == (change > tolerance).sum()
    assert reference.n_farther == (change < -tolerance).sum()
    assert reference.n_unchanged == (abs(change) <= tolerance).sum()
    assert (reference.bootstrap_seed, reference.bootstrap_draws) == (CFG["m4_reference"]["seed"], CFG["m4_reference"]["draws"])

    # Independent scalar reconstruction with one shared reestimated center for each paired resample.
    rng = np.random.default_rng(reference.bootstrap_seed)
    draws = reference.bootstrap_draws
    centers = [hc[idx].mean() for idx in rng.integers(0, len(hc), (draws, len(hc)))]
    patient_indices = rng.integers(0, n, (draws, n))
    estimator = [(sum(abs(off[i] - c) - abs(on[i] - c) for i in indices) / n)
                 for c, indices in zip(centers, patient_indices, strict=True)]
    assert np.ptp(centers) > 0
    same(np.percentile(estimator, [2.5, 97.5]), [reference.ci95_low_s, reference.ci95_high_s])


def verify_artifacts():
    frozen_receipt()
    manifest = read_json(RESULTS / "Round2_Run_Manifest.json")
    assert manifest["reference_commit"] == BASE
    for name, expected in manifest["scripts_sha256"].items():
        assert digest(ROOT / "scripts" / name, text=True) == expected
    for name, expected in manifest["public_outputs_sha256"].items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT)
        assert digest(path, text=path.suffix in {".md", ".csv", ".json"}) == expected
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for file in RESULTS.glob("*.csv"):
        headers = pd.read_csv(file, nrows=0).columns
        assert not {"patient_id", "scan_id", "study_code", "subject", "assessment_date", "on_scan_id", "off_scan_id"}.intersection(headers)
    for name, expected in read_json(PRIVATE / "Round2_Input_Hashes.json")["file_sha256"].items():
        assert digest(name) == expected, f"Changed source: {Path(name).name}"
    git("check-ignore", "--", (PRIVATE / "Round2_Input_Hashes.json").relative_to(REPO).as_posix())
    assert not git("ls-files", "--", PRIVATE.relative_to(REPO).as_posix())
    return manifest["missing_input_dependencies"]


def main():
    issues = verify_artifacts()
    pairs = read_pairs()
    verify_clinical(pairs, "clinical_composition" not in issues)
    verify_brain(pairs, "m4_hc_reference" not in issues)
    print("Verified frozen files, clinical arithmetic and missing-rate handling, M2/M4 directions, "
          "release artifacts and private boundaries.")
    if "m4_hc_reference" not in issues:
        print("Reconstructed the M4 interval using 9,999 draws with a shared HC center per draw.")


if __name__ == "__main__":
    main()
