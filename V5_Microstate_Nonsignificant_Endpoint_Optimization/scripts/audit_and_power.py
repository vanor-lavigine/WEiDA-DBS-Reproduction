"""Audit the archived visit evidence and calculate the prespecified power scenarios."""

from hashlib import sha256
import json
import math

import numpy as np
import openpyxl
import pandas as pd
from statsmodels.stats.power import TTestIndPower, TTestPower

from v5_data import LOCAL_V4, RELIABILITY, RESULT, WORKSPACE, csv, dump


AUDIT = RESULT / "covariates"
POWER = RESULT / "power"
ALPHA = .05 / 6
LOSS = .15


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def clinical_scan_audit(frozen):
    visit_book, = WORKSPACE.glob("V100*/结论/**/105原始访视逐人核查.xlsx")
    workbook = openpyxl.load_workbook(visit_book, read_only=True, data_only=True)
    try:
        visits = pd.DataFrame(workbook["48次访视"].iter_rows(min_row=5, values_only=True)).dropna(subset=[0, 1])
    finally:
        workbook.close()
    visits["scan_key"] = visits[11].astype(str).str.extract(r"(?i)(sub-STN(?:on|off)\d+)")[0].str.casefold()
    scans = frozen.assign(scan_key=frozen.scan.str.casefold())
    linked = scans.merge(visits, on="scan_key", validate="one_to_one")
    assert linked.condition.eq(linked[1]).all() and linked.groupby("subject")[0].nunique().eq(1).all()
    paired = linked.pivot(index="subject", columns="condition", values=2).dropna()
    intervals = (paired.OFF-paired.ON).dt.total_seconds()/60
    hours_since_dose = pd.to_numeric(linked[6], errors="coerce").notna().sum()
    data = dict(pairs_in_frozen_cohort=25, matched_scans=len(linked), matched_pairs=len(paired),
                on_first=int(intervals.gt(0).sum()), off_first=int(intervals.lt(0).sum()),
                same_day=int(paired.OFF.dt.normalize().eq(paired.ON.dt.normalize()).sum()),
                interval_minutes={name: float(value) for name,value in zip(["min", "q1", "median", "q3", "max"],
                                                                             intervals.quantile([0,.25,.5,.75,1.]))},
                verified_medication_intervals=int(hours_since_dose), source_visit_workbook_sha256=sha(visit_book),
                paired_scan_match_basis="Original DBS scan sidecar key, condition and frozen paired participant")
    return data


def audit():
    frozen = pd.read_csv(LOCAL_V4 / "private/Subject_Level_Microstate_Features.csv")
    quality = pd.read_csv(RELIABILITY / "private/scan_quality.csv")
    merged = frozen.merge(quality, on=["scan", "condition"], validate="one_to_one", suffixes=("", "_quality"))
    assert len(merged) == len(frozen) == 65 and np.allclose(merged.fd, merged.fd_quality, atol=1e-14, rtol=0)
    provenance_path = RELIABILITY / "results/Frame_Quality_Input_Provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert merged.n_dvars.sum() == provenance["total_defined_dvars_frames"] and provenance["source_confounds_sha256_matches"] == 65
    visits = clinical_scan_audit(frozen)
    dump(visits, AUDIT / "Scan_Order_and_Intervals.json")
    names = {"fd":"Mean FD (mm)", "mean_raw_dvars":"Mean DVARS", "mean_std_dvars":"Mean standardized DVARS"}
    rows = []
    for name, label in names.items():
        for condition, group in merged.groupby("condition"):
            values = group[name].dropna()
            rows.append(dict(field=label, condition=condition, n=len(values), mean=values.mean(), sd=values.std(),
                             median=values.median(), q1=values.quantile(.25), q3=values.quantile(.75)))
    csv(pd.DataFrame(rows), AUDIT / "Motion_Quality_Group_Summary.csv")

    source_root = WORKSPACE / "WEiDA-raw-audit-private-20261116"
    roster_path = source_root / "source_participant_evidence.json"
    source = json.loads(roster_path.read_text(encoding="utf-8"))
    inventory_path = source_root / "source105_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    sidecars = {r["subject_label"].lower().removeprefix("sub-"): r["json_metadata"]
                for r in inventory["scan_records"] if r["modality"]=="bold" and r["dataset"]=="DBS"}
    frozen_labels = frozen.scan.str.casefold().str.removeprefix("sub-")
    if not set(frozen_labels).issubset(sidecars):
        # Older inventory records use the original directory designation instead of DBS.
        sidecars = {r["subject_label"].lower().removeprefix("sub-"): r["json_metadata"]
                    for r in inventory["scan_records"] if r["modality"]=="bold" and r["subject_label"].lower().removeprefix("sub-") in set(frozen_labels)}
    assert len(set(frozen_labels) & sidecars.keys()) == 65
    age, sex = (sum(any(k.casefold() in field and value not in (None,"", "n/a") for k,value in sidecars[s].items())
                    for s in frozen_labels) for field in ({"age", "patientage"}, {"sex", "gender", "patientsex"}))
    confirmed_hc = sum(x["group"]=="HC" for x in source["dbs_confirmed"].values())
    assert confirmed_hc == 15 and len(source["DBS_unverified"]) == 1
    recovery = json.loads((WORKSPACE / "WEiDA_HC13_candidate_20261117/candidate_provenance.json").read_text(encoding="utf-8"))
    assert recovery
    dump(dict(paper_doi="10.1038/s41531-026-01354-3", paper_hc=16, frozen_hc=15,
              registered_source_hc=confirmed_hc, additional_unregistered_hc_scan=len(source["DBS_unverified"]),
              candidate_derived_roi_exists=True, uniquely_identified_and_eligible=False, paper_inclusion_verified=False,
              expanded_cohort_run=False, roster_evidence_sha256=sha(roster_path), inventory_evidence_sha256=sha(inventory_path),
              conclusion="Candidate acquisition exists, but the source tables do not verify identity and paper inclusion. The protected cohort was not expanded."),
         AUDIT / "Sixteenth_HC_Audit.json")

    availability = [
        ("Age", age, 65, "Confirmed acquisition sidecars and original roster lack consistent age data", "no"),
        ("Sex", sex, 65, "Confirmed acquisition sidecars and original roster lack consistent sex data", "no"),
        ("Mean FD", int(merged.fd.notna().sum()), 65, "Frozen V4 mean FD and independently archived scan-quality rows agree", "Model 1"),
        ("DVARS and standardized DVARS", provenance["scans_with_dvars"], 65, "All original confound sources had matching hashes. 13,585 defined points per measure", "diagnostic"),
        ("tSNR", 0, 65, "A comparable pre-normalization measure for the frozen ROI input was unavailable", "no"),
        ("Scan order", visits["matched_pairs"], 25, "Verified original DICOM time via linked scan sidecars", "order confounded with condition"),
        ("Medication timing", visits["verified_medication_intervals"], 50, "No verified last-dose timestamps or elapsed hours in linked visit audit", "no"),
        ("Scan interval", visits["matched_pairs"], 25, "OFF and ON DICOM timestamp difference", "diagnostic"),
    ]
    csv(pd.DataFrame(availability, columns=["field", "verified_units", "expected_units", "evidence", "use"]), AUDIT / "Covariate_Availability.csv")
    dump(dict(model_0="unadjusted", model_1="Mean FD", model_2="unavailable because individual age and sex remain incomplete",
              raw_dvars_archive_sha256=provenance["dvars_payload_sha256"], frozen_confounds_sha256_matches=65), AUDIT / "Model_Eligibility.json")


def recruitment(n):
    return math.ceil(n/(1-LOSS))


def independent_size(effect, target, ratio, solver):
    continuous = solver.solve_power(effect, alpha=ALPHA, power=target, ratio=ratio)
    hc = math.ceil(continuous)
    return hc, math.ceil(hc*ratio)


def power():
    independent, paired = TTestIndPower(), TTestPower()
    restoration = pd.read_csv(RESULT / "restoration/Restoration_Results.csv").iloc[0]
    measured = abs(restoration.dz)
    scenarios = [("standardized target", .35), ("standardized target", .40)]
    for origin, effect in [("Observed restoration |dz|", measured), ("Half observed restoration |dz|, capped at 0.35", min(.35, .5*measured))]:
        if effect>1e-5:
            scenarios.append((origin, effect))
    rows = []
    for origin, effect in scenarios:
        for target in (.80, .90):
            pair_count = math.ceil(paired.solve_power(effect, alpha=ALPHA, power=target))
            def add(design, hc, pd_count, achieved):
                n_hc, patients = recruitment(hc), recruitment(pd_count)
                rows.append(dict(effect_source=origin, standardized_effect=effect, design=design, intended_power=target,
                                 alpha=ALPHA, attrition=LOSS, retained_hc=hc, retained_pd_pairs=pd_count,
                                 retained_scans=hc+2*pd_count, recruit_hc=n_hc, recruit_pd_pairs=patients,
                                 recruit_scans=n_hc+2*patients, achieved_power=achieved))
            if origin=="standardized target":
                for design, ratio in [("Independent HC:PD 1:1",1.), ("Independent HC:PD 15:25",25/15)]:
                    n, m = independent_size(effect,target,ratio,independent)
                    add(design,n,m,independent.power(effect,n,ALPHA,ratio=m/n))
                hc, patients = independent_size(effect,target,1.,independent)
                joint_pd = max(patients,pair_count)
                add("Independent and paired goals",hc,joint_pd,min(independent.power(effect,hc,ALPHA,ratio=joint_pd/hc),
                                                                     paired.power(effect,joint_pd,ALPHA)))
            add("Paired ON/OFF",0,pair_count,paired.power(effect,pair_count,ALPHA))
    csv(pd.DataFrame(rows), POWER / "Sample_Size_Planning.csv")
    curve = [dict(hc=hc, pd_pairs=25, total_scans=hc+50, standardized_effect=effect, alpha=ALPHA,
                  power=independent.power(effect,hc,ALPHA,ratio=25/hc))
             for _, effect in scenarios[:2] for hc in (15,16,25,40,60,100,200,500)]
    csv(pd.DataFrame(curve), POWER / "Fixed_PD25_HC_Increase.csv")


if __name__ == "__main__":
    audit()
    power()
