"""Join the independent 48-scan fit to archived UPDRS-III. Emit grouped results only."""
import argparse
import io
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from clinical_archive_rerun import ENGLISH_DOMAINS, SHARE_FIELDS, VERSIONS, check_effects, clinical_visits
from stats import analyze, holm_adjust


def aggregate(payload, project, tmp):
    visits=pd.DataFrame(payload["visits"]).rename(columns={"person":"subject_id"})
    measures=pd.read_csv(io.StringIO(payload["metrics_tsv"]),sep="\t",na_values=["n/a"])
    if (set(measures.method)!={"AAHC","signed"} or len(measures)!=384 or
            measures.duplicated(["method","run_id","state"]).any() or
            set(measures.run_id)!=set(visits.run_id) or not visits.run_id.is_unique or not visits.scan_key.is_unique
            or len(visits)!=48 or set(visits.condition)!={"OFF","ON"} or visits.subject_id.nunique()!=24):
        raise ValueError("Expected both complete fitted models on exactly 48 selected visits")
    visits["cohort"]="DBS"
    visits["mapping_source"]="verified paired raw-scan identity and archived visit linkage"
    primary,duplicate,alternate=[clinical_visits(project,folder,series,payload["salt"]) for folder,series in VERSIONS]
    pd.testing.assert_series_equal(primary.score,duplicate.score)
    for table in (primary,duplicate):
        np.testing.assert_array_equal(table.score,table[list(ENGLISH_DOMAINS)].sum(axis=1))
    pd.testing.assert_frame_equal(primary[list(ENGLISH_DOMAINS)],alternate[list(ENGLISH_DOMAINS)],check_dtype=False)
    same=primary.score.eq(alternate.score)
    if not visits.scan_key.isin(primary.index).all() or not visits.scan_key.isin(alternate.index).all():
        raise ValueError("A fitted visit is missing from the clinical tables")
    by_person=visits.assign(same=visits.scan_key.map(same)).groupby("subject_id").same
    agreed=set(by_person.all().loc[lambda x:x].index)
    if len(agreed)!=17 or (by_person.any()!=by_person.all()).any():
        raise ValueError("Clinical version agreement differs from the source audit")
    rows,checked=[],[]
    for name,scores,subset in (("archive_main",primary.score,None),("archive_alternate",alternate.score,None),
                               ("concordant_pairs",primary.score,agreed)):
        meta=visits.copy()
        if subset is not None:
            meta=meta[meta.subject_id.isin(subset)].copy()
        meta["clinical_score"]=meta.scan_key.map(scores)
        meta["clinical_scale"]="UPDRS3"
        if meta.clinical_score.isna().any():
            raise ValueError("Missing clinical outcome")
        for method in ("AAHC","signed"):
            data=measures[measures.method.eq(method)&measures.run_id.isin(meta.run_id)]
            with TemporaryDirectory(dir=tmp) as work:
                counts=analyze(data,meta,work,k=4,clinical_scale="UPDRS3",n_permutations=10000,n_bootstrap=2000,
                               seed=20261121)
                result=pd.read_csv(Path(work)/"stats.csv",keep_default_na=False)
            if counts["n_eligible_pairs"]!=len(meta)//2 or counts["n_clinically_eligible_pairs"]!=len(meta)//2:
                raise ValueError("Unexpected number of eligible paired patients")
            if len(result)!=28:
                raise ValueError("An inference family is incomplete")
            if name!="archive_main":
                result=result[result.family.ne("on_off")]
            err,_=check_effects(data,meta,result)
            checked.append({"method":method,"scenario":name,"effects_checked":len(result),"max_abs_error":err})
            rows.extend(dict(scenario=name,method=method,**r) for r in result[list(SHARE_FIELDS)].to_dict("records"))
    out=pd.DataFrame(rows)
    for _,group in out.groupby(["scenario","family"],sort=False):
        out.loc[group.index,"p_holm_both_methods"]=holm_adjust(group.p_raw.to_numpy(dtype=float))
    if len(out)!=120 or not np.isfinite(out[["estimate","p_raw","p_holm","p_holm_both_methods"]]).all().all():
        raise ValueError("Incomplete numerical results")
    return {"statistics":out.to_dict("records"),"independent_effect_checks":checked,
            "environment":{"python":platform.python_version(),**{p:version(p) for p in ("numpy","scipy","pandas")}},
            "cohort":{"scored_pairs":24,"score_versions_concordant_pairs":17,"score_versions_discordant_pairs":7,
                      "individually_confirmed_scan_date_and_medication_pairs":0}}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--private-tmp",type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(aggregate(json.load(sys.stdin),args.project,args.private_tmp),allow_nan=False))
