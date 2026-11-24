"""Fit EEG-style amplitude states and clinical contrasts in the 25-pair fMRIpost cohort."""
from collections import Counter
from itertools import groupby
import argparse
import json
from pathlib import Path
import platform
import re
import secrets

import numpy as np
import pandas as pd
import scipy

from clinical_archive_rerun import clinical_visits, ENGLISH_DOMAINS, VERSIONS, scan_key
from core import sequence_metrics
from li_clinical_stats import analyze
from peak_states import prepare_scan, group_maps, interpolate_peaks
from stats import METRICS, _bootstrap_mean, _rng, holm_adjust, signflip_pvalue


def paired_effects(metrics, visits):
    data = metrics.merge(visits[['run_id','person','condition']], validate='many_to_one')
    results = []
    for variant, subset in data.groupby('variant'):
        family = []
        for state, section in subset.groupby('state'):
            for j, metric in enumerate(METRICS):
                pairs = section.pivot(index='person', columns='condition', values=metric).dropna()
                delta = (pairs.ON - pairs.OFF).to_numpy()
                low, high, _ = _bootstrap_mean(delta, 2000, _rng(20261121,1,state,j,1))
                family.append(dict(variant=variant, state=int(state), metric=metric, n=len(delta),
                                   off=pairs.OFF.mean(), on=pairs.ON.mean(), delta=delta.mean(),
                                   ci_low=low, ci_high=high,
                                   p=signflip_pvalue(delta,10000,_rng(20261121,1,state,j,0))))
        for row, adjusted in zip(family, holm_adjust([r['p'] for r in family])):
            row['p_holm12'] = adjusted
        results.extend(family)
    return results


def run(scans, motion, project, salt, axis):
    names = sorted(scans)
    identities = [re.fullmatch(r'sub-STN(off|on)(\d+)', n, re.I) for n in names]
    if len(names)!=50 or not all(identities):
        raise ValueError('Expected all 50 STN scans')
    visits = pd.DataFrame([dict(run_id=name, person=m[2], condition=m[1].upper()) for name,m in zip(names,identities)])
    if not visits.groupby('person').condition.apply(lambda s: sorted(s)==['OFF','ON']).all():
        raise ValueError('OFF and ON must be present exactly once per participant')
    if len({np.ascontiguousarray(x).tobytes() for x in scans.values()}) != 50:
        raise ValueError('Duplicate input time series')
    prepared = [prepare_scan(scans[n], motion[n]) for n in names]
    centers, fit = group_maps([s.individual for s in prepared], np.ones(len(names)))
    metrics, exposure = [], []
    for name, scan, cond in zip(names, prepared, visits.condition):
        valid = scan.gfp>0
        if not valid.all(): raise ValueError('Unexpected invalid input frame')
        similarity = np.abs(scan.units @ centers.T)
        direct = similarity.argmax(axis=1)
        peak, _ = interpolate_peaks(len(direct), scan.peaks, direct[scan.peaks], valid, similarity)
        for variant, labels in [('peak',peak),('direct',direct)]:
            measures = sequence_metrics(labels, 2.1, 4)
            retained = measures['n_valid_frames']
            episodes = Counter(s for s,_ in groupby(labels) if s>=0)
            for row in measures['states']:
                state = row['state']-1
                if row['n_episodes']!=episodes[state] or row['n_state_frames']!=int(np.sum(labels==state)):
                    raise ValueError('Independent episode counter disagrees')
                metrics.append(dict(variant=variant, run_id=name, n_valid_frames=retained, mean_fd=motion[name], **row))
            exposure.append(dict(variant=variant,condition=cond,retained=int(retained),episodes=sum(episodes.values())))
    # Join scores only after estimating all group maps, assignments and temporal metrics.
    primary, duplicate = [clinical_visits(project, folder, series, salt) for folder,series in VERSIONS[:2]]
    pd.testing.assert_series_equal(primary.score,duplicate.score)
    np.testing.assert_array_equal(primary.score,primary[list(ENGLISH_DOMAINS)].sum(axis=1))
    visits['score'] = [primary.score.at[scan_key(n,salt)] for n in names]
    scores = visits.pivot(index='person',columns='condition',values='score')
    measured = pd.DataFrame(metrics)
    result = analyze(measured,visits,expected_patients=25)
    result.update(paired=paired_effects(measured,visits),score_means=dict(off=scores.OFF.mean(),on=scores.ON.mean()),
                  templates=centers.tolist(),axis=axis,
                  exposure=pd.DataFrame(exposure).groupby(['variant','condition']).sum().reset_index().to_dict('records'),
                  peaks=sum(len(s.peaks) for s in prepared),fit=fit)
    return result


def read_archive(bids_root, project):
    packed_dir = project/'data'/VERSIONS[0][1]
    packed_files = sorted(packed_dir.glob('sub-STN*_BOLD_AAL1+PD25_MNI_basic36_sm6.csv'))
    labels = [p.name.split('_')[0] for p in packed_files]
    if len(set(labels))!=50 or len(labels)!=50:
        raise ValueError('Expected 50 scored STN source files')
    for encoding in ('utf-8-sig','gb18030'):
        try:
            roster = pd.read_csv(bids_root/'code/demographics_Chs.csv',encoding=encoding).set_index('subject_id')
            break
        except UnicodeDecodeError:
            if encoding=='gb18030': raise
    roster.index = roster.index.str.lower().str.removeprefix('sub-')
    selected = roster.loc[[n.lower().removeprefix('sub-') for n in labels]]
    if selected.name.isna().any() or selected.name.nunique()!=25:
        raise ValueError('Incomplete archived identity register')
    for _, pair in selected.groupby('name'):
        if sorted(pair.DBS_status.str.upper())!=['OFF','ON'] or len(set(pair.index.str.extract(r'(\d+)$',expand=False)))!=1:
            raise ValueError('Patient identity or visit pairing disagrees with archive')
        for label,stored in zip(pair.index,pair.DBS_status):
            if stored.lower() not in label: raise ValueError('Stimulation label differs from register')
    axis = pd.read_csv(packed_files[0],index_col=0,nrows=0).columns.tolist()
    if len(axis)!=78 or len(set(axis))!=78:
        raise ValueError('Expected 78 distinct AAL regions')
    scans, motion, copy_checks = {}, {}, []
    for label, packed_path in zip(labels,packed_files):
        path = bids_root/'derivatives/fmripost'/label/(label+'_BOLD_AAL1ctx_MNI_basic36_sm6.csv')
        frame = pd.read_csv(path,index_col=0,float_precision='round_trip')
        packed = pd.read_csv(packed_path,index_col=0,float_precision='round_trip')
        raw, = (bids_root/label/'func').glob('*_bold.json')
        confounds, = (bids_root/'derivatives/fmriprep'/label/'func').glob('*desc-confounds_timeseries.tsv')
        conf = pd.read_csv(confounds,sep='\t')
        if json.loads(raw.read_text())['RepetitionTime']!=2.1 or list(frame.columns)!=axis or list(packed.columns)!=axis:
            raise ValueError('Unexpected region axis or TR')
        np.testing.assert_array_equal(frame.index,np.arange(len(frame)))
        fd = conf.framewise_displacement.to_numpy()
        if len(fd)!=210 or not np.isnan(fd[0]) or not np.isfinite(fd[1:]).all() or (fd[1:]<0).any():
            raise ValueError('Unexpected motion record')
        missing = 210-len(frame)
        excluded = np.flatnonzero(conf.filter(regex='^non_steady_state_outlier').any(axis=1)).tolist()
        if missing not in (0,1) or excluded!=list(range(missing)) or len(packed)!=210:
            raise ValueError('Frame support disagrees with the non-steady-state record')
        x = frame.to_numpy()
        if not np.isfinite(x).all() or (np.std(x,axis=0)==0).any(): raise ValueError('Invalid ROI time series')
        copy_checks.append(dict(frames=len(x),max_abs_diff_after_start_trim=float(abs(packed.to_numpy()[missing:]-x).max()),
                                max_abs_diff_after_end_trim=float(abs(packed.to_numpy()[:len(x)]-x).max())))
        scans[label],motion[label]=x,float(fd[1:].mean())
    return scans, motion, axis, copy_checks


def analyze_archive(bids_root, project):
    scans, motion, axis, copy_checks = read_archive(bids_root, project)
    result = run(scans,motion,project,secrets.token_hex(32),axis)
    result.update(summary=dict(patients=25,visits=50,frames=sum(map(len,scans.values())),
                               lengths=dict(Counter(map(len,scans.values()))),data_kind='real',tr_seconds=2.1),
                  copy_checks=copy_checks,environment={'python':platform.python_version(),'numpy':np.__version__,
                                                       'scipy':scipy.__version__,'pandas':pd.__version__})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-root',type=Path,required=True)
    parser.add_argument('--clinical-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(analyze_archive(args.bids_root,args.clinical_root),allow_nan=False))
