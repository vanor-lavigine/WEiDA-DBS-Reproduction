"""Export and check group-level results of the clinical EEG comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


def export(result, out):
    out.mkdir(parents=True, exist_ok=True)
    tests = pd.DataFrame(result['tests'])
    paired = pd.DataFrame(result['paired'])
    assert len(tests)==84 and len(paired)==24 and tests.n.eq(25).all()
    assert not tests.duplicated(['variant','contrast','state','metric']).any()
    assert not paired.duplicated(['variant','state','metric']).any()
    for _, family in tests.groupby('variant'):
        assert len(family)==42
        ranked = family.sort_values('p_raw')
        unadjusted = 2*t.sf(np.abs(ranked.r)*np.sqrt(23/(1-ranked.r**2)),23)
        np.testing.assert_allclose(unadjusted,ranked.p_raw,rtol=0,atol=2e-14)
        bh = np.minimum.accumulate((ranked.p_raw.to_numpy()*42/np.arange(1,43))[::-1])[::-1]
        np.testing.assert_allclose(bh,ranked.q_bh42,atol=2e-14)
    for _, family in paired.groupby('variant'):
        ranked = family.sort_values('p')
        assert len(ranked)==12
        holm = np.minimum(1,np.maximum.accumulate(ranked.p.to_numpy()*np.arange(12,0,-1)))
        np.testing.assert_allclose(holm,ranked.p_holm12,atol=2e-14)
        np.testing.assert_allclose(family.on-family.off,family.delta,atol=1e-12)
    tests.to_csv(out/'clinical.csv',index=False)
    paired.to_csv(out/'paired.csv',index=False)
    pd.DataFrame(result['templates'],columns=result['axis'],index=['A1','A2','A3','A4']).to_csv(out/'templates.tsv',sep='\t',index_label='state')
    qa={k:result[k] for k in ('summary','exposure','peaks','fit','primary_sensitivity','environment')}
    (out/'verification.json').write_text(json.dumps(qa,indent=2,allow_nan=False)+'\n',encoding='utf8')
    fig, ax = plt.subplots(1,2,figsize=(8.5,3.4),sharey=True)
    fig.subplots_adjust(left=.215,right=.96,bottom=.25,top=.77,wspace=.17)
    fig.suptitle('DBS OFF motor severity and state dynamics',fontweight='bold',fontsize=15)
    for panel,(metric,title,ref) in zip(ax,[('duration_sec','Mean episode duration',.68),('occurrence_per_sec','Total occurrences per second',-.76)]):
        panel.set_title(title,fontsize=11,loc='left')
        panel.axvline(0,color='#7d8287',linestyle='--',lw=.9)
        panel.plot(ref,2,'D',color='#705739')
        for i,(mode,color) in enumerate([('peak','#086b72'),('direct','#52648a')]):
            row=tests[(tests.variant==mode)&(tests.state=='all')&(tests.contrast=='off_severity')&(tests.metric==metric)].iloc[0]
            panel.errorbar(row.r,1-i,xerr=[[row.r-row.ci_low],[row.ci_high-row.r]],fmt='o',capsize=3,color=color)
        panel.set_xlim(-1,1)
        panel.set_ylim(-.45,2.45)
        panel.set_yticks([2,1,0],['Published EEG (20)','105 peak-based (25)','105 all-frame (25)'])
        panel.set_xlabel('Pearson r')
        panel.spines[['top','right','left']].set_visible(False)
        panel.tick_params(axis='y',length=0)
    fig.text(.03,.045,'EEG points from Li et al. (2021), Figure 4. BOLD bars show pointwise 95% confidence intervals.',fontsize=9)
    fig.savefig(out/'off_correlations.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aggregate',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    export(json.loads(args.aggregate.read_text(encoding='utf-8')),args.output)
