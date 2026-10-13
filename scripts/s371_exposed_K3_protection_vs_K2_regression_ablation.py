"""S371 label-blind K3-protect ablation on ALREADY EXPOSED 36 WAV test data.

A deterministic counterfactual only: IF original official K3, source S18
parent K2, and final S80 remains K2, restore original baseline K3.
THIS IS NOT A RECOMMENDED FIX. Diagnose unavoidable loss of the
precious true K2 corrections, with all other classes, per cohort.
This is assessed post-hoc; never claim independent gain, no promotion.
"""
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
S18={'row_id','freeze_K','original_fold_ids','original_S18_by_fold_K',
      'S18_piece_neutral_consensus_K','S18_quorum_of_3_supported',
      'S18_vote_strength','S18_abstention_to_freeze'}
LABEL={'row_id','member','historical_attack_K_if_XML_complete',
      'acoustic_anchored_K_or_unknown','official_freeze_K',
      'S80_original_K','S80_twofold_K'}
def need(ok,msg):
    if not ok:raise ValueError(msg)
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def read(p,req):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==req,'original frozen native stage schema differs')
        return {k:z[k].copy() for k in z.files}
def ledger(y,base,candidate,mask):
    idx=np.flatnonzero(mask)
    if not len(idx):return dict(n=0)
    b=base[idx]==y[idx];a=candidate[idx]==y[idx]
    return dict(n=int(len(idx)),correct_baseline=int(b.sum()),
       correct_counterfactual=int(a.sum()),
       fixes=int((~b&a).sum()),regressions=int((b&~a).sum()),
       net=int(a.sum()-b.sum()))
def run(s18first,labfirst,s18second,labsecond,out):
    need(not out.exists(),'immutable experimental counterfactual audit')
    data=[];cases=[]
    for src,truthfile,n,label in [(s18first,labfirst,477,'S354_12'),
        (s18second,labsecond,967,'S363_24')]:
        a=read(src,S18);t=read(truthfile,LABEL)
        ids=a['row_id'];y=t['historical_attack_K_if_XML_complete']
        freeze=t['official_freeze_K']
        final=t['S80_original_K']
        parent=a['S18_piece_neutral_consensus_K']
        need(len(y)==n and np.array_equal(ids,t['row_id'])
             and np.array_equal(a['freeze_K'],freeze),
             'cannot match original native fourfold S18 consensus to scored frozen S80')
        # This test deliberately relies only on predictions, NOT true-K
        mask=(freeze==3)&(parent==2)&(final==2)
        alternative=final.copy();alternative[mask]=3
        perK={f'K{k}':ledger(y,final,alternative,y==k) for k in range(7)}
        totals=ledger(y,final,alternative,np.ones(n,bool))
        vsfreeze=ledger(y,freeze,alternative,np.ones(n,bool))
        affected=np.flatnonzero(mask)
        data.append(dict(group=label,n=n,trigger_count=len(affected),
            triggered_trueK=np.bincount(y[affected],minlength=7).tolist(),
            baseline_S80_vs_protected_counterfactual=totals,
            alternative_vs_official_freeze=vsfreeze,
            S80_to_K3_reversions_per_trueK=perK))
        for i in affected:
            cases.append(dict(cohort=label,member_sha=str(t['member'][i]),
              original_ID=str(ids[i]),original_trueK=int(y[i]),
              original_reference_K=3,original_S18_K=2,
              original_S80_K=2,counterfactual_K=3,
              original_4fold_S18_K=list(map(int,a['original_S18_by_fold_K'][:,i]))))
    relevant=np.bincount(np.array([c['original_trueK'] for c in cases],int),minlength=7).tolist()
    need(sum(x['trigger_count'] for x in data)==len(cases)
         and relevant[3]>=6 and relevant[2]>=0,'expected six original K3 lost cases should be in counterfactual trigger')
    report=dict(status='S371_EXPOSED_36_WAV_K3_BASELINE_PROTECTION_COUNTERFACTUAL_NOT_PROMOTABLE',
      sources_sha256={p.name:sha(p) for p in [s18first,labfirst,s18second,labsecond]},
      original_native_events=1444,original_wavs=36,
      trigger_definition='original freeze K3 and source S18 K2 and full S80 K2 -> revert to freeze K3',
      nontruth_trigger_model_policy_precommitted_before_scoring_within_this_RUN=True,
      trigger_sample_total=len(cases),
      trueK_histogram_of_triggered_events=relevant,
      each_original_cohort=data,
      triggered_change_ids_readonly=cases,
      trueK3_restorations_expected=6,
      source_prelabel_frozen_models_modified=False,
      any_new_training=False,
      validation_xml_previously_exposed=True,
      rule_would_require_new_independent_future_poly_test=True,
      official_S80_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='triggered_change_ids_readonly'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s18first','labfirst','s18second','labsecond','out'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();run(a.s18first,a.labfirst,a.s18second,a.labsecond,a.out)
