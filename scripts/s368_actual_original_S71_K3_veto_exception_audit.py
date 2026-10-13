"""S368: exact ORIGINAL K3→K2 regression cause in S69→S72 routing.

Read 24 Dataset1 WAVs already exposed by S363 and their unmodified
pre-XML S352 final K. Independently replay saved trained S69,S70,S71,S72
source checkpoints to reproduce S352's exact s72_K array before
comparing ANY ground truth. Posthoc compare original *risk* vs *poly*
S71/S72 routes, especially intentional suppression of S71 veto for
parent=K3, S70=K2. Never fit, change final S80 or promote on exposed XML.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
LAB={'row_id','member','historical_attack_K_if_XML_complete',
     'acoustic_anchored_K_or_unknown','official_freeze_K',
     'S80_original_K','S80_twofold_K'}
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
STAGE={'row_id','s69_polymax_K','s70_proposed_poly_K',
       's71_risk_K','s71_poly_K','s71_veto_three_folds',
       's72_veto_three_folds','s71_fold_veto','s72_fold_veto',
       'original_pr16_geometry46','s72_risk_all_K',
       's72_risk_poly_guard_K','s72_poly_all_K',
       's72_poly_poly_guard_K'}
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'different genuine historical checkpoint archive schema')
        return {k:z[k].copy() for k in z.files}
def row(x,i,y,official):
    keys=('s69_polymax_K','s70_proposed_poly_K','s71_risk_K','s71_poly_K',
          's72_risk_all_K','s72_risk_poly_guard_K',
          's72_poly_all_K','s72_poly_poly_guard_K')
    r={k:int(x[k][i]) for k in keys}
    r.update(row_id=str(x['row_id'][i]),true_K=int(y[i]),
             freeze_original_K=int(official[i]),
             S71_threefold_veto=bool(x['s71_veto_three_folds'][i]),
             S72_threefold_veto=bool(x['s72_veto_three_folds'][i]),
             S71_folds_veto_count=int(x['s71_fold_veto'][:,i].sum()),
             S72_folds_veto_count=int(x['s72_fold_veto'][:,i].sum()))
    return r
def tally(k,y,mask):
    return dict(n=int(mask.sum()),
                correct=int(np.count_nonzero(k[mask]==y[mask])),
                predicted_K_hist=np.bincount(k[mask].astype(int),minlength=7).tolist())
def run(stages,original_labels,final,out):
    need(not out.exists(),'immutable forensic on same SHAs and native V88 row IDs')
    src=load(stages,STAGE)
    truth=load(original_labels,LAB)
    old=load(final,PRED)
    y=truth['historical_attack_K_if_XML_complete']
    b=truth['official_freeze_K']
    z=old['s80_K']
    need(len(y)==967 and np.array_equal(truth['row_id'],src['row_id'])
         and np.array_equal(truth['row_id'],old['row_id'])
         and np.array_equal(src['s72_poly_poly_guard_K'],old['s72_K'])
         and np.array_equal(b,old['incumbent_freeze_K'])
         and np.array_equal(z,truth['S80_original_K']),
         'original saved S80 preXML final K does not bit-match replayed ORIGINAL S72 chain')
    mask_k3=(y==3)
    lost=mask_k3&(b==3)&(z!=3)
    need(lost.sum()==6 and np.all(z[lost]==2),'S367 six K3->K2 cases not reconstructed')
    parent3to2=(src['s69_polymax_K']==3)&(src['s70_proposed_poly_K']==2)
    suppressed=parent3to2 & src['s71_veto_three_folds'] & (src['s71_risk_K']==3) & (src['s71_poly_K']==2)
    lossrows=np.flatnonzero(lost)
    fixrows=np.flatnonzero((y==2)&(b!=2)&(z==2))
    summary={
       'K3_regressions_total':int(lost.sum()),
       'K3_parent_S69_three':int(np.count_nonzero(lost&(src['s69_polymax_K']==3))),
       'K3_original_S70_proposed_two':int(np.count_nonzero(lost&(src['s70_proposed_poly_K']==2))),
       'K3_S71_veto_suppressed_on_parent3to2':int(np.count_nonzero(lost&suppressed)),
       'K3_S71_risk_restores_correct_three':int(np.count_nonzero(lost&(src['s71_risk_K']==3))),
       'K3_S71_poly_keeps_wrong_two':int(np.count_nonzero(lost&(src['s71_poly_K']==2))),
       'K3_S72_risk_guard_restores_correct_three':int(np.count_nonzero(lost&(src['s72_risk_poly_guard_K']==3))),
       'K3_S72_poly_guard_keeps_wrong_two':int(np.count_nonzero(lost&(src['s72_poly_poly_guard_K']==2))),
       'all_original_source_parent3to2_rows':int(parent3to2.sum()),
       'all_source_S71_k3to2_veto_overridden_rows':int(suppressed.sum())}
    by_K={}
    for K in [0,1,2,3,4,5,6]:
        mask=y==K
        by_K[f'K{K}']={
            tag:tally(src[key],y,mask) for tag,key in (
                ('S69','s69_polymax_K'),('S70','s70_proposed_poly_K'),
                ('S71_risk','s71_risk_K'),('S71_poly','s71_poly_K'),
                ('S72_risk_poly_guard','s72_risk_poly_guard_K'),
                ('S72_poly_poly_guard','s72_poly_poly_guard_K'))}
    result=dict(status='S368_EXACT_SOURCE_S69_S70_S71_S72_K3_VETO_DESIGN_AUDIT',
        original_src_S351_sha256=sha(stages),
        original_postfrozen_S363_truth_sha256=sha(original_labels),
        original_original_S352_prelabel_preds_sha256=sha(final),
        original_S72_stage_matches_prelabel_original_S80_archive=True,
        true_K3_regression_source_summary=summary,
        regression_K3_row_evidence=[row(src,int(i),y,b) for i in lossrows],
        corrected_K2_row_evidence=[row(src,int(i),y,b) for i in fixrows],
        original_stage_accuracy_by_true_K_if_XML_complete=by_K,
        source_specific_design_exception_from_S351='veto71poly=veto71 & ~((parent==3)&(s70==2))',
        source_code_modified=False,original_frozen_candidate_modified=False,
        true_labels_are_EXPOSED_NOT_INDEPENDENT_for_design_selection=True,
        K3_fix_performance_if_deployed_on_new_songs_unverified=True,
        official_promotion_not_authorized=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()
          if k not in ('regression_K3_row_evidence',
                       'corrected_K2_row_evidence','original_stage_accuracy_by_true_K_if_XML_complete')},
           indent=2),flush=True)
    print('K3_SOURCE',json.dumps(by_K['K3'],indent=2),flush=True)
    print('K2_SOURCE',json.dumps(by_K['K2'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s351','truth','original','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.s351,a.truth,a.original,a.out)
