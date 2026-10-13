"""S367 K3 regression provenance audit across TWO SHA frozen independent-file cohorts.

Inspect actual S72→S79→S80 stage predictions on each already labelled
V88 row. Compare K2 corrections versus K3 regressions, and genuine
original S147 K0/K2 action flags. READ ONLY: do not tune thresholds,
reclassify XML, refit or promote based on exposed test labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
LAB={'row_id','member','historical_attack_K_if_XML_complete','acoustic_anchored_K_or_unknown','official_freeze_K','S80_original_K','S80_twofold_K'}
PRED={'row_id','incumbent_freeze_K','s72_K','s79_K','s80_K',
      'original_PR16_policy22_K','K1_by_fold_threshold_action',
      'K1_by_fold_threshold_scores','action_K1','action_K0','action_K2'}
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path,expected):
    with np.load(path,allow_pickle=False) as z:
        need(set(z.files)==expected,'unverified saved label/prediction archive schema')
        return {k:z[k].copy() for k in z.files}
def hist(x):return np.bincount(np.asarray(x,dtype=int),minlength=7).tolist()
def score(a,truth,mask):
    i=np.flatnonzero(mask)
    return dict(n=len(i),correct=int(np.count_nonzero(a[i]==truth[i])),
        predicted_K0toK6=hist(a[i]))
def row_snapshot(cohort,truth,ref,d,i):
    return dict(cohort=cohort,row_id=str(d['row_id'][i]),
        true_K=int(truth[i]),freeze_K=int(ref[i]),
        original_S72_K=int(d['s72_K'][i]),
        original_S79_K=int(d['s79_K'][i]),
        original_S80_K=int(d['s80_K'][i]),
        original_PR16_policy22_K=int(d['original_PR16_policy22_K'][i]),
        final_S147_K1_action=bool(d['action_K1'][i]),
        final_S147_K0_action=bool(d['action_K0'][i]),
        final_S147_K2_action=bool(d['action_K2'][i]))
def run(a_lab,a_pred,b_lab,b_pred,out):
    need(not out.exists(),'S367 immutable fixed-prospective cohorts only')
    records=[]
    summaries=[]
    for group,labelpath,predpath,n in [
       ('S354_twelve_first',a_lab,a_pred,477),
       ('S363_twentyfour_next',b_lab,b_pred,967)]:
        labels=load(labelpath,LAB)
        p=load(predpath,PRED)
        y=labels['historical_attack_K_if_XML_complete']
        ref=labels['official_freeze_K']
        final=labels['S80_original_K']
        need(len(y)==n and np.array_equal(labels['row_id'],p['row_id'])
             and np.array_equal(ref,p['incumbent_freeze_K'])
             and np.array_equal(final,p['s80_K']),
             'S367 cannot match original unlabelled S80 archived pre-XML SHA event IDs')
        k3=y==3
        k2=y==2
        reg3=k3&(ref==3)&(final!=3)
        fixes2=k2&(ref!=2)&(final==2)
        k4_reg=(y==4)&(ref==4)&(final!=4)
        for i in np.flatnonzero(reg3|fixes2|k4_reg):
            rec=row_snapshot(group,y,ref,p,i)
            rec['category']=('K3_regression' if reg3[i] else
                              'K2_correction' if fixes2[i] else 'K4_regression')
            records.append(rec)
        summaries.append(dict(cohort=group,source_label_sha256=sha(labelpath),
          original_S80_preXML_archive_SHA256=sha(predpath),
          rows=n,
          truth_hist_K0toK6=hist(y),
          K2_fixes=int(fixes2.sum()),K3_regressions=int(reg3.sum()),
          K4_regressions=int(k4_reg.sum()),
          stages_by_trueK={
            f'K{k}':{stage:score(p[field],y,y==k) for stage,field in
                 [('freeze','incumbent_freeze_K'),('S72','s72_K'),
                  ('S79','s79_K'),('S80','s80_K')]}
            for k in [1,2,3,4,5,6]},
          K3_regression_source=dict(
            regressed_n=int(reg3.sum()),
            S72_was_K3=int(np.count_nonzero(reg3&(p['s72_K']==3))),
            S79_was_K3=int(np.count_nonzero(reg3&(p['s79_K']==3))),
            S72_was_K2=int(np.count_nonzero(reg3&(p['s72_K']==2))),
            S79_was_K2=int(np.count_nonzero(reg3&(p['s79_K']==2))),
            changed_S72_to_S79=int(np.count_nonzero(reg3&(p['s72_K']!=p['s79_K']))),
            changed_S79_to_S80=int(np.count_nonzero(reg3&(p['s79_K']!=p['s80_K']))),
            tail_final_K2_action=int(np.count_nonzero(reg3&p['action_K2'])),
            tail_final_K0_action=int(np.count_nonzero(reg3&p['action_K0'])),
            tail_final_K1_action=int(np.count_nonzero(reg3&p['action_K1'])))))
    k3rows=[v for v in records if v['category']=='K3_regression']
    k2rows=[v for v in records if v['category']=='K2_correction']
    k4rows=[v for v in records if v['category']=='K4_regression']
    need(len(k3rows)==6 and len(k2rows)==15 and len(k4rows)==1,
         'original S366 per-K K2 +13 and K3 −6 evidence corrupted')
    r=dict(status='S367_ORIGINAL_S72_S79_S80_FINAL_REAL_K3_REGRESSIONS',
       original_frozen_cohorts=summaries,
       pooled_K2_corrections=len(k2rows),
       pooled_K3_regressions=len(k3rows),
       pooled_K4_regressions=len(k4rows),
       K3_regression_final_K_histogram=hist([x['original_S80_K'] for x in k3rows]),
       K3_regressions_already_K3_at_S72=sum(x['original_S72_K']==3 for x in k3rows),
       K3_regressions_already_K3_at_S79=sum(x['original_S79_K']==3 for x in k3rows),
       K3_regressions_due_to_final_S79_to_S80_change=sum(
           x['original_S79_K']!=x['original_S80_K'] for x in k3rows),
       per_changed_case=records,
       historic_training_K_definition='20ms new note attack count per V88 group',
       original_S147_tail_or_upstream_stage_isolated_posthoc=True,
       no_postlabel_model_weights_or_gates_tuned=True,
       cannot_validate_K3_fix_on_exposed_audio=True,
       official_reference_unchanged=True)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in
            ('original_frozen_cohorts','per_changed_case')},indent=2),flush=True)
    print('STAGES',json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('first-labels','first-model','second-labels','second-model','out'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    run(a.first_labels,a.first_model,a.second_labels,a.second_model,a.out)
