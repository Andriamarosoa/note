"""S370 trace six TRUE K3 errors to genuine S5/S8/S9/S13/S14/S15/S16/S17/S18 heads.

This audit uses the SHA-frozen prelabel model stage artifacts with the
already EXPOSED 24 WAV labels. Compare all four trained heldfold raw S5
probabilities and original risk cascade outputs, do not train or apply
a correction tuned on seen labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
STAGES=['S5','S8','S9','S13','S14','S15','S16','S17','S18']
SRC5={'row_id','incumbent_freeze_K','fold_ids',
      'original_S5_raw_probability_by_fold',
      'original_S5_prior_probability_by_fold',
      'original_S5_K_by_fold','all_four_folds_agree'}
TRUTH={'row_id','member','historical_attack_K_if_XML_complete',
       'acoustic_anchored_K_or_unknown','official_freeze_K',
       'S80_original_K','S80_twofold_K'}
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
def load(p,expected=None):
    with np.load(p,allow_pickle=False) as z:
        if expected is not None:need(set(z.files)==expected,'SHA frozen model output schema drift')
        return {k:z[k].copy() for k in z.files}
def frequency(k):return np.bincount(np.asarray(k,dtype=int),minlength=7).tolist()
def run(s5,s18,truth,out):
    need(not out.exists(),'S370 forensic immutability')
    a=load(s5,SRC5);b=load(s18);c=load(truth,TRUTH)
    y=c['historical_attack_K_if_XML_complete']
    frozen=c['official_freeze_K'];model=c['S80_original_K']
    n=len(y)
    need(n==967 and all(np.array_equal(a['row_id'],x['row_id']) for x in (b,c))
         and np.array_equal(a['incumbent_freeze_K'],frozen)
         and np.array_equal(b['freeze_K'],frozen)
         and a['original_S5_K_by_fold'].shape==(4,n),
         'S370 source native prelabel K0..K6 alignment drift')
    values={'S5':a['original_S5_K_by_fold']}
    for k in STAGES[1:]:
        t=b[k+'_K']
        need(t.shape==(4,n),f'actual four ORIGINAL folds missing {k}')
        values[k]=t
    wrong=(y==3)&(frozen==3)&(model==2)
    need(int(wrong.sum())==6,'S369 six K3->K2 root events do not match original labels')
    true2=(y==2)&(frozen!=2)&(model==2)
    rows=[]
    for i in np.flatnonzero(wrong|true2):
        pred={k:list(map(int,v[:,i])) for k,v in values.items()}
        stage_first=None
        for k in STAGES:
            if sum(x==2 for x in pred[k])>=3:
                stage_first=k
                break
        rows.append(dict(row_id=str(a['row_id'][i]),true_K=int(y[i]),freeze_K=int(frozen[i]),
             S80_K=int(model[i]),class_of_audit='K3_regression' if wrong[i] else 'K2_correction',
             first_strong_K2_majority_stage=stage_first,
             four_fold_K_by_original_stage=pred,
             S5_four_fold_original_raw_K3_probability=
               list(map(float,a['original_S5_raw_probability_by_fold'][:,i,3])),
             S5_four_fold_original_raw_K2_probability=
               list(map(float,a['original_S5_raw_probability_by_fold'][:,i,2]))))
    byclass={}
    for k in (2,3):
        mask=y==k
        byclass['K'+str(k)]={}
        for stage,seq in values.items():
            pred=seq[:,mask]
            byclass['K'+str(k)][stage]=dict(
                four_fold_each_accuracy=[float(np.mean(z==k)) for z in pred],
                original_fourfold_majority_for_trueK=int(np.sum((pred==k).sum(0)>=3)),
                unanimous_correct_trueK=int(np.sum((pred==k).sum(0)==4)),
                four_fold_per_model_K_histogram=[frequency(z) for z in pred])
    report=dict(status='S370_EARLIEST_REAL_S5_TO_S18_K3_UNDERCOUNT_CAUSE_READONLY',
       original_trained_S5_output_SHA256=sha(s5),
       original_133_S8_S18_output_SHA256=sha(s18),
       original_postfrozen_XML_SHA256=sha(truth),
       total_orig_native_rows=n,
       six_K3_regressions=int(wrong.sum()),
       K2_fixes_on_this_new24=int(true2.sum()),
       stage_summary_by_true_class=byclass,
       per_contrasting_K2_K3_cases=rows,
       no_model_or_threshold_changed=True,
       label_set_already_exposed_not_independent_for_feature_design=True,
       official_promotion=False)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
          ('per_contrasting_K2_K3_cases','stage_summary_by_true_class')},indent=2))
    print('EACH ORIGINAL STAGE',json.dumps([{k:v for k,v in row.items()
          if k not in ('row_id','S5_four_fold_original_raw_K3_probability',
          'S5_four_fold_original_raw_K2_probability')} for row in rows],indent=2))
    print('K2/K3 CASCADE',json.dumps(byclass,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s5','s18','truth','out'):
        p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();run(a.s5,a.s18,a.truth,a.out)
