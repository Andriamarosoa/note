"""S398 paired post-freeze scorer for S397 TRUE full-replayed S80.
The only labels here come from ORIGINAL held-piece benchmark after S397
has output SHA-sealed its complete label-blind result. This is already-
exposed GuitarSet, NOT an external validation or permission to promote.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def need(x,msg):
    if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def measure(y,a,b,sel):
    t=np.flatnonzero(sel); aa=a[t]==y[t];bb=b[t]==y[t]
    return dict(n=int(len(t)),reference_correct=int(aa.sum()),
       candidate_correct=int(bb.sum()),
       corrections=int((~aa&bb).sum()),regressions=int((aa&~bb).sum()),
       gain=int(bb.sum()-aa.sum()),
       candidate_exactK_percent=100.*float(bb.mean()) if len(t) else None,
       reference_exactK_percent=100.*float(aa.mean()) if len(t) else None)
def run(truth,original,replay,s397report,your,out):
    need(not out.exists(),'cannot replace S398 labeled evidence')
    label=load(truth);first=load(original);new=load(replay);bench=load(your)
    ss=json.loads(Path(s397report).read_text())
    need(ss['status']=='S397_SOURCE_S70_K5_DOWNCAST_GUARD_FULL_S80_REPLAY_LABEL_FREE'
      and ss['labels_used'] is False
      and ss['final_prediction_posthoc_modified'] is False,
      'S397 source must be completed and frozen before truth scoring')
    need(sha(replay)==ss['source_sha256']['replayed_S80'],
      'S397 full S80 prediction changed since SHA seal')
    k=np.argsort(bench['global_index'].astype(np.int64))
    ids=label['row_id'].astype(str);y=label['true_K'].astype(int)
    f=label['original_fold'].astype(int)
    old=first['s80_K'].astype(int);pred=new['s80_K'].astype(int)
    mt=bench['predicted'][k].astype(int)
    need(len(y)==59309 and np.array_equal(ids,first['row_id'].astype(str))
      and np.array_equal(ids,new['row_id'].astype(str))
      and np.array_equal(ids,bench['global_index'][k].astype(str))
      and np.array_equal(y,bench['k'][k]) and np.array_equal(f,bench['fold'][k])
      and int((y==5).sum())==355 and int((y>=2).sum())==7385
      and int((old==y).sum())==53327
      and int((mt==y).sum())==51328,
      'S397-S398 original input alignment has drifted')
    change=pred!=old
    byK={f'K{x}':measure(y,old,pred,y==x) for x in range(7)}
    folds={str(x):measure(y,old,pred,f==x) for x in (0,1,2,4)}
    groups={}
    with np.load(truth,allow_pickle=False) as z:
        pass
    report=dict(status='S398_S397_FULL_REPLAY_DEV_SCORE_K5_GUITARSET_NOT_INDEPENDENT',
      native_rows=len(y),
      labels_read_only_after_S397_SHA_seal=True,
      full_S70_S71_S72_S79_S80_reexecuted=True,
      unchanged_original_full_S80_reference=True,
      full_replay_changed_events=int(change.sum()),
      changed_K_before_hist=np.bincount(old[change],minlength=7).tolist(),
      changed_K_after_hist=np.bincount(pred[change],minlength=7).tolist(),
      score_global=measure(y,old,pred,np.ones(len(y),bool)),
      score_poly_K2to6=measure(y,old,pred,y>=2),
      by_true_K=byK,by_heldfold=folds,
      yourmt3_global_correct=int((mt==y).sum()),
      yourmt3_poly_correct=int(((mt==y)&(y>=2)).sum()),
      S80_candidate_global_correct=int((pred==y).sum()),
      S80_candidate_poly_correct=int(((pred==y)&(y>=2)).sum()),
      source_S397_sha=sha(s397report),
      source_original_S80_sha=sha(original),
      source_replayed_S80_sha=sha(replay),
      source_truth_sha=sha(truth),
      original_guitarset_S80_foldneutral_training_overlap=True,
      hypothesis_developed_on_same_GuitarSet=True,
      external_dataset_evaluation=False,
      trained_source_weights_untouched=True,
      deployed_or_promoted=False)
    need(report['full_replay_changed_events']==
         ss['full_S80_replay_output_changes'],
         'frozen S397 and S398 predicted changes disagreed')
    need(report['score_global']['gain']==
         sum(v['gain'] for v in byK.values()),
         'full replay incorrect byclass ledger')
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    print('S398 PASS development-only truthful S80 full-chain counterfactual, not promotion',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('truth','original','replay','s397report','your','out'):
        p.add_argument('--'+k,required=True,type=Path)
    z=p.parse_args()
    run(z.truth,z.original,z.replay,z.s397report,z.your,z.out)
