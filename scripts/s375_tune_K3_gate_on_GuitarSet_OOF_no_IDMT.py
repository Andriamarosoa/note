"""S375 source ONLY GuitarSet heldfold OOF tuning, NO IDMT labels or audio.

Original S373 104D/1133D trained eight fold-shared acoustic checkpoints
were fitted using GuitarSet other-fold samples. HERE threshold/variant
is selected from declared deterministic TRAINING/DEVELOPMENT OOF only:
by original heldfold K3 corrections minus all regressions, with a strict
K2-regression cap. This is hyperparameter selection, NOT independent
OOF score: only the eight S372 UNOPENED XMLs can validate it.

Models unchanged, no IDMT features, no test outcome inspected.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
TRUTH={'row_id','true_K','original_fold','original_freeze_K'}
S373={'row_id','original_fold','original_S18_OOF_K','eligible_trigger',
      'heldfold_pred_probability_true_K3','fixed_trigger_above_threshold'}
T=[round(float(x),3) for x in np.arange(.10,.801,.025)]
def need(v,msg):
    if not v:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==fields,'saved fold OOF schema drift')
        return {k:z[k].copy() for k in z.files}
def run(original,low,expanded,out):
    need(not out.exists(),'immutable original training-only threshold evidence')
    yt=load(original,TRUTH)
    need(len(yt['row_id'])==59309,'source training original fold count invalid')
    yy={str(row):int(k) for row,k in zip(yt['row_id'],yt['true_K'])}
    folds={str(row):int(k) for row,k in zip(yt['row_id'],yt['original_fold'])}
    views={}
    for name,path in [('base104',low),('expanded1133',expanded)]:
        p=load(path,S373);ids=p['row_id'].astype(str)
        need(len(ids)==len(set(ids)) and all(x in yy for x in ids),
             'source trained expert heldfold IDs not in GuitarSet')
        y=np.array([yy[x] for x in ids],np.int8)
        fold=np.array([folds[x] for x in ids],np.int8)
        need(np.array_equal(fold,p['original_fold'])
             and np.array_equal(p['eligible_trigger'],p['original_S18_OOF_K']==2)
             and np.isfinite(p['heldfold_pred_probability_true_K3']).all(),
             'true heldfold condition missing or OOF probabilities not finite')
        eligible=p['eligible_trigger']
        probs=p['heldfold_pred_probability_true_K3']
        trials=[]
        for t in T:
            sel=eligible&(probs>=t)
            h=np.bincount(y[sel],minlength=7)
            fixes=int(h[3]);reg=int(h.sum()-h[3])
            k2=int(h[2])
            per_fold=[]
            for f in [0,1,2,4]:
                m=sel&(fold==f)
                correct=int(np.count_nonzero(y[m]==3))
                regress=int(np.count_nonzero(m))-correct
                per_fold.append(dict(fold=f,trigger=int(m.sum()),
                    K3_fixes=correct,regressions=regress,net=correct-regress))
            trials.append(dict(threshold=t,actions=int(sel.sum()),
                original_actual_K_histogram=h.tolist(),
                fixes_K3=fixes,regressions_all_otherK=reg,
                K2_regressions=k2,net_correct=fixes-reg,
                candidate_allowed=(fixes>=3 and fixes>reg and k2<=10),
                per_fold=per_fold))
        views[name]=dict(heldfold_trigger_count=int(eligible.sum()),
              candidate_thresholds=trials,
              trained_model_not_refit=True,trained_probability_SHA256=sha(path))
    valid=[(name,t) for name,v in views.items() for t in v['candidate_thresholds']
        if t['candidate_allowed']]
    # Pre-registered tie-break: prefer max net, fewer total changes, then
    # highest threshold to avoid overcounting, then expanded feature view
    # only on a perfect tie to test whether morphology adds signal.
    ranked=sorted(valid,key=lambda pair:(
      -pair[1]['net_correct'],pair[1]['actions'],-pair[1]['threshold'],
      0 if pair[0]=='base104' else 1))
    selected=None
    if ranked:
        name,t=ranked[0]
        selected=dict(variant=name,threshold=t['threshold'],
            development_only_net=t['net_correct'],
            development_actions=t['actions'],development_K3_fixes=t['fixes_K3'],
            development_regressions=t['regressions_all_otherK'],
            K2_regressions=t['K2_regressions'])
    r=dict(status='S375_GUITARSET_ONLY_OOF_K3_EARLY_EXPERT_CANDIDATE_SELECTION',
       original_truth_SHA256=sha(original),
       original_two_expert_OOF_SHAs={'base104':sha(low),'expanded1133':sha(expanded)},
       fixed_threshold_grid=T,
       candidate_selection_constraints='K3 fixes>=3, net improves, K2 losses<=10 of 261 original OOF-trigger K2; rank highest net then fewer actions then higher threshold then base104',
       original_GuitarSet_heldfold_dev_only=True,
       original_training_members_withheld_from_each_validation_fold=True,
       external_IDMT_S372_unopened_audio_XML_NOT_USED=True,
       selected_frozen_external_candidate=selected,
       selected_candidate_valid_for_prospective_test=selected is not None,
       dev_threshold_scan_requires_genuinely_separate_external_validation=True,
       trained_models_changed=False,
       no_implicit_S80_official_promotion=True,
       view_results=views)
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='view_results'},indent=2),flush=True)
    for name,v in views.items():
        print('ALL source-only OOF GRID',name,json.dumps([dict(t=t['threshold'],n=t['actions'],
              fixes=t['fixes_K3'],reg=t['regressions_all_otherK'],
              K2_losses=t['K2_regressions'],net=t['net_correct'],
              pass_train_gate=t['candidate_allowed']) for t in v['candidate_thresholds']],indent=2))
if __name__=='__main__':
    a=argparse.ArgumentParser()
    for k in ('truth','base','expanded','out'):
        a.add_argument('--'+k,type=Path,required=True)
    args=a.parse_args();run(args.truth,args.base,args.expanded,args.out)
