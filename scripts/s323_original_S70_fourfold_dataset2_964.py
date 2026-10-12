"""S323 S70 original four FROZEN trained fold logistic heads on new WAV.

Reuses exact-original S222 checkpoints (historically 59309/59309),
original PR16 12x7 + 36 votes from S85, fold-free S221 S69
polymax and balanced parents plus piece-neutral S217 S18. Native
IDMT 57 events already exposed: NO accuracy validation.
Original threshold is stored separately for each held-fold checkpoint.
New deployment identity: four heads vote equally; >=3 matching
intervention votes required, otherwise retain S69 parent. All
probabilities, raw folds, and abstention masks saved for audit.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np
from scripts.audit_v273_s70_pr16_poly_meta import POLICIES,propose,permitted
FOLDS=(0,1,2,4)
BAD={'true_K','truth','label','labels','midi','target','correct','regression'}
def need(ok,why):
    if not ok:raise ValueError(why)
def get(p,fields):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(fields) and not BAD.intersection(z.files),
             'only label-free source '+str(p))
        return {k:z[k].copy() for k in z.files}
def run(s221,s85,s222,out):
    need(not out.exists(),'immutable S323 new audio result')
    a=get(s221,{'row_id','S18_parent_K','balanced_s46_K',
        'balanced_protected_K','polymax_s46_K','polymax_protected_K',
        'balanced_fold_veto','balanced_veto_three_folds',
        'polymax_fold_veto','polymax_veto_three_folds'})
    p=get(s85,{'member','start_samples','end_samples','incumbent_freeze_K',
      'pr16_policy_K','pr16_vote_count','pr16_policy22_K',
      'pr16_network_probabilities'})
    ids=list(map(str,a['row_id']));n=len(ids)
    keys=[f'{member}:{int(start)}' for member,start in zip(p['member'],p['start_samples'])]
    need(n==964 and len(set(keys))==len(keys) and len(set(ids))==n,
         'source IDMT native audio event IDs')
    pos={key:i for i,key in enumerate(keys)}
    need(all(key in pos for key in ids),'cannot match original PR16 to S69')
    order=[pos[key] for key in ids]
    probs=p['pr16_network_probabilities'][order].astype(np.float32)
    votes=p['pr16_policy_K'][order].astype(np.int8)
    need(probs.shape==(n,12,7) and votes.shape==(n,36) and
         np.allclose(probs.sum(2),1,atol=1e-4),
         'real PR16 neural original 12-prob 36-decision contract')
    ct=np.column_stack([(votes==k).sum(1) for k in range(7)])
    s18=a['S18_parent_K'].astype(np.int8)
    b=a['polymax_protected_K'].astype(np.int8)
    balanced=a['balanced_protected_K'].astype(np.int8)
    chosen_rows=np.flatnonzero((b>=2)|(votes>=2).any(1))
    bb=b[chosen_rows];bal=balanced[chosen_rows];s=s18[chosen_rows]
    prob=probs[chosen_rows];m=len(chosen_rows)
    X=np.column_stack((prob.reshape(m,-1),prob.mean(1),prob.std(1),
        ct[chosen_rows]/36.,np.eye(7)[bb],np.eye(7)[s],
        np.eye(7)[bal],
        np.column_stack([bb,s,bb-s,bb>=2,s>=2]))).astype(np.float32)
    need(np.isfinite(X).all(),'S70 original feature matrix not finite')
    models={}
    for fold in FOLDS:
        cp=s222/'models'/f'original_S70_poly_meta_heldfold{fold}.joblib'
        need(cp.is_file(),'missing reconstructed original S70 checkpoint '+str(fold))
        st=joblib.load(cp)
        need(int(st['fold'])==fold and st['feature_width']==X.shape[1] and
             st['train_piece_fold_excluded'] and
             st['training_uses_only_historical_labels'],
             'S70 checkpoint schema drift')
        clf=st['model']
        raw=clf.predict_proba(X)
        q=np.zeros((m,7),np.float64)
        for c,key in enumerate(clf[-1].classes_):
            q[:,int(key)]=raw[:,c]
        need(q.shape==(m,7) and np.allclose(q.sum(1),1,atol=1e-5),
             'S70 real frozen checkpoint normalized class probabilities')
        models[fold]=(q,st)
    by_policy={}
    evidence={}
    for name in POLICIES:
        fold_decisions=np.broadcast_to(b,(4,n)).copy()
        fold_actions=np.zeros((4,n),bool)
        for fi,fold in enumerate(FOLDS):
            posterior,st=models[fold]
            proposal,margin=propose(posterior,bb)
            threshold=float(st['thresholds'][name])
            active=permitted(proposal,bb,name)&(margin>=threshold)
            fold_decisions[fi,chosen_rows[active]]=proposal[active]
            fold_actions[fi,chosen_rows[active]]=True
        result=b.copy()
        agreed=np.zeros(n,bool)
        for k in range(7):
            active=(fold_decisions==k)&fold_actions
            enough=(active.sum(axis=0)>=3)&(b!=k)
            need(not np.any(agreed & enough),
                 'S70 conflicting >=3/4 fold policies')
            result[enough]=k;agreed|=enough
        need(np.all((result==b)|(agreed)),
             'S70 K changed without 3 agreeing fold logits')
        by_policy[name+'_K']=result
        by_policy[name+'_fold_K']=fold_decisions
        by_policy[name+'_fold_active']=fold_actions
        by_policy[name+'_vote_accepted']=agreed
        evidence[name]=dict(original_checkpoint_fold_count=4,
            eligible_rows=m,
            folded_proposals_by_fold=list(map(int,fold_actions.sum(axis=1))),
            actions_accepted_by_three_folds=int(agreed.sum()),
            changes_vs_s69_polymax=int((result!=b).sum()))
    out.mkdir(parents=True)
    np.savez_compressed(out/'S323_original_S70_4fold_no_oracle_IDMT.npz',
        row_id=np.asarray(ids),s69_polymax_K=b,s69_balanced_K=balanced,
        s18_K=s18,**by_policy)
    manifest=dict(status='S323_ORIGINAL_TRAINED_S70_4FOLDS_UNSEEN_964_NATIVE_IDMT',
        rows=n,original_S70_fitted_models=4,original_archived_model_feature_width=X.shape[1],
        original_PR16_neural_heads=12,original_PR16_policies=36,
        fold_rule='only act if 3 of 4 original S70 fold classifiers match active K; otherwise S69 parent',
        model_weights_retrained=False,external_true_K_read=False,
        S70_historical_identical_new_piece_policy=False,
        original_S70_training_prediction_verified_run=38296428889,
        policies=evidence,independent_ExactK_measured=False,
        S71_S72_unexecuted=True,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for key in ('s221','s85','s222','out'):
        ap.add_argument('--'+key,type=Path,required=True)
    z=ap.parse_args()
    run(z.s221,z.s85,z.s222,z.out)
