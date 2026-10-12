"""S222: reconstruct and SEAL authentic S70 historical fold-excluded fitted models.

This is explicitly a TRAINING/RECOVERY tool using historical training
labels, NEVER the external new-audio inference runtime. Recreate the
original exact S70 source matrices, original fold heldpiece logistic fits,
training-only OOF threshold optimization. Compare all 59309 saved S70
decisions against the original immutable artifact before persisting each
fold's trained model, selected thresholds and source model diagnostics.
No GOAT labels, no new-audio accuracy, and NO S80 promotion.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from scripts.audit_v273_s70_pr16_poly_meta import (
    FOLDS,POLICIES,propose,permitted,choose_threshold)
def need(v,why):
    if not v:raise ValueError(why)
def strict(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(z.files)==set(keys),'S222 original historic inputs drift '+str(p))
        return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forward(model,x):
    raw=model.predict_proba(x)
    out=np.zeros((len(x),7),np.float64)
    for i,k in enumerate(model[-1].classes_):
        out[:,int(k)]=raw[:,i]
    need(np.allclose(out.sum(1),1,atol=1e-6),'S70 probability normalization')
    return out
def fit(x,y,idx):
    mdl=make_pipeline(StandardScaler(),
        LogisticRegression(C=.01,max_iter=400,solver='lbfgs'))
    mdl.fit(x[idx],y[idx])
    return mdl
def run(s69file,prfile,probfile,s70file,out):
    need(not out.exists(),'immutable recovered S70 artifact')
    a=strict(s69file,{'global_index','true_K','fold','s18_K',
        'balanced_s46_K','balanced_protected_K',
        'polymax_s46_K','polymax_protected_K'})
    pr=strict(prfile,{'global_index','true_K','fold','variant_ids','predictions'})
    # Original PR16 archive may include additional metadata. Strict no
    # assumptions beyond exact keys for S69 and fold alignment.
    with np.load(probfile,allow_pickle=False) as z:
        need('global_index' in z and 'probabilities' in z,
             'original PR16 12-way neural probabilities')
        ids_prob=z['global_index'].copy()
        prob=z['probabilities'].copy()
    with np.load(s70file,allow_pickle=False) as z:
        expected={'global_index','fold','true_K','s69_poly_K',
                  'proposed_poly_K','downward_poly_K'}
        need(set(z.files)==expected,'original S70 saved predictions fields changed')
        saved={k:z[k].copy() for k in z.files}
    ids=a['global_index'];y=a['true_K'];fold=a['fold']
    n=len(ids);b=a['polymax_protected_K'];balanced=a['balanced_protected_K']
    need(n==59309 and np.array_equal(ids,pr['global_index']) and
         np.array_equal(ids,ids_prob) and
         np.array_equal(pr['true_K'],y) and np.array_equal(pr['fold'],fold),
         'exact S69/PR16 historical native position, fold and truth joins')
    need(np.array_equal(ids,saved['global_index']) and
         np.array_equal(fold,saved['fold']) and
         np.array_equal(y,saved['true_K']) and
         np.array_equal(b,saved['s69_poly_K']),
         'held-fold original S70 parent/replay mismatch')
    vote=pr['predictions'];need(vote.shape==(n,36) and prob.shape==(n,12,7),
         'S70 original model 36 vote and 12x7 probability contract')
    counts=np.stack([(vote==k).sum(1) for k in range(7)],axis=1)
    selected=np.flatnonzero((b>=2)|(vote>=2).any(1))
    bb=b[selected];bal=balanced[selected];s18=a['s18_K'][selected]
    truth=y[selected];ft=fold[selected];P=prob[selected];nselect=len(selected)
    x=np.column_stack((P.reshape(nselect,-1),P.mean(1),P.std(1),
         counts[selected]/36.,np.eye(7)[bb],np.eye(7)[s18],np.eye(7)[bal],
         np.column_stack([bb,s18,bb-s18,bb>=2,s18>=2]))).astype(np.float32)
    need(np.isfinite(x).all(),'original S70 trained feature matrix contains NaNs')
    all_probs=np.full((nselect,7),np.nan)
    oof_models={}
    fold_models={};metrics=[]
    for held in FOLDS:
        tr=np.flatnonzero(ft!=held);te=np.flatnonzero(ft==held)
        model=fit(x,truth,tr)
        all_probs[te]=forward(model,x[te])
        inner=np.full((nselect,7),np.nan)
        for h in FOLDS:
            if h==held:continue
            innertrain=np.flatnonzero((ft!=held)&(ft!=h))
            innerte=np.flatnonzero(ft==h)
            inner[innerte]=forward(fit(x,truth,innertrain),x[innerte])
        need(np.isfinite(inner[tr]).all(),
             'S70 original nested training probabilities incomplete')
        oof_models[held]=inner
        fold_models[held]=model
        metrics.append({'held_fold':held,'train_rows':len(tr),'held_rows':len(te),
             'original_weight_type':'logistic_pipeline_trained_on_3_other_folds',
             'n_features':int(x.shape[1])})
    need(np.isfinite(all_probs).all(),'S70 trained fold outputs incomplete')
    proposed,margins=propose(all_probs,bb)
    thresholds={label:{} for label in POLICIES}
    replay={}
    for label,cost in POLICIES.items():
        accepted=np.zeros(nselect,bool)
        domain=label
        for held in FOLDS:
            ownp,ownmargin=propose(oof_models[held],bb)
            cut=choose_threshold(ownmargin,ownp,bb,truth,ft!=held,cost,domain)
            thresholds[label][str(held)]=float(cut)
            accepted[(ft==held)&permitted(proposed,bb,domain)&(margins>=cut)]=True
        result=b.copy()
        result[selected[accepted]]=proposed[accepted]
        reference=saved[label+'_K']
        mismatch=int(np.count_nonzero(result!=reference))
        need(mismatch==0,'original S70 decisions NOT bit-identical: '+label+': '+str(mismatch))
        replay[label]={'historical_exact_predictions_checked':int(n),
                       'mismatches':mismatch,
                       'actions':int(np.count_nonzero(result!=b)),
                       'thresholds':thresholds[label]}
    out.mkdir(parents=True)
    (out/'models').mkdir()
    checks=[]
    for fold_id in FOLDS:
        dest=out/'models'/f'original_S70_poly_meta_heldfold{fold_id}.joblib'
        joblib.dump({'fold':fold_id,'model':fold_models[fold_id],
            'feature_width':int(x.shape[1]),
            'thresholds':{label:thresholds[label][str(fold_id)] for label in POLICIES},
            'train_piece_fold_excluded':True,
            'training_uses_only_historical_labels':True},dest)
        state=joblib.load(dest)
        te=np.flatnonzero(ft==fold_id)
        need(np.allclose(forward(state['model'],x[te]),all_probs[te],atol=1e-10),
             'reload original fold classifier changes probability')
        checks.append({'fold':fold_id,'checkpoint_sha256':sha(dest),
                       'heldout_forward_probability_reloaded_exact':True,
                       'stored_thresholds':state['thresholds']})
    report=dict(status='S222_ORIGINAL_S70_FOLD_CHECKPOINTS_RECONSTRUCTED',
       historical_event_count=59309,eligible_for_meta=int(nselect),
       original_model_feature_width=int(x.shape[1]),
       recovered_fold_models=4,replay=replay,models=metrics,checks=checks,
       historical_train_labels_used=True,external_audio_labels_used=False,
       original_S70_crossfit_predictions_bit_exact=True,
       deploy_four_fold_selection_for_new_song_not_original=True,
       S71_S72_models_not_yet_persisted=True,
       full_original_S80_deployed=False,independent_ExactK_measured=False,
       S80_promoted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='models' and k!='checks'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('s69','pr','probs','s70','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.s69,a.pr,a.probs,a.s70,a.out)
