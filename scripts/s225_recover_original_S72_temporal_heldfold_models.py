"""S225: recover exact historical S72 fold-excluded temporal risk checkpoints.

Recreate 4 original S72 trained HistGradientBoosting regress-veto models
on 1029 S71 source interventions. Preserve 450D original
PR16 summary58 + temporal trajectories (8 groups of 49) plus old/new
onehots + delta => 465D. Determine original outer-fold training-only
thresholds and validate BOTH original S72 risk/poly pathways with all
59309 historic rows, before persisting the four genuine trained models.
Historical training labels allowed in this recovery job; new audio labels
never read. No new-song score and no official promotion.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from scripts.audit_v273_s72_temporal_poly_guard import FOLDS,GROUP,Q,acoustic,choose
def need(v,msg):
    if not v:raise ValueError(msg)
def read(p,keys):
    with np.load(p,allow_pickle=False) as z:
        need(set(keys).issubset(z.files),'S225 missing real original historical fields '+str(p))
        return {k:z[k].copy() for k in keys}
def model_scores_or_constant(x,y,w,tr,te):
    if len(set(y[tr].tolist()))<2:
        value=float(y[tr].mean())
        return np.full(len(te),value),dict(constant_probability=value)
    model=HistGradientBoostingClassifier(max_iter=45,max_leaf_nodes=7,
        min_samples_leaf=30,l2_regularization=30.,
        learning_rate=.06,random_state=72,early_stopping=False)
    model.fit(x[tr],y[tr],sample_weight=w[tr])
    return model.predict_proba(x[te])[:,1],dict(model=model)
def run(s71file,features_root,referencefile,out):
    need(not out.exists(),'immutable S225 checkpoint identity')
    a=read(s71file,{'global_index','true_K','fold','s69_poly_K','s70_K','s71_risk_K','s71_poly_K'})
    original=read(referencefile,{'global_index','true_K','fold','s69_poly_K','s70_K','s71_risk_K','s71_poly_K',
        's72_risk_all_K','s72_risk_poly_guard_K','s72_poly_all_K','s72_poly_poly_guard_K'})
    for k in ('global_index','true_K','fold','s69_poly_K','s70_K','s71_risk_K','s71_poly_K'):
        need(np.array_equal(a[k],original[k]),'S72 expected S71 original provenance mismatch '+k)
    ids=a['global_index'];y=a['true_K'];fold=a['fold'];b=a['s69_poly_K']
    p=a['s71_risk_K'];other=a['s71_poly_K']
    need(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'historical native 4 folds')
    pos=np.flatnonzero(p!=b);need(len(pos)==1029,'original S71 changed-K inventory')
    old=b[pos];new=p[pos];actual=y[pos];folds=fold[pos]
    acoust=acoustic(features_root,ids,pos,y,fold)
    need(acoust.shape==(1029,450),'original temporal 450D view 58+8x49')
    x=np.column_stack([acoust,np.eye(7)[old],np.eye(7)[new],new-old]).astype(np.float32)
    need(x.shape==(1029,465),'original S72 trained risk feature width')
    regress=(old==actual)&(new!=actual)
    fix=(old!=actual)&(new==actual)
    need((int(regress.sum()),int(fix.sum()))==(308,380),
         'original S71 regressions/fixes unchanged')
    weight=np.where(~regress&~fix,.15,1.)
    scores=np.full(len(pos),np.nan)
    inner={}
    models={}
    for held in FOLDS:
        tr=np.flatnonzero(folds!=held)
        te=np.flatnonzero(folds==held)
        scored,fit=model_scores_or_constant(x,regress.astype(int),weight,tr,te)
        scores[te]=scored;models[held]=fit
        cross=np.full(len(pos),np.nan)
        for vi in FOLDS:
            if vi==held:continue
            fi=np.flatnonzero((folds!=held)&(folds!=vi))
            tei=np.flatnonzero(folds==vi)
            pp,_=model_scores_or_constant(x,regress.astype(int),weight,fi,tei)
            cross[tei]=pp
        need(np.isfinite(cross[tr]).all(),'original S72 inner OOF not fully scored')
        inner[held]=cross
    need(np.isfinite(scores).all(),'S72 held-fold training predictions nonfinite')
    family=np.logical_or.reduce([(old==i)&(new==j) for i,j in GROUP])
    veto=np.zeros(len(pos),bool)
    thresholds={}
    for held in FOLDS:
        cut=choose(inner[held],(folds!=held)&family,regress,fix)
        thresholds[held]=None if cut is None else float(cut)
        if cut is None:continue
        veto|=(folds==held)&family&(scores>=cut)
    poly_veto=veto&(old>=2)
    result={}
    for label,upstream in (('risk',p),('poly',other)):
        for key,mask in (('all',veto),('poly_guard',poly_veto)):
            outpred=upstream.copy()
            outpred[pos[mask]]=b[pos[mask]]
            name=f'{label}_{key}'
            diffs=int(np.count_nonzero(outpred!=original[f's72_{name}_K']))
            result[name]=dict(historical_mismatches=diffs,
                changed_from_S71=int(np.count_nonzero(outpred!=upstream)))
            need(diffs==0,'S72 original checkpoint replay mismatch '+name+': '+str(diffs))
    out.mkdir(parents=True);(out/'models').mkdir()
    proofs=[]
    for held,model_info in models.items():
        cp=out/'models'/f'original_S72_temporal_veto_fold{held}.joblib'
        joblib.dump(dict(original_held_fold=held,original_trained_feature_width=465,
            threshold=thresholds[held],model_info=model_info,
            training_labels_only_from_other_original_folds=True),cp)
        state=joblib.load(cp)
        te=np.flatnonzero(folds==held)
        if 'model' in state['model_info']:
            q=state['model_info']['model'].predict_proba(x[te])[:,1]
        else:q=np.full(len(te),state['model_info']['constant_probability'])
        need(np.allclose(q,scores[te],atol=1e-10),'reloaded S72 trained checkpoint probability')
        proofs.append(dict(fold=held,sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
             original_threshold=thresholds[held],checkpoint_fits_original=True))
    rep=dict(status='S225_ORIGINAL_S72_TEMPORAL_RISK_4_MODELS_RECOVERED',
       historical_event_count=59309,original_S71_interventions=1029,
       feature_original_acoustic_view=450,model_input_feature_width=465,
       recovered_model_count=4,source_K_transitions=list(map(list,GROUP)),
       original_decisions_replayed_exact=result,fold_thresholds=thresholds,
       checkpoint_proof=proofs,training_labels_used_on_historical_folds=True,
       external_song_labels_read=False,new_song_accuracy_measured=False,
       historical_equivalence_on_new_piece_not_yet_claimed=True,
       original_S80_not_yet_complete=True,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rep.items() if k!='checkpoint_proof'},indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s71','features','s72','out'):p.add_argument('--'+k,type=Path,required=True)
    q=p.parse_args()
    run(q.s71,q.features,q.s72,q.out)
