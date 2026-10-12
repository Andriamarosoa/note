"""S224 reconstruct original trained S71 fold-excluded acoustic risk models.

Training/recovery on archival 59309 historical events, not external music.
Replay original S71 1263 interventions, 104D PR16 acoustics + onehot old/new
(119D) and 4 train-fold-only LogisticRegression heldfold critics.
For each original edge preserve 4x6 original selected veto thresholds.
Require 0 differences against genuine archived original S71 59309 rows.
Only after exact replay save checkpoints for later new audio; no promotion.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.audit_v273_s71_acoustic_risk import (
 FOLDS,EDGES,threshold,features)
def need(v,msg):
    if not v:raise ValueError(msg)
def read(path,required):
    with np.load(path,allow_pickle=False) as a:
        need(set(required).issubset(a.files),'S224 missing original fields '+str(path))
        return {k:a[k].copy() for k in required}
def train(x,y,w,tr):
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.05,max_iter=500,solver='lbfgs'))
    model.fit(x[tr],y[tr],logisticregression__sample_weight=w[tr])
    return model
def score(model,x):
    return model.predict_proba(x)[:,1]
def run(s70file,acoustic_root,referencefile,out):
    need(not out.exists(),'do not overwrite S224 forensic proof')
    d=read(s70file,{'global_index','true_K','fold','s69_poly_K','proposed_poly_K'})
    orig=read(referencefile,{'global_index','true_K','fold','s69_poly_K','s70_K',
            's71_risk_K','s71_poly_K'})
    ids=d['global_index'];truth=d['true_K'];fold=d['fold']
    b=d['s69_poly_K'];p=d['proposed_poly_K']
    need(len(ids)==59309 and set(fold.tolist())==set(FOLDS) and
         all(np.array_equal(d[k],orig[k]) for k in
             ('global_index','true_K','fold','s69_poly_K')) and
         np.array_equal(p,orig['s70_K']),
         'archival S70 to original S71 exact native identity')
    positions=np.flatnonzero(b!=p)
    need(len(positions)==1263,'S71 source eligible actions 1263 expected')
    acoustic,member=features(acoustic_root,ids,positions,truth,fold)
    need(acoustic.shape==(1263,104),'original S71 summary58 and geometry46')
    old=b[positions];proposed=p[positions];yp=truth[positions];fp=fold[positions]
    x=np.column_stack([acoustic,np.eye(7)[old],np.eye(7)[proposed],
                       proposed-old]).astype(np.float32)
    regression=(old==yp)&(proposed!=yp)
    correction=(old!=yp)&(proposed==yp)
    neutral=(~regression)&(~correction)
    weights=np.where(neutral,.2,1.).astype(float)
    need((int(correction.sum()),int(regression.sum()))==(451,397),
         'original S71 correction regression census')
    scores=np.full(len(positions),np.nan)
    inner_by_fold={}
    held_models={}
    for held in FOLDS:
        tr=np.flatnonzero(fp!=held);te=np.flatnonzero(fp==held)
        model=train(x,regression.astype(int),weights,tr)
        scores[te]=score(model,x[te])
        inner_scores=np.full(len(positions),np.nan)
        for inner in FOLDS:
            if inner==held:continue
            it=np.flatnonzero((fp!=held)&(fp!=inner))
            iv=np.flatnonzero(fp==inner)
            inner_scores[iv]=score(train(x,regression.astype(int),weights,it),x[iv])
        need(np.isfinite(inner_scores[tr]).all(),'original S71 nested train scores')
        held_models[held]=model
        inner_by_fold[held]=inner_scores
    need(np.isfinite(scores).all(),'original S71 trained probabilities missing')
    veto=np.zeros(len(positions),bool)
    thresholds={}
    for held in FOLDS:
        cuts={}
        for source,target in EDGES:
            subset=(old==source)&(proposed==target)
            cut=threshold(inner_by_fold[held],(fp!=held)&subset,regression,correction)
            active=(fp==held)&subset&(scores>=cut)
            veto|=active
            cuts[f'{source}->{target}']=float(cut)
        thresholds[held]=cuts
    safer=p.copy();safer[positions[veto]]=b[positions[veto]]
    veto_poly=veto&~((old==3)&(proposed==2))
    poly=p.copy();poly[positions[veto_poly]]=b[positions[veto_poly]]
    differences={'risk':int(np.count_nonzero(safer!=orig['s71_risk_K'])),
                 'poly':int(np.count_nonzero(poly!=orig['s71_poly_K']))}
    need(differences=={'risk':0,'poly':0},
         'S71 historical checkpoint reconstruction NOT exact: '+str(differences))
    out.mkdir(parents=True);(out/'models').mkdir()
    proof=[]
    for held,model in held_models.items():
        file=out/'models'/f'original_S71_risk_heldfold{held}.joblib'
        joblib.dump(dict(original_fold=held,model=model,
             original_feature_width=int(x.shape[1]),
             edge_cuts=thresholds[held],
             no_unseen_labels_used=True,historic_training_labels_used=True),file)
        again=joblib.load(file)['model']
        test=np.flatnonzero(fp==held)
        need(np.allclose(score(again,x[test]),scores[test],atol=1e-10),
             'reloaded S71 model disagrees')
        proof.append(dict(fold=held,checkpoint_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            threshold_count=len(thresholds[held]),
            outer_probability_reloaded=True))
    rep=dict(status='S224_REBUILT_ORIGINAL_S71_FOLD_ACOUSTIC_RISK',
        historical_rows=59309,original_eligible_actions=len(positions),
        acoustic_feature_width=104,original_trained_feature_width=int(x.shape[1]),
        original_checkpoint_count=len(held_models),
        original_veto_edges=[f'{a}->{b}' for a,b in EDGES],
        original_vetoed_actions=int(veto.sum()),
        zero_historical_decision_mismatch=differences,
        checkpoint_proof=proof,
        original_training_labels_used=True,external_truth_used=False,
        original_S71_models_recovered_bit_exact=True,
        S72_original_models_not_yet_reconstructed=True,
        independent_accuracy_measured=False,S80_promoted=False)
    (out/'report.json').write_text(json.dumps(rep,indent=2)+'\n')
    print(json.dumps({k:v for k,v in rep.items() if k!='checkpoint_proof'},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('s70','features','s71','out'):p.add_argument('--'+k,type=Path,required=True)
    z=p.parse_args()
    run(z.s70,z.features,z.s71,z.out)
