"""Verify frozen S80 final gate against every original historical prediction.

NO training, NO access to truth when predicting, NO external promotion claim.
S80 is a composition-dependent pipeline: this file audits its final-stage
weights, not external S45/S46/S69-S72 deployment.
"""
import argparse,hashlib,json
from pathlib import Path
import joblib,numpy as np
FOLDS=(0,1,2,4)
def need(ok,why):
    if not ok:raise ValueError(why)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def run(a):
    need(not a.output.exists(),'refuse overwrite')
    historical=load(a.s80/'predictions.npz')
    a72=load(a.s72/'predictions.npz')
    pr=load(a.pr16/'predictions.npz')
    prob=load(a.pr16/'probabilities.npz')
    s46=load(a.s46/'predictions.npz')
    report=json.loads((a.s80/'report.json').read_text())
    ids=historical['global_index'];fold=historical['fold'];b=historical['s72_K']
    n=len(ids)
    need(n==59309 and len(np.unique(ids))==n and set(fold)==set(FOLDS),'native identity')
    for x in (a72,pr,prob,s46):
        need(np.array_equal(x['global_index'],ids),'source ID misalignment')
    need(np.array_equal(a72['s72_poly_poly_guard_K'],b),'S72 drift')
    ps=prob['probabilities'];r=pr['predictions'];v=s46['predictions']
    need(ps.shape==(n,12,7) and r.shape==(n,36) and v.shape==(n,193),'producer count')
    sm=np.empty((n,58),np.float32);geo=np.empty((n,46),np.float32)
    index={int(x):i for i,x in enumerate(ids)};seen=np.zeros(n,bool);fingerprints={}
    for f in FOLDS:
        matches=list(a.features.rglob('features-fold-%s.npz'%f))
        need(len(matches)==1,'feature inventory '+str(f))
        fp=matches[0];rep=json.loads(fp.with_name('report.json').read_text())
        hashv=hashlib.sha256(fp.read_bytes()).hexdigest()
        need(hashv==rep['feature_sha256'] and rep['labels_used_in_features'] is False
             and rep['baseline_used_in_features'] is False and rep['yourmt3_used_in_features'] is False,'feature source')
        with np.load(fp,allow_pickle=False) as z:
            ix=np.array([index[int(k)] for k in z['global_index']])
            need(np.all(fold[ix]==f) and not seen[ix].any(),'feature folds')
            seen[ix]=True;sm[ix]=z['summary'];geo[ix]=z['geometry']
        fingerprints[str(f)]=hashv
    need(seen.all(),'incomplete features')
    avg=ps.mean(1)
    vote=np.stack([(r==k).sum(1) for k in range(7)],1)/36.
    sv=np.stack([(v==k).sum(1) for k in range(7)],1)/193.
    q=r[:,22]
    xextra=np.column_stack([avg[np.arange(n),q],avg[np.arange(n),b],vote,sv]).astype(np.float32)
    combined=np.column_stack([sm,geo,ps.reshape(n,-1),xextra])
    scores=np.full(n,np.nan,np.float32)
    p79=b.copy();det={}
    for f in FOLDS:
        idx=np.flatnonzero((fold==f)&(b==1)&(q>b))
        cfg=report['train_fold_only_K1'][str(f)]['training_only']
        need(cfg['name']=='combined' and cfg['C']==.1,'saved policy mismatch')
        model=joblib.load(a.s80/'models'/('K1_gate_fold%s.joblib'%f))
        need(model.named_steps['logisticregression'].C==.1,'saved checkpoint mismatch')
        pred=model.predict_proba(combined[idx])[:,1]
        scores[idx]=pred
        accept=pred>=cfg['cut']
        p79[idx[accept]]=q[idx[accept]]
        det[str(f)]={'eligible':len(idx),'threshold':cfg['cut'],'accepted':int(accept.sum())}
    mix=.7*avg+.2*vote+.1*sv
    q0=avg.argmax(1);q2=sv.argmax(1)
    take0=(b==0)&(q0>=2)
    take2=(b==2)&(q2!=b)&(mix[np.arange(n),q2]>=mix[np.arange(n),b]+.4)
    p80=p79.copy();p80[take0]=q0[take0];p80[take2]=q2[take2]
    for x,ref in ((p79,'s79_K'),(p80,'s80_K'),(take0,'action_K0'),(take2,'action_K2')):
        need(np.array_equal(x,historical[ref]),'S80 exact result mismatch: '+ref)
    saved=historical['pr16_probability_k1']
    need(np.array_equal(np.isnan(scores),np.isnan(saved)),'K1 mask drift')
    difference=np.abs(scores[~np.isnan(scores)]-saved[~np.isnan(saved)])
    maxerr=float(np.max(difference)) if len(difference) else 0.
    need(maxerr<1e-5,'frozen K1 logits drift')
    y=historical['true_K']
    need(int((p80==y).sum())==49281 and int(((p80==y)&(y>=2)).sum())==3240,
         'historical metrics drift')
    data={'status':'FROZEN_S80_LAST_STAGE_EXACT_REPLAY','full_external_S80_validation':False,
          'official_reference_changed':False,'input_rows':n,
          'frozen_checkpoint_count':4,'fold_decisions':det,
          'K1_score_max_abs_error':maxerr,
          'global_correct':int((p80==y).sum()),'poly_correct':int(((p80==y)&(y>=2)).sum()),
          'feature_sha256':fingerprints,
          'missing_inputs_on_new_audio':['S46 193 policy predictions',
          'S72 parent predictions','one prespecified gate deployment policy']}
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'frozen_s80_replayed.npz',global_index=ids,s80_K=p80)
    print(json.dumps(data,sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('s80','s72','pr16','s46','features','output'):
        p.add_argument('--'+key,required=True,type=Path)
    run(p.parse_args())
