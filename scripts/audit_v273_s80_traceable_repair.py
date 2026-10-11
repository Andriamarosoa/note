"""S80 reproducible experimental K1 gate, K0/K2 routes; historical folds only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
FOLDS=(0,1,2,4)
def req(c,msg):
    if not c:raise ValueError(msg)
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as file:
        for part in iter(lambda:file.read(1<<20),b''):h.update(part)
    return h.hexdigest()
def measure(y,b,p,mask):
    changed=(b!=p)&mask
    fix=changed&(p==y)&(b!=y)
    reg=changed&(p!=y)&(b==y)
    return dict(n=int(mask.sum()),s72_correct=int(((b==y)&mask).sum()),correct=int(((p==y)&mask).sum()),actions=int(changed.sum()),fixes=int(fix.sum()),regressions=int(reg.sum()),net=int(fix.sum()-reg.sum()),wrong_to_wrong=int((changed&(p!=y)&(b!=y)).sum()))
def run(a):
    req(not a.output.exists(),'refuse overwriting experiment')
    old=load(a.s72);pr=load(a.pr16/'predictions.npz');npz=load(a.pr16/'probabilities.npz');s46=load(a.s46/'predictions.npz')
    ids=old['global_index'];y=old['true_K'];fold=old['fold'];b=old['s72_poly_poly_guard_K'];n=len(y)
    req(n==len(np.unique(ids))==59309 and int((y>=2).sum())==7385,'native cohort drift')
    req(set(map(int,fold))==set(FOLDS),'fold drift')
    for z in (pr,npz,s46):req(np.array_equal(ids,z['global_index']),'ID alignment failure')
    for z in (pr,s46):req(np.array_equal(y,z['true_K']),'truth alignment failure')
    req(np.array_equal(fold,s46['fold']),'S46 fold mismatch')
    req(pr['predictions'].shape==(n,36) and s46['predictions'].shape==(n,193),'source policy count')
    ps=npz['probabilities'];req(ps.shape==(n,12,7),'PR16 producers count')
    req(int((b==y).sum())==49238 and int(((b==y)&(y>=2)).sum())==3202,'S72 drift')
    summ=np.zeros((n,58),np.float32);geom=np.zeros((n,46),np.float32);members=np.empty(n,dtype='U96')
    lookup={int(v):i for i,v in enumerate(ids)};seen=np.zeros(n,bool);hashes={}
    for f in FOLDS:
        files=list(a.features.rglob('features-fold-%s.npz'%f))
        req(len(files)==1,'missing/duplicate feature fold')
        filename=files[0];rep=json.loads(filename.with_name('report.json').read_text())
        req(rep.get('fold')==f and rep.get('status')=='completed' and
            rep.get('labels_used_in_features') is False and
            rep.get('baseline_used_in_features') is False and
            rep.get('yourmt3_used_in_features') is False and
            rep.get('feature_sha256')==sha(filename),'source integrity '+str(f))
        with np.load(filename,allow_pickle=False) as z:
            dst=np.asarray([lookup[int(i)] for i in z['global_index']])
            req(not seen[dst].any() and np.all(fold[dst]==f) and np.array_equal(y[dst],z['k']),'feature identity')
            seen[dst]=True;summ[dst]=z['summary'];geom[dst]=z['geometry'];members[dst]=z['member']
        hashes[str(f)]=rep['feature_sha256']
    req(seen.all() and np.isfinite(summ).all() and np.isfinite(geom).all(),'feature gap')
    pavg=ps.mean(1)
    votes=np.stack([(pr['predictions']==k).sum(1) for k in range(7)],1)/36.
    sv=np.stack([(s46['predictions']==k).sum(1) for k in range(7)],1)/193.
    proposed=pr['predictions'][:,22]
    eligible=(b==1)&(proposed>b);idx=np.flatnonzero(eligible)
    signed=np.where(y==proposed,1,np.where(y==b,-1,0))
    extra=np.column_stack((pavg[np.arange(n),proposed],pavg[np.arange(n),b],votes,sv)).astype(np.float32)
    acoust=np.column_stack((summ,geom))
    flat=ps.reshape(n,-1)
    features={'prob':np.column_stack((flat,extra)),'acoustic':np.column_stack((acoust,extra)),
        'combined':np.column_stack((acoust,flat,extra))}
    def train(name,C,tr,te):
        m=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=1000,solver='liblinear',random_state=79))
        m.fit(features[name][tr],(signed[tr]==1).astype(int),logisticregression__sample_weight=np.where(signed[tr]==0,.15,1.))
        return m.predict_proba(features[name][te])[:,1],m
    result79=b.copy();scores=np.full(n,np.nan,np.float32);decisions={};models={}
    for f in FOLDS:
        tr=idx[fold[idx]!=f];te=idx[fold[idx]==f];choices=[]
        for name in features:
            for C in (.02,.1,.5):
                inner_scores=np.empty(len(tr))
                for ff in FOLDS:
                    if ff==f:continue
                    it=tr[fold[tr]!=ff];i_test=np.flatnonzero(fold[tr]==ff)
                    inner_scores[i_test]=train(name,C,it,tr[i_test])[0]
                for cut in (.45,.5,.55,.6,.65,.7,.75,.8,.85,.9):
                    active=inner_scores>=cut
                    fixes=int(np.sum((signed[tr]==1)&active));reg=int(np.sum((signed[tr]==-1)&active))
                    folds=[int(np.sum((signed[tr]==1)&active&(fold[tr]==ff))-np.sum((signed[tr]==-1)&active&(fold[tr]==ff))) for ff in FOLDS if ff!=f]
                    if fixes>reg and fixes>=4 and min(folds)>=-3:
                        choices.append(dict(name=name,C=C,cut=cut,net=fixes-reg,fixes=fixes,reg=reg,folds=folds))
        chosen=max(choices,key=lambda v:(v['net'],v['fixes'],-v['reg'])) if choices else dict(name='combined',C=.1,cut=2.,net=0,fixes=0,reg=0,folds=[0])
        held,model=train(chosen['name'],chosen['C'],tr,te)
        scores[te]=held;accept=held>=chosen['cut'];result79[te[accept]]=proposed[te[accept]]
        decisions[str(f)]={'training_only':chosen,'outer_actions':int(accept.sum()),'outer_fixes':int(np.sum((signed[te]==1)&accept)),'outer_regressions':int(np.sum((signed[te]==-1)&accept))}
        models[str(f)]=model
    mix=.7*pavg+.2*votes+.1*sv
    q0=pavg.argmax(1);q2=sv.argmax(1)
    action0=(b==0)&(q0>=2)
    action2=(b==2)&(q2!=b)&(mix[np.arange(n),q2]>=mix[np.arange(n),b]+.4)
    req(not np.any((result79!=b)&(action0|action2)),'overlapping routes')
    result80=result79.copy();result80[action0]=q0[action0];result80[action2]=q2[action2]
    def audit(p):
        return {'global':measure(y,b,p,np.ones(n,bool)),'poly':measure(y,b,p,y>=2),
            'by_K':{str(k):measure(y,b,p,y==k) for k in range(7)},
            'by_fold':{str(f):measure(y,b,p,fold==f) for f in FOLDS},
            'by_transition':{f'{i}->{j}':measure(y,b,p,(b==i)&(p==j)) for i in range(7) for j in range(7) if i!=j and np.any((b==i)&(p==j))}}
    rep={'status':'RESEARCH_ONLY_NOT_PROMOTED','s79':audit(result79),'s80':audit(result80),'train_fold_only_K1':decisions,'feature_hashes':hashes,
        'actions_by_source':{'K1_guard':int((result79!=b).sum()),'K0_head':int(action0.sum()),'K2_head':int(action2.sum())},
        'caveats':['K0/K2 source routes chosen after auditing historical folds','PR16/S46 upstream producers not fully nested','No fresh unseen-composition test']}
    req(rep['s79']['global']['correct']==49244 and rep['s79']['poly']['correct']==3232,'S79 replay mismatch')
    req(rep['s80']['global']['correct']==49281 and rep['s80']['poly']['correct']==3240,'S80 replay mismatch')
    req(all(rep['s80']['by_fold'][str(f)]['net']>0 for f in FOLDS),'per-fold regression')
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(rep,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,fold=fold,member=members,s72_K=b,s79_K=result79,s80_K=result80,pr16_policy22_K=proposed,pr16_probability_k1=scores,action_K0=action0,action_K2=action2)
    (a.output/'models').mkdir()
    for f,m in models.items():joblib.dump(m,a.output/'models'/('K1_gate_fold'+f+'.joblib'))
    print('S79',rep['s79']['global'],rep['s79']['poly'])
    print('S80',rep['s80']['global'],rep['s80']['poly'])
    print('Official reference unchanged.')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for arg in ('s72','pr16','s46','features','output'):p.add_argument('--'+arg,required=True,type=Path)
    run(p.parse_args())
