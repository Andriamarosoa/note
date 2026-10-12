"""S72: research-only temporal anti-regression head after S71.

All outer labels excluded from training and threshold choice. Upstream neural
producers not nested, and the transition family is ex-post selected.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FOLDS=(0,1,2,4)
GROUP=((1,2),(3,4),(4,5))
Q=(40,50,60,70,75,80,85,90,95,97,99)

def check(ok,msg):
    if not ok: raise ValueError(msg)

def metrics(y,b,p):
    a=b==y;z=p==y
    return dict(correct=int(z.sum()),poly=int((z&(y>=2)).sum()),
        fixes=int((~a&z).sum()),regressions=int((a&~z).sum()))

def acoustic(root,ids,pos,y,f):
    lookup={int(ids[i]):j for j,i in enumerate(pos)}
    n=len(pos)
    summary=np.full((n,58),np.nan,np.float32)
    sequence=np.full((n,42,49),np.nan,np.float32)
    seen=np.zeros(n,bool);covered=set()
    for outer in FOLDS:
        files=list(root.rglob(f'features-fold-{outer}.npz'))
        check(len(files)==1,'missing feature fold')
        file=files[0];rep=json.loads(file.with_name('report.json').read_text())
        check(rep['status']=='completed' and rep['fold']==outer and
            rep['labels_used_in_features'] is False and
            rep['baseline_used_in_features'] is False and
            rep['yourmt3_used_in_features'] is False and
            hashlib.sha256(file.read_bytes()).hexdigest()==rep['feature_sha256'],
            'feature source integrity/provenance')
        with np.load(file,allow_pickle=False) as a:
            fid=a['global_index'];newids=set(map(int,fid))
            check(len(fid)==len(newids) and not (covered&newids),
                  'feature duplication')
            covered|=newids
            choose=np.flatnonzero(np.isin(fid,ids[pos]))
            dest=np.array([lookup[int(fid[i])] for i in choose],int)
            check(not seen[dest].any() and
                  np.array_equal(a['k'][choose],y[pos[dest]]) and
                  np.array_equal(a['fold'][choose],f[pos[dest]]),
                  'feature-to-native identity mismatch')
            summary[dest]=a['summary'][choose]
            sequence[dest]=a['sequence'][choose]
            seen[dest]=True
    check(len(covered)==len(ids)==59309 and seen.all()
          and np.isfinite(sequence).all() and np.isfinite(summary).all(),
          'full cohort feature coverage')
    t=sequence
    temporal=np.column_stack([t[:,:6].mean(1),t[:,6:14].mean(1),
        t[:,14:28].mean(1),t[:,28:].mean(1),t.std(1),t.max(1),
        t[:,-1]-t[:,0],np.abs(np.diff(t,axis=1)).mean(1)])
    return np.column_stack([summary,temporal]).astype(np.float32)

def model_scores(x,l,w,tr,te):
    if len(set(l[tr].tolist()))<2:return np.full(len(te),float(l[tr].mean()))
    model=HistGradientBoostingClassifier(max_iter=45,max_leaf_nodes=7,
        min_samples_leaf=30,l2_regularization=30.,learning_rate=.06,
        random_state=72,early_stopping=False)
    model.fit(x[tr],l[tr],sample_weight=w[tr])
    return model.predict_proba(x[te])[:,1]

def choose(score,mask,reg,fix):
    subset=np.flatnonzero(mask&np.isfinite(score))
    if len(subset)<25:return None
    cuts=np.unique(np.percentile(score[subset],Q))
    best=(0,0,0,None)
    for cut in cuts:
        selected=subset[score[subset]>=cut]
        if len(selected)<5:continue
        gain=int(reg[selected].sum()-fix[selected].sum())
        if gain<=0:continue
        key=(gain,-int(fix[selected].sum()),-len(selected),float(cut))
        if key>best:best=key
    return best[3]

def run(parent,root,output):
    check(not output.exists(),'cannot overwrite a prior experiment')
    with np.load(parent,allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    ids=a['global_index'];y=a['true_K'];fold=a['fold']
    b=a['s69_poly_K'];s71=a['s71_risk_K'];other=a['s71_poly_K']
    check(len(ids)==len(set(ids.tolist()))==59309 and
          set(fold.tolist())==set(FOLDS),'native identity')
    check(metrics(y,b,s71)['correct']==49220 and
          metrics(y,b,s71)['poly']==3184 and
          metrics(y,b,other)['poly']==3190,'frozen S71 baseline changed')
    pos=np.flatnonzero(s71!=b);check(len(pos)==1029,'S71 action mismatch')
    old=b[pos];new=s71[pos];yt=y[pos];f=fold[pos]
    raw=acoustic(root,ids,pos,y,fold)
    x=np.column_stack([raw,np.eye(7)[old],np.eye(7)[new],new-old])
    reg=(old==yt)&(new!=yt);fix=(old!=yt)&(new==yt)
    weights=np.where(~reg&~fix,.15,1.)
    check((int(reg.sum()),int(fix.sum()))==(308,380),'S71 regression count')
    family=np.logical_or.reduce([(old==i)&(new==j) for i,j in GROUP])
    scores=np.full(len(pos),np.nan)
    inner={}
    for held in FOLDS:
        tr=np.flatnonzero(f!=held);te=np.flatnonzero(f==held)
        scores[te]=model_scores(x,reg.astype(int),weights,tr,te)
        cross=np.full(len(pos),np.nan)
        for vi in FOLDS:
            if vi==held:continue
            fi=np.flatnonzero((f!=held)&(f!=vi))
            tei=np.flatnonzero(f==vi)
            cross[tei]=model_scores(x,reg.astype(int),weights,fi,tei)
        check(np.isfinite(cross[tr]).all(),'inner OOF incomplete')
        inner[held]=cross
    veto=np.zeros(len(pos),bool);rules=[]
    for held in FOLDS:
        threshold=choose(inner[held],(f!=held)&family,reg,fix)
        if threshold is None:continue
        decision=(f==held)&family&(scores>=threshold)
        veto|=decision
        rules.append(dict(fold=held,threshold=threshold,
                          held_actions=int(decision.sum())))
    poly_veto=veto&(old>=2)
    out=dict(global_index=ids,true_K=y,fold=fold,s69_poly_K=b,
             s70_K=a['s70_K'],s71_risk_K=s71,s71_poly_K=other)
    results={}
    for label,parent_pred in [('risk',s71),('poly',other)]:
        for case,mask in [('all',veto),('poly_guard',poly_veto)]:
            p=parent_pred.copy();p[pos[mask]]=b[pos[mask]]
            check(np.all((p==parent_pred)|(p==b)),'unexpected changed K')
            name=label+'_'+case
            out['s72_'+name+'_K']=p
            m=metrics(y,b,p)
            results[name]=dict(**m,abstentions=int((p!=parent_pred).sum()),
                poly_gain=int(((p==y)&(y>=2)).sum()-((parent_pred==y)&(y>=2)).sum()),
                gain=int((p==y).sum()-(parent_pred==y).sum()),
                by_fold={str(k):int((p[fold==k]==y[fold==k]).sum()-
                                    (parent_pred[fold==k]==y[fold==k]).sum())
                         for k in FOLDS})
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',**out)
    report=dict(status='completed',promotion=False,independent_validation=False,
        native_rows=len(ids),decisions_parent=len(pos),veto_all=int(veto.sum()),
        veto_poly=int(poly_veto.sum()),rules=rules,results=results,
        caveats=['K transition family chosen post hoc on exposed folds',
          'Source PR16/S70 training is not nested with the meta-learner',
          'Requires unseen music and nested upstream verification'])
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--s71',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.s71,args.features,args.output)
