"""S36b crosspiece learned veto on S35. H9 remains excluded.

Train per held music piece on its fold's OTHER music pieces (the S35
producer for that fold was itself trained outside the entire fold).
Labels never enter prediction-time risk features or the decision.
"""
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

BASES=('series35__lambda1__threshold0.05',
       'series35__lambda4__threshold0')
LAMS=(1.,2.,4.)
CUTS=(0.,.005,.02)
VIEWS=('audio_flow','full_path')
    
def read(f):
    with np.load(f,allow_pickle=False) as x:return {k:x[k] for k in x.files}

def get(z,k):
    names=list(map(str,z['variant_ids']))
    require(k in names,'source prediction missing '+k)
    return z['predictions'][:,names.index(k)].astype(np.int8)

def build(x,parent,candidate,path,view):
    """Unchanged audio, observable K proposals, and optionally past actions."""
    one=np.eye(7,dtype=np.float32)
    base=np.column_stack((x[:,32:335],one[parent],one[candidate]))
    if view=='audio_flow':return base.astype(np.float32)
    require(path.shape==(len(parent),3),'bad saved head path')
    seen=np.zeros((len(parent),19),np.float32)
    for t in range(3):
        at=np.flatnonzero(path[:,t]>=0)
        seen[at,path[at,t]]=1.
    depth=np.sum(path>=0,axis=1).astype(np.float32)[:,None]/3.
    return np.column_stack((x,one[parent],one[candidate],seen,depth)).astype(np.float32)

def data_pool(z,routes,old):
    """Pair dedupe: one training row per native event and proposed K."""
    found={}
    paths=list(map(str,routes['variant_ids']))
    for name in paths:
        if not name.startswith('series35__'):continue
        choice=get(z,name)
        trace=routes['selected_head'][:,paths.index(name)]
        for i in np.flatnonzero(choice!=old):
            key=(int(i),int(choice[i]))
            if key not in found: found[key]=trace[i].copy()
    keys=sorted(found)
    require(len(keys)>100,'insufficient native proposed cases')
    return (np.array([k[0] for k in keys],np.int64),
            np.array([k[1] for k in keys],np.int8),
            np.array([found[k] for k in keys],np.int8))

def selftest():
    x=np.zeros((3,335),np.float32)
    parent=np.array([2,3,4]);new=np.array([1,2,3])
    p=np.array([[0,1,-1],[2,-1,-1],[3,4,6]],np.int8)
    assert build(x,parent,new,p,'audio_flow').shape==(3,317)
    assert build(x,parent,new,p,'full_path').shape==(3,369)
    print('PASS: held-piece candidate-based veto, source-audio/no-H9 observations')

def run(a):
    require(not a.output.exists(),'no overwrite completed S36b')
    began=time.monotonic()
    d=prepare(a)
    z=read(a.s35)
    r=read(a.routes)
    require(np.array_equal(z['global_index'],d['ids']) and
        np.array_equal(r['global_index'],d['ids']),'native event ID mismatch')
    assert len(d['y'])==59309
    old=get(z,'series18_parent')
    require(np.array_equal(old,d['parent']) and
        (old==d['y']).sum()==49178,'S18 benchmark mutated')
    heads=list(map(str,r['action_names']))
    require(len(heads)==19 and
        not any('H9' in h for h in heads),'forbidden head detected')
    ix,tar,seq=data_pool(z,r,old)
    songs=np.asarray(d['pieces'])
    policies={}
    source_paths=list(map(str,r['variant_ids']))
    for base in BASES:
        for view in VIEWS:
            for lam in LAMS:
                for cut in CUTS:
                    policies[f'series36b__{base}__{view}__lambda{lam:g}__threshold{cut:g}']=old.copy()
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    records=[]
    for piece in sorted(set(songs.tolist())):
        on=(songs==piece)
        f=int(np.unique(d['fold'][on])[0])
        fit=(d['fold'][ix]==f)&(songs[ix]!=piece)
        require(fit.sum()>=40,'not enough same-fold proposals: '+piece)
        good=(tar[fit]==d['y'][ix[fit]])&(old[ix[fit]]!=d['y'][ix[fit]])
        bad=(tar[fit]!=d['y'][ix[fit]])&(old[ix[fit]]==d['y'][ix[fit]])
        target=np.column_stack([good,bad]).astype(np.float32)
        # No inferred-risk producer sees held-fold labels; S35 specialists
        # trained on the other 3 outer folds, and this fit excludes piece.
        for view in VIEWS:
            feat=build(d['X'][ix[fit]],old[ix[fit]],tar[fit],seq[fit],view)
            scaler=StandardScaler().fit(feat)
            train=np.clip(scaler.transform(feat),-6,6)
            gate=Ridge(alpha=150.,fit_intercept=True)
            with threadpool_limits(limits=2):gate.fit(train,target)
            file=f'{piece}__{view}.joblib'
            joblib.dump(dict(model=gate,scaler=scaler,
                fold=f,piece=piece,view=view,
                train_native_ids=d['ids'][ix[fit]],
                train_candidates=tar[fit],
                train_correct_count=int(good.sum()),
                train_break_count=int(bad.sum())),
                a.output/'models'/file,compress=3)
            records.append(dict(file=file,piece=piece,fold=f,
                training_proposals=int(fit.sum()),
                held_piece_excluded=True,corrected=int(good.sum()),
                regressed=int(bad.sum()),
                fit_sha256=hashlib.sha256(d['ids'][ix[fit]].astype('<i8').tobytes()).hexdigest()))
            for base in BASES:
                cand=get(z,base)
                held=np.flatnonzero(on&(cand!=old))
                if not len(held):continue
                path=r['selected_head'][held,source_paths.index(base)]
                zz=build(d['X'][held],old[held],cand[held],path,view)
                with threadpool_limits(limits=2):
                    pred=np.clip(gate.predict(np.clip(
                        scaler.transform(zz),-6,6)),0,1)
                for lam in LAMS:
                    gain=pred[:,0]-lam*pred[:,1]
                    for cut in CUTS:
                        name=f'series36b__{base}__{view}__lambda{lam:g}__threshold{cut:g}'
                        keep=(gain>cut)
                        policies[name][held[keep]]=cand[held[keep]]
        print(json.dumps(dict(piece=piece,fold=f,fit=int(fit.sum()),
             stored=len(records),sec=round(time.monotonic()-began,1))),flush=True)
    out={'series18_parent':old.copy()}
    for b in BASES:out[b]=get(z,b).copy()
    out.update(policies)
    measures={}
    for name,pred in out.items():
        fixed=(old!=d['y'])&(pred==d['y'])
        broken=(old==d['y'])&(pred!=d['y'])
        neutral=(old!=d['y'])&(pred!=d['y'])&(old!=pred)
        measures[name]=dict(metrics=metrics(d['y'],pred),
            versus_parent=paired(d['y'],old,pred),
            corrected=int(fixed.sum()),regressed=int(broken.sum()),
            neutral=int(neutral.sum()),
            by_true_K={str(k):dict(
                fixed=int(((d['y']==k)&fixed).sum()),
                broken=int(((d['y']==k)&broken).sum()))
                for k in range(7)},
            per_fold={str(f):dict(
                fixed=int(((d['fold']==f)&fixed).sum()),
                broken=int(((d['fold']==f)&broken).sum())) for f in FOLDS})
    safe=[k for k in policies if measures[k]['corrected']>0 and
        measures[k]['regressed']==0 and
        measures[k]['metrics']['poly']['correct']>=2998]
    ranked=sorted(measures,key=lambda k:(
        measures[k]['metrics']['correct'],
        measures[k]['metrics']['poly']['correct']),reverse=True)
    report=dict(status='completed',S18_unchanged=True,H9_excluded=True,
        held_music_piece_train_exclusion=True,
        source_S35_fold_exclusion=True,
        independent_new_compositions=False,
        model_count=len(records),models=records,
        policy_count=len(out),best_global=ranked[0],
        strict_zero_loss_candidates=safe,audits=measures,
        no_production_promotion=True)
    (a.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    lines=['# S36b crosspiece trained risk veto of dynamic 18-head S35',
        '', '36 learned policies, 38 piece-held Ridge risk judges, S18 unchanged.',
        'No oracle labels in test-time risk; source specialists excluded held outer fold.',
        'No new compositions, no production promotion.',
        '| Policy | Global | Poly | Fixed | Lost |',
        '|---|---:|---:|---:|---:|']
    for key in ranked:
        v=measures[key];m=v['metrics']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{v['corrected']} | {v['regressed']} |")
    lines.extend(['',f'Strict zero-loss candidates: {len(safe)}'])
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s35',type=Path)
    p.add_argument('--routes',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    x=p.parse_args()
    if x.self_test:selftest()
    else:
        require(x.features and x.s18 and x.s35 and x.routes and x.output,
            'native acoustic features and S35 evidence required')
        run(x)
