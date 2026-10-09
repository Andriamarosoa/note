"""S29: actually dynamic A/B scheduler which chooses next head after each step.

The head execution changes the observable state; the same learned gate is
requeried at every step. STOP, A (reuse B), B→A (refresh B first) are real
per-event actions, not preselected fixed entire paths.

Development research: S20 experts and S18 parent previously exposed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_aba_residual import ResidualAB,onehot_prior
from scripts.loop_v273_native_risk import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27329
MAX_STEPS=3
ACTIONS=('A','BA')
LAMBDA=(1.,2.,4.)
CUTS=(0.,.005,.020)
PREFIXES=('', 'A','D','AA','AD','DA','DD')
BY_STAGE=(('',),('A','D'),('AA','AD','DA','DD'))
# fixed margin: keep parent unless the new proposal has at least .60 more probability.
MARGIN=.60


def digest_rows(v):
    return hashlib.sha256(np.asarray(v).astype('<i8').tobytes()).hexdigest()


def reset_state(base):
    prior=onehot_prior(base)
    return prior,torch.zeros_like(prior),torch.zeros_like(prior)


def act(net,x,prior,parent,p,msg,actcode):
    """One actually executed action; outputs new (probability, message, delta)."""
    if actcode==2:
        newmsg=torch.tanh(net.B(torch.cat((x,prior,p),dim=1)))
    else:
        newmsg=msg
    logits=torch.log(prior)+net.A(torch.cat((x,prior,p,newmsg),dim=1))
    nextp=logits.softmax(1)
    return nextp,newmsg,nextp-p


def state_features(x,prior,p,msg,delta,stage):
    """No truth, fold, future expert outputs, or piece identity."""
    n=len(x)
    assert x.ndim==2 and x.shape[1]==335
    assert all(z.shape==(n,7) for z in (prior,p,msg,delta))
    assert 0<=stage<MAX_STEPS
    orig=(prior.argmax(1)).long()
    predicted=p.argmax(1).long()
    ids=torch.arange(n)
    stat=torch.stack([
        -(p*torch.log(p.clamp(min=1e-9))).sum(1),
        p.max(1).values,
        p[ids,orig],
        p[ids,predicted]-p[ids,orig],
        torch.abs(msg).mean(1)
    ],dim=1)
    oh=torch.zeros((n,MAX_STEPS),dtype=p.dtype)
    oh[:,stage]=1.
    feats=torch.cat([x[:,:90],prior,p,msg,delta,oh,stat],dim=1)
    assert feats.shape==(n,126)
    return feats


def finish(p,parent):
    n=len(p)
    proposal=p.argmax(1)
    gain=p[torch.arange(n),proposal]-p[torch.arange(n),parent]
    return torch.where((proposal!=parent)&(gain>MARGIN),proposal,parent)


def grid_features(net,x,base):
    """7 states under all paths of length<=2; training labels are assigned later."""
    n=len(base)
    y=torch.from_numpy(base.astype(np.int64))
    prior=onehot_prior(y)
    init=reset_state(y)
    nodes={'':init}
    features=np.empty((n,7,126),np.float32)
    values=np.empty((n,7,2),np.int8)
    with torch.no_grad():
        for i,path in enumerate(PREFIXES):
            p,msg,delta=nodes[path]
            depth=len(path)
            # Every node is evaluated exactly under its actually simulated history.
            features[:,i]=state_features(x,prior,p,msg,delta,depth).numpy()
            for j,letter in enumerate(('A','D')):
                nex=act(net,x,prior,y,p,msg,1 if letter=='A' else 2)
                values[:,i,j]=finish(nex[0],y).numpy().astype(np.int8)
                if depth<2:
                    nodes[path+letter]=nex
    require(set(nodes)==set(PREFIXES),'missing explored state branch')
    return features,values


def choose_q(qA,qD,lam,cut):
    """Inputs are [fix, break] estimator outputs; choose STOP/A/BA."""
    a=qA[:,0]-lam*qA[:,1]-.0005
    b=qD[:,0]-lam*qD[:,1]-.001
    output=np.zeros(len(a),np.int8)
    useA=(a>b)&(a>cut)
    useD=(b>=a)&(b>cut)
    output[useA]=1
    output[useD]=2
    return output


def execute_routing(net,X,base,scaler,models,lam,cut):
    """Batch and call ONLY selected heads. Policies re-decide after each pass."""
    parent=torch.from_numpy(base.astype(np.int64))
    prior=onehot_prior(parent)
    p,msg,delta=reset_state(base)
    n=len(X)
    alive=np.ones(n,bool)
    routes=np.zeros((n,MAX_STEPS),np.int8)
    counts=np.zeros((n,2),np.int8)
    net.eval()
    with torch.no_grad():
        for stage in range(MAX_STEPS):
            ix=np.flatnonzero(alive)
            if len(ix)==0:break
            features=state_features(
                X[ix],prior[ix],p[ix],msg[ix],delta[ix],stage).numpy()
            transformed=np.clip(scaler.transform(features),-6,6)
            pA=np.clip(models[0].predict(transformed),0,1)
            pD=np.clip(models[1].predict(transformed),0,1)
            decisions=choose_q(pA,pD,lam,cut)
            # stop is irreversible: no A/B after STOP for this event
            routes[ix,stage]=decisions
            alive[ix[decisions==0]]=False
            for action in (1,2):
                chosen=ix[decisions==action]
                if not len(chosen):continue
                pp,mm,dd=act(net,X[chosen],prior[chosen],parent[chosen],
                             p[chosen],msg[chosen],action)
                p[chosen]=pp
                msg[chosen]=mm
                delta[chosen]=dd
                counts[chosen,0]+=1
                if action==2:
                    counts[chosen,1]+=1
    prediction=finish(p,parent).numpy().astype(np.int8)
    return prediction,routes,counts,p.numpy()


def selftest():
    torch.manual_seed(SEED)
    m=ResidualAB()
    torch.nn.init.normal_(m.A[-1].weight,std=.05)
    m.eval()
    x=torch.randn(6,335)
    base=np.asarray([0,1,2,3,4,5],np.int8)
    initial=reset_state(base)
    prior=onehot_prior(torch.from_numpy(base.astype(np.int64)))
    with torch.no_grad():
        nodes,vals=grid_features(m,x,base)
        assert nodes.shape==(6,7,126) and vals.shape==(6,7,2)
        pA,ma,da=act(m,x,prior,torch.from_numpy(base.astype(np.int64)),*initial,1)
        pD,md,dd=act(m,x,prior,torch.from_numpy(base.astype(np.int64)),*initial,2)
        assert pA.shape==pD.shape==(6,7)
        assert not torch.allclose(pA,pD),'B first does not change A'
        assert torch.allclose(pA.sum(1),torch.ones(6),atol=1e-6)
        assert torch.allclose(pD.sum(1),torch.ones(6),atol=1e-6)
    qA=np.array([[.6,.1],[0,.5],[.2,.2]],np.float32)
    qD=np.array([[.3,.3],[.7,.1],[.2,.2]],np.float32)
    assert choose_q(qA,qD,2,.01).tolist()==[1,2,0]
    # Test a true initial STOP and a dynamic route that takes A, then BA.
    class ScriptedEstimator:
        def __init__(self,kind):self.kind=kind
        def predict(self,z):
            out=np.zeros((len(z),2),np.float32)
            stage=np.argmax(z[:,-8:-5],axis=1)  # stage one-hot
            if self.kind==0:
                out[(stage==0)&(z[:,0]>0),0]=.8
            if self.kind==1:
                out[stage==1,0]=.8
            return out
    # Do not rely on standardizer of synthetic features for this test.
    class IdentityScaler:
        def transform(self,v):return v
    x2=torch.zeros(3,335)
    x2[0,0]=1
    pred,route,cost,pp=execute_routing(m,x2,base[:3],
        IdentityScaler(),[ScriptedEstimator(0),ScriptedEstimator(1)],1,.01)
    assert route[0,:].tolist()==[1,2,0],route[0,:]
    assert route[1,:].tolist()==[0,0,0]
    assert cost[0].tolist()==[2,1] and cost[1].tolist()==[0,0]
    print(json.dumps(dict(test='PASS',states=7,actions=3,
        dynamic_action_after_new_state=True,actual_head_counts=True)),flush=True)


def build_exploration(args,d):
    """Out-of-piece source checkpoint for every training node, no labels entered."""
    n=len(d['y'])
    feats=np.full((n,7,126),np.nan,np.float32)
    proposals=np.full((n,7,2),-1,np.int8)
    source=json.loads((args.s20/'report.json').read_text())
    require(source['status']=='completed' and len(source['weights'])==19,
            '19 S20 checkpoints required')
    records=[]
    for spec in source['weights']:
        piece=spec['piece'];fold=int(spec['fold'])
        held=d['pieces']==piece
        fit=(d['fold']==fold)&~held
        require(held.any() and not np.any(fit&held) and
                piece not in spec['fit_pieces'],'source split compromised')
        require(spec['held_sha256']==digest_rows(d['ids'][held]) and
                spec['fit_sha256']==digest_rows(d['ids'][fit]),
                'source checkpoint training IDs changed')
        file=args.s20/'models'/spec['model_file']
        require(file.is_file(),'missing saved S20 model')
        pack=torch.load(file,weights_only=False,map_location='cpu')
        require(pack['piece']==piece and pack['fold']==fold and
                np.array_equal(pack['held_ids'],d['ids'][held]) and
                np.array_equal(pack['fit_ids'],d['ids'][fit]),
                'fit provenance mismatch')
        m=ResidualAB();m.load_state_dict(pack['model']);m.eval()
        scaled=np.clip((d['X'][held]-pack['scaler_mean'])/
            pack['scaler_scale'],-5,5).astype(np.float32)
        parent=d['parent'][held]
        positions=np.flatnonzero(held)
        with torch.no_grad():
            for start in range(0,len(positions),512):
                sl=slice(start,start+512)
                xi=positions[sl]
                f,v=grid_features(m,torch.from_numpy(scaled[sl]),parent[sl])
                feats[xi]=f
                proposals[xi]=v
        records.append(dict(piece=piece,fold=fold,checkpoint=spec['model_file'],
            count=int(held.sum()),model_training_hash=spec['fit_sha256']))
        print(json.dumps(dict(phase='exploration',piece=piece,
            completed=len(records))),flush=True)
    require(np.isfinite(feats).all() and
            np.all((proposals>=0)&(proposals<=6)),
            'incomplete source exploration')
    return feats,proposals,records


def fit_scheduler(features,action_outcomes,y,parent,fit,held):
    """Two action regressors; stage/node is a feature, not an oracle label."""
    target=[]
    Xfit=features[fit].reshape(-1,126)
    labels=y[fit]
    old=parent[fit]
    for action in (0,1):
        cand=action_outcomes[fit,:,action]
        fixed=((cand==labels[:,None])&(old[:,None]!=labels[:,None])).astype(np.float32)
        broken=((cand!=labels[:,None])&(old[:,None]==labels[:,None])).astype(np.float32)
        target.append(np.column_stack([fixed.reshape(-1),broken.reshape(-1)]))
    scaler=StandardScaler().fit(Xfit)
    xx=np.clip(scaler.transform(Xfit),-6,6)
    estimators=[]
    with threadpool_limits(limits=2):
        for t in target:
            model=Ridge(alpha=200.0)
            model.fit(xx,t)
            estimators.append(model)
    return scaler,estimators


def experiment(args):
    require(not args.output.exists(),'research evidence must not be overwritten')
    torch.set_num_threads(2)
    np.random.seed(SEED);torch.manual_seed(SEED)
    start=time.monotonic()
    d=prepare(args)
    n=len(d['y'])
    feats,actions,originals=build_exploration(args,d)
    preds={f'series29__lambda{lam:g}__threshold{cut:g}':
           np.full(n,-1,np.int8)
           for lam in LAMBDA for cut in CUTS}
    routed={name:np.full((n,MAX_STEPS),-1,np.int8) for name in preds}
    costs={name:np.full((n,2),-1,np.int8) for name in preds}
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    held_records=[]
    srcspec={entry['piece']:entry for entry in
        json.loads((args.s20/'report.json').read_text())['weights']}
    for piece in sorted(set(d['pieces'].tolist())):
        held=d['pieces']==piece
        fold=int(np.unique(d['fold'][held])[0])
        fit=(d['fold']==fold)&~held
        require(not (fit&held).any() and
            piece not in set(d['pieces'][fit].tolist()),'held piece leaked')
        scaler,models=fit_scheduler(feats,actions,d['y'],d['parent'],fit,held)
        names=[]
        for j,letter in enumerate(ACTIONS):
            file=f'{piece}__{letter}_Q.joblib'
            joblib.dump(dict(estimator=models[j],scaler=scaler,
                training_event_ids=d['ids'][fit],held_event_ids=d['ids'][held],
                held_piece=piece,fold=fold,action=letter),
                args.output/'models'/file,compress=3)
            names.append(file)
        spec=srcspec[piece]
        saved=torch.load(args.s20/'models'/spec['model_file'],
                         weights_only=False,map_location='cpu')
        model=ResidualAB();model.load_state_dict(saved['model']);model.eval()
        X=np.clip((d['X'][held]-saved['scaler_mean'])/
                  saved['scaler_scale'],-5,5).astype(np.float32)
        indexes=np.flatnonzero(held)
        for begin in range(0,len(indexes),512):
            batch=indexes[begin:begin+512]
            x=torch.from_numpy(X[begin:begin+512])
            prior=d['parent'][batch]
            for lam in LAMBDA:
                for cut in CUTS:
                    key=f'series29__lambda{lam:g}__threshold{cut:g}'
                    with torch.no_grad(),threadpool_limits(limits=2):
                        p,route,cost,_=execute_routing(
                            model,x,prior,scaler,models,lam,cut)
                    preds[key][batch]=p
                    routed[key][batch]=route
                    costs[key][batch]=cost
        held_records.append(dict(piece=piece,fold=fold,
            fit_count=int(fit.sum()),held_count=int(held.sum()),
            fit_SHA256=digest_rows(d['ids'][fit]),
            held_SHA256=digest_rows(d['ids'][held]),learned_models=names))
        print(json.dumps(dict(phase='learned_dynamic',piece=piece,
            completed=len(held_records),seconds=round(time.monotonic()-start,1))),
            flush=True)
    out={'series18_parent':d['parent'].copy(),
         'freeze_parent':d['freeze'].copy()}
    out.update(preds)
    audits={}
    for name,p in out.items():
        require((p>=0).all() and (p<7).all(),'unfilled decision '+name)
        changed=(p!=d['parent'])
        routeinfo=None
        if name in routed:
            r=routed[name]
            c=costs[name]
            require(np.all(r>=0) and np.all(c>=0),'missing route '+name)
            assert np.array_equal(c[:,0],(r!=0).sum(1))
            assert np.array_equal(c[:,1],(r==2).sum(1))
            routeinfo=dict(
                A_calls=int(c[:,0].sum()),
                B_calls=int(c[:,1].sum()),
                A_per_event=float(c[:,0].mean()),
                B_per_event=float(c[:,1].mean()),
                cost_ratio_vs_S20_full=float(c.sum()/(7*n)),
                stopped_without_any_head=int((c[:,0]==0).sum()),
                first_action={str(k):int((r[:,0]==k).sum())
                              for k in (0,1,2)},
                by_step=[{str(k):int((r[:,stage]==k).sum()) for k in (0,1,2)}
                         for stage in range(MAX_STEPS)],
                route_sequences={
                    ''.join('S' if z==0 else 'A' if z==1 else 'D' for z in seq):
                    int(count) for seq,count in zip(*np.unique(
                        r,axis=0,return_counts=True))})
        audits[name]=dict(metrics=metrics(d['y'],p),
            versus_S18=paired(d['y'],d['parent'],p),
            versus_freeze=paired(d['y'],d['freeze'],p),
            versus_YourMT3=paired(d['y'],d['yourmt3'],p),
            corrected=int((changed&(p==d['y'])&(d['parent']!=d['y'])).sum()),
            regressed=int((changed&(p!=d['y'])&(d['parent']==d['y'])).sum()),
            neutral=int((changed&(p!=d['y'])&(d['parent']!=d['y'])).sum()),
            routes=routeinfo,
            by_true_K={str(k):dict(total=int((d['y']==k).sum()),
                correct=int(((d['y']==k)&(p==d['y'])).sum()),
                corrected=int(((d['y']==k)&changed&
                    (p==d['y'])&(d['parent']!=d['y'])).sum()),
                regressed=int(((d['y']==k)&changed&
                    (p!=d['y'])&(d['parent']==d['y'])).sum()))
                for k in range(7)},
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                p[d['fold']==f]),versus_parent=paired(
                d['y'][d['fold']==f],d['parent'][d['fold']==f],
                p[d['fold']==f])) for f in FOLDS})
    safe=[k for k in preds if audits[k]['corrected']>0 and
          audits[k]['regressed']==0 and
          audits[k]['metrics']['poly']['correct']>=2998]
    ranked=sorted(audits,key=lambda k:(
        audits[k]['metrics']['correct'],
        audits[k]['metrics']['poly']['correct']),reverse=True)
    report=dict(status='completed',architecture='stateful_stepwise_A_or_BA_or_STOP',
        actions=list(ACTIONS)+['STOP'],max_decision_stages=MAX_STEPS,
        seven_training_states_enumerated=True,
        true_test_labels_never_used_to_route=True,
        crosspiece_Q_fit=True,upstream_meta_contamination_possible=True,
        historical_parent_selected_posthoc=True,
        unseen_music_validation=False,production_promotion=False,
        source_S20_run=37858383973,
        models_count=len(held_records)*2,
        source_checkpoints=originals,
        fit_records=held_records,
        all_policies=audits,
        strict_no_lost_corrections_candidates=safe,
        best_global=ranked[0],elapsed_seconds=round(time.monotonic()-start,2))
    (args.output/'report.json').write_text(
        json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    np.savez_compressed(args.output/'routes.npz',global_index=d['ids'],
        variant_ids=np.asarray(list(preds)),
        actions=np.stack([routed[k] for k in preds],axis=1),
        head_calls=np.stack([costs[k] for k in preds],axis=1))
    lines=['# Série29 — scheduler dynamique A / B→A / STOP, décision après chaque étape','',
        '**Réseau à deux têtes S20, cohortes déjà explorées ; aucune validation totalement inédite.**',
        'Les appels A/B sont réellement exécutés seulement après choix de la route.',
        '', '| Route apprise | Exact global | Exact poly | Corrections | Régressions | Appels A+B relatifs |',
        '|---|---:|---:|---:|---:|---:|']
    for k in ranked:
        v=audits[k];m=v['metrics'];cost=v['routes']
        lines.append(f"| {k} | {m['exact']*100:.4f}% | "
          f"{m['poly']['exact']*100:.4f}% | {v['corrected']} | "
          f"{v['regressed']} | "
          f"{cost['cost_ratio_vs_S20_full']*100:.2f}%" if cost else
          f"| {k} | {m['exact']*100:.4f}% | {m['poly']['exact']*100:.4f}% | "
          f"{v['corrected']} | {v['regressed']} | 0% |")
    lines+=['',f"Strict no-lost-correction candidates: {len(safe)}",
        'Voir report.json pour corrections/régressions K0-K6, fold et décisions à chaque étape.',
        'Production unchanged.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s20',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:selftest()
    else:
        require(args.features and args.s18 and args.s20 and args.output,
                'original source checkpoints required')
        experiment(args)
