"""S20 — residual recurrent A/B: B's soft exclusions change next A's input.

All input features are original fold-excluded sources, except a separately declared
posthoc S18 initial prior (NO labels supplied during inference). Research only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require
from scripts.train_v273_aba_recurrent import (
    SEED as PREVIOUS_SEED, EPOCHS, BATCH, LR, PASSPRIOR,
    prepare, class_balancing)

SEED=27320
MARGINS=(.10,.25,.40,.60)
PASS_NAMES=('pass1','pass2','pass3','pass4','no_B_pass4')
BASES=('series18_parent','freeze_parent')


def onehot_prior(base):
    y=torch.as_tensor(base,dtype=torch.long)
    return F.one_hot(y,7).float()*.825+.025


class ResidualAB(nn.Module):
    def __init__(self,num_features=335):
        super().__init__()
        self.num_features=num_features
        self.A=nn.Sequential(
            nn.Linear(num_features+21,128),nn.GELU(),
            nn.Linear(128,64),nn.GELU(),nn.Linear(64,7))
        self.B=nn.Sequential(
            nn.Linear(num_features+14,96),nn.GELU(),
            nn.Linear(96,64),nn.GELU(),nn.Linear(64,7))
        # With no evidence, A's additive residual begins exactly at zero.
        nn.init.zeros_(self.A[-1].weight)
        nn.init.zeros_(self.A[-1].bias)

    def forward(self,x,parent,passes=4,no_B=False,override=None):
        assert x.ndim==2 and x.shape[1]==self.num_features
        assert 1<=passes<=4 and len(parent)==len(x)
        prior=onehot_prior(parent).to(device=x.device,dtype=x.dtype)
        base=torch.log(prior)
        old=prior
        message=torch.zeros_like(prior)
        As=[];Bs=[];Ps=[]
        for t in range(passes):
            delta=self.A(torch.cat([x,prior,old,message],dim=1))
            logits=base+delta
            p=torch.softmax(logits,dim=1)
            B=self.B(torch.cat([x,prior,p],dim=1))
            As.append(logits);Bs.append(B);Ps.append(p)
            old=p
            message=torch.zeros_like(p) if no_B else torch.tanh(B)
            if t==0 and override is not None:
                require(override.shape==message.shape,'B intervention shape')
                message=override
        return As,Bs,Ps


def selftest():
    torch.manual_seed(SEED)
    net=ResidualAB()
    x=torch.randn(8,335)
    parent=torch.tensor([0,1,2,3,4,5,6,4])
    with torch.no_grad():
        A,B,P=net(x,parent,passes=4)
        assert len(P)==4 and P[-1].shape==(8,7)
        assert torch.allclose(P[0],onehot_prior(parent),atol=1e-6)
        # Activate nonzero last-layer residual to probe the B-message channel.
        nn.init.normal_(net.A[-1].weight,mean=0,std=.03)
        a,b,p=net(x,parent,passes=4)
        msgs=torch.tanh(b[0]).clone()
        msgs[:,4]=-1.
        p2other=net(x,parent,passes=2,override=msgs)[2][1]
        delta=(p2other-p[1]).abs().max().item()
        assert delta>1e-6,'B exclusions do not change A2'
        nop=net(x,parent,passes=4,no_B=True)[2][-1]
        assert (nop-p[-1]).abs().max().item()>1e-6, 'B-off control ineffective'
        assert net.A[0].in_features==356 and net.B[0].in_features==349
    print(json.dumps(dict(test='PASS',zero_initialized_prior=True,
        B_reenters_A=True,B_intervention_changes_A2=float(delta))),flush=True)


def train_model(model,x,y,parent,seed):
    target=torch.from_numpy(y.astype(np.int64))
    base=torch.from_numpy(parent.astype(np.int64))
    X=torch.from_numpy(x.astype(np.float32))
    cw=torch.from_numpy(class_balancing(y))
    posweight=torch.ones(7)*6
    optim=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=.001)
    history=[]
    model.train()
    for epoch in range(EPOCHS):
        gen=torch.Generator().manual_seed(seed+epoch)
        order=torch.randperm(len(y),generator=gen)
        total=0
        for start in range(0,len(y),BATCH):
            idx=order[start:start+BATCH]
            xx=X[idx];yy=target[idx];pp=base[idx]
            A,B,_=model(xx,pp,passes=4)
            truths=F.one_hot(yy,7).float()
            ce=sum(w*F.cross_entropy(z,yy,weight=cw)
                   for w,z in zip(PASSPRIOR,A))
            good=(yy==pp).float()
            preserve=torch.stack([
                (F.cross_entropy(z,pp,reduction='none')*good).sum()/
                  torch.clamp(good.sum(),min=1) for z in A]).mean()
            compatibility=sum(F.binary_cross_entropy_with_logits(
                z,truths,pos_weight=posweight) for z in B)
            loss=ce+preserve+0.06*compatibility
            require(bool(torch.isfinite(loss).item()),'nonfinite recurrent fit')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            optim.step()
            total+=float(loss.item())*len(idx)
        history.append(round(total/len(y),6))
    return history


def evaluate_model(model,X,parent,index,out,probe):
    model.eval()
    with torch.no_grad():
        for st in range(0,len(X),1024):
            xx=torch.from_numpy(X[st:st+1024])
            pp=torch.from_numpy(parent[st:st+1024].astype(np.int64))
            A,B,P=model(xx,pp,passes=4)
            nob=model(xx,pp,passes=4,no_B=True)[2][-1]
            i=index[st:st+len(xx)]
            for t in range(4):
                out['A'][i,t]=P[t].numpy()
                out['B'][i,t]=torch.sigmoid(B[t]).numpy()
            out['A'][i,4]=nob.numpy()
            selected=torch.topk(P[0][:,4],k=min(len(xx),16)).indices
            m=torch.tanh(B[0][selected]).clone()
            m[:,4]=-1.
            alt=model(xx[selected],pp[selected],passes=2,override=m)[2][1]
            natural=P[1][selected]
            change=torch.mean(torch.abs(alt-natural),dim=1).numpy()
            probe['checked']+=len(selected)
            probe['change_sum']+=float(change.sum())
            probe['flip_count']+=int((alt.argmax(1)!=natural.argmax(1)).sum())
            probe['k4_less']+=int((alt[:,4]<natural[:,4]).sum())


def proposals(A,parent):
    out={'series18_parent':parent.copy()}
    for pass_index,pass_name in enumerate(PASS_NAMES):
        p=A[:,pass_index]
        proposed=p.argmax(axis=1).astype(np.int8)
        out[f'series20__{pass_name}__raw']=proposed
        ii=np.arange(len(parent))
        best=p[ii,proposed]
        original=p[ii,parent]
        for c in MARGINS:
            selected=(proposed!=parent)&((best-original)>c)
            name=f'series20__{pass_name}__margin{c:g}'
            out[name]=np.where(selected,proposed,parent).astype(np.int8)
    require(len(out)==26,'25 neural policies + S18')
    return out


def runner(a):
    require(not a.output.exists(),'do not overwrite trained evidence')
    torch.set_num_threads(2)
    np.random.seed(SEED);torch.manual_seed(SEED)
    started=time.monotonic()
    d=prepare(a)
    n=len(d['y'])
    arr={'A':np.full((n,5,7),np.nan,dtype=np.float32),
         'B':np.full((n,4,7),np.nan,dtype=np.float32)}
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    records=[]
    probe=dict(checked=0,change_sum=0.,flip_count=0,k4_less=0)
    for j,piece in enumerate(sorted(set(d['pieces'].tolist()))):
        test=(d['pieces']==piece)
        fold=int(np.unique(d['fold'][test])[0])
        fit=(d['fold']==fold)&~test
        require(not np.any(fit&test),'fit/test overlap')
        scaler=StandardScaler().fit(d['X'][fit])
        train=np.clip(scaler.transform(d['X'][fit]),-5,5).astype(np.float32)
        held=np.clip(scaler.transform(d['X'][test]),-5,5).astype(np.float32)
        torch.manual_seed(SEED+100*j)
        model=ResidualAB()
        with threadpool_limits(limits=2):
            history=train_model(model,train,d['y'][fit],
                                d['parent'][fit],SEED+100*j)
            evaluate_model(model,held,d['parent'][test],
                           np.flatnonzero(test),arr,probe)
        filename=f'{piece}__residual_A_B.pt'
        torch.save(dict(model=model.state_dict(),scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,piece=piece,fold=fold,
            fit_ids=d['ids'][fit],held_ids=d['ids'][test],seed=SEED+100*j),
            a.output/'models'/filename)
        records.append(dict(piece=piece,fold=fold,
            fit_rows=int(fit.sum()),held_rows=int(test.sum()),
            fit_pieces=sorted(set(d['pieces'][fit].tolist())),
            fit_sha256=hashlib.sha256(d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            held_sha256=hashlib.sha256(d['ids'][test].astype('<i8').tobytes()).hexdigest(),
            training_losses=history,model_file=filename))
        print(json.dumps(dict(piece=piece,fold=fold,trained=len(records),
            held=int(test.sum()),last_loss=history[-1],
            seconds=round(time.monotonic()-started,1))),flush=True)
    require(np.isfinite(arr['A']).all() and np.isfinite(arr['B']).all(),
            'incomplete recurrent probability collection')
    require(np.allclose(arr['A'].sum(2),1,atol=1e-4),'bad probability normalization')
    require(probe['checked']>0,'missing B intervention events')
    probe['mean_absolute_A2_change']=probe['change_sum']/probe['checked']
    require(probe['mean_absolute_A2_change']>1e-7,
            'B counterfactual had no impact on A2')
    policies=proposals(arr['A'],d['parent'])
    policies['freeze_parent']=d['freeze'].copy()
    require(len(policies)==27,'26 neural policies plus freeze')
    results={}
    for name,pred in policies.items():
        results[name]=dict(metrics=metrics(d['y'],pred),
            versus_series18=paired(d['y'],d['parent'],pred),
            versus_freeze=paired(d['y'],d['freeze'],pred),
            versus_yourmt3=paired(d['y'],d['yourmt3'],pred),
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],pred[d['fold']==f]),
                vs_parent=paired(d['y'][d['fold']==f],d['parent'][d['fold']==f],
                    pred[d['fold']==f])) for f in FOLDS})
    transitions={}
    for left,right in zip(PASS_NAMES[:3],PASS_NAMES[1:4]):
        v=policies[f'series20__{left}__raw']
        t=policies[f'series20__{right}__raw']
        mat=np.zeros((7,7),dtype=np.int64)
        np.add.at(mat,(v,t),1)
        transitions[f'{left}_to_{right}']=dict(paired=paired(d['y'],v,t),
            changed=int((v!=t).sum()),matrix=mat.tolist())
    sorted_results=sorted(results,key=lambda k:results[k]['metrics']['correct'],
                          reverse=True)
    conservative=[name for name in policies if name.startswith('series20__')
        and results[name]['versus_series18']['global']['regressions']==0
        and results[name]['versus_series18']['global']['corrections']>0
        and results[name]['metrics']['poly']['correct']>=2998]
    summary=dict(status='completed',independent_validation=False,
        source_parent_posthoc_selected=True,automatic_promotion=False,
        supervised_labels_exclude_evaluated_piece=True,
        recurrent_A_conditional_on_B=True,seed=SEED,epochs=EPOCHS,
        feature_sources=d['features_source'],source=d['source'],
        weights=records,B_K4_intervention=probe,policies=results,
        transitions=transitions,zero_loss_candidates=conservative,
        best_global_candidate=sorted_results[0],
        elapsed_seconds=round(time.monotonic()-started,2))
    (a.output/'report.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(policies)),
        predictions=np.column_stack(list(policies.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=d['ids'],
        policy_names=np.asarray(list(PASS_NAMES)),A_probs=arr['A'],B_compatibility=arr['B'])
    lines=['# S20 — Anchored recurrent A↔B↔A corrections','',
        'Research on previously exposed pieces, NOT independent validation.',
        '', '| Candidate | Global | Poly | New corrections vs S18 | Regressions vs S18 | Regressions vs freeze |',
        '|---|---:|---:|---:|---:|---:|']
    for name in sorted_results:
        m=results[name]['metrics']
        x=results[name]['versus_series18']['global']
        b=results[name]['versus_freeze']['global']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {x['corrections']} | "
            f"{x['regressions']} | {b['regressions']} |")
    lines+=['','## Corrections and regressions per recurrent passage',
        '| From → to | Corrections | Regressions | Net | Changed |',
        '|---|---:|---:|---:|---:|']
    for name,entry in transitions.items():
        paired_global=entry['paired']['global']
        lines.append(f"| {name} | {paired_global['corrections']} | "
            f"{paired_global['regressions']} | {paired_global['net']} | "
            f"{entry['changed']} |")
    lines+=['','## Signal from B excluding K4',
        f"Interventions tested: {probe['checked']}",
        f"Mean A2 probability difference: {probe['mean_absolute_A2_change']:.9f}",
        f"Changed A2 argmax decisions: {probe['flip_count']}",
        f"K4 probability decreased: {probe['k4_less']}",
        '','Prototype only, no production model changed.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        selftest()
    else:
        require(args.features is not None and args.s18 is not None and
                args.output is not None,'missing sources')
        runner(args)
