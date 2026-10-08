"""S25: supervised B-first A/B recurrence, matching S20 model capacity and fit splits.

Four B predictions are learned at t=0 (before any A), then after A1/A2/A3.
Four A predictions are CE-trained. B's previous output is input to next A.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_aba_residual import (
    ResidualAB,onehot_prior,train_model,proposals,SEED,EPOCHS,BATCH,LR)
from scripts.loop_v273_native_risk import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require


class TrainedBfirst(ResidualAB):
    def forward(self,x,parent,passes=4,no_B=False,override=None):
        assert x.ndim==2 and x.shape[1]==self.num_features
        assert 1<=passes<=4 and len(x)==len(parent)
        prior=onehot_prior(parent).to(dtype=x.dtype,device=x.device)
        base=torch.log(prior)
        old=prior
        b=self.B(torch.cat((x,prior,old),dim=1))
        Bs=[b]
        message=torch.zeros_like(b) if no_B else torch.tanh(b)
        if override is not None:
            require(override.shape==message.shape,'B-first intervention dimensionality')
            message=override
        As=[];Ps=[]
        for t in range(passes):
            logits=base+self.A(torch.cat((x,prior,old,message),dim=1))
            p=torch.softmax(logits,dim=1)
            As.append(logits);Ps.append(p)
            old=p
            if t<passes-1:
                b=self.B(torch.cat((x,prior,p),dim=1))
                Bs.append(b)
                message=torch.zeros_like(b) if no_B else torch.tanh(b)
        return As,Bs,Ps


def selftest():
    torch.manual_seed(27320)
    m=TrainedBfirst()
    x=torch.randn(9,335)
    parent=torch.tensor([0,1,2,3,4,5,6,4,2])
    with torch.no_grad():
        A,B,P=m(x,parent,passes=4)
        assert len(A)==len(B)==len(P)==4
        assert torch.allclose(P[0],onehot_prior(parent),atol=1e-6)
        torch.nn.init.normal_(m.A[-1].weight,std=.04)
        normal=m(x,parent,passes=2)[2][0]
        disabled=m(x,parent,passes=2,no_B=True)[2][0]
        assert (normal-disabled).abs().max().item()>1e-6
        b=m(x,parent,passes=2)[1][0]
        intervention=torch.tanh(b).clone();intervention[:,4]=-1.
        changed=m(x,parent,passes=2,override=intervention)[2][0]
        assert (normal-changed).abs().max().item()>1e-6
        assert m.A[0].in_features==356 and m.B[0].in_features==349
    print('PASS: B initial impacts A1, shared weights, four B/CE losses and K4 intervention')


def evaluate(model,xtest,pbase,output,held,probe):
    model.eval()
    with torch.no_grad():
        for start in range(0,len(xtest),512):
            xx=torch.from_numpy(xtest[start:start+512])
            yy=torch.from_numpy(pbase[start:start+512].astype(np.int64))
            A,B,P=model(xx,yy,passes=4)
            off=model(xx,yy,passes=4,no_B=True)[2][-1]
            ii=held[start:start+len(xx)]
            for step in range(4):
                output['A'][ii,step]=P[step].numpy()
                output['B'][ii,step]=torch.sigmoid(B[step]).numpy()
            output['A'][ii,4]=off.numpy()
            best=torch.topk(P[0][:,4],k=min(16,len(xx))).indices
            change=torch.tanh(B[0][best]).clone()
            change[:,4]=-1.
            alt=model(xx[best],yy[best],passes=1,override=change)[2][0]
            old=P[0][best]
            probe['tested']+=len(best)
            probe['mean_absolute_change_sum']+=float(torch.mean(
                torch.abs(old-alt),dim=1).sum().item())
            probe['argmax_changed']+=int((alt.argmax(1)!=old.argmax(1)).sum().item())
            probe['K4_probability_reduced']+=int((alt[:,4]<old[:,4]).sum().item())


def run(args):
    require(not args.output.exists(),'no overwriting experiment')
    t0=time.monotonic()
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    d=prepare(args)
    source=read(args.s20_probs)
    require(np.array_equal(source['global_index'],d['ids']),'S20 cohort alignment')
    old=np.asarray(source['A_probs'],np.float32)
    require(old.shape==(59309,5,7),'old S20 A matrix schema')
    n=len(d['ids'])
    output={'A':np.full((n,5,7),np.nan,np.float32),
            'B':np.full((n,4,7),np.nan,np.float32)}
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    ledger=[];probe=dict(tested=0,mean_absolute_change_sum=0.,
                         argmax_changed=0,K4_probability_reduced=0)
    for j,piece in enumerate(sorted(set(d['pieces'].tolist()))):
        held=d['pieces']==piece
        fold=int(np.unique(d['fold'][held])[0])
        fit=(d['fold']==fold)&~held
        require(fit.sum()>200 and len(set(d['y'][fit].tolist()))>3,
                'insufficient training labels')
        require(not (fit&held).any(),'piece leaked')
        scaler=StandardScaler().fit(d['X'][fit])
        xtrain=np.clip(scaler.transform(d['X'][fit]),-5,5).astype(np.float32)
        xtest=np.clip(scaler.transform(d['X'][held]),-5,5).astype(np.float32)
        torch.manual_seed(SEED+100*j)
        m=TrainedBfirst()
        with threadpool_limits(limits=2):
            history=train_model(m,xtrain,d['y'][fit],d['parent'][fit],
                                SEED+100*j)
            evaluate(m,xtest,d['parent'][held],output,np.flatnonzero(held),probe)
        name=f'{piece}__trained_Bfirst.pt'
        torch.save(dict(model=m.state_dict(),scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,piece=piece,fold=fold,
            fit_ids=d['ids'][fit],held_ids=d['ids'][held],seed=SEED+100*j),
            args.output/'models'/name)
        ledger.append(dict(piece=piece,fold=fold,fit_rows=int(fit.sum()),
            held_rows=int(held.sum()),train_pieces=sorted(set(d['pieces'][fit].tolist())),
            fit_sha256=hashlib.sha256(d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            held_sha256=hashlib.sha256(d['ids'][held].astype('<i8').tobytes()).hexdigest(),
            training_losses=history,model_file=name))
        print(json.dumps(dict(piece=piece,fold=fold,models=len(ledger),
            loss=history[-1],seconds=round(time.monotonic()-t0,1))),flush=True)
    require(np.isfinite(output['A']).all() and np.isfinite(output['B']).all(),
            'unfinished models')
    require(np.allclose(output['A'].sum(axis=2),1,atol=1e-4),
            'A output is not distribution')
    require(probe['tested']>0,'no K4 perturbations')
    probe['mean_absolute_A1_change']=probe['mean_absolute_change_sum']/probe['tested']
    require(probe['mean_absolute_A1_change']>1e-7,
            'trained A insensitive to prior B message')
    ps=proposals(output['A'],d['parent'])
    preds={name.replace('series20__','series25__'):pred
           for name,pred in ps.items()}
    preds['freeze_reference']=d['freeze'].copy()
    alone=output['B'][:,0,:].argmax(1).astype(np.int8)
    preds['Bbefore_A0__raw']=alone
    require(len(preds)==28,'expected parent+25 policies+freeze+B')
    results={}
    for name,pred in preds.items():
        results[name]=dict(metrics=metrics(d['y'],pred),
            versus_S18=paired(d['y'],d['parent'],pred),
            versus_freeze=paired(d['y'],d['freeze'],pred),
            versus_YourMT3=paired(d['y'],d['yourmt3'],pred),
            per_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                pred[d['fold']==f]),vs_S18=paired(d['y'][d['fold']==f],
                d['parent'][d['fold']==f],pred[d['fold']==f]))
                for f in FOLDS})
    bypass={}
    for t,prefix in enumerate(('pass1','pass2','pass3','pass4')):
        before=old[:,t,:].argmax(1).astype(np.int8)
        reverse=output['A'][:,t,:].argmax(1).astype(np.int8)
        byclass={}
        for k in range(7):
            at=d['y']==k
            byclass[str(k)]=dict(Afirst_correct=int((at&(before==d['y'])).sum()),
                Bfirst_correct=int((at&(reverse==d['y'])).sum()),
                Bfirst_corrections=int((at&(before!=d['y'])&
                                       (reverse==d['y'])).sum()),
                Bfirst_regressions=int((at&(before==d['y'])&
                                      (reverse!=d['y'])).sum()))
        bypass[prefix]=dict(paired_vs_S20_Afirst=paired(d['y'],before,reverse),
            raw_changed=int((before!=reverse).sum()),per_true_K=byclass)
    b_to_a=dict(B_correct=int((alone==d['y']).sum()),
        B_wrong_to_A1_correct=int(((alone!=d['y'])&
            (output['A'][:,0,:].argmax(1)==d['y'])).sum()),
        B_correct_to_A1_wrong=int(((alone==d['y'])&
            (output['A'][:,0,:].argmax(1)!=d['y'])).sum()))
    names=list(preds)
    ranking=sorted(names,key=lambda k:(
        results[k]['metrics']['correct'],results[k]['metrics']['poly']['correct']),reverse=True)
    safe=[name for name in names if name.startswith('series25__') and
        results[name]['versus_S18']['global']['regressions']==0 and
        results[name]['versus_S18']['global']['corrections']>0 and
        results[name]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',prototype_only=True,
        independent_validation=False,already_exposed_cohort=True,
        parent_posthoc_selected=True,
        A_B_sequential_order='B,A,B,A,B,A,B,A',
        B_outputs_before_A=True,
        training_seed_matched_S20=SEED,
        same_train_budget_as_S20=True,
        no_production_promotion=True,
        fit_models=ledger,fit_count=len(ledger),
        B_only_versus_A1=b_to_a,
        K4_message_intervention=probe,
        per_pass_vs_trained_Afirst=bypass,
        policy_count=len(preds),policies=results,
        strict_zero_loss_candidates=safe,
        best_global=ranking[0],
        seconds=round(time.monotonic()-t0,2))
    (args.output/'report.json').write_text(json.dumps(report,indent=2,
        sort_keys=True)+'\n')
    np.savez_compressed(args.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(names),predictions=np.column_stack(list(preds.values())))
    np.savez_compressed(args.output/'probabilities.npz',
        global_index=d['ids'],A_probs=output['A'],B_compatibility=output['B'])
    lines=['# Série25 — B-first avec apprentissage à l ordre inverse','',
        '**Cohorte de développement exposée, pas de validation indépendante.**',
        'Mêmes 19 splits, epochs, capacité et seeds que S20 A-first.',
        '','## B faible en premier : diagnostic exact',
        f"B seule correcte: {b_to_a['B_correct']}",
        f"B se trompe et A1 corrige: {b_to_a['B_wrong_to_A1_correct']}",
        f"B correcte et A1 casse: {b_to_a['B_correct_to_A1_wrong']}",
        f"Intervention B avant A1: changement moyen de distribution {probe['mean_absolute_A1_change']:.8f}",
        '','## Ordres à nombre égal de passages de A',
        '| Passage | Corrections B-first vs A-first S20 | Régressions B-first vs A-first S20 | Net |',
        '|---|---:|---:|---:|']
    for name,v in bypass.items():
        pair=v['paired_vs_S20_Afirst']['global']
        lines.append(f"| {name} | {pair['corrections']} | {pair['regressions']} | {pair['net']:+d} |")
    lines+=['','## Toutes les politiques, y compris négatives',
        '| Politique | Exact global | Exact poly | Corrections vs S18 | Régressions vs S18 |',
        '|---|---:|---:|---:|---:|']
    for key in ranking:
        m=results[key]['metrics'];r=results[key]['versus_S18']['global']
        lines.append(f"| {key} | {100*m['exact']:.4f}% | {100*m['poly']['exact']:.4f}% | "
            f"{r['corrections']} | {r['regressions']} |")
    lines+=['',f'Strict zero-loss candidates: {len(safe)}',
        'No independent unseen validation or promotion.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s20-probs',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s20_probs and a.output,
                'required acoustic and source parent missing')
        run(a)
