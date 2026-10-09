"""S27 — empirical test: must every recurrent selection run again?

Original A-first S20 weights, 19 piece-held models, 14 frozen scheduling policies.
Compute actual per-example A and B calls; no labels used in the decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_aba_residual import ResidualAB,onehot_prior
from scripts.loop_v273_native_risk import read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

POLICIES=(
 'full_4','full_2','full_3',
 'B_initial_only','B_if_K_changed',
 'B_if_delta_002','B_if_delta_005','B_if_delta_010',
 'msg_only_parent_candidate','msg_only_top2','msg_only_top3',
 'early_stable_002','early_stable_005','early_stable_010')
MARGINS=(.25,.40,.60)
PASSES={'full_2':2,'full_3':3}
POLY=range(2,7)


def sha_file(file):
    h=hashlib.sha256()
    with Path(file).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def masked_message(b_logits, current, parent, mode):
    msg=torch.tanh(b_logits)
    if mode=='msg_only_parent_candidate':
        arg=current.argmax(1)
        support=torch.zeros_like(msg)
        support.scatter_(1,arg[:,None],1.)
        support.scatter_(1,parent[:,None],1.)
        return msg*support
    if mode.startswith('msg_only_top'):
        k=int(mode[-1])
        cols=current.topk(k,dim=1).indices
        support=torch.zeros_like(msg)
        support.scatter_(1,cols,1.)
        return msg*support
    return msg


def run_policy(net,signal,parent,variant,check_sparsity=False):
    """Count only executed A/B sub-batch evaluations, not theoretical heads."""
    assert variant in POLICIES
    total=len(signal)
    assert signal.shape==(total,335) and parent.shape==(total,)
    prior=onehot_prior(parent)
    log_prior=torch.log(prior)
    previous=prior.clone()
    message=torch.zeros_like(prior)
    active=torch.ones(total,dtype=torch.bool)
    nA=nB=0
    states=[]
    max_steps=PASSES.get(variant,4)
    sparse_nonzero_max=0
    for t in range(max_steps):
        indices=torch.flatnonzero(active)
        if len(indices)==0:
            states.extend([previous.clone() for _ in range(max_steps-t)])
            break
        prev=previous[indices].clone()
        logits=log_prior[indices]+net.A(torch.cat(
            [signal[indices],prior[indices],prev,message[indices]],dim=1))
        updated=logits.softmax(1)
        previous[indices]=updated
        nA+=len(indices)
        states.append(previous.clone())
        if t==max_steps-1:
            continue
        # Early stopping is detected immediately after A2 or A3. No head is
        # executed on the stopped event after this decision.
        if variant.startswith('early_stable') and t>=1:
            cutoff=int(variant[-3:])/1000.
            distance=(updated-prev).abs().mean(1)
            stable=(updated.argmax(1)==prev.argmax(1))&(distance<cutoff)
            active[indices[stable]]=False
        indices=torch.flatnonzero(active)
        if len(indices)==0:
            continue
        # First B after A1 is mandatory; subsequent B can be skipped.
        need=torch.ones(len(indices),dtype=torch.bool)
        if t>=1:
            if variant=='B_initial_only':
                need[:]=False
            elif variant=='B_if_K_changed':
                # The proposal before the present A iteration is saved via
                # the previous whole-vector snapshot.
                old=states[-2][indices]
                need=(previous[indices].argmax(1)!=old.argmax(1))
            elif variant.startswith('B_if_delta_'):
                cutoff=int(variant[-3:])/1000.
                old=states[-2][indices]
                need=(previous[indices]-old).abs().mean(1)>=cutoff
        target=indices[need]
        if len(target):
            newB=net.B(torch.cat(
                [signal[target],prior[target],previous[target]],dim=1))
            bmessage=masked_message(newB,previous[target],parent[target],variant)
            message[target]=bmessage
            if variant.startswith('msg_only_'):
                sparse_nonzero_max=max(sparse_nonzero_max,int(
                    (bmessage.abs()>1e-12).sum(1).max().item()))
            nB+=len(target)
    assert len(states)==max_steps
    assert nA<=total*max_steps and nB<=total*max(0,max_steps-1)
    last=states[-1]
    if check_sparsity and variant.startswith('msg_only_'):
        assert sparse_nonzero_max<=2 if variant in (
            'msg_only_top2','msg_only_parent_candidate') else sparse_nonzero_max<=3
    return last,dict(A=nA,B=nB,A_per_event=nA/total,B_per_event=nB/total,
          max_sparse_components=sparse_nonzero_max),states


def decisions(parent,p,variant):
    ii=np.arange(len(parent))
    chosen=p.argmax(1).astype(np.int8)
    out={f'series27__{variant}__raw':chosen}
    gain=p[ii,chosen]-p[ii,parent]
    for cut in MARGINS:
        yes=(chosen!=parent)&(gain>cut)
        out[f'series27__{variant}__margin{cut:g}']=np.where(yes,chosen,parent).astype(np.int8)
    return out


def selftest():
    torch.manual_seed(27327)
    net=ResidualAB()
    torch.nn.init.normal_(net.A[-1].weight,mean=0,std=.03)
    net.eval()
    x=torch.randn(64,335)
    y=torch.tensor([i%7 for i in range(64)])
    with torch.no_grad():
        full,c,stages=run_policy(net,x,y,'full_4')
        standard=net(x,y,passes=4)[2]
        assert c['A']==4*len(y) and c['B']==3*len(y)
        for a,b in zip(stages,standard):
            assert torch.allclose(a,b,atol=1e-6)
        initial,ci,_=run_policy(net,x,y,'B_initial_only')
        assert ci['A']==4*len(y) and ci['B']==len(y)
        for variant in ('msg_only_top2','msg_only_top3',
                        'msg_only_parent_candidate'):
            p,ct,_=run_policy(net,x,y,variant,check_sparsity=True)
            assert ct['A']==4*len(y) and ct['B']==3*len(y)
        stopped,cstop,_=run_policy(net,x,y,'early_stable_010')
        assert cstop['A']<=4*len(y)
    stable=ResidualAB()
    zero=torch.zeros((60,335))
    y=torch.tensor([i%7 for i in range(60)])
    with torch.no_grad():
        p,ct,_=run_policy(stable,zero,y,'early_stable_002')
        assert ct['A']==2*len(y) and ct['B']==len(y)
        assert torch.allclose(p,onehot_prior(y))
    assert len(POLICIES)==14
    print('PASS: exact full-4 reproduction, true B reuse, sparse messages, early stop and 14 policies')


def evaluate(args):
    require(not args.output.exists(),'refuse overwrite')
    torch.set_num_threads(2)
    started=time.monotonic()
    d=prepare(args)
    source=read(args.s20_probs)
    require(np.array_equal(source['global_index'],d['ids']),'S20 misaligned')
    historic=source['A_probs']
    require(historic.shape==(59309,5,7),'incomplete source distributions')
    original_report=json.loads((args.s20/'report.json').read_text())
    require(len(original_report['weights'])==19 and
            original_report['status']=='completed','incorrect prior trained model list')
    n=len(d['ids'])
    distribution={m:np.full((n,7),np.nan,np.float32) for m in POLICIES}
    costs={m:dict(A=0,B=0) for m in POLICIES}
    max_repro=0.
    checkpoints=[]
    for rec in original_report['weights']:
        name=rec['piece'];fold=rec['fold']
        held=d['pieces']==name
        fit=(d['fold']==fold)&~held
        require(held.any() and int(held.sum())==rec['held_rows'] and
                int(fit.sum())==rec['fit_rows'] and
                name not in rec['fit_pieces'],'piece excluded fit violated')
        require(rec['fit_sha256']==hashlib.sha256(
            d['ids'][fit].astype('<i8').tobytes()).hexdigest(),'fit hash drift')
        require(rec['held_sha256']==hashlib.sha256(
            d['ids'][held].astype('<i8').tobytes()).hexdigest(),'hold hash drift')
        file=args.s20/'models'/rec['model_file']
        require(file.exists(),'missing frozen checkpoint')
        checkpoint=torch.load(file,map_location='cpu',weights_only=False)
        require(checkpoint['piece']==name and checkpoint['fold']==fold and
                np.array_equal(checkpoint['fit_ids'],d['ids'][fit]) and
                np.array_equal(checkpoint['held_ids'],d['ids'][held]),
                'source weights use different events')
        net=ResidualAB()
        net.load_state_dict(checkpoint['model'])
        net.eval()
        xx=np.clip((d['X'][held]-checkpoint['scaler_mean'])/
            checkpoint['scaler_scale'],-5,5).astype(np.float32)
        yy=d['parent'][held].astype(np.int64)
        ix=np.flatnonzero(held)
        with torch.no_grad():
            for start in range(0,len(xx),256):
                at=ix[start:start+256]
                X=torch.from_numpy(xx[start:start+256])
                y=torch.from_numpy(yy[start:start+256])
                for mode in POLICIES:
                    p,c,stages=run_policy(net,X,y,mode,check_sparsity=True)
                    distribution[mode][at]=p.numpy()
                    costs[mode]['A']+=c['A']
                    costs[mode]['B']+=c['B']
                    if mode=='full_4':
                        for t,stage in enumerate(stages):
                            max_repro=max(max_repro,float(np.max(np.abs(
                                stage.numpy()-historic[at,t]))))
        checkpoints.append(dict(piece=name,fold=fold,sha256=sha_file(file),
            samples=int(held.sum())))
        print(json.dumps(dict(piece=name,fold=fold,completed=len(checkpoints),
            source_replay_maxdiff=max_repro,seconds=round(time.monotonic()-started,1))),
            flush=True)
    require(max_repro<5e-5,
        'reference full-4 no longer reproduces original S20 distributions')
    for mode in POLICIES:
        require(np.isfinite(distribution[mode]).all(),'incomplete policy '+mode)
        require(np.allclose(distribution[mode].sum(1),1,atol=1e-5),
                'non-normalized prediction '+mode)
    require(costs['full_4']==dict(A=4*n,B=3*n),
            'true A/B reference cost drift')
    require(costs['B_initial_only']==dict(A=4*n,B=n),
            'B caching did not save real evaluations')
    out={'series18_parent':d['parent'].copy(),'freeze_parent':d['freeze'].copy()}
    comparisons={}
    full=distribution['full_4']
    for mode,p in distribution.items():
        out.update(decisions(d['parent'],p,mode))
        costs[mode]['A_per_event']=costs[mode]['A']/n
        costs[mode]['B_per_event']=costs[mode]['B']/n
        costs[mode]['equivalent_head_calls_vs_full']=(costs[mode]['A']+
            costs[mode]['B'])/(costs['full_4']['A']+costs['full_4']['B'])
        comparisons[mode]=dict(
            changed_K_vs_full=int((p.argmax(1)!=full.argmax(1)).sum()),
            exactly_equal_argmax=bool(np.array_equal(p.argmax(1),full.argmax(1))),
            max_absolute_probability_deviation=float(np.abs(p-full).max()),
            mean_absolute_probability_deviation=float(np.abs(p-full).mean()),
            distribution_equal_tol_1e_5=bool(np.allclose(p,full,atol=1e-5)),
            costs=costs[mode])
    require(len(out)==58,'14×4 policies plus two parents')
    audits={}
    for name,pred in out.items():
        changed=pred!=d['parent']
        audits[name]=dict(
            metrics=metrics(d['y'],pred),
            vs_S18=paired(d['y'],d['parent'],pred),
            vs_freeze=paired(d['y'],d['freeze'],pred),
            vs_YourMT3=paired(d['y'],d['yourmt3'],pred),
            corrected=int((changed&(pred==d['y'])&(d['parent']!=d['y'])).sum()),
            regressed=int((changed&(pred!=d['y'])&(d['parent']==d['y'])).sum()),
            neutral=int((changed&(pred!=d['y'])&(d['parent']!=d['y'])).sum()),
            by_true_K={str(k):dict(total=int((d['y']==k).sum()),
                correct=int(((d['y']==k)&(pred==d['y'])).sum()),
                corrections=int(((d['y']==k)&changed&(pred==d['y'])&
                    (d['parent']!=d['y'])).sum()),
                regressions=int(((d['y']==k)&changed&(pred!=d['y'])&
                    (d['parent']==d['y'])).sum()))
                for k in range(7)},
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                pred[d['fold']==f]),vs_S18=paired(d['y'][d['fold']==f],
                d['parent'][d['fold']==f],pred[d['fold']==f]))
                for f in FOLDS})
    ranked=sorted(audits,key=lambda name:(
        audits[name]['metrics']['correct'],
        audits[name]['metrics']['poly']['correct']),reverse=True)
    no_loss=[name for name in out if name.startswith('series27__') and
        audits[name]['corrected']>0 and audits[name]['regressed']==0 and
        audits[name]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',question='Is every A/B revisit necessary?',
        posthoc_exposed_development_cohort=True,independent_validation=False,
        source_model_run=37858383973,
        original_source_max_replay_difference=max_repro,
        not_promoted=True,truth_not_used_in_policy=True,
        model_SHA256=checkpoints,comparisons=comparisons,candidates=audits,
        strict_no_loss_candidates=no_loss,
        policies=POLICIES,total_policy_decisions=len(out),
        best_global=ranked[0],elapsed_seconds=round(time.monotonic()-started,1))
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(args.output/'distributions.npz',global_index=d['ids'],
        variant_ids=np.asarray(POLICIES),
        final_probabilities=np.stack(list(distribution.values()),axis=1))
    np.savez_compressed(args.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    lines=['# Série27 — repassage complet ou partiel de A/B ?','',
        '**Même poids S20, même signal, chaque morceau exclus du fit initial.**',
        'Cohorte déjà explorée : pas de validation externe ni promotion.',
        f'Rejeu complet S20: max |p_new−p_source| = {max_repro:.9g}.',
        'Les coûts sont des évaluations *échantillon × tête* réelles, pas des ms CPU/GPU.',
        '', '| Mode | A / événement | B / événement | Coût relatif | K différents du full-4 | Distribution identique |',
        '|---|---:|---:|---:|---:|---|']
    for name,c in comparisons.items():
        cost=c['costs']
        lines.append(f"| {name} | {cost['A_per_event']:.3f} | "
            f"{cost['B_per_event']:.3f} | "
            f"{cost['equivalent_head_calls_vs_full']*100:.2f}% | "
            f"{c['changed_K_vs_full']} | "
            f"{'oui' if c['distribution_equal_tol_1e_5'] else 'non'} |")
    lines+=['','## Qualité de chaque politique, y compris négatives',
        '| Variante | Global | Poly | Corrigées vs S18 | Régressées vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name in ranked:
        x=audits[name];m=x['metrics']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | {x['corrected']} | {x['regressed']} |")
    lines+=['',f'Strict no-loss candidates: {len(no_loss)}',
        'No policies are selected using the truth of an evaluated event.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--features',type=Path)
    parser.add_argument('--s18',type=Path)
    parser.add_argument('--s20',type=Path)
    parser.add_argument('--s20-probs',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s20 and
            a.s20_probs and a.output,'all original feature sources required')
        evaluate(a)
