"""S19 — a true unrolled neural A→B→A multi-pass Exact-K selector.

A receives the ORIGINAL audio and a learned 7-dimensional exclusion message from B,
so the second A evaluation is NOT just renormalizing a masked first-pass output.
All supervised fits exclude the evaluated piece. Research, no model promotion.
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

from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require
from scripts.loop_v273_native_risk import read, full_members_and_geometry, designs
from scripts.loop_v273_birth_flow import collect
from scripts.loop_v273_policy_gate import GLOBAL, POLY

SEED = 27319
EPOCHS = 6
BATCH = 512
LR = 1e-3
PASSPRIOR = (0.20, 0.30, 0.50, 1.00)
NAMES = ('A_pass1', 'ABA_pass2', 'ABABA_pass3', 'ABABABA_pass4',
         'A_no_B_pass4')
S18 = 'series18__flow_logistic__k21__pbase_gt0.95'
DEVICE = 'cpu'


class RecurrentSelector(nn.Module):
    """Shared A and B: B scores are fed as variables of the NEXT A pass."""

    def __init__(self, num_features=335):
        super().__init__()
        self.num_features = num_features
        self.A = nn.Sequential(
            nn.Linear(num_features + 14, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 7))
        self.B = nn.Sequential(
            nn.Linear(num_features + 7, 96), nn.GELU(),
            nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 7))

    def forward(self, x, passes=4, no_B=False, second_message_override=None):
        assert x.ndim == 2 and x.shape[1] == self.num_features
        assert 1 <= passes <= 4
        before = torch.zeros((x.shape[0], 7), dtype=x.dtype, device=x.device)
        message = torch.zeros_like(before)
        logits_A, logits_B, soft_A = [], [], []
        for t in range(passes):
            # A reinterprets acoustic X in light of previous A and B outputs.
            a = self.A(torch.cat((x, before, message), dim=1))
            p = torch.softmax(a, dim=1)
            b = self.B(torch.cat((x, p), dim=1))
            logits_A.append(a)
            logits_B.append(b)
            soft_A.append(p)
            before = p
            message = torch.zeros_like(b) if no_B else torch.tanh(b)
            if t == 0 and second_message_override is not None:
                require(second_message_override.shape == message.shape,
                        'intervention dimensionality')
                message = second_message_override
        return logits_A, logits_B, soft_A


def static_selftest():
    torch.manual_seed(SEED)
    selector = RecurrentSelector(335)
    x = torch.randn(5, 335)
    with torch.no_grad():
        a, b, p = selector(x, passes=4)
        assert len(p) == 4 and all(q.shape == (5, 7) for q in p)
        assert all(torch.allclose(q.sum(1), torch.ones(5)) for q in p)
        assert torch.isfinite(p[-1]).all()
        assert not torch.allclose(p[0], p[1]), 'second A is not re-evaluated'
        # Same acoustic X and same A at t1; only B's message changes.
        m = torch.tanh(b[0]).clone()
        m[:, 4] = -4.0
        a2_ex = selector(x, passes=2, second_message_override=m)[2][1]
        changed = (a2_ex - p[1]).abs().max().item()
        assert changed > 1e-5, 'B cannot influence second A!'
        base4 = selector(x, passes=4, no_B=True)[2][-1]
        assert (base4 - p[-1]).abs().max().item() > 1e-5, 'B ablation ineffective'
        assert selector.A[0].in_features == 349
        assert selector.B[0].in_features == 342
    print(json.dumps(dict(test='PASS', feedback_changes_A2=True,
                          feedback_effect_initial=float(changed),
                          four_passes=True, no_B_ablation=True)), flush=True)


def lookup(z, policy):
    ids=list(map(str,z['variant_ids']))
    require(policy in ids,'missing '+policy)
    return z['predictions'][:,ids.index(policy)].astype(np.int8)


def prepare(a):
    core=a.root/'analysis/evidence'
    d=read(core/'v273-regression-loops/prepared/inputs.npz')
    ids,y,b,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==59309 and set(fold.tolist())==set(FOLDS),'cohort drift')
    old=read(a.s18)
    require(np.array_equal(old['global_index'],ids),'S18 IDs changed')
    ref=lookup(old,S18)
    require(metrics(y,ref)['correct']==49178,'S18 reference global changed')
    require(metrics(y,ref)['poly']['correct']==2998,'S18 reference poly changed')
    require(paired(y,b,ref)['global']['corrections']==2934 and
            paired(y,b,ref)['global']['regressions']==2210,'S18 baseline drift')
    five=read(core/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(five['global_index'],ids),'G/P source alignment')
    g=lookup(five,GLOBAL);p=lookup(five,POLY)
    # Only input is original audio and fold-excluded producer predictions.
    pieces,geom,provenance=full_members_and_geometry(a.root,ids,y,b,fold)
    sound,flow,feature_sources=collect(a.features,ids,y,b,fold,d)
    votes=designs(d,ids,b,g,p,geom)['votes_time']
    features=np.column_stack([votes,sound,flow]).astype(np.float32)
    require(features.shape==(len(ids),335) and np.isfinite(features).all(),
            'signal feature dimensions or nonfinite signal')
    comp=read(core/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(comp['global_index'],ids),'competitor alignment')
    target=comp['yourmt3_K']
    require(int((target==y).sum())==51328,'YourMT3+ reference changed')
    return dict(ids=ids,y=y,freeze=b,fold=fold,parent=ref,
                yourmt3=target,pieces=pieces,
                X=features,source=provenance,features_source=feature_sources)


def class_balancing(target):
    counts=np.bincount(target,minlength=7).astype(np.float64)
    raw=np.sqrt(counts.sum()/(7*np.maximum(counts,1)))
    raw=np.clip(raw,.3,4.0)
    raw=raw/np.mean(raw[target])
    return raw.astype(np.float32)


def optimize(model,x,labels,seed):
    torch.manual_seed(seed)
    weights=class_balancing(labels)
    class_tensor=torch.from_numpy(weights)
    targets=torch.from_numpy(labels.astype(np.int64))
    samples=torch.from_numpy(x.astype(np.float32))
    model.train()
    optimizer=torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.001)
    posweight=torch.full((7,),6.,dtype=torch.float32)
    history=[]
    for epoch in range(EPOCHS):
        shuffled=torch.randperm(len(x),generator=torch.Generator().manual_seed(seed+epoch))
        total=0.; count=0
        for start in range(0,len(x),BATCH):
            i=shuffled[start:start+BATCH]
            xx=samples[i]; yy=targets[i]
            a,b,_=model(xx,passes=4)
            onehot=F.one_hot(yy,7).float()
            ce=sum(w*F.cross_entropy(z, yy, weight=class_tensor)
                   for w,z in zip(PASSPRIOR,a))
            bce=sum(F.binary_cross_entropy_with_logits(
                        z,onehot,pos_weight=posweight) for z in b)
            loss=ce+0.06*bce
            require(bool(torch.isfinite(loss).item()),'bad loss in training')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.)
            optimizer.step()
            total+=float(loss.item())*len(i)
            count+=len(i)
        history.append(round(total/count,6))
    return history,weights.tolist()


def evaluate_piece(model,x,out,indices,probe):
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),1024):
            chunk=torch.from_numpy(x[start:start+1024])
            a,b,p=model(chunk,passes=4)
            no_b=model(chunk,passes=4,no_B=True)[2][-1]
            for t in range(4):
                out['probs'][indices[start:start+len(chunk)],t]=p[t].numpy()
                out['B'][indices[start:start+len(chunk)],t]=torch.sigmoid(b[t]).numpy()
            out['probs'][indices[start:start+len(chunk)],4]=no_b.numpy()
            # Intervention test is only a diagnostic: NEVER used as a prediction.
            # Probe the highest-K4 events in EVERY held piece. The
            # network may never argmax K4 early in training.
            sel=torch.topk(p[0][:,4],k=min(16,len(chunk))).indices
            if len(sel):
                msg=torch.tanh(b[0][sel]).clone()
                msg[:,4]=-1.0
                alt=model(chunk[sel],passes=2,
                    second_message_override=msg)[2][1]
                natural=p[1][sel]
                impact=torch.mean(torch.abs(alt-natural),dim=1).numpy()
                probe['tested']+=len(sel)
                probe['total_shift']+=float(impact.sum())
                probe['max_shift']=max(probe['max_shift'],float(impact.max()))
                probe['K4_reduced']+=int((alt[:,4]<natural[:,4]).sum().item())
                probe['decision_changed']+=int((alt.argmax(1)!=natural.argmax(1)).sum().item())


def metrics_rows(y,parent,freeze,ref,name):
    m=metrics(y,ref)
    return dict(metrics=m,
        paired_vs_series18=paired(y,parent,ref),
        paired_vs_freeze=paired(y,freeze,ref),
        changed_vs_series18=int((ref!=parent).sum()),
        predicted_distribution=np.bincount(ref,minlength=7).tolist())


def train(a):
    require(not a.output.exists(),'never overwrite experiment')
    clock=time.monotonic()
    torch.set_num_threads(2)
    np.random.seed(SEED);torch.manual_seed(SEED)
    d=prepare(a)
    n=len(d['y'])
    arr=dict(probs=np.full((n,5,7),np.nan,dtype=np.float32),
             B=np.full((n,4,7),np.nan,dtype=np.float32))
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    records=[];probe=dict(tested=0,total_shift=0.,max_shift=0.,
                          K4_reduced=0,decision_changed=0)
    for j,piece in enumerate(sorted(set(d['pieces'].tolist()))):
        held=d['pieces']==piece
        require(held.any(), 'empty test piece')
        fold=int(np.unique(d['fold'][held])[0])
        fit=(d['fold']==fold)&(~held)
        require(fit.sum()>200 and len(set(d['y'][fit].tolist()))>=4,
                'underpopulated label-excluded fit')
        require(not np.any(fit & held),'train/test overlap')
        scaler=StandardScaler().fit(d['X'][fit])
        xfit=np.clip(scaler.transform(d['X'][fit]),-5,5).astype(np.float32)
        xtest=np.clip(scaler.transform(d['X'][held]),-5,5).astype(np.float32)
        # Reinitialization is essential: do not transfer labels across folds.
        torch.manual_seed(SEED+100*j)
        network=RecurrentSelector(d['X'].shape[1])
        with threadpool_limits(limits=2):
            losses,classweights=optimize(network,xfit,d['y'][fit],seed=SEED+100*j)
            evaluate_piece(network,xtest,arr,np.flatnonzero(held),probe)
        modelfile=f'{piece}__A_B_A.pt'
        torch.save(dict(state_dict=network.state_dict(),
                        scaler_mean=scaler.mean_.astype(np.float32),
                        scaler_scale=scaler.scale_.astype(np.float32),
                        seed=SEED+100*j,piece=piece,fold=fold,
                        fit_ids=d['ids'][fit],held_ids=d['ids'][held],
                        class_weights=np.asarray(classweights)),
                   a.output/'models'/modelfile)
        records.append(dict(piece=piece,fold=fold,fit_rows=int(fit.sum()),
            held_rows=int(held.sum()),train_pieces=sorted(set(d['pieces'][fit].tolist())),
            training_loss=losses,class_weights=classweights,
            fit_sha256=hashlib.sha256(d['ids'][fit].astype('<i8').tobytes()).hexdigest(),
            held_sha256=hashlib.sha256(d['ids'][held].astype('<i8').tobytes()).hexdigest(),
            model_file=modelfile))
        print(json.dumps(dict(piece=piece,fold=fold,trained=len(records),
                    held_rows=int(held.sum()),last_loss=losses[-1],
                    seconds=round(time.monotonic()-clock,1))),flush=True)
    require(np.isfinite(arr['probs']).all() and np.isfinite(arr['B']).all(),
            'incomplete crosspiece inference')
    require(np.allclose(arr['probs'].sum(2),1,atol=1e-4),
            'probability normalization')
    require(probe['tested']>0,'no K4 stress-test examples for B input')
    probe['mean_prob_absolute_change']=probe['total_shift']/probe['tested']
    require(probe['mean_prob_absolute_change']>1e-7,
            'B intervention had no measurable effect on second A')
    predictions=dict(zip(NAMES,
        [arr['probs'][:,t].argmax(1).astype(np.int8) for t in range(5)]))
    predictions['series18_reference']=d['parent']
    predictions['freeze_reference']=d['freeze']
    results={}
    for key,pred in predictions.items():
        results[key]=metrics_rows(d['y'],d['parent'],d['freeze'],pred,key)
        results[key]['paired_vs_yourmt3']=paired(d['y'],d['yourmt3'],pred)
        results[key]['by_fold']={
            str(f):dict(metrics=metrics(d['y'][d['fold']==f],pred[d['fold']==f]),
                versus_S18=paired(d['y'][d['fold']==f],
                    d['parent'][d['fold']==f],pred[d['fold']==f]))
            for f in FOLDS}
    paths=list(NAMES[:4])
    transitions={}
    for left,right in zip(paths[:-1],paths[1:]):
        p0=predictions[left];p1=predictions[right]
        mat=np.zeros((7,7),dtype=np.int64)
        np.add.at(mat,(p0,p1),1)
        transitions[f'{left}__to__{right}']=dict(
            matrix=mat.tolist(),
            corrections=paired(d['y'],p0,p1),
            changed=int((p0!=p1).sum()),
            new_corrections_by_true_K={
                str(k):int(np.sum((d['y']==k)&(p0!=d['y'])&(p1==d['y'])))
                for k in range(7)},
            newly_wrong_by_true_K={
                str(k):int(np.sum((d['y']==k)&(p0==d['y'])&(p1!=d['y'])))
                for k in range(7)})
    poly_wins=[key for key in NAMES if
        results[key]['paired_vs_series18']['global']['regressions']==0 and
        results[key]['paired_vs_series18']['global']['corrections']>0 and
        results[key]['metrics']['poly']['correct']>=2998]
    report=dict(status='completed',training_is_recurrent=True,
        learned_B_feedback_input_to_A=True,
        optimization_backpropagates_through_four_passes=True,
        original_acoustic_features_only=True,independent_validation=False,
        evaluation_cohort_previously_exposed=True,automatic_promotion=False,
        test_piece_excluded_from_fit=True,no_yourmt3_prediction_in_features=True,
        prototype_only=True,unseen_composition_validation_required=True,
        experiment='V273_SERIE19_ABA',seed=SEED,epochs=EPOCHS,
        batch=BATCH,learning_rate=LR,num_training_models=len(records),
        feature_sources=d['features_source'],source=d['source'],
        B_intervention_K4=probe,policies=results,
        transitions=transitions,
        research_candidates_not_losing_parent_corrections=poly_wins,
        fit_records=records,elapsed_seconds=round(time.monotonic()-clock,2))
    (a.output/'report.json').write_text(json.dumps(report,indent=2,
        sort_keys=True,allow_nan=False)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=d['ids'],
        true_K=d['y'],fold=d['fold'],variant_ids=np.asarray(list(predictions)),
        predictions=np.column_stack(list(predictions.values())))
    np.savez_compressed(a.output/'probabilities.npz',global_index=d['ids'],
        variant_ids=np.asarray(NAMES),A_probs=arr['probs'],B_compatibility=arr['B'])
    lines=['# S19 — True recurrent A→B→A neural selection','',
        '**Development-exposed cohort. No independent unseen validation. No promotion.**',
        '', '| Policy | Exact global | Exact poly | Corrections vs S18 | Regressions vs S18 | Corrections vs freeze | Regressions vs freeze |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key in predictions:
        m=results[key]['metrics']
        p=results[key]['paired_vs_series18']['global']
        f=results[key]['paired_vs_freeze']['global']
        lines.append(f"| {key} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {p['corrections']} | "
            f"{p['regressions']} | {f['corrections']} | {f['regressions']} |")
    lines+=['','## Iteration-to-iteration correction balance','',
        '| Transition | Corrected | Newly regressed | Net | Rows changed |',
        '|---|---:|---:|---:|---:|']
    for key,value in transitions.items():
        p=value['corrections']['global']
        lines.append(f"| {key} | {p['corrections']} | {p['regressions']} | "
            f"{p['net']:+d} | {value['changed']} |")
    lines+=['','## B counterfactual exclusion of K4','',
        f"Top-K4 probability candidates tested: {probe['tested']}",
        f"Mean absolute distribution change in A2: {probe['mean_prob_absolute_change']:.8f}",
        f"A2 class decision changed by only the B intervention: {probe['decision_changed']}",
        f"K4 probability decreased: {probe['K4_reduced']}/{probe['tested']}",
        '','All five neural prediction variants, per-pass class distributions and B outputs retained.',
        'No post-hoc cherry-picking may be presented as validation.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--features',type=Path)
    parser.add_argument('--s18',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        static_selftest()
    else:
        require(args.features is not None and args.s18 is not None and
                args.output is not None,'need all original inputs')
        train(args)
