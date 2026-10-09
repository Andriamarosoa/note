"""S35: connect 18 non-H9 expert/action heads to an event-wise dynamic router.

All genuine acoustic heads are fitted with outer-fold isolation. Router
quality targets come from INNER OOF specialist predictions that never train
on the current outer test fold. A/B stateful S29 is an additional existing
producer. STOP and each expert are re-considered after every choice.

H8 cannot be active without full aligned real-waveform pitch-shift OOF
probabilities; H9 is NOT in any action, model input or registry.
No efficiency claim: expert posteriors are currently precomputed per fold.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_v273_aba_recurrent import prepare
from scripts.v273_extended_head_bank import (
    ALL_HEADS,FEATURE_HEADS,KEEP_HEADS,K_COUNT,REGISTRY,
    action_candidates,audio_temporal_views,bank_feature_views,
    waveform_pitch_source,inspect_registry)
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27335
# These five are the historical K2–K6 specialists. New refits preserve
# their original structural support rather than pretending they detect K0.
POLY_HEADS=FEATURE_HEADS[:5]
NONPITCH=[h for h in FEATURE_HEADS if h!='H8_pitch_shift']
TRAINABLE=tuple(NONPITCH)+('E12_second_note',)
EXTRA_ACTION='AB_stateful_S29'
ACTIONS=ALL_HEADS+(EXTRA_ACTION,)
RISK_LAMBDAS=(1.,2.,4.)
RISK_CUTS=(0.,.02,.05)
MAX_PASS=3
assert len(ACTIONS)==19 and len(TRAINABLE)==9

def read(file):
    with np.load(file,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def get(z,name):
    keys=list(map(str,z['variant_ids']))
    require(name in keys, 'missing native source action '+name)
    return z['predictions'][:,keys.index(name)].astype(np.int8)

def hsh(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()

def original_temporal_data(folder,d):
    """Load real aligned 42×49 trajectories; never invent morphology inputs."""
    n=len(d['ids'])
    morph=np.full((n,490),np.nan,np.float32)
    physics=np.full((n,294),np.nan,np.float32)
    index={int(i):j for j,i in enumerate(d['ids'])}
    manifests=[]
    for fold in FOLDS:
        paths=list(folder.rglob(f'features-fold-{fold}.npz'))
        require(len(paths)==1,'one original trajectory file per fold required')
        with np.load(paths[0],allow_pickle=False) as z:
            source=z['global_index']
            require(len(source)==int((d['fold']==fold).sum()),'fold size drift')
            ix=np.asarray([index[int(k)] for k in source],np.int64)
            require(np.array_equal(d['fold'][ix],np.full(len(ix),fold)),
                    'original waveform sequence aligned to wrong fold')
            sequence=z['sequence']
            for start in range(0,len(ix),256):
                zz=np.asarray(sequence[start:start+256],np.float32)
                m,p=audio_temporal_views(zz)
                at=ix[start:start+len(m)]
                morph[at]=m
                physics[at]=p
            manifests.append(dict(fold=int(fold),rows=int(len(ix)),
                source_file=paths[0].name,
                original_row_sha256=hsh(source)))
    require(np.isfinite(morph).all() and np.isfinite(physics).all(),
            'incomplete 42-frame morphology')
    return morph,physics,manifests

def fit_head(name,features,y,train_index,val_index,seed):
    x=features[name]
    if name=='E12_second_note':
        allowed=(y[train_index]==1)|(y[train_index]==2)
        ix=train_index[allowed]
    elif name in POLY_HEADS:
        ix=train_index[y[train_index]>=2]
    else:
        ix=train_index
    labels=y[ix]
    expected=(1,2) if name=='E12_second_note' else (
        (2,3,4,5,6) if name in POLY_HEADS else tuple(range(7)))
    require(set(np.unique(labels).tolist())==set(expected),
        f'{name}: missing an expert class in training folds')
    scaler=StandardScaler().fit(x[ix])
    xx=np.clip(scaler.transform(x[ix]),-5,5)
    held=np.clip(scaler.transform(x[val_index]),-5,5)
    model=SGDClassifier(loss='log_loss',penalty='l2',alpha=0.001,
        max_iter=36,tol=0.002,random_state=seed,average=True)
    with threadpool_limits(limits=2):
        model.fit(xx,labels)
        p=model.predict_proba(held)
    require(np.array_equal(model.classes_,np.asarray(expected)),
           'expert class alignment drift')
    p7=np.zeros((len(val_index),7),np.float32)
    p7[:,list(expected)]=p.astype(np.float32)
    require(np.isfinite(p7).all() and
            np.allclose(p7.sum(1),1,atol=1e-4),'bad head posterior')
    return p7,dict(model=model,scaler=scaler,
                   name=name,fit_ids=dummy_id_tag(train_index),
                   held_ids=dummy_id_tag(val_index))

def dummy_id_tag(index):
    # This function is intentionally only used with INDEX provenance.
    return np.asarray(index,dtype=np.int64)

def source_oof_for_outer(features,d,outer,model_dir):
    """Nested cross-fold predictions, excluding *outer* test fold entirely.

    Outer held experts train on three other folds.
    Inner expert outputs for router training fit on the other two folds.
    """
    n=len(d['y'])
    test_idx=np.flatnonzero(d['fold']==outer)
    fit_idx=np.flatnonzero(d['fold']!=outer)
    require(len(test_idx)>0 and len(fit_idx)>0,'empty outer fold')
    outer_p={}
    inner_p={}
    manifests=[]
    other=[int(k) for k in FOLDS if k!=outer]
    for inner in [None,*other]:
        if inner is None:
            train=fit_idx;hold=test_idx
            state='outer'
        else:
            train=np.flatnonzero((d['fold']!=outer)&(d['fold']!=inner))
            hold=np.flatnonzero(d['fold']==inner)
            state=f'inner{inner}'
        require((d['fold'][train]!=outer).all() and
                not np.intersect1d(train,hold).size,
                'nested specialist leakage')
        for j,name in enumerate(TRAINABLE):
            pr,record=fit_head(name,features,d['y'],train,hold,
                               SEED+outer*100+ (0 if inner is None else 10+inner)*11+j)
            where=outer_p if inner is None else inner_p
            if name not in where:
                where[name]=np.full((n,7),np.nan,np.float32)
            where[name][hold]=pr
            fname=f'fold{outer}__{state}__{name}.joblib'
            joblib.dump(dict(**record, outer_fold=int(outer),
                             inner_fold=inner, source_train_SHA256=hsh(d['ids'][train]),
                             source_hold_SHA256=hsh(d['ids'][hold]),
                             source_train_native_ids=d['ids'][train],
                             source_held_native_ids=d['ids'][hold]),
                        model_dir/fname,compress=3)
            manifests.append(dict(outer=int(outer),inner=inner,name=name,
                trained=int(len(train)),held=int(len(hold)),file=fname,
                fit_SHA256=hsh(d['ids'][train]),held_SHA256=hsh(d['ids'][hold])))
    for name in TRAINABLE:
        require(np.isfinite(outer_p[name][test_idx]).all() and
                np.isfinite(inner_p[name][fit_idx]).all(),
                'incomplete nested OOF '+name)
    return test_idx,fit_idx,outer_p,inner_p,manifests

def add_optional_head(bank,pitch,indices):
    if pitch is None:
        return bank
    bank['H8_pitch_shift']=pitch
    require(np.isfinite(bank['H8_pitch_shift'][indices]).all(),
            'H8 waveform OOF scores missing')
    return bank

def source_actions(current,base,experts,ab_pred):
    proposal,confidence,mask=action_candidates(
        current,experts,base_parent=base)
    n=len(current)
    proposal=np.column_stack([proposal,ab_pred]).astype(np.int8)
    confidence=np.column_stack([confidence,np.full(n,.70,np.float32)])
    mask=np.column_stack([mask,ab_pred!=current])
    # H1–H5 historically covered only K2..K6.
    for name in POLY_HEADS:
        j=ACTIONS.index(name)
        mask[:,j] &= ((current>=2)&(proposal[:,j]>=2))
    mask &= proposal!=current[:,None]
    assert proposal.shape==confidence.shape==mask.shape==(n,len(ACTIONS))
    assert not np.any(mask[:,ACTIONS.index('H9')] if 'H9' in ACTIONS else False)
    return proposal,confidence,mask

def risk_features(x,base,current,proposed,confidence,stage,previous_head):
    """Observable gate state only; labels/fold/YourMT3+ categorically absent."""
    n=len(current)
    one=np.eye(7,dtype=np.float32)
    result=np.column_stack([
        x[:,:90],
        one[np.asarray(base,np.int64)],
        one[np.asarray(current,np.int64)],
        one[np.asarray(proposed,np.int64)],
        np.asarray(confidence,np.float32),
        np.full(n,float(stage)/3,np.float32),
        np.asarray(previous_head,np.float32)/len(ACTIONS),
        (np.asarray(current)!=np.asarray(base)).astype(np.float32),
    ]).astype(np.float32)
    require(result.shape==(n,115),'router condition dimensions changed')
    require(np.isfinite(result).all(),'non-finite routing inputs')
    return result

def fit_risk_models(x,base,y,train_idx,inner_proba,ab,model_dir,outer):
    """Fit on INNER OOF proposed actions excluding outer fold at producer level."""
    sources=inner_proba
    parent=base[train_idx]
    alternative=sources['H3_harmonics'][train_idx].argmax(1).astype(np.int8)
    previous=np.full(len(train_idx),-1,np.int8)
    stored=[]
    gates={}
    for j,head in enumerate(ACTIONS):
        feats=[];goals=[]
        for stage,current in ((0,parent),(1,alternative)):
            pp,conf,mask=source_actions(current,parent,
                {k:v[train_idx] for k,v in sources.items()},
                ab[train_idx])
            eligible=mask[:,j]
            if not np.any(eligible):continue
            pos=np.flatnonzero(eligible)
            cand=pp[pos,j]
            before=current[pos]
            truth=y[train_idx[pos]]
            target=np.column_stack([
                ((cand==truth)&(before!=truth)).astype(np.float32),
                ((cand!=truth)&(before==truth)).astype(np.float32),
            ])
            feat=risk_features(x[train_idx[pos]],parent[pos],before,cand,
                               conf[pos,j],stage,
                               np.full(len(pos),-1 if stage==0 else
                                       ACTIONS.index('H3_harmonics'),np.int8))
            feats.append(feat);goals.append(target)
        if not feats:continue
        matrix=np.vstack(feats)
        target=np.vstack(goals)
        require(len(matrix)==len(target),'Q labels misaligned')
        scaler=StandardScaler().fit(matrix)
        xx=np.clip(scaler.transform(matrix),-5,5)
        estimator=Ridge(alpha=180.,solver='auto')
        with threadpool_limits(limits=2):
            estimator.fit(xx,target)
        gates[head]=dict(model=estimator,scaler=scaler)
        fname=f'outer{outer}__Q__{head}.joblib'
        joblib.dump(dict(model=estimator,scaler=scaler,head=head,
                         outer=int(outer),fit_ids=dummy_id_tag(train_idx)),
                    model_dir/fname,compress=3)
        stored.append(dict(head=head,training_action_rows=int(len(matrix)),
             observed_fixes=int(target[:,0].sum()),
             observed_regressions=int(target[:,1].sum()),
             file=fname))
    return gates,stored

def dynamic_route(x,base,bank,ab_pred,gates,lam,cut):
    """Re-score all actions after each change; apply only selected action.

    Costs for experts are NOT considered selective yet: all expert outputs
    are precomputed. Actions are sequential decisions, not 18× lazy inference.
    """
    n=len(base)
    current=base.copy().astype(np.int8)
    seen=np.zeros((n,len(ACTIONS)),bool)
    routes=np.full((n,MAX_PASS),-1,np.int8)
    alive=np.ones(n,bool)
    prev=np.full(n,-1,np.int8)
    for step in range(MAX_PASS):
        active=np.flatnonzero(alive)
        if len(active)==0:break
        probs={key:v[active] for key,v in bank.items()}
        proposal,confidence,mask=source_actions(current[active],
              base[active],probs,ab_pred[active])
        mask &= ~seen[active]
        utility=np.full(mask.shape,-np.inf,np.float32)
        for j,head in enumerate(ACTIONS):
            gate=gates.get(head)
            ok=mask[:,j]
            if gate is None or not np.any(ok):continue
            at=np.flatnonzero(ok)
            feat=risk_features(x[active[at]],base[active[at]],
                current[active[at]],proposal[at,j],
                confidence[at,j],step,prev[active[at]])
            transformed=np.clip(gate['scaler'].transform(feat),-5,5)
            with threadpool_limits(limits=2):
                q=np.clip(gate['model'].predict(transformed),0,1)
            utility[at,j]=q[:,0]-float(lam)*q[:,1]-.0005
        best=utility.argmax(1)
        utility_best=utility[np.arange(len(active)),best]
        viable=np.isfinite(utility_best)&(utility_best>cut)
        # Other events immediately stop; not all heads re-run.
        alive[active[~viable]]=False
        chosen=active[viable]
        selected=best[viable]
        if len(chosen):
            routes[chosen,step]=selected.astype(np.int8)
            current[chosen]=proposal[viable,selected]
            seen[chosen,selected]=True
            prev[chosen]=selected.astype(np.int8)
    return current,routes

def check_selftest():
    from scripts.v273_extended_head_bank import selftest
    selftest()
    parent=np.array([2,3,1,4,2],np.int8)
    proba=np.zeros((len(parent),7),np.float32)
    proba[:,3]=.9
    proba[:,1]=.1
    p,c,m=source_actions(parent,parent,{'H1_spectral':proba,
         'H3_harmonics':proba},parent)
    assert p.shape==(len(parent),19)
    assert len(ACTIONS)==19 and 'H9' not in ACTIONS
    assert m[0,ACTIONS.index('C23')]
    assert not m[0,ACTIONS.index('F_keep_any')]
    modified=parent.copy();modified[0]=3
    _,_,m2=source_actions(modified,parent,{'H1_spectral':proba},parent)
    assert m2[0,ACTIONS.index('F_keep_any')]
    x=np.zeros((len(parent),335),np.float32)
    q=risk_features(x,parent,parent,parent,np.ones(len(parent)),
            0,np.full(len(parent),-1))
    assert q.shape==(len(parent),115)
    print('PASS: 18 registered non-H9 specialists + pre-existing AB stateful route; KEEP reverts S18, all K eligibility masked')

def run(args):
    require(not args.output.exists(),'no overwrite of native experiment')
    start=time.monotonic()
    d=prepare(args)
    n=len(d['ids'])
    source=read(args.s29)
    require(np.array_equal(source['global_index'],d['ids']) and
            np.array_equal(source['true_K'],d['y']),'S29 source drift')
    ab=get(source,'series29__lambda2__threshold0.02')
    base=d['parent'].astype(np.int8)
    require(np.array_equal(base,get(source,'series18_parent')),
            'S18 parent changed')
    morph,physics,temporal_provenance=original_temporal_data(args.features,d)
    views=bank_feature_views(d['X'],morph,physics)
    active_optional=None
    if args.h8 is not None:
        active_optional=waveform_pitch_source(read(args.h8),d['ids'])
    args.output.mkdir(parents=True)
    (args.output/'models').mkdir()
    variants={f'series35__lambda{lam:g}__threshold{cut:g}':
              np.full(n,-1,np.int8)
              for lam in RISK_LAMBDAS for cut in RISK_CUTS}
    paths={key:np.full((n,MAX_PASS),-2,np.int8) for key in variants}
    fitted=[]
    protected_head_counts=np.zeros((len(ACTIONS),),np.int64)
    outer_reports=[]
    for outer in FOLDS:
        test,fit,held,train,manifest=source_oof_for_outer(
            views,d,outer,args.output/'models')
        fitted.extend(manifest)
        if active_optional is not None:
            train['H8_pitch_shift']=active_optional
            held['H8_pitch_shift']=active_optional
        # For each router quality model the reference is STRICTLY inner OOF,
        # not the outer-fold predictions.
        gates,gatespec=fit_risk_models(d['X'],base,d['y'],fit,
            train,ab,args.output/'models',outer)
        outer_reports.append(dict(fold=int(outer),fit_count=int(len(fit)),
            held_count=int(len(test)),source_estimators=len(manifest),
            risk_estimators=gatespec,train_sha256=hsh(d['ids'][fit]),
            held_sha256=hsh(d['ids'][test])))
        held_local={head:p[test] for head,p in held.items()}
        for lam in RISK_LAMBDAS:
            for cut in RISK_CUTS:
                key=f'series35__lambda{lam:g}__threshold{cut:g}'
                preds,routes=dynamic_route(d['X'][test],base[test],
                    held_local,ab[test],gates,lam,cut)
                variants[key][test]=preds
                paths[key][test]=routes
        print(json.dumps(dict(fold=int(outer),outer_test=int(len(test)),
            nested_models=len(manifest),gates=len(gates),
            seconds=round(time.monotonic()-start,1))),flush=True)
    for key,y in variants.items():
        require((y>=0).all() and (y<7).all(),'some native predictions missing')
        require((paths[key]>=-1).all(),'routing path incomplete')
    audits={}
    outputs={'series18_parent':base.copy(),
        'series29_stateful_A_B':ab.copy()}
    outputs.update(variants)
    for name,pred in outputs.items():
        changed=pred!=base
        active_path=None
        if name in paths:
            route=paths[name]
            active_path=dict(
                head_usage={head:int((route==j).sum()) for j,head in enumerate(ACTIONS)},
                steps_used={
                    str(i):int(((route!=-1).sum(1)==i).sum())
                    for i in range(MAX_PASS+1)},
                stage_actions=[{head:int((route[:,t]==j).sum())
                    for j,head in enumerate(ACTIONS)} for t in range(MAX_PASS)],
                H9_calls=0,
                H8_calls=int((route==ACTIONS.index('H8_pitch_shift')).sum()))
        audits[name]=dict(metrics=metrics(d['y'],pred),
            versus_S18=paired(d['y'],base,pred),
            versus_freeze=paired(d['y'],d['freeze'],pred),
            corrected=int((changed&(pred==d['y'])&(base!=d['y'])).sum()),
            regressed=int((changed&(pred!=d['y'])&(base==d['y'])).sum()),
            neutral=int((changed&(pred!=d['y'])&(base!=d['y'])).sum()),
            route=active_path,
            by_true_K={str(k):dict(
                correct=int(((d['y']==k)&(pred==d['y'])).sum()),
                corrections=int(((d['y']==k)&changed&(pred==d['y'])&
                    (base!=d['y'])).sum()),
                regressions=int(((d['y']==k)&changed&(pred!=d['y'])&
                    (base==d['y'])).sum()))
                for k in range(7)},
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                   pred[d['fold']==f]),versus_S18=paired(d['y'][d['fold']==f],
                   base[d['fold']==f],pred[d['fold']==f]))
                  for f in FOLDS})
    safe=[k for k in variants if audits[k]['corrected']>0 and
         audits[k]['regressed']==0 and
         audits[k]['metrics']['poly']['correct']>=2998]
    rank=sorted(audits,key=lambda name:(
          audits[name]['metrics']['correct'],
          audits[name]['metrics']['poly']['correct']),reverse=True)
    active_in_source=[h for h in TRAINABLE]
    if active_optional is not None:active_in_source.append('H8_pitch_shift')
    report=dict(status='completed',architecture='statewise_19_actions_with_AB_S29',
        head_registry=inspect_registry(),dynamic_actions=list(ACTIONS),
        total_added_head_contracts=len(ALL_HEADS),
        source_available_acoustic_models=active_in_source,
        corrected_keep_and_transition_adapters=8,
        H9_excluded_from_all_actions=True,
        H8_real_waveform_evidence_available=active_optional is not None,
        H8_not_fabricated_from_frequency_transform=True,
        no_measured_runtime_cost_saving=True,
        all_expert_posteriors_precomputed=True,
        nested_fold_fit=True,
        independent_never_seen_music=False,
        historical_parent_selected_on_development=True,
        no_production_promotion=True,
        temporal_sources=temporal_provenance,
        native_source_estimators_saved=len(fitted),
        fitted_source_provenance=fitted,
        router_models_by_fold=outer_reports,
        retained_S18_correct=int((base==d['y']).sum()),
        strict_no_loss_candidates=safe,
        best_global=rank[0],
        audits=audits,elapsed_seconds=round(time.monotonic()-start,1))
    (args.output/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    np.savez_compressed(args.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(outputs)),
        predictions=np.column_stack(list(outputs.values())))
    np.savez_compressed(args.output/'routing.npz',
        global_index=d['ids'],variant_ids=np.asarray(list(paths)),
        selected_head=np.stack(list(paths.values()),axis=1),
        action_names=np.asarray(ACTIONS))
    text=['# Série35 — 18 modules historiques raccordés au routeur dynamique, H9 exclu',
        '', 'Architecture: 18 heads + ancien AB_stateful_S29; choisissez de nouveau après chaque tête.',
        '**H8 sans sortie WAV pitch-shift sur la cohorte entière: intégré comme adaptateur masqué, non annoncé comme exécuté.**',
        'H7 utilise de vrais 42×49 profils temporels; H10 des descripteurs de dissipation, pas une simulation Navier–Stokes.',
        f"144 fit estimators nested/outer saved: {len(fitted)}",
        '**Toutes sorties d experts pré-calculées : les étapes de la politique varient, mais les économies d inférence ne sont PAS mesurées.**',
        'Recherche sur cohorte développée antérieurement, aucun résultat promu.',
        '', '| Politique | Exact global | Exact poly | Corrigées | Régressions |',
        '|---|---:|---:|---:|---:|']
    for key in rank:
        m=audits[key];stats=m['metrics']
        text.append(f"| {key} | {100*stats['exact']:.4f}% | "
            f"{100*stats['poly']['exact']:.4f}% | "
            f"{m['corrected']} | {m['regressed']} |")
    text+=['',f"Strict no-loss research candidates: {len(safe)}"]
    (args.output/'report.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=Path('.'))
    a.add_argument('--features',type=Path)
    a.add_argument('--s18',type=Path)
    a.add_argument('--s29',type=Path)
    a.add_argument('--output',type=Path)
    a.add_argument('--h8',type=Path,help='genuine full-cohort pitch-shift OOF prediction archive')
    a.add_argument('--self-test',action='store_true')
    args=a.parse_args()
    if args.self_test:check_selftest()
    else:
        require(args.features and args.s18 and args.s29 and args.output,
                'source S18/S29 and native acoustic inputs mandatory')
        run(args)
