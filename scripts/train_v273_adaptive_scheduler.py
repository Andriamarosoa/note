"""S28 — learn, per event, the order and partial revisit schedule of A/B heads.

An action chooses A-first full, A-first partial, or trained B-first.
Decision uses only original signal/votes and initial S18 parent class.
All 19 scheduler fits exclude the scored piece and use multioutput Ridge
to model corrected vs newly regressed events separately.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

ACTIONS=(
 'series27__full_4__margin0.6',
 'series27__full_2__margin0.6',
 'series27__B_initial_only__margin0.6',
 'series27__B_if_delta_005__margin0.6',
 'series27__msg_only_parent_candidate__margin0.6',
 'series27__early_stable_005__margin0.6',
 'series25__pass2__margin0.6',
 'series25__pass3__margin0.6',
)
LAMBDAS=(1,2,4)
CUTS=(0.,.02,.05,.10)
SEED=27328


def read(file):
    with np.load(file,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def get_pred(data,name):
    keys=list(map(str,data['variant_ids']))
    require(name in keys,'source action unavailable '+name)
    return data['predictions'][:,keys.index(name)].astype(np.int8)


def digest(x):
    return hashlib.sha256(np.asarray(x).astype('<i8').tobytes()).hexdigest()


def decode(parent,actions,scores):
    n=len(parent)
    require(actions.shape==(n,8) and scores.shape==(n,8,2),
            'source/action matrix drift')
    outputs={'series18_parent':parent.copy()}
    route={}
    for lam in LAMBDAS:
        util=scores[:,:,0]-lam*scores[:,:,1]
        ix=util.argmax(1)
        best=util[np.arange(n),ix]
        for cut in CUTS:
            accepts=best>cut
            selected=np.where(accepts,ix,-1).astype(np.int8)
            values=np.where(accepts,actions[np.arange(n),ix],parent).astype(np.int8)
            key=f'series28__learned_lambda{lam}__benefit_gt{cut:g}'
            outputs[key]=values
            route[key]=selected
    require(len(outputs)==13,'12 learned order policies plus parent')
    return outputs,route


def selftest():
    n=16
    parent=np.arange(n)%7
    acts=np.column_stack([(parent+i+1)%7 for i in range(8)]).astype(np.int8)
    scores=np.zeros((n,8,2),np.float32)
    scores[:,3,0]=.8
    scores[:,3,1]=.2
    o,r=decode(parent.astype(np.int8),acts,scores)
    assert len(o)==13 and len(r)==12
    assert np.all(o['series28__learned_lambda1__benefit_gt0.1']==acts[:,3])
    assert np.all(o['series28__learned_lambda4__benefit_gt0.1']==parent)
    print('PASS: fixed 12 scheduler policies, 8 routes, learned correction/regression tradeoff')


def run(a):
    require(not a.output.exists(),'no overwriting learned scheduler research')
    d=prepare(a)
    source27=read(a.s27)
    source25=read(a.s25)
    for z in (source27,source25):
        require(np.array_equal(z['global_index'],d['ids']),
                'the two selection-order producers have different event IDs')
        require(np.array_equal(z['true_K'],d['y']) and
                np.array_equal(z['fold'],d['fold']),
                'producer label/fold mismatch')
    require(np.array_equal(get_pred(source27,'series18_parent'),d['parent']) and
            np.array_equal(get_pred(source25,'series18_parent'),d['parent']),
            'initial parent changed')
    require(np.array_equal(get_pred(source27,'freeze_parent'),d['freeze']) and
            np.array_equal(get_pred(source25,'freeze_reference'),d['freeze']),
            'freeze parent changed')
    n=len(d['y'])
    action_prediction=np.column_stack([get_pred(
        source27 if name.startswith('series27__') else source25,name)
        for name in ACTIONS]).astype(np.int8)
    parent=d['parent']
    # The route is chosen before any new A/B inference: 90 original
    # signal/vote features and 7 one-hot probabilities of the parent.
    X=np.column_stack([
        d['X'][:,:90],
        np.eye(7,dtype=np.float32)[parent]
    ]).astype(np.float32)
    require(X.shape==(59309,97) and np.isfinite(X).all(),
            '97 initial routing signals unavailable')
    scores=np.full((n,len(ACTIONS),2),np.nan,np.float32)
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    splits=[]
    started=time.monotonic()
    for piece in sorted(set(d['pieces'].tolist())):
        test=d['pieces']==piece
        fold=int(np.unique(d['fold'][test])[0])
        fit=(d['fold']==fold)&~test
        require(fit.sum()>200 and
                piece not in set(d['pieces'][fit].tolist()),
                'scheduler uses evaluation piece')
        scale=StandardScaler().fit(X[fit])
        fitX=np.clip(scale.transform(X[fit]),-6,6)
        tstX=np.clip(scale.transform(X[test]),-6,6)
        models=[]
        for k,name in enumerate(ACTIONS):
            candidate=action_prediction[fit,k]
            old=parent[fit]
            truth=d['y'][fit]
            corrected=((candidate==truth)&(old!=truth)).astype(np.float32)
            regressed=((candidate!=truth)&(old==truth)).astype(np.float32)
            target=np.column_stack((corrected,regressed))
            model=Ridge(alpha=100.0,fit_intercept=True)
            with threadpool_limits(limits=2):
                model.fit(fitX,target)
                predictions=model.predict(tstX)
            scores[test,k]=np.clip(predictions,0,1).astype(np.float32)
            fname=f'{piece}__route{k}.joblib'
            joblib.dump(dict(model=model,scaler=scale,action=name,
                held_piece=piece,fold=fold,train_ids=d['ids'][fit],
                test_ids=d['ids'][test]),a.output/'models'/fname,compress=3)
            models.append(dict(action=name,file=fname,
                corrections_train=int(corrected.sum()),
                regressions_train=int(regressed.sum())))
        splits.append(dict(piece=piece,fold=fold,
            train_count=int(fit.sum()),held_count=int(test.sum()),
            fit_SHA256=digest(d['ids'][fit]),
            held_SHA256=digest(d['ids'][test]),
            actions=models))
        print(json.dumps(dict(piece=piece,fold=fold,
                completed=len(splits),seconds=round(time.monotonic()-started,1))),
              flush=True)
    require(np.isfinite(scores).all() and
        np.all((scores>=0)&(scores<=1)),'scheduler output invalid')
    out,route=decode(parent,action_prediction,scores)
    out['freeze_reference']=d['freeze'].copy()
    audits={}
    for name,pred in out.items():
        changed=pred!=parent
        route_info={}
        if name in route:
            c=route[name]
            route_info={'parent_or_abstention':int((c<0).sum()),
                'chosen_actions':{ACTIONS[k]:int((c==k).sum())
                    for k in range(len(ACTIONS))}}
        audits[name]=dict(metrics=metrics(d['y'],pred),
            vs_parent=paired(d['y'],parent,pred),
            vs_freeze=paired(d['y'],d['freeze'],pred),
            vs_YourMT3=paired(d['y'],d['yourmt3'],pred),
            corrected=int((changed&(pred==d['y'])&(parent!=d['y'])).sum()),
            regressed=int((changed&(pred!=d['y'])&(parent==d['y'])).sum()),
            neutral=int((changed&(pred!=d['y'])&(parent!=d['y'])).sum()),
            routes=route_info,
            by_true_K={str(k):dict(
                correct=int(((d['y']==k)&(pred==d['y'])).sum()),
                corrected=int(((d['y']==k)&changed&
                    (pred==d['y'])&(parent!=d['y'])).sum()),
                regressed=int(((d['y']==k)&changed&
                    (pred!=d['y'])&(parent==d['y'])).sum()))
                for k in range(7)},
            by_fold={str(f):dict(metrics=metrics(d['y'][d['fold']==f],
                pred[d['fold']==f]),vs_parent=paired(d['y'][d['fold']==f],
                parent[d['fold']==f],pred[d['fold']==f]))
                for f in FOLDS})
    ranking=sorted(audits,key=lambda name:(
        audits[name]['metrics']['correct'],audits[name]['metrics']['poly']['correct']),
        reverse=True)
    safe=[name for name in out if name.startswith('series28__')
        and audits[name]['corrected']>0 and audits[name]['regressed']==0
        and audits[name]['metrics']['poly']['correct']>=2998]
    result=dict(status='completed',independent_validation=False,
        cohort_previously_exposed=True,
        no_production_promotion=True,
        held_piece_excluded_from_scheduler_fit=True,
        upstream_S18_parent_selected_posthoc_on_same_cohort=True,
        possible_online_scheduler_features_97=True,
        source_runs=dict(partial=37863961918,
           Afirst=37858383973,Bfirst=37862417522),
        models_count=len(splits)*len(ACTIONS),
        actions=list(ACTIONS),Lambdas=list(LAMBDAS),
        cutoffs=list(CUTS),trained_splits=splits,
        policy_count=len(out),
        audits=audits,best_global=ranking[0],
        strict_no_loss_candidates=safe,
        elapsed_seconds=round(time.monotonic()-started,1))
    (a.output/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=d['ids'],
        fold=d['fold'],true_K=d['y'],variant_ids=np.asarray(list(out)),
        predictions=np.column_stack(list(out.values())))
    np.savez_compressed(a.output/'scores.npz',global_index=d['ids'],
        action_names=np.asarray(ACTIONS),
        estimated_fix_and_regress=scores,
        action_predictions=action_prediction)
    np.savez_compressed(a.output/'routes.npz',global_index=d['ids'],
        variant_ids=np.asarray(list(route)),
        chosen_action=np.column_stack(list(route.values())))
    lines=['# Série28 — ordre et repassage choisis par ordonnanceur entraîné','',
        'Les 8 actions incluent A-first full/partiel et B-first. Séparation stricte du morceau évalué dans le fit de la porte.',
        '**Corpus antérieurement exploré, prédiction S18 posthoc, PAS de validation indépendante.**',
        '', '| Politique | Exact global | Exact poly | Corrigées vs S18 | Régressées vs S18 |',
        '|---|---:|---:|---:|---:|']
    for name in ranking:
        t=audits[name];m=t['metrics']
        lines.append(f"| {name} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | {t['corrected']} | {t['regressed']} |")
    lines+=['',f"Strict no-loss candidates: {len(safe)}",
        'All 152 trained route models, their splits, all decisions and routes archived.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--s27',type=Path)
    p.add_argument('--s25',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:
        require(a.features and a.s18 and a.s27 and
                a.s25 and a.output,'S27/S25 and native audio required')
        run(a)
