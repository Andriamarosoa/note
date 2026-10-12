"""S38: independently held-fold acoustic/polyphonic specialist, with no H9.

Two estimators × two genuine acoustic views, each trained on other three folds.
The new classifier proposes K1..K6; old S18 may reject proposal label-blind.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.train_v273_aba_recurrent import prepare
from scripts.train_v273_extended_dynamic_router import original_temporal_data,read
from scripts.yourmt3_exactk_common import FOLDS,metrics,paired,require

SEED=27338
VIEWS=('raw335','morph825')
ALGOS=('hgb','extra_trees')
DOMAINS=('source_K1to6','source_poly_K2to6')
GATES=('argmax','delta_gt0.10','delta_gt0.25')

def sha(x):
    return hashlib.sha256(np.asarray(x,dtype='<i8').tobytes()).hexdigest()

def score_predictions(base,proposed,prob,domain,mode):
    n=len(base)
    eligible=(base>=1) if domain=='source_K1to6' else (base>=2)
    cand=proposed!=base
    if mode!='argmax':
        t=.10 if mode=='delta_gt0.10' else .25
        gain=prob[np.arange(n),proposed]-prob[np.arange(n),base]
        cand&=(gain>t)
    return np.where(eligible&cand,proposed,base).astype(np.int8)

def unit_test():
    a=np.asarray([0,1,2,3],np.int8)
    b=np.asarray([2,2,3,4],np.int8)
    pp=np.zeros((4,7),np.float32)
    pp[np.arange(4),b]=.80
    pp[np.arange(4),a]=.15
    assert score_predictions(a,b,pp,'source_poly_K2to6','argmax').tolist()==[0,1,3,4]
    assert score_predictions(a,b,pp,'source_K1to6','delta_gt0.25').tolist()==[0,2,3,4]
    assert 0 not in (1,2,3,4,5,6)
    print('PASS: 24 preregistered proposal policies; no hidden true-K access or H9')

def run(a):
    require(not a.output.exists(),'no overwrite')
    started=time.monotonic()
    d=prepare(a)
    n=len(d['ids'])
    old=d['parent'].astype(np.int8)
    require(n==59309 and metrics(d['y'],old)['correct']==49178
            and metrics(d['y'],old)['poly']['correct']==2998,
            'source native reference drift')
    morph,physics,provenance=original_temporal_data(a.features,d)
    views={'raw335':d['X'].astype(np.float32),
           'morph825':np.column_stack((d['X'],morph)).astype(np.float32)}
    assert views['morph825'].shape==(59309,825)
    a.output.mkdir(parents=True)
    (a.output/'models').mkdir()
    predictions={f'{view}__{algo}':np.full(n,-1,np.int8)
        for view in VIEWS for algo in ALGOS}
    confidences={k:np.full((n,7),np.nan,np.float32)
        for k in predictions}
    ledger=[]
    for fold in FOLDS:
        train=np.flatnonzero((d['fold']!=fold)&(d['y']>=1))
        hold=np.flatnonzero(d['fold']==fold)
        require(len(train)>2000 and len(hold)>1000 and
             not np.intersect1d(train,hold).size,'outer-fold train/test leakage')
        labels=d['y'][train]
        require(set(labels.tolist())==set(range(1,7)),
                'missing K1-K6 class')
        freq=np.bincount(labels,minlength=7)
        balance=np.sqrt(max(freq[1:])/freq[labels])
        balance=np.minimum(balance,5).astype(np.float32)
        for view in VIEWS:
            xx=views[view]
            scaler=StandardScaler().fit(xx[train])
            trainx=np.clip(scaler.transform(xx[train]),-5,5).astype(np.float32)
            holdx=np.clip(scaler.transform(xx[hold]),-5,5).astype(np.float32)
            for algo in ALGOS:
                model=HistGradientBoostingClassifier(
                    max_iter=150,max_depth=5,max_leaf_nodes=24,
                    min_samples_leaf=25,l2_regularization=20,
                    learning_rate=.065,random_state=SEED+fold) if algo=='hgb' else (
                    ExtraTreesClassifier(n_estimators=220,max_features=.7,
                        min_samples_leaf=3,n_jobs=2,
                        random_state=SEED+fold,class_weight=None))
                with threadpool_limits(limits=2):
                    model.fit(trainx,labels,sample_weight=balance)
                    pp=model.predict_proba(holdx).astype(np.float32)
                require(np.array_equal(model.classes_,np.arange(1,7)),
                        'expert output labels 1..6 misaligned')
                matrix=np.zeros((len(hold),7),np.float32)
                matrix[:,1:7]=pp
                key=f'{view}__{algo}'
                confidences[key][hold]=matrix
                predictions[key][hold]=matrix.argmax(1).astype(np.int8)
                name=f'fold{fold}__{key}.joblib'
                joblib.dump(dict(estimator=model,scaler=scaler,fold=int(fold),
                    view=view,algo=algo,
                    train_native_ids=d['ids'][train],
                    held_native_ids=d['ids'][hold],
                    training_classes=np.unique(labels)),
                    a.output/'models'/name,compress=3)
                ledger.append(dict(file=name,fold=int(fold),view=view,
                    model=algo,fit_count=int(len(train)),held_count=int(len(hold)),
                    fit_ids_sha256=sha(d['ids'][train]),
                    held_ids_sha256=sha(d['ids'][hold]),
                    true_poly_held=int((d['y'][hold]>=2).sum()),
                    diagnostic_poly_accurate=int(((d['y'][hold]>=2)&
                       (predictions[key][hold]==d['y'][hold])).sum())))
        print(json.dumps(dict(held_fold=int(fold),models=len(ledger),
              elapsed=round(time.monotonic()-started,1))),flush=True)
    output={'series18_parent':old.copy()}
    for key,pred in predictions.items():
        require((pred>=1).all() and np.isfinite(confidences[key]).all(),
                'unscored held-fold predictions '+key)
        for domain in DOMAINS:
            for gate in GATES:
                name=f'series38__{key}__{domain}__{gate}'
                output[name]=score_predictions(old,pred,confidences[key],domain,gate)
    require(len(output)==25,'exactly 24 alternatives plus parent')
    audits={}
    diagnostics={}
    for key,pred in predictions.items():
        mask=(d['y']>=2)
        diffs=confidences[key][np.arange(n),d['y']]
        diagnostics[key]=dict(
            held_poly_correct=int((pred[mask]==d['y'][mask]).sum()),
            held_poly_rows=int(mask.sum()),
            held_poly_exact=float((pred[mask]==d['y'][mask]).mean()),
            poly_K_class_correct={
                str(k):int(((d['y']==k)&(pred==k)).sum())
                for k in range(2,7)},
            all_K1_K6_correct=int(((d['y']>=1)&(pred==d['y'])).sum()),
            all_K1_K6_rows=int((d['y']>=1).sum()))
    for key,pred in output.items():
        ch=pred!=old
        fix=ch&(pred==d['y'])&(old!=d['y'])
        loss=ch&(pred!=d['y'])&(old==d['y'])
        neutral=ch&(pred!=d['y'])&(old!=d['y'])
        audits[key]=dict(metrics=metrics(d['y'],pred),
            paired_vs_S18=paired(d['y'],old,pred),
            corrected=int(fix.sum()),regressed=int(loss.sum()),
            neutral=int(neutral.sum()),
            per_true_K={str(k):dict(
                fixed=int(((d['y']==k)&fix).sum()),
                lost=int(((d['y']==k)&loss).sum())) for k in range(7)},
            by_fold={str(f):dict(fixes=int(((d['fold']==f)&fix).sum()),
                losses=int(((d['fold']==f)&loss).sum())) for f in FOLDS})
    ranking=sorted(audits,key=lambda k:(audits[k]['metrics']['correct'],
        audits[k]['metrics']['poly']['correct']),reverse=True)
    safe=[k for k in output if k.startswith('series38__')
        and audits[k]['corrected']>0 and audits[k]['regressed']==0
        and audits[k]['metrics']['poly']['correct']>=2998]
    meta=dict(status='completed',new_polyphonically_trained_models=len(ledger),
        no_H9_input_or_action=True,H8_still_masked=True,
        held_fold_train_excluded=True,
        original_S18_unchanged=True,
        benchmark_YourMT3_is_external_only=True,
        previously_used_development_cohort=True,
        not_independent_new_composition_validation=True,
        no_production_promotion=True,
        input_provenance=provenance,
        model_provenance=ledger,
        poly_diagnostics=diagnostics,
        analyses=audits,best_global=ranking[0],
        strict_no_loss_candidates=safe,
        elapsed_seconds=round(time.monotonic()-started,1))
    (a.output/'report.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',
        global_index=d['ids'],true_K=d['y'],fold=d['fold'],
        variant_ids=np.asarray(list(output)),
        predictions=np.column_stack(list(output.values())))
    np.savez_compressed(a.output/'new_expert_probs.npz',
        global_index=d['ids'],variant_ids=np.asarray(list(predictions)),
        poly_probs=np.stack(list(confidences.values()),axis=1))
    lines=['# S38 — genuine acoustic K1..6 specialist, not another veto','',
        '**Exploratory: source compositions already used in research. No H9 input. No model promotion.**',
        'Each new expert trained on other 3 folds, original 335 dimensions or true 42×49 trajectory summaries.',
        '', '## Standalone K2–K6 accuracy (diagnostic only)','',
        '| New classifier | Correct poly | Exact-K poly | S18 poly |',
        '|---|---:|---:|---:|']
    for k,q in diagnostics.items():
        lines.append(f"| {k} | {q['held_poly_correct']}/{q['held_poly_rows']} | "
            f"{q['held_poly_exact']*100:.4f}% | 40.5958% |")
    lines+=['','## Paired S18 interventions','',
        '| Variant | Global | Poly | New fixes | New losses |',
        '|---|---:|---:|---:|---:|']
    for k in ranking:
        v=audits[k];m=v['metrics']
        lines.append(f"| {k} | {m['exact']*100:.4f}% | "
            f"{m['poly']['exact']*100:.4f}% | "
            f"{v['corrected']} | {v['regressed']} |")
    lines+=['',f"Positive zero-loss exploratory variants: {len(safe)}"]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--features',type=Path)
    p.add_argument('--s18',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:unit_test()
    else:
        require(a.features and a.s18 and a.output,'native folds mandatory')
        run(a)
