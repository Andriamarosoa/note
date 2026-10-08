"""Native paired evaluation and permanent inventory of every announced candidate."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS, digest, metrics, paired, require
from scripts.audit_v273_yourmt3_target import align


def main(a):
    require(not a.output.exists(),'refusing overwrite')
    evidence=a.root/'analysis/evidence';old=evidence/'v273-regression-loops'
    with np.load(old/'prepared/inputs.npz',allow_pickle=False) as z:
        ids,y,base,fold=(z[k] for k in ['native_global_index','native_truth','native_baseline','native_fold'])
    with np.load(evidence/'v273-yourmt3-target/comparison/row-evidence.npz',allow_pickle=False) as z:
        require(np.array_equal(z['global_index'],ids),'old comparison alignment')
        current=z['current_best_K'];prior_reachable=z['combined_reachable']
        competitor=z['yourmt3_K']
    arrays=[];source_records=[]
    for f in FOLDS:
        paths=list(a.models.rglob(f'fold-{f}/predictions.npz'))
        require(len(paths)==1,f'model fold inventory {f}')
        path=paths[0];r=json.loads(path.with_name('report.json').read_text())
        require(r['status']=='completed' and r['held_fold']==f,'model fold status')
        require(digest(path)==r['prediction_sha256'],'model prediction checksum')
        with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        require(len(r['records'])==12,'checkpoint inventory')
        for m in r['records']:
            require(f not in m['train_folds'] and not set(m['train_pieces']) & set(m['held_pieces']),
                    'model exclusion violation')
            require(digest(path.parent/m['weights'])==m['weights_sha256'],'weights checksum')
            require(digest(path.parent/m['normalizer'])==m['normalizer_sha256'],'normalizer checksum')
        source_records.append(r)
    require(all(np.array_equal(arrays[0]['variant_ids'],z['variant_ids']) for z in arrays),'model IDs drift')
    combined={k:np.concatenate([z[k] for z in arrays]) for k in ['global_index','truth','fold','probabilities']}
    order=align(combined['global_index'],ids)
    require(np.array_equal(combined['truth'][order],y) and np.array_equal(combined['fold'][order],fold),
            'model cohort mismatch')
    q=combined['probabilities'][order]
    require(q.shape==(59309,12,7) and np.isfinite(q).all() and np.allclose(q.sum(2),1,atol=1e-6),
            'model probability schema')
    new={}
    for j,name in enumerate(arrays[0]['variant_ids']):
        for alpha in (.6,.8,1.):
            p=alpha*q[:,j]+(1-alpha)*np.eye(7,dtype=np.float32)[base]
            winner=p.argmax(1)
            pred=np.where(p[np.arange(len(y)),winner]>p[np.arange(len(y)),base],winner,base)
            new[f'open_k0k6_series2__{name}__mix{alpha:g}']=pred
    previous_names=[];previous=[]
    for path,prefix in [(old/'all-candidate-decisions.npz',''),
                        (old/'posthoc-audit/appended-candidates.npz',''),
                        (evidence/'v273-open-k0-k6/series1/predictions.npz','open_k0k6_series1__')]:
        with np.load(path,allow_pickle=False) as z:
            require(np.array_equal(z['global_index'],ids),'previous policy alignment')
            previous_names.extend(prefix+str(x) for x in z['variant_ids']);previous.append(z['predictions'])
    previous=np.column_stack(previous)
    require(len(previous_names)==225,'previous policy inventory')
    names=previous_names+list(new);new_predictions=np.column_stack(list(new.values())).astype(np.int8)
    all_predictions=np.column_stack([previous,new_predictions])
    require(len(names)==len(set(names))==261,'final policy inventory')
    reports={}
    for name,pred in new.items():
        reports[name]=dict(metrics=metrics(y,pred),versus_freeze=paired(y,base,pred),
            versus_previous_best=paired(y,current,pred),versus_yourmt3=paired(y,competitor,pred),
            newly_correct_outside_previous_coverage=int(np.sum((pred==y)&~prior_reachable)),
            folds={str(f):dict(metrics=metrics(y[fold==f],pred[fold==f]),
                               versus_freeze=paired(y[fold==f],base[fold==f],pred[fold==f])) for f in FOLDS})
    good=all_predictions==y[:,None];best=int(np.argmax(good.sum(0)))
    new_reachable=prior_reachable|good.any(1)
    oracle=np.where(new_reachable,y,base)
    target_pass=[names[j] for j in range(len(names)) if good[:,j].sum()>51328 and good[y>=2,j].sum()>4028]
    distinct=len({all_predictions[:,j].tobytes() for j in range(len(names))})
    report=dict(status='completed',independent_validation=False,automatic_promotion=False,
        protocol_commit='22b55fb96e3a27348fa8b51c8b2bbfe99b28e2ea',
        announced_new_policies=36,previous_policies=225,total_policies=261,distinct_prediction_vectors=distinct,
        new_models=48,folds=list(FOLDS),policies=reports,
        best_global=dict(id=names[best],metrics=metrics(y,all_predictions[:,best]),
                         versus_previous_best=paired(y,current,all_predictions[:,best]),
                         versus_yourmt3=paired(y,competitor,all_predictions[:,best])),
        oracle=dict(not_a_model=True,metrics=metrics(y,oracle),
                    new_errors_covered=int(np.sum(new_reachable&~prior_reachable))),
        descriptive_target_exceeded_by=target_pass,yourmt3=metrics(y,competitor),
        previous_best=metrics(y,current),model_sources=source_records)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,true_K=y,baseline_K=base,fold=fold,
                        variant_ids=np.asarray(list(new)),predictions=new_predictions)
    np.savez_compressed(a.output/'probabilities.npz',global_index=ids,variant_ids=arrays[0]['variant_ids'],probabilities=q)
    np.savez_compressed(a.output/'coverage.npz',global_index=ids,previous_reachable=prior_reachable,
                        new_reachable=new_reachable,best_global_K=all_predictions[:,best])
    report['files_sha256']={p.name:digest(p) for p in a.output.glob('*.npz')}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    registry=dict(total_candidates=261,distinct_prediction_vectors=distinct,
        previous_registry='../../v273-regression-loops/posthoc-audit/candidate-registry.json',
        previous_candidates=211,series1_candidates=14,series2_candidates=36,
        candidate_ids=names,downstream_ready=False,independent_validation=False,
        promotion=False,all_candidates_retained=True)
    (a.output/'candidate-registry.json').write_text(json.dumps(registry,indent=2,sort_keys=True)+'\n')
    lines=['# Ouverture K0–K6 : résultats de développement','',
        'Chaque réseau exclut son fold évalué. Les données ont déjà servi au développement.',
        'Aucune promotion et aucune validation indépendante revendiquée.','',
        '| Candidat | Global | Poly | Corrections / freeze | Régressions / freeze | Net |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in reports.items():
        m=r['metrics'];p=r['versus_freeze']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | {100*m['poly']['exact']:.4f}% | {p['corrections']} | {p['regressions']} | {p['net']:+d} |")
    lines+=['',f"Meilleur global parmi les 261 politiques : {names[best]}.",
        f"Couverture oracle, non réalisable revendiquée : {100*report['oracle']['metrics']['exact']:.4f}% global, {100*report['oracle']['metrics']['poly']['exact']:.4f}% poly.",
        f"Nouvelles erreurs couvertes hors du catalogue précédent : {report['oracle']['new_errors_covered']}.",
        f"Politiques dépassant YourMT3+ en global et poly : {len(target_pass)}."]
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--models',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
