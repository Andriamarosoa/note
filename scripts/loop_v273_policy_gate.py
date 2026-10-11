"""Exploratory V27.3 regression loop: fixed, label-blind series-5 hybrid gates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require

GLOBAL = 'open_k0k6_series5__prior_corrected__tilt1.5__freeze__mix1'
POLY = 'open_k0k6_series5__raw__tilt1.5__freeze__mix1'
GLOBAL08 = 'open_k0k6_series5__prior_corrected__tilt1.5__previous__mix0.8'
POLY08 = 'open_k0k6_series5__raw__tilt1.5__previous__mix0.8'

def hybrid_policies(base, global_pred, poly_pred, global08, poly08):
    """Return 16 predeclared policies without labels, folds or audit outcomes."""
    b, g, p, g8, p8 = [np.asarray(x) for x in
                          (base, global_pred, poly_pred, global08, poly08)]
    require(all(x.shape == b.shape and x.ndim == 1 for x in (g,p,g8,p8)),
            'hybrid shape')
    tests = {
        'poly_if_both_poly': (p>=2)&(g>=2),
        'poly_if_both_poly_near': (p>=2)&(g>=2)&(abs(p-g)<=1),
        'poly_if_base_poly': (p>=2)&(b>=2),
        'poly_if_base_poly_near': (p>=2)&(b>=2)&(abs(p-b)<=1),
        'poly_if_both_and_base_poly': (p>=2)&(g>=2)&(b>=2),
        'poly_if_both_and_base_poly_near': (p>=2)&(g>=2)&(b>=2)&(abs(p-b)<=1),
        'poly_if_upcount': (p>=2)&(g>=2)&(p>g),
        'poly_if_downcount': (p>=2)&(g>=2)&(p<g),
        'poly_if_high_and_base_poly': (p>=4)&(b>=2),
        'poly_if_base_agrees': (p==b)&(b>=2),
        'poly_if_poly08_agrees': (p==p8)&(p>=2),
        'poly_if_global08_disagrees': (g!=g8)&(p>=2),
    }
    out = {GLOBAL:g.copy(),POLY:p.copy(),GLOBAL08:g8.copy(),POLY08:p8.copy()}
    for name,mask in tests.items():
        out['hybrid5__'+name]=np.where(mask,p,g).astype(np.int8)
    require(len(out)==16,'policy inventory changed')
    return out

def load(path):
    with np.load(path,allow_pickle=False) as z:
        return {key:z[key] for key in z.files}

def main(a):
    src=a.root/'analysis/evidence'
    require(not a.output.exists(),'refusing to overwrite historical outputs')
    d=load(src/'v273-regression-loops/prepared/inputs.npz')
    ids,y,base,fold=(d[k] for k in
        ('native_global_index','native_truth','native_baseline','native_fold'))
    require(len(ids)==len(y)==len(base)==len(fold)==59309,'cohort size')
    require(len(set(ids.tolist()))==len(ids) and set(fold.tolist())==set(FOLDS),
            'fold/identity drift')
    archive=load(src/'v273-open-k0-k6/series5/predictions.npz')
    require(np.array_equal(archive['global_index'],ids),'series5 ID alignment')
    names=[str(s) for s in archive['variant_ids']]
    require(len(names)==len(set(names)),'duplicate policy names')
    matrix=archive['predictions']
    require(matrix.shape==(len(ids),len(names)),'prediction matrix shape')
    for key,expected in [('true_K',y),('baseline_K',base),('fold',fold)]:
        if key in archive:
            require(np.array_equal(archive[key],expected),'series5 '+key+' drift')
    def get(name):
        require(name in names,'missing series5 policy: '+name)
        return matrix[:,names.index(name)]
    policies=hybrid_policies(base,get(GLOBAL),get(POLY),get(GLOBAL08),get(POLY08))
    target=load(src/'v273-yourmt3-target/comparison/row-evidence.npz')
    require(np.array_equal(target['global_index'],ids),'YourMT3 row mismatch')
    competitor=target['yourmt3_K']; previous=target['current_best_K']
    n_global=int(np.sum(competitor==y))
    n_poly=int(np.sum((competitor==y)&(y>=2)))
    require(n_global==51328 and n_poly==4028,'YourMT3 reference changed')
    results={}
    for name,pred in policies.items():
        results[name]=dict(metrics=metrics(y,pred),
            versus_freeze=paired(y,base,pred),
            versus_previous=paired(y,previous,pred),
            versus_yourmt3=paired(y,competitor,pred),
            versus_global_parent=paired(y,policies[GLOBAL],pred),
            versus_poly_parent=paired(y,policies[POLY],pred),
            folds={str(f):dict(
                metrics=metrics(y[fold==f],pred[fold==f]),
                versus_global_parent=paired(
                    y[fold==f],policies[GLOBAL][fold==f],pred[fold==f]))
                for f in FOLDS})
    pareto=[]
    for name,entry in results.items():
        global_correct=entry['metrics']['correct']
        poly_correct=entry['metrics']['poly']['correct']
        if not any((r['metrics']['correct']>=global_correct and
                    r['metrics']['poly']['correct']>=poly_correct) and
                   (r['metrics']['correct']>global_correct or
                    r['metrics']['poly']['correct']>poly_correct)
                   for r in results.values()):
            pareto.append(name)
    ordered=sorted(results,key=lambda name:(
        results[name]['metrics']['poly']['correct'],
        results[name]['metrics']['correct']),reverse=True)
    report=dict(status='completed',
        population='59,309 exposed development examples',
        validation='not independent; no model promoted',
        method='fixed label-blind source-level decision gates',
        target_global_correct=n_global,target_poly_correct=n_poly,
        policies=results,pareto=pareto,best_poly_this_loop=ordered[0],
        count=len(results),promotion=False)
    a.output.mkdir(parents=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    np.savez_compressed(a.output/'predictions.npz',global_index=ids,
        true_K=y,fold=fold,variant_ids=np.asarray(list(policies)),
        predictions=np.column_stack(list(policies.values())))
    lines=['# Series 6: regression-aware, fixed hybrid gate controls','',
        'All policies are predeclared and ignore target truth; exploratory development data only.','',
        '| Policy | Exact-K global | Exact-K poly | Corrections vs global parent | Regressions vs global parent | K5 | K6 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ordered:
        record=results[name];m=record['metrics']
        gain=record['versus_global_parent']['global']
        lines.append(f"| {name} | {100*m['exact']:.4f}% | "
            f"{100*m['poly']['exact']:.4f}% | {gain['corrections']} | "
            f"{gain['regressions']} | {100*m['by_k']['5']['exact']:.2f}% | "
            f"{100*m['by_k']['6']['exact']:.2f}% |")
    passed=sum(r['metrics']['correct']>n_global and
        r['metrics']['poly']['correct']>n_poly for r in results.values())
    lines+=['',f'Pareto candidates: {len(pareto)}',
        f'Best poly in this loop: {ordered[0]}',
        f'Policies beating YourMT3+ on both metrics: {passed}',
        'No independent validation or automatic promotion.']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)

def selftest():
    b=np.array([0,1,2,3,4,3,2,1,5,2,3],dtype=np.int8)
    g=np.array([0,1,1,3,0,4,2,3,5,1,2],dtype=np.int8)
    p=np.array([0,2,2,2,4,3,3,2,6,2,4],dtype=np.int8)
    g8=g.copy();g8[6]=4
    p8=p.copy();p8[9]=3
    out=hybrid_policies(b,g,p,g8,p8)
    assert len(out)==16
    assert np.array_equal(out[GLOBAL],g) and np.array_equal(out[POLY],p)
    assert out['hybrid5__poly_if_base_poly'][1]==g[1]
    assert out['hybrid5__poly_if_base_poly'][4]==p[4]
    assert out['hybrid5__poly_if_upcount'][10]==p[10]
    assert out['hybrid5__poly_if_downcount'][3]==p[3]
    for name,pred in out.items():
        if name.startswith('hybrid5__'):
            assert np.logical_or(pred==g,pred==p).all(),name
    print('PASS: 16 deterministic policies and regression constraints')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--output',type=Path,default=Path('model/v273-policy-loop'))
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:selftest()
    else:main(args)
