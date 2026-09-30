"""Independently recount the fixed-epoch native pair from saved probabilities."""
import argparse
import json
from pathlib import Path

import numpy as np
from scripts.rebuild_v273_sources import digest,write_json
from scripts.train_boundaries import group_stem
from scripts.v273_high_pitch_native import ARMS
from scripts.v273_high_pitch_registers import masks as register_masks
from scripts.v273_window_experiment import require


def count(k,p):
    pred=np.argmax(p,axis=1)
    confusion=np.bincount(k*7+pred,minlength=49).reshape(7,7)
    rows=confusion.sum(1);good=np.diag(confusion)
    return dict(rows=int(rows.sum()),correct=int(good.sum()),exact=float(good.sum()/rows.sum()),
        poly_rows=int(rows[2:].sum()),poly_correct=int(good[2:].sum()),
        poly_exact=float(good[2:].sum()/rows[2:].sum()),
        over=int(np.sum(pred>k)),under=int(np.sum(pred<k)),
        over_k_lt4=int(np.sum((k<4)&(pred>k))),
        by_true_k={str(v):dict(rows=int(rows[v]),correct=int(good[v]),
            exact=float(good[v]/rows[v]),over=int(np.sum((k==v)&(pred>k))),
            under=int(np.sum((k==v)&(pred<k)))) for v in range(7)},
        confusion_true_by_predicted=confusion.tolist())


def bootstrap(k,member,a,b,mask):
    compositions=np.asarray([group_stem(m) for m in member])
    names=np.unique(compositions);denom=[];difference=[]
    for name in names:
        selected=(compositions==name)&mask
        denom.append(int(selected.sum()))
        difference.append(int(np.sum((b==k)&selected))-int(np.sum((a==k)&selected)))
    rng=np.random.default_rng(9302026)
    weights=np.array([np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names)) for _ in range(2000)])
    total=weights@np.asarray(denom);valid=total>0
    values=(weights@np.asarray(difference))[valid]/total[valid]
    return dict(compositions=names.tolist(),valid_draws=int(valid.sum()),seed=9302026,
        difference=float(np.sum(difference)/np.sum(denom)),
        percentile95=np.quantile(values,[.025,.975]).tolist(),scope='descriptive internal development result')


def subset(k,pred,take):
    n=int(np.sum(take))
    return dict(rows=n,correct=int(np.sum((pred==k)&take)),
        exact=float(np.mean(pred[take]==k[take])) if n else None,
        over=int(np.sum((pred>k)&take)),under=int(np.sum((pred<k)&take)))


def stratify(k,pitches,member,a,b):
    selections=register_masks(k,pitches)
    strata={name:{ARMS[0]:subset(k,a,take),ARMS[1]:subset(k,b,take)} for name,take in selections.items()}
    names=np.asarray([group_stem(m) for m in member])
    by_composition={name:{ARMS[0]:subset(k,a,names==name),ARMS[1]:subset(k,b,names==name)} for name in np.unique(names)}
    return strata,by_composition,selections


def run(args):
    require(not args.output.exists(),'refusing to overwrite an audit')
    config=json.loads(args.config.read_text())
    allowed={m for m,f in config['member_folds'].items() if f==0}
    source={arm:json.loads((args.root/('high-pitch-'+arm)/'report.json').read_text()) for arm in ARMS}
    identity=('source_sha','launch_sha256','protocol_sha256','config_sha256','feature_builder_sha256',
        'register_builder_sha256','seed','epochs','primary_epoch','snapshots','batch_size',
        'tensorflow','numpy','parameters','initial_sha256','preflight_sha256','prepared_report_sha256',
        'maps_sha256','geometry_report_sha256','bundle_sha256','fit_indices_sha256','val_indices_sha256',
        'decode','checkpoint_selection','weighting','epoch_orders','script_sha256','model_builder_sha256')
    for key in identity:
        require(source[ARMS[0]][key]==source[ARMS[1]][key],'paired conditions differ: '+key)
    for arm,r in source.items():
        require(r['arm']==arm and r['status']=='completed' and r['epochs']==12 and
                r['primary_epoch']==12 and r['outer_rows_evaluated']==0 and not r['output_corrector'],
                'incomplete or altered training')
        require([h['epoch'] for h in r['history']]==list(range(1,13)) and len(r['epoch_orders'])==12,
                'wrong number of epochs')
    epochs={};saved={}
    for epoch in (4,8,12):
        result={};previous=None
        for arm in ARMS:
            root=args.root/('high-pitch-'+arm);r=source[arm];checkpoint=r['checkpoints'][str(epoch)]
            path=root/f'epoch-{epoch:02d}-validation.npz'
            require(digest(path)==checkpoint['predictions_sha256'] and
                    digest(root/f'epoch-{epoch:02d}.weights.h5')==checkpoint['weights_sha256'],
                    'checkpoint changed')
            require(checkpoint['optimizer_iterations']==epoch*339,'update budget differs')
            with np.load(path,allow_pickle=False) as z:
                data={name:z[name] for name in z.files}
            k=data['k'];p=data['probability']
            require(len(k)==15952 and set(data['member'])==allowed,'wrong validation rows')
            require(np.isfinite(p).all() and (p>=0).all() and (p<=1).all(),'invalid probability')
            np.testing.assert_allclose(p.sum(1),1,atol=1e-6)
            np.testing.assert_array_equal(data['predicted'],p.argmax(1))
            require(np.all(np.diff(data['global_index'])>0),'duplicate/reordered indices')
            if previous is not None:
                for key in ('k','member','global_index'):
                    np.testing.assert_array_equal(data[key],previous[key])
            previous=data
            result[arm]=count(k,p)
            for key in ('rows','correct','exact','poly_rows','poly_correct','poly_exact','under','over'):
                np.testing.assert_allclose(result[arm][key],checkpoint['validation'][key],atol=1e-8)
            if epoch==12:saved[arm]=data
        epochs[str(epoch)]=result
    control,treatment=(epochs['12'][a] for a in ARMS)
    k=saved[ARMS[0]]['k'];member=saved[ARMS[0]]['member']
    a,b=(saved[arm]['predicted'] for arm in ARMS)
    prepared_path=args.audit_inputs/'report.json'
    require(digest(prepared_path)==source[ARMS[0]]['prepared_report_sha256'],'audit metadata comes from another preparation')
    prepared=json.loads(prepared_path.read_text())
    preflight_path=args.preflight/'report.json'
    require(digest(preflight_path)==source[ARMS[0]]['preflight_sha256'],'runtime/preflight report changed')
    preflight=json.loads(preflight_path.read_text())
    pitch_path=args.audit_inputs/'validation-registers.npz'
    require(digest(pitch_path)==prepared['validation_registers']['sha256'],'register metadata changed')
    with np.load(pitch_path,allow_pickle=False) as z:
        for key in ('k','member','global_index'):
            np.testing.assert_array_equal(z[key],saved[ARMS[0]][key])
        pitches=z['pitches_midi']
    strata,compositions,selections=stratify(k,pitches,member,a,b)
    require(selections['k3_bass'].sum()==187 and selections['any_high'].sum()==263,'frozen register population changed')
    gate=dict(poly_gain=treatment['poly_exact']>control['poly_exact'],
        global_no_regression=treatment['exact']>=control['exact'],
        k1_k2_k3_no_regression=all(treatment['by_true_k'][str(v)]['exact']>=
            control['by_true_k'][str(v)]['exact'] for v in (1,2,3)),
        fewer_overcounts_k_lt4=treatment['over_k_lt4']<control['over_k_lt4'],
        k3_bass_gain=strata['k3_bass'][ARMS[1]]['correct']>strata['k3_bass'][ARMS[0]]['correct'],
        high_no_regression=strata['any_high'][ARMS[1]]['correct']>=strata['any_high'][ARMS[0]]['correct'])
    gate['favorable_to_replication']=all(gate.values())
    args.output.mkdir(parents=True)
    report=dict(status='completed',experiment='native_high_pitch_pair',primary_epoch=12,
        epochs=epochs,decision=gate,automatic_promotion=False,output_corrector=False,outer_rows_evaluated=0,
        register_strata=strata,by_composition=compositions,validation_registers=prepared['validation_registers'],
        runtime=dict(features=preflight['feature_runtime'],inference=preflight['inference_single_group'],scope=preflight['latency_scope']),
        paired=dict(corrected=int(np.sum((a!=k)&(b==k))),regressed=int(np.sum((a==k)&(b!=k))),
            unchanged=int(np.sum(a==b)),changed=int(np.sum(a!=b))),
        bootstrap=dict(global_exact=bootstrap(k,member,a,b,np.ones(len(k),bool)),
                       poly_exact=bootstrap(k,member,a,b,k>=2),
                       k3_bass=bootstrap(k,member,a,b,selections['k3_bass']),
                       any_high=bootstrap(k,member,a,b,selections['any_high'])),
        source_reports={arm:digest(args.root/('high-pitch-'+arm)/'report.json') for arm in ARMS},
        common_conditions={key:source[ARMS[0]][key] for key in identity},
        audit_sha256=digest(__file__),limitations=[
            'one seed, five previously examined internal validation compositions',
            'no outer-fold or Locked12 evaluation',
            'not a full V27.3 score replacement',
            'MIDI strata overlap; K counts assigned attacks, not all sounding notes',
            'rare high-register notes; centered filtering uses synthetic zero frame boundaries',
            'fixed source support tests representation, not acquisition of extra periods'])
    write_json(args.output/'report.json',report)
    lines=['# Comparaison native du high pitch rescue — résultat interne', '',
        'Époque 12 fixée avant entraînement, une graine, 15 952 groupes. Aucun correcteur de sortie.', '',
        '| Mesure | Observé seul | Avec high pitch rescue |','|---|---:|---:|',
        f"| Exact K global | {100*control['exact']:.4f} % | {100*treatment['exact']:.4f} % |",
        f"| Exact K polyphonique | {100*control['poly_exact']:.4f} % | {100*treatment['poly_exact']:.4f} % |",
        f"| Surcomptages K < 4 | {control['over_k_lt4']} | {treatment['over_k_lt4']} |",
        f"| Sous-comptages | {control['under']} | {treatment['under']} |", '',
        '| Vrai K | Groupes | Observé seul, Exact K | Avec high pitch rescue, Exact K |','|---|---:|---:|---:|']
    for v in range(7):
        c,t=(x['by_true_k'][str(v)] for x in (control,treatment))
        lines.append(f"| {v} | {c['rows']} | {100*c['exact']:.4f} % | {100*t['exact']:.4f} % |")
    lines+=['','| Registre du groupe | Groupes | Observé seul, Exact K | Avec high pitch rescue, Exact K |','|---|---:|---:|---:|']
    labels=dict(any_bass='Au moins une note sous C3',any_high='Au moins une note à partir de C5',
        only_bass='Graves seuls',only_mid='C3–B4 seuls',only_high='Aigus seuls',mixed_registers='Registres mixtes',
        octave_pair='Doublure à une ou plusieurs octaves',k3_bass='K=3 avec grave',k3_no_bass='K=3 sans grave')
    for name,values in strata.items():
        c,t=(values[arm] for arm in ARMS)
        left='n/a' if c['exact'] is None else f"{100*c['exact']:.4f} %"
        right='n/a' if t['exact'] is None else f"{100*t['exact']:.4f} %"
        lines.append(f"| {labels[name]} | {c['rows']} | {left} | {right} |")
    lines+=['','Les strates se chevauchent ; elles ne doivent pas être additionnées. K=0 ne signifie pas silence.']
    lines+=['','| Construction des vues sur CPU, mesure descriptive | p50 ms/groupe | p95 ms/groupe |','|---|---:|---:|']
    for arm,timing in preflight['feature_runtime'].items():
        lines.append(f"| {arm} | {timing['p50_ms']:.3f} | {timing['p95_ms']:.3f} |")
    lines+=['',f"Critère favorable à une réplication : **{gate['favorable_to_replication']}**.", '',
        'Ce résultat de développement sur une seule graine ne prouve pas une généralisation externe.',
        'Source audio et horodatages identiques entre bras ; aucune réduction de latence réelle n’est revendiquée.','']
    (args.output/'report.md').write_text('\n'.join(lines))
    print(json.dumps(dict(primary=epochs['12'],decision=gate,paired=report['paired'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','output','audit-inputs','preflight'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
