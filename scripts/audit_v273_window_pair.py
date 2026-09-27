"""Independent audit of trained 23/31-frame native count outputs on fold 3."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_window_experiment import require, WEIGHTINGS
from scripts.v273_native_protocol import load_config
from scripts.train_boundaries import group_stem


def metrics(k, prediction):
    k, prediction = np.asarray(k), np.asarray(prediction)
    n = len(k)
    poly = k >= 2
    require(k.shape == prediction.shape and np.all((k >= 0) & (k <= 6)) and
            np.all((prediction >= 0) & (prediction <= 6)), 'invalid count arrays')
    correct, over, under = (int(np.sum(prediction == k)), int(np.sum(prediction > k)), int(np.sum(prediction < k)))
    return dict(rows=n, correct=correct, exact=correct/n if n else None, over=over, under=under,
        poly_rows=int(poly.sum()), poly_correct=int(np.sum(poly & (k == prediction))),
        poly_exact=float(np.mean(k[poly] == prediction[poly])) if poly.any() else None,
        poly_over=int(np.sum(poly & (prediction > k))), poly_under=int(np.sum(poly & (prediction < k))),
        confusion=np.bincount(k*7+prediction, minlength=49).reshape(7,7).tolist())


def effect(k, before, after, mask):
    k, before, after = k[mask], before[mask], after[mask]
    fixed = (before != k) & (after == k)
    broken = (before == k) & (after != k)
    wrong_to_wrong = (before != k) & (after != k) & (before != after)
    net = int(fixed.sum() - broken.sum())
    require(net == int((after == k).sum() - (before == k).sum()), 'paired accounting mismatch')
    return dict(rows=len(k), corrected=int(fixed.sum()), degraded=int(broken.sum()),
        wrong_to_wrong=int(wrong_to_wrong.sum()), net_correct=net,
        percentage_points=100*net/len(k) if len(k) else None,
        before=metrics(k, before), after=metrics(k, after))


def read_predictions(root, phase):
    report = json.loads((root/'report.json').read_text())
    require(report['status'] == 'completed', 'incomplete training job')
    evaluation = json.loads((root/phase/'evaluation.json').read_text())
    require(digest(root/phase/'predictions.npz') == evaluation['predictions_sha256'], 'prediction digest mismatch')
    require(digest(root/phase/'training.json') == evaluation['frozen_training_sha256'], 'post-evaluation training change')
    record = json.loads((root/phase/'training.json').read_text())
    require(record == report['phases'][phase]['training'] and record['epochs'] == 8 and len(record['history']) == 8,
            'training completion evidence mismatch')
    require(digest(root/phase/'latest.weights.h5') == record['weights_sha256'], 'model weights changed')
    with np.load(root/phase/'predictions.npz', allow_pickle=False) as z:
        p = {key: np.asarray(z[key]) for key in z.files}
    probability = p['probability']
    require(probability.shape == (len(p['k']),7) and np.isfinite(probability).all() and
            (probability >= 0).all() and np.allclose(probability.sum(1),1.,atol=1e-5), 'invalid saved probabilities')
    scalar = np.asarray([max(range(7), key=row.__getitem__) for row in probability.tolist()], np.int32)
    np.testing.assert_array_equal(scalar, p['predicted'])
    actual = metrics(p['k'], scalar)
    require(actual['exact'] == evaluation['metrics']['exact'] and actual['poly_correct'] == evaluation['metrics']['poly_correct'],
            'saved metric disagrees with independent audit')
    return report, record, p


def audit(root, config, coverage, output):
    cfg = load_config(config)
    covered = json.loads(coverage.read_text())
    affected = {(e['member'],e['track_row']) for e in covered['recovered_events']}
    results = {}
    for weighting in WEIGHTINGS:
        phases = {}
        for phase in ('inner', 'final'):
            a, at, old = read_predictions(root/f'frames-23-{weighting}', phase)
            b, bt, new = read_predictions(root/f'frames-31-{weighting}', phase)
            for record, frames in ((a,23),(b,31)):
                protocol = record['protocol']
                require(protocol['frames'] == frames and protocol['weighting'] == weighting and
                        protocol['outer_fold'] == 3 and not protocol['output_corrector'] and
                        protocol['config_sha256'] == digest(config), 'wrong experiment identity')
            for key in ('bundle_sha256','source_sha','partitions','epochs_per_phase','batch_size','decoder','epoch_selection'):
                require(a['protocol'][key] == b['protocol'][key], 'unpaired protocol: '+key)
            for key in ('seed','epochs','fit_rows','predict_rows','fit_indices_sha256','predict_indices_sha256',
                        'class_weights','initial_weights_sha256','epoch_order_sha256','observed_epoch_order_sha256','batch_size'):
                require(at[key] == bt[key], 'unpaired training: '+key)
            for key in ('global_index','member','cluster_start_samples','k'):
                np.testing.assert_array_equal(old[key],new[key])
            require(len(np.unique(old['global_index'])) == len(old['global_index']), 'duplicate predicted rows')
            expected_fold = 0 if phase == 'inner' else 3
            names = old['member'].astype(str)
            require(all(cfg['member_folds'][m] == expected_fold for m in names), 'wrong evaluation fold')
            require(set(names) == {m for m,f in cfg['member_folds'].items() if f == expected_fold}, 'incomplete evaluation tracks')
            k, before, after = old['k'], old['predicted'], new['predicted']
            poly = k >= 2
            phases[phase] = dict(all=effect(k,before,after,np.ones(len(k),bool)),
                poly=effect(k,before,after,poly),
                by_true_k={str(i):effect(k,before,after,k == i) for i in range(7)},
                pairing_verified=True, probability_decoder_verified=True)
            if phase == 'final':
                require(len(k) == 15279 and int(poly.sum()) == 1969, 'outer population drift')
                track_row = np.zeros(len(k),np.int32)
                for m in set(names):
                    ids = np.flatnonzero(names == m)
                    track_row[ids] = np.arange(len(ids))
                changed_input = np.asarray([(m,int(i)) in affected for m,i in zip(names,track_row)])
                require(changed_input.sum() == 87 and (changed_input & poly).sum() == 42, 'coverage subgroup mapping changed')
                phases[phase]['previously_outside_window'] = effect(k,before,after,changed_input)
                phases[phase]['previously_outside_window_poly'] = effect(k,before,after,changed_input & poly)
                compositions = np.asarray([group_stem(m) for m in names])
                per_group = {g:effect(k,before,after,(compositions == g) & poly) for g in sorted(set(compositions))}
                phases[phase]['by_composition_poly'] = per_group
                numerators = np.asarray([r['net_correct'] for r in per_group.values()])
                denominators = np.asarray([r['rows'] for r in per_group.values()])
                draws = np.random.default_rng(27331).integers(0,len(per_group),size=(10000,len(per_group)))
                delta = 100*numerators[draws].sum(1)/denominators[draws].sum(1)
                phases[phase]['descriptive_composition_bootstrap_95_pp'] = np.percentile(delta,[2.5,97.5]).tolist()
                phases[phase]['residual_errors'] = dict(total=int((after != k).sum()),
                    poly=int((poly & (after != k)).sum()), over=int((after > k).sum()), under=int((after < k).sum()))
        results[weighting] = phases
    report = dict(status='audited', experiment='native_count_window_23_vs_31', outer_fold=3,
        primary='uniform', secondary='weighted', results=results, models_promoted=False,
        full_v273_reproduced=False, additional_corrector=False,
        limitations=['One training seed; only five outer compositions; interval is descriptive.',
                     'Fold 3 was already used for diagnostic audits; this is a development comparison.',
                     'The common historical proposal stack was not retrained separately within each fold.',
                     'These scores concern the native seven-class count component, not the complete V27.3 chain.'])
    write_json(output,report)
    print(json.dumps({w:results[w]['final']['poly'] for w in WEIGHTINGS},indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','coverage','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    audit(a.root,a.config,a.coverage,a.output)
