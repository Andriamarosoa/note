"""Audit overcounting for true K=0..3 on frozen fold 3, without retraining.

Saved decisions, annotated context and diagnostic interventions are separate.
Interventions are sensitivity probes, never a proposed output corrector.
"""
import argparse
import gc
import json
from pathlib import Path
import numpy as np

from causal_note.guitarset import load_boundary_slots
from scripts.audit_v273_control_inputs import nearest_assignment
from scripts.audit_v273_window_pair import read_predictions
from scripts.candidate_timing import full_samples
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.spectral_window import LEGACY, COVERED, cache_window
from scripts.train_boundaries import group_stem
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import require

VARIANTS = ('frames-23-uniform', 'frames-31-uniform', 'frames-23-weighted', 'frames-31-weighted')
STAT_NAMES = ('candidate_count_log', 'group_width', 'proposal_mean', 'proposal_max',
              'router_mean', 'local_count_mean_norm', 'local_count_weighted_norm', 'local_birth_max')


def decisions(k, predicted, select=None):
    k, predicted = np.asarray(k), np.asarray(predicted)
    require(k.shape == predicted.shape and k.ndim == 1, 'invalid decision arrays')
    take = k < 4 if select is None else np.asarray(select, bool)
    y, p = k[take], predicted[take]
    over = p > y
    return dict(rows=len(y), correct=int((p == y).sum()), over=int(over.sum()),
                under=int((p < y).sum()), over_rate=float(over.mean()) if len(y) else None,
                confusion=np.bincount(y * 7 + p, minlength=49).reshape(7, 7).tolist(),
                excess_histogram=np.bincount((p-y)[over], minlength=7).tolist())


def transitions(k, before, after, take=None):
    take = k < 4 if take is None else take
    y, a, b = k[take], before[take], after[take]
    old_over, new_over = a > y, b > y
    old_code, new_code = np.sign(a-y).astype(int)+1, np.sign(b-y).astype(int)+1
    matrix = np.bincount(old_code * 3 + new_code, minlength=9).reshape(3, 3)
    return dict(rows=len(y), status_order=['under', 'correct', 'over'], transition_matrix=matrix.tolist(),
        before_over=int(old_over.sum()), after_over=int(new_over.sum()),
        persistent_over=int((old_over & new_over).sum()),
        new_over=int((~old_over & new_over).sum()), removed_over=int((old_over & ~new_over).sum()),
        over_to_correct=int((old_over & (b == y)).sum()), over_to_under=int((old_over & (b < y)).sum()),
        correct_to_over=int(((a == y) & new_over).sum()), under_to_over=int(((a < y) & new_over).sum()),
        net_over=int(new_over.sum()-old_over.sum()))


def describe(a):
    a = np.asarray(a)
    return dict(rows=len(a), mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p10=float(np.percentile(a, 10)) if len(a) else None,
                p90=float(np.percentile(a, 90)) if len(a) else None)


def probability_summary(k, probability):
    pred = probability.argmax(1)
    result = {}
    for value in range(4):
        rows = np.flatnonzero((k == value) & (pred > k))
        p = probability[rows]
        sorted_p = np.sort(p, axis=1)
        true = p[:, value]
        rank = 1 + np.sum(p > true[:, None], axis=1)
        result[str(value)] = dict(rows=len(rows), winner=describe(sorted_p[:, -1]),
            top_two_margin=describe(sorted_p[:, -1]-sorted_p[:, -2]), true_probability=describe(true),
            true_rank_histogram=np.bincount(rank, minlength=8).tolist(),
            winner_at_least_075=int((sorted_p[:, -1] >= .75).sum()),
            predicted_histogram=np.bincount(pred[rows], minlength=7).tolist())
    return result


def event_context(starts, groups, events, exact):
    """Recompute labels; distinguish assigned births from other audible notes.

    events: (onset_sample, offset_sample, string) in annotation order.
    Foreign means assigned to another group OR unassigned, not a wrong label.
    """
    n = len(starts)
    events = np.asarray(events, np.int64).reshape(-1, 3)
    onset, offset = events[:, 0], events[:, 1]
    flat = np.concatenate(groups)
    row_ids = np.repeat(np.arange(n), [len(g) for g in groups])
    order = np.argsort(flat, kind='stable')
    assigned = np.full(len(events), -1, np.int32)
    for i, t in enumerate(onset):
        answer = nearest_assignment(int(t), flat[order], row_ids[order])
        if answer is not None:
            assigned[i] = answer[0]
    counts = np.bincount(assigned[assigned >= 0], minlength=n)
    np.testing.assert_array_equal(counts, exact)
    names = ('foreign_births_23', 'foreign_births_31', 'foreign_births_added_tail',
             'own_births_added_tail', 'carried_notes_at_start', 'overlapping_notes_31',
             'offsets_31', 'foreign_assigned_births_31', 'unassigned_births_31')
    values = {key: np.zeros(n, np.int32) for key in names}
    for row, start in enumerate(starts):
        left, right = int(start)-COVERED.pre_samples, int(start)+COVERED.post_samples
        born31 = (onset >= left) & (onset < right)
        born23 = (onset >= left) & (onset < int(start)+LEGACY.post_samples)
        tail = (onset >= int(start)+LEGACY.post_samples) & (onset < right)
        foreign = assigned != row
        values['foreign_births_23'][row] = np.sum(born23 & foreign)
        values['foreign_births_31'][row] = np.sum(born31 & foreign)
        values['foreign_births_added_tail'][row] = np.sum(tail & foreign)
        values['own_births_added_tail'][row] = np.sum(tail & ~foreign)
        values['carried_notes_at_start'][row] = np.sum((onset < start) & (offset > start) & foreign)
        values['overlapping_notes_31'][row] = np.sum((onset < right) & (offset > left))
        values['offsets_31'][row] = np.sum((offset >= left) & (offset < right))
        values['foreign_assigned_births_31'][row] = np.sum(born31 & foreign & (assigned >= 0))
        values['unassigned_births_31'][row] = np.sum(born31 & (assigned < 0))
    return values, dict(annotations=len(events), assigned=int((assigned >= 0).sum()),
                       unassigned=int((assigned < 0).sum()), labels_recomputed=True)


def load_outer(root, annotations, config):
    cfg = load_config(config)
    coverage = json.loads((root/'report.json').read_text())
    hashes = {r['member']: r['covered_sha256'] for r in coverage['tracks']}
    require(coverage['status'] == 'passed' and coverage['outer_fold'] == 3, 'wrong coverage source')
    required = {m for m, f in cfg['member_folds'].items() if f == 3}
    records = []
    seen = set()
    for path in sorted(root.rglob('v100-spectral-shard-*.npz')):
        with np.load(path, allow_pickle=False) as z:
            cache = {key: np.asarray(z[key]) for key in z.files}
        members = set(cache['members'].astype(str))
        require(len(members) == 1 and not members & seen and members <= required, 'wrong outer shard')
        member = next(iter(members))
        seen |= members
        require(digest(path) == hashes[member] and cache_window(cache, version=3) == COVERED, 'outer source changed')
        groups = full_samples(cache)
        events = [(b.onset_sample, b.offset_sample, s)
                  for s, boundaries in enumerate(load_boundary_slots(annotations, member)) for b in boundaries]
        context, check = event_context(cache['cluster_start_samples'], groups, events, cache['exact'])
        context['track_row'] = np.arange(len(groups), dtype=np.int32)
        context['retained_candidates'] = cache['mask'].sum(1).astype(np.int32)
        context['full_candidates'] = np.asarray([len(g) for g in groups], np.int32)
        for i, key in enumerate(STAT_NAMES):
            context[key] = np.asarray(cache['stats'][:, i], np.float32)
        records.append((member, cache, context, check))
    require(seen == required and len(records) == 50, 'incomplete outer fold')
    records.sort(key=lambda r: r[0])
    keys = ('sequence', 'mask', 'stats', 'spectral', 'exact', 'members', 'cluster_start_samples')
    cache = {key: np.concatenate([r[1][key] for r in records]) for key in keys}
    context = {key: np.concatenate([r[2][key] for r in records]) for key in records[0][2]}
    checks = {key: sum(r[3][key] for r in records) for key in ('annotations', 'assigned', 'unassigned')}
    checks.update(labels_recomputed=True, tracks=50, rows=len(cache['exact']))
    require(checks['rows'] == 15279 and checks['assigned'] == 8812 and checks['unassigned'] == 454,
            'annotation or outer population drift')
    return cache, context, checks


def associations(k, pred, context):
    result = {}
    def compact(select):
        return {key: value for key, value in decisions(k, pred, select).items()
                if key not in ('confusion', 'excess_histogram')}
    for value in range(4):
        take = k == value
        flags = {key: context[key] > 0 for key in (
            'foreign_births_31', 'foreign_births_added_tail', 'own_births_added_tail',
            'carried_notes_at_start', 'offsets_31', 'foreign_assigned_births_31', 'unassigned_births_31')}
        flags['no_annotated_note_overlap_31'] = context['overlapping_notes_31'] == 0
        flags['candidate_truncation'] = context['full_candidates'] > context['retained_candidates']
        result[str(value)] = {'flags': {key: {
            'present': compact(take & flag), 'absent': compact(take & ~flag)
        } for key, flag in flags.items()}, 'features': {key: {
            'over': describe(context[key][take & (pred > k)]),
            'correct': describe(context[key][take & (pred == k)])
        } for key in (*STAT_NAMES, 'retained_candidates', 'full_candidates')}}
    return result


def spectral_probe(x, kind):
    """Return a copy; preserve all three channels of unmodified frames."""
    require(x.shape[1] == 31, '31-frame interventions only')
    z = np.array(x, copy=True)
    if kind == 'tail_zero':
        z[:, 23:] = 0
    elif kind == 'tail_hold':
        z[:, 23:] = z[:, 22:23]
    elif kind == 'head_zero':
        z[:, :8] = 0
    else:
        raise ValueError(kind)
    return z


def replay_and_probe(cache, root, saved, output):
    import tensorflow as tf
    from scripts.train_v260_count_weighting import build_model
    require(tf.__version__ == '2.15.1', 'pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    records = {}
    n, k = len(cache['exact']), np.minimum(cache['exact'].astype(np.int32), 6)
    for variant in VARIANTS:
        frames, weighting = int(variant.split('-')[1]), variant.split('-')[2]
        report, training, pred = saved[variant]
        model = build_model(weighting, training['seed'], time_frames=frames)
        model.load_weights(root/variant/'final/latest.weights.h5')
        fused = model.get_layer('v240_cardinality_context')
        widths = [int(t.shape[-1]) for t in fused.input]
        probe = tf.keras.Model(model.inputs, [model.output, fused.output])
        h = tf.keras.Input(shape=(sum(widths),))
        z = h
        for name in ('v240_cardinality_hidden1', 'v240_cardinality_dropout', 'v240_cardinality_hidden2', 'cardinality'):
            z = model.get_layer(name)(z)
        head = tf.keras.Model(h, z)
        def forward(kind=None):
            probability, latent = [], []
            for start in range(0, n, 128):
                stop = min(n, start+128)
                x = {'candidate_set': cache['sequence'][start:stop].astype(np.float32),
                     'candidate_mask': cache['mask'][start:stop].astype(np.float32),
                     'cluster_stats': cache['stats'][start:stop].astype(np.float32),
                     'spectral_map': cache['spectral'][start:stop, :frames].astype(np.float32)}
                if kind:
                    x['spectral_map'] = spectral_probe(x['spectral_map'], kind)
                p, l = probe(x, training=False)
                probability.append(np.asarray(p)); latent.append(np.asarray(l))
            return np.concatenate(probability), np.concatenate(latent)
        baseline, latent = forward()
        np.testing.assert_allclose(baseline, pred['probability'], rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(baseline.argmax(1), pred['predicted'])
        reconstructed = np.asarray(head(latent, training=False))
        np.testing.assert_allclose(reconstructed, baseline, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(reconstructed.argmax(1), pred['predicted'])
        outputs = dict(baseline=baseline)
        for name, section in (('zero_candidate_context', slice(sum(widths[:2]), None)),
                              ('zero_spectral_context', slice(0, sum(widths[:2])))):
            changed = latent.copy(); changed[:, section] = 0
            outputs[name] = np.asarray(head(changed, training=False))
        if frames == 31:
            for kind in ('tail_zero', 'tail_hold', 'head_zero'):
                outputs[kind] = forward(kind)[0]
        for name, p in outputs.items():
            require(p.shape == (n, 7) and np.isfinite(p).all() and np.allclose(p.sum(1), 1, atol=1e-5),
                    'invalid intervention probability')
        np.savez_compressed(output/(variant+'-probes.npz'), **outputs)
        layer_names = [layer.name for layer in model.layers]
        records[variant] = dict(replay_argmax_identical=True,
            replay_max_probability_difference=float(np.max(np.abs(baseline-pred['probability']))),
            head_replay_identical=True, context_widths=widths, layers=layer_names,
            interventions={name: {
                'low_k': transitions(k, pred['predicted'], p.argmax(1)),
                'by_true_k': {str(i): transitions(k, pred['predicted'], p.argmax(1), k == i) for i in range(4)}
            } for name, p in outputs.items() if name != 'baseline'})
        # Detailed joint effects for genuine newly introduced low-K overcounts.
        if frames == 31:
            before = saved['frames-23-'+weighting][2]['predicted']
            old, current = before > k, pred['predicted'] > k
            for subgroup, take in (('new_over_from_23', (k < 4) & ~old & current),
                                   ('persistent_over_from_23', (k < 4) & old & current)):
                records[variant][subgroup] = {name: transitions(k, pred['predicted'], p.argmax(1), take)
                    for name, p in outputs.items() if name != 'baseline'}
        print(json.dumps(dict(variant=variant, replay='passed', interventions=list(outputs))), flush=True)
        del model, probe, head, latent, outputs
        tf.keras.backend.clear_session(); gc.collect()
    return records


def run(args):
    require(not args.output.exists(), 'refusing to overwrite audit')
    args.output.mkdir(parents=True)
    cfg = load_config(args.config)
    saved = {}
    for variant in VARIANTS:
        validate_original_inventory(args.results/variant)
        saved[variant] = read_predictions(args.results/variant, 'final')
    first = saved[VARIANTS[0]][2]
    k = first['k']
    require(len(k) == 15279 and int((k < 4).sum()) == 14744, 'wrong population')
    require(all(cfg['member_folds'][str(m)] == 3 for m in first['member']), 'not fold 3')
    for variant, (r, t, p) in saved.items():
        for key in ('global_index', 'member', 'cluster_start_samples', 'k'):
            np.testing.assert_array_equal(p[key], first[key])
        for key in ('seed', 'epochs', 'initial_weights_sha256', 'fit_indices_sha256', 'predict_indices_sha256',
                    'epoch_order_sha256', 'observed_epoch_order_sha256', 'batch_size'):
            require(t[key] == saved[VARIANTS[0]][1][key], 'unpaired experiment: '+key)
        require(r['protocol']['source_sha'] == 'cdb11b6cfe50beac86194fc6a2ddca06149e8467', 'wrong training source')
        require(r['protocol']['bundle_sha256'] == saved[VARIANTS[0]][0]['protocol']['bundle_sha256'], 'different bundle')
    report = dict(status='decision_audit_completed', outer_fold=3, scope='true K=0,1,2,3; overcount means predicted K > true K',
                  full_rows=len(k), low_k_rows=int((k < 4).sum()), variants={}, paired_effects={}, training=False,
                  output_corrector=False, model_promotion=False, full_v273_pipeline=False,
                  audit_script_sha256=digest(__file__), config_sha256=digest(args.config))
    compositions = np.asarray([group_stem(str(m)) for m in first['member']])
    for variant, (r, t, p) in saved.items():
        report['variants'][variant] = dict(low_k=decisions(k, p['predicted']),
            by_true_k={str(i): decisions(k, p['predicted'], k == i) for i in range(4)},
            probability=probability_summary(k, p['probability']), class_weights=t['class_weights'],
            by_composition={g: decisions(k, p['predicted'], (k < 4) & (compositions == g)) for g in sorted(set(compositions))},
            model_sha256=t['weights_sha256'], predictions_sha256=digest(args.results/variant/'final/predictions.npz'))
    for name, a, b in (
        ('window_uniform', 'frames-23-uniform', 'frames-31-uniform'),
        ('window_weighted', 'frames-23-weighted', 'frames-31-weighted'),
        ('weighting_23', 'frames-23-uniform', 'frames-23-weighted'),
        ('weighting_31', 'frames-31-uniform', 'frames-31-weighted')):
        report['paired_effects'][name] = dict(low_k=transitions(k, saved[a][2]['predicted'], saved[b][2]['predicted']),
            by_true_k={str(i): transitions(k, saved[a][2]['predicted'], saved[b][2]['predicted'], k == i) for i in range(4)})
    if args.outer is not None:
        require(args.annotations is not None, 'annotations required with outer inputs')
        require(digest(args.annotations, 'md5') == 'b39b78e63d3446f2e54ddb7a54df9b10', 'annotation archive differs')
        validate_original_inventory(args.outer)
        cache, context, checks = load_outer(args.outer, args.annotations, args.config)
        for cached, predicted in (('members','member'),('cluster_start_samples','cluster_start_samples'),('exact','k')):
            np.testing.assert_array_equal(cache[cached], first[predicted])
        report['annotation_checks'] = checks
        np.savez_compressed(args.output/'row-context.npz', **context, k=k, member=first['member'],
                            global_index=first['global_index'], cluster_start_samples=first['cluster_start_samples'])
        for variant, (_, _, p) in saved.items():
            report['variants'][variant]['context_associations'] = associations(k, p['predicted'], context)
        if args.probe:
            report['replay_and_sensitivity'] = replay_and_probe(cache, args.results, saved, args.output)
        report['status'] = 'completed'
    else:
        require(not args.probe, 'input probes require outer inputs')
    report['limitations'] = [
        'One development fold and one training seed; no inference on other outer folds.',
        'K counts assigned new onsets, not all audible or sustaining notes; K=0 does not establish silence.',
        'Context associations are observational, including within-true-K comparisons.',
        'Occlusion and zero-latent interventions may leave the training distribution: sensitivity, not an acoustic proof or a correction.',
        'No raw waveform source separation; harmonic confusion is not established by these tests.',
        'This audits the native count component, not the complete V27.3 pipeline.']
    write_json(args.output/'report.json', report)
    print(json.dumps({v: x['low_k']['over'] for v, x in report['variants'].items()}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--outer', type=Path)
    p.add_argument('--annotations', type=Path)
    p.add_argument('--probe', action='store_true')
    run(p.parse_args())
