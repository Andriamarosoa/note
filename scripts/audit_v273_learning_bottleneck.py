"""Diagnose frozen count models on fit data and their original held-out split.

No training, threshold selection, label replacement or outer-fold expansion.
Truth-conditioned rankings are diagnostic oracles, never deployable scores.
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_v273_window_pair import batches, weight_hash
from scripts.v273_window_experiment import array_hash, load_bundle, require


def metrics(k, probability, class_weights):
    k = np.asarray(k, np.int32)
    p = np.asarray(probability, np.float64)
    require(p.shape == (len(k), 7) and len(k) > 0, 'invalid probability shape')
    require(np.isfinite(p).all() and (p >= 0).all(), 'invalid probabilities')
    np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
    pred = p.argmax(1)
    poly = k >= 2
    loss = -np.log(np.maximum(p[np.arange(len(k)), k], 1e-12))
    weighted_loss = loss * np.asarray(class_weights)[k]
    top2 = np.any(np.argsort(-p, axis=1, kind='stable')[:, :2] == k[:, None], axis=1)
    conditional = p[:, 2:].argmax(1) + 2
    by_k = {}
    for value in range(7):
        same = k == value
        n = int(same.sum())
        by_k[str(value)] = dict(rows=n, correct=int((same & (pred == k)).sum()),
            under=int((same & (pred < k)).sum()), over=int((same & (pred > k)).sum()),
            exact=float(np.mean(pred[same] == k[same])) if n else None,
            nll=float(loss[same].mean()) if n else None,
            weighted_loss_total=float(weighted_loss[same].sum()),
            sample_weight_total=float(n * class_weights[value]),
            mean_true_class_probability=float(p[same, value].mean()) if n else None,
            predicted_as_class=int((pred == value).sum()))
    return dict(rows=len(k), correct=int((pred == k).sum()), exact=float(np.mean(pred == k)),
        under=int((pred < k).sum()), over=int((pred > k).sum()),
        poly_rows=int(poly.sum()), poly_correct=int((poly & (pred == k)).sum()),
        poly_exact=float(np.mean(pred[poly] == k[poly])) if poly.any() else None,
        poly_under=int((poly & (pred < k)).sum()), poly_over=int((poly & (pred > k)).sum()),
        poly_predicted_nonpoly=int((poly & (pred < 2)).sum()),
        poly_correct_if_true_poly_known=int((poly & (conditional == k)).sum()),
        poly_top2_correct=int((poly & top2).sum()),
        nll=float(loss.mean()), weighted_nll=float(weighted_loss.mean()),
        confusion_true_by_predicted=np.bincount(k * 7 + pred, minlength=49).reshape(7, 7).tolist(),
        by_true_k=by_k, oracle_uses_truth=True, oracle_not_a_valid_model_score=True)


def predict(model, cache, ids, output, phase, split):
    chunks = []
    for begin in range(0, len(ids), 4096):
        part = ids[begin:begin + 4096]
        p = np.asarray(model.predict(batches(cache, part, 31), workers=0,
                                     max_queue_size=1, verbose=0), np.float32)
        chunks.append(p)
        write_json(output / 'progress.json', dict(phase=phase, split=split,
            completed_rows=min(begin + 4096, len(ids)), expected_rows=len(ids)))
        print(json.dumps(dict(phase=phase, split=split,
            completed_rows=min(begin + 4096, len(ids)), expected_rows=len(ids))), flush=True)
    return np.concatenate(chunks)


def run(args):
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(), 'refusing to overwrite an audit')
    validate_original_inventory(args.model)
    validate_original_inventory(args.bundle)
    protocol = json.loads((args.model / 'protocol.json').read_text())
    require(protocol['frames'] == 31 and protocol['weighting'] == args.weighting and
            protocol['outer_fold'] == 3, 'wrong model scope')
    cache, parts, bundle = load_bundle(args.bundle, args.config)
    require(digest(args.bundle / 'bundle.json') == protocol['bundle_sha256'], 'wrong training inputs')
    k = np.minimum(np.asarray(cache['exact'], np.int32), 6)
    args.output.mkdir(parents=True)
    result = dict(status='running', weighting=args.weighting, frames=31, outer_fold=3,
        tensorflow=tf.__version__, source_training_run=36351028493,
        protocol_sha256=digest(args.model / 'protocol.json'),
        bundle_sha256=digest(args.bundle / 'bundle.json'), script_sha256=digest(__file__),
        training_performed=False, output_corrector=False, model_promoted=False,
        only_original_fit_and_heldout_partitions=True, phases={})
    for phase, fit_name, held_name in (('inner', 'inner_fit', 'inner_val'),
                                      ('final', 'final_fit', 'outer')):
        root = args.model / phase
        record = json.loads((root / 'training.json').read_text())
        require(record['status'] == 'trained' and record['epochs'] == 8, 'incomplete frozen training')
        require(array_hash(parts[fit_name]) == record['fit_indices_sha256'], 'fit identity changed')
        require(array_hash(parts[held_name]) == record['predict_indices_sha256'], 'held-out identity changed')
        require(digest(root / 'latest.weights.h5') == record['weights_sha256'], 'weights checksum differs')
        table = v260.arm_weights(args.weighting, k[parts[fit_name]])
        np.testing.assert_array_equal(table, np.asarray(record['class_weights'], np.float32))
        model = v260.build_model(args.weighting, record['seed'], time_frames=31)
        model.load_weights(root / 'latest.weights.h5')
        before = weight_hash(model)
        phase_result = dict(frozen_training=record, weights_sha256=record['weights_sha256'], splits={})
        for split, partition in (('fit', fit_name), ('heldout', held_name)):
            ids = parts[partition]
            probability = predict(model, cache, ids, args.output, phase, split)
            phase_result['splits'][split] = metrics(k[ids], probability, table)
            if split == 'heldout':
                with np.load(root / 'predictions.npz', allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['global_index'], ids)
                    np.testing.assert_array_equal(z['k'], k[ids])
                    np.testing.assert_array_equal(z['member'], cache['members'][ids])
                    np.testing.assert_allclose(probability, z['probability'], rtol=2e-4, atol=2e-5)
                    np.testing.assert_array_equal(probability.argmax(1), z['predicted'])
                    phase_result['heldout_replay_max_abs_difference'] = float(np.max(np.abs(probability-z['probability'])))
                if phase == 'inner':
                    np.testing.assert_allclose(phase_result['splits']['heldout']['weighted_nll'],
                        record['history'][-1]['val_loss'], rtol=2e-5, atol=2e-5)
            np.savez_compressed(args.output / f'{phase}-{split}.npz', global_index=ids,
                member=cache['members'][ids], k=k[ids], probability=probability,
                predicted=probability.argmax(1).astype(np.int32))
        require(weight_hash(model) == before, 'inference mutated model weights')
        phase_result['weights_unchanged'] = True
        result['phases'][phase] = phase_result
        write_json(args.output / 'report.json', result)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
    result.update(status='completed', interpretation_limits=[
        'Fit-set inference measures learning of seen rows; it is not generalization.',
        'Training-loop loss includes dropout and changing weights; compare frozen inference losses instead.',
        'Truth-conditioned rankings diagnose class competition and cannot be deployed as reported scores.',
        'A fit/heldout gap locates generalization loss but does not distinguish every physical cause.',
        'This count component is not the full V27.3 pipeline; no official score replacement.'])
    write_json(args.output / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'bundle', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--weighting', choices=('uniform', 'weighted'), required=True)
    run(parser.parse_args())
