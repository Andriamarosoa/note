"""Train the native seven-class count network, 23 vs 31 frames, fold 3 only.

This isolates the acoustic window in the V26 count component used by V27.3.
It does not reproduce the complete V27.3 fusion/transition pipeline.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, batch_inputs, epoch_order, array_hash, require, FRAMES, WEIGHTINGS


def weight_hash(model):
    h = hashlib.sha256()
    for a in model.get_weights():
        h.update(str((a.shape, str(a.dtype))).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def batches(cache, indices, frames, *, seed=0, shuffle=False, k=None, weights=None, batch_size=128):
    import tensorflow as tf
    base = np.asarray(indices, np.int64)
    require(len(base) > 0 and len(np.unique(base)) == len(base), 'empty/duplicate batch population')

    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(base, seed, 0) if shuffle else base.copy()

        def __len__(self):
            return (len(base) + batch_size - 1) // batch_size

        def __getitem__(self, batch):
            ids = self.order[batch*batch_size:(batch+1)*batch_size]
            x = batch_inputs(cache, ids, frames)
            if k is None:
                return x
            target = k[ids]
            return x, target, weights[target]

        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(base, seed, self.epoch)

    return Batches()


def train(args):
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(), 'refusing to overwrite training')
    cfg = load_config(args.config)
    budget = cfg['folds'][3]['epochs']['v260_'+args.weighting]
    require(budget == 8, 'frozen eight-epoch budget changed')
    cache, parts, manifest = load_bundle(args.bundle, args.config)
    k = np.minimum(cache['exact'].astype(np.int32), 6)
    args.output.mkdir(parents=True)
    protocol = dict(experiment='native_count_window_23_vs_31', frames=args.frames, weighting=args.weighting,
        outer_fold=3, inner_validation_fold=0, epochs_per_phase=budget, batch_size=128,
        bundle_sha256=digest(args.bundle/'bundle.json'), config_sha256=digest(args.config),
        source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        decoder='argmax P(K=0..6)', output_corrector=False, historical_v273_reproduced=False,
        epoch_selection='fixed historical budget; final epoch only; no outer selection',
        partitions=manifest['partitions'], tensorflow=tf.__version__, numpy=np.__version__)
    write_json(args.output/'protocol.json', protocol)
    completed = {}
    for phase, fit_name, predict_name, seed in (
        ('inner', 'inner_fit', 'inner_val', v260.SEED+103),
        ('final', 'final_fit', 'outer', v260.SEED+1003)):
        fit, predict = parts[fit_name], parts[predict_name]
        require(not np.intersect1d(fit, predict).size, 'fit/predict overlap')
        weights = v260.arm_weights(args.weighting, k[fit])
        root = args.output/phase
        root.mkdir()
        model = v260.build_model(args.weighting, seed, time_frames=args.frames)
        order = hashlib.sha256()
        for epoch in range(budget):
            order.update(epoch_order(fit, seed, epoch).tobytes())
        record = dict(seed=seed, epochs=budget, fit_rows=len(fit), predict_rows=len(predict),
            fit_indices_sha256=array_hash(fit), predict_indices_sha256=array_hash(predict),
            class_weights=weights.tolist(), initial_weights_sha256=weight_hash(model),
            epoch_order_sha256=order.hexdigest(), batch_size=128, final_checkpoint_selected=True)
        # Commit the configuration before model.fit or held-out prediction.
        write_json(root/'frozen-training.json', record)
        history = []
        observed_orders = []

        class PreserveEpoch(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                expected = epoch_order(fit, seed, epoch)
                np.testing.assert_array_equal(train_seq.order, expected)
                observed_orders.append(array_hash(train_seq.order))

            def on_epoch_end(self, epoch, logs=None):
                entry = dict(epoch=int(epoch)+1, **{key:float(value) for key,value in (logs or {}).items()})
                require(all(np.isfinite(v) for v in entry.values()), 'nonfinite epoch')
                history.append(entry)
                self.model.save_weights(root/'latest.weights.h5')
                write_json(root/'progress.json', dict(status='training', completed_epochs=len(history), expected_epochs=budget, history=history))

        train_seq = batches(cache, fit, args.frames, seed=seed, shuffle=True, k=k, weights=weights)
        kwargs = {}
        if phase == 'inner':
            kwargs['validation_data'] = batches(cache, predict, args.frames, k=k, weights=weights)
        print(json.dumps(dict(phase=phase, frames=args.frames, weighting=args.weighting, **record)), flush=True)
        result = model.fit(train_seq, epochs=budget, shuffle=False, workers=0, max_queue_size=1,
            verbose=2, callbacks=[tf.keras.callbacks.TerminateOnNaN(), PreserveEpoch()], **kwargs)
        require(len(history) == budget and all(np.isfinite(v).all() for v in result.history.values()), 'incomplete training')
        weight_file = root/'latest.weights.h5'
        # Save completion and weight identity before reading held-out predictions.
        record.update(status='trained', weights_sha256=digest(weight_file), history=history,
                      observed_epoch_order_sha256=observed_orders)
        write_json(root/'training.json', record)
        frozen_hash = digest(root/'training.json')
        probability = np.asarray(model.predict(batches(cache, predict, args.frames),
                                    workers=0, max_queue_size=1, verbose=0), np.float32)
        require(probability.shape == (len(predict), 7) and np.isfinite(probability).all() and
                np.all(probability >= 0) and np.allclose(probability.sum(1), 1., atol=1e-5), 'invalid output probabilities')
        predicted = probability.argmax(1).astype(np.int32)
        np.savez_compressed(root/'predictions.npz', global_index=predict, member=cache['members'][predict],
            cluster_start_samples=cache['cluster_start_samples'][predict], k=k[predict], probability=probability, predicted=predicted)
        metrics = v260.cardinality(k[predict], predicted)
        write_json(root/'evaluation.json', dict(metrics=metrics, predictions_sha256=digest(root/'predictions.npz'),
                    frozen_training_sha256=frozen_hash, evaluated_after_training_complete=True))
        completed[phase] = dict(training=record, metrics=metrics)
        write_json(root/'progress.json', dict(status='completed', completed_epochs=budget, history=history))
        del model, train_seq, kwargs, result
        tf.keras.backend.clear_session()
        gc.collect()
    write_json(args.output/'report.json', dict(status='completed', protocol=protocol, phases=completed,
                                              automatic_promotion=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'config', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--frames', type=int, choices=FRAMES, required=True)
    p.add_argument('--weighting', choices=WEIGHTINGS, required=True)
    train(p.parse_args())
