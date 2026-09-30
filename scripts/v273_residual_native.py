"""Fixed paired native count model and batches for the residual-map experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v250_count_only import build_model
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_spectral_normalization import (
    SEED, DROPOUT_SEED, EPOCHS, CHECKPOINTS, HISTORICAL_INITIAL_HASH,
    compile_metrics, layer_hashes)
from scripts.v273_window_experiment import epoch_order, require

ARMS = ('observed_only', 'with_residual')
LAUNCH = Path('analysis/v273-residual-native-launch.json')
PROTOCOL = Path('analysis/v273-residual-native-protocol.md')


def experiment_identity():
    cfg = json.loads(LAUNCH.read_text())
    require(cfg['arms'] == list(ARMS) and cfg['seed'] == SEED and
            cfg['dropout_seed'] == DROPOUT_SEED and cfg['epochs'] == EPOCHS and
            cfg['primary_epoch'] == 12 and cfg['snapshots'] == list(CHECKPOINTS),
            'frozen training settings changed')
    require(cfg['config_sha256'] == digest('analysis/v273-native-paired-config.json'),
            'partition changed')
    require(cfg['feature_builder_sha256'] == digest('scripts/v273_residual_map.py') and
            cfg['stable_estimator_sha256'] == digest('scripts/v273_stable_prediction.py'),
            'audited feature builder changed')
    return dict(source_sha=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        launch_sha256=digest(LAUNCH), protocol_sha256=digest(PROTOCOL),
        config_sha256=cfg['config_sha256'], feature_builder_sha256=cfg['feature_builder_sha256'],
        stable_estimator_sha256=cfg['stable_estimator_sha256'])


def treatment_maps(features, arm):
    require(arm in ARMS, 'unknown arm')
    x = np.asarray(features, np.float32)
    require(x.ndim == 4 and x.shape[1:] == (31,64,4) and np.isfinite(x).all()
            and (x >= 0).all(), 'invalid or outer feature maps')
    return x[:, :, :, [0,0,2,2]] if arm == 'observed_only' else x.copy()


def build():
    return build_model('categorical', SEED, time_frames=31,
        spectral_normalization='fixed_scale', count_dropout_seed=DROPOUT_SEED,
        ownership_context=True, spectral_channels=4)


def inputs(cache, maps, geometry, ids, arm):
    owned = np.asarray(geometry[ids, :, 1:2], np.float32)
    require(np.isfinite(owned).all(), 'outer/missing geometry')
    return dict(candidate_set=np.asarray(cache['sequence'][ids], np.float32),
        candidate_mask=np.asarray(cache['mask'][ids], np.float32),
        cluster_stats=np.asarray(cache['stats'][ids], np.float32),
        spectral_map=treatment_maps(maps[ids], arm), ownership_map=owned)


def batches(cache, maps, geometry, ids, arm, *, labels=None, shuffle=False, batch_size=128):
    import tensorflow as tf
    base = np.asarray(ids, np.int64)
    require(len(base) and len(np.unique(base)) == len(base), 'invalid indices')
    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch = 0
            self.order = epoch_order(base, SEED, 0) if shuffle else base.copy()
        def __len__(self):
            return (len(base)+batch_size-1)//batch_size
        def __getitem__(self, index):
            selected = self.order[index*batch_size:(index+1)*batch_size]
            x = inputs(cache, maps, geometry, selected, arm)
            return x if labels is None else (x, labels[selected])
        def on_epoch_end(self):
            self.epoch += 1
            if shuffle:
                self.order = epoch_order(base, SEED, self.epoch)
    return Batches()


def preflight(output):
    import tensorflow as tf
    require(tf.__version__ == '2.15.1', 'pinned TensorFlow required')
    require(not output.exists(), 'preflight output exists')
    tf.config.experimental.enable_op_determinism()
    identity = experiment_identity()
    historical = build_model('categorical', SEED, time_frames=31)
    require(weight_hash(historical) == HISTORICAL_INITIAL_HASH, 'historical default changed')
    del historical
    previous = previous_probability = previous_dropout = None
    arms = {}
    for arm in ARMS:
        model = build(); initial = weight_hash(model)
        if previous is None:
            previous = initial
        require(initial == previous, 'paired initialization differs')
        shapes = {t.name.split(':')[0]:tuple(t.shape.as_list()) for t in model.inputs}
        require(shapes['spectral_map'][1:] == (31,64,4), 'four channels not connected')
        rng = np.random.default_rng(90273)
        x = {key:rng.uniform(0,1,(4,*shape[1:])).astype(np.float32) for key,shape in shapes.items()}
        x['candidate_mask'].fill(1)
        p = model(x, training=False).numpy()
        if previous_probability is None:
            previous_probability = p
        np.testing.assert_array_equal(p, previous_probability)
        normalizer = model.get_layer('v240_dense_channel_norm')
        require(isinstance(normalizer, tf.keras.layers.Rescaling), 'wrong normalization')
        np.testing.assert_allclose(normalizer(x['spectral_map']), x['spectral_map']/12., atol=1e-7)
        tensors = {key:tf.convert_to_tensor(value) for key,value in x.items()}
        with tf.GradientTape() as tape:
            tape.watch([tensors['spectral_map'],tensors['ownership_map']])
            loss = tf.reduce_sum(model(tensors, training=False)[:,2])
        gradients = tape.gradient(loss, [tensors['spectral_map'],tensors['ownership_map']])
        require(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients),
                'input gradient missing')
        require(np.any(gradients[0].numpy()[:,:,:,[1,3]] != 0) and
                np.any(gradients[1].numpy() != 0), 'residual or geometry ignored by graph')
        dropout = {name:model.get_layer(name)(tf.ones((2,192)),training=True).numpy()
                   for name in ('cluster_dropout','v240_cardinality_dropout')}
        if previous_dropout is None:
            previous_dropout = dropout
        for name in dropout:
            np.testing.assert_array_equal(dropout[name], previous_dropout[name])
        compile_metrics(model)
        require(int(model.optimizer.iterations.numpy()) == 0, 'optimizer not fresh')
        log = model.train_on_batch(x, np.array([0,1,2,3]), return_dict=True)
        require(all(np.isfinite(v) for v in log.values()) and weight_hash(model) != initial,
                'synthetic update failed')
        arms[arm] = dict(initial_sha256=initial, parameters=model.count_params(),
            residual_gradient_nonzero=True, ownership_gradient_nonzero=True,
            synthetic_update_passed=True, all_initial_dropout_masks_identical=True)
    require(arms[ARMS[0]] == arms[ARMS[1]], 'unequal preflight results')
    # Descriptive benchmark only: does not choose architecture or training budget.
    sample = {key:np.tile(value,(32,*([1]*(value.ndim-1)))) for key,value in x.items()}
    y = np.tile(np.arange(4),32)
    model.train_on_batch(sample,y)
    started = time.monotonic()
    model.train_on_batch(sample,y)
    batch_seconds = time.monotonic()-started
    output.mkdir(parents=True)
    report = dict(status='passed', **identity, tensorflow=tf.__version__, numpy=np.__version__,
        arms=arms, same_input_same_prediction=True, historical_default_unchanged=True,
        epoch_budget=EPOCHS, checkpoints=list(CHECKPOINTS), benchmark_batch128_seconds=batch_seconds,
        script_sha256=digest(__file__), training_performed=False, output_corrector=False)
    write_json(output/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    preflight(p.parse_args().output)
