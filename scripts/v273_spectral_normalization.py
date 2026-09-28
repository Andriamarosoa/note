"""One native intervention: local channel LayerNorm versus fixed division by 12."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts import train_v250_count_only as v250
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_window_experiment import require

ARMS=('channel_norm','fixed_scale')
SEED=16164
DROPOUT_SEED=46164
EPOCHS=12
CHECKPOINTS=(4,8,12)
NORMALIZER='v240_dense_channel_norm'
HISTORICAL_INITIAL_HASH='e3d36af85c33601c4e3183633e688f335d031e31006d6875857bb4cb22a05b5d'


def layer_hashes(model, *, omit_normalizer=False):
    result={}
    for layer in model.layers:
        if omit_normalizer and layer.name==NORMALIZER:
            continue
        h=hashlib.sha256()
        h.update(layer.__class__.__name__.encode())
        for a in layer.get_weights():
            h.update(str((a.shape,str(a.dtype))).encode())
            h.update(np.ascontiguousarray(a).tobytes())
        result[layer.name]=h.hexdigest()
    return result


def shared_hash(model):
    return hashlib.sha256(json.dumps(layer_hashes(model,omit_normalizer=True),
                                     sort_keys=True).encode()).hexdigest()


def build(arm):
    require(arm in ARMS,'unknown normalization arm')
    return v250.build_model('categorical',SEED,time_frames=31,
        spectral_normalization=arm,count_dropout_seed=DROPOUT_SEED)


def compile_metrics(model):
    import tensorflow as tf

    class SubsetAccuracy(tf.keras.metrics.Metric):
        def __init__(self,value,name):
            super().__init__(name=name)
            self.value=value
            self.good=self.add_weight(name='good',initializer='zeros')
            self.total=self.add_weight(name='total',initializer='zeros')

        def update_state(self,y_true,y_pred,sample_weight=None):
            # These are unweighted scores. The paired experiment uses unit
            # example weights; losses and metrics must not be conflated.
            truth=tf.cast(tf.reshape(y_true,[-1]),tf.int32)
            pred=tf.argmax(y_pred,axis=-1,output_type=tf.int32)
            keep=(truth>=2) if self.value is None else (truth==self.value)
            self.good.assign_add(tf.reduce_sum(tf.cast(keep&(pred==truth),self.dtype)))
            self.total.assign_add(tf.reduce_sum(tf.cast(keep,self.dtype)))

        def result(self):
            return tf.math.divide_no_nan(self.good,self.total)

        def reset_state(self):
            self.good.assign(0.)
            self.total.assign(0.)

    measures=[tf.keras.metrics.SparseCategoricalAccuracy(name='exact'),
              SubsetAccuracy(None,'poly_exact')]
    measures.extend(SubsetAccuracy(k,f'k{k}_exact') for k in range(7))
    model.compile(optimizer=model.optimizer,loss='sparse_categorical_crossentropy',metrics=measures)
    return measures


def interpretation(poly_delta,by_k_delta):
    require(set(by_k_delta)==set(map(str,range(7))),'incomplete per-K result')
    if poly_delta<=0:
        return 'no_polyphonic_validation_gain'
    if any(by_k_delta[str(k)]<0 for k in (1,2,3)):
        return 'polyphonic_gain_with_low_k_regressions'
    return 'polyphonic_gain_without_low_k_regression_on_this_split'


def preflight(output):
    import tensorflow as tf
    require(tf.__version__=='2.15.1','pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not output.exists(),'preflight output exists')
    project=Path(__file__).resolve().parents[1]
    launch_path=project/'analysis/v273-normalization-launch.json'
    launch=json.loads(launch_path.read_text())
    require(launch['arms']==list(ARMS) and launch['primary_epoch']==EPOCHS and
        launch['snapshots']==list(CHECKPOINTS) and launch['initialization_seed']==SEED and
        launch['dropout_seed']==DROPOUT_SEED and launch['weighting']=='uniform' and
        launch['historical_initial_hash']==HISTORICAL_INITIAL_HASH,'launch protocol differs')
    require(launch['config_sha256']==digest(project/'analysis/v273-native-paired-config.json'),
            'frozen partition configuration differs')
    old=v250.build_model('categorical',SEED,time_frames=31)
    require(weight_hash(old)==HISTORICAL_INITIAL_HASH,'historical default initialization changed')
    del old
    result=dict(status='running',tensorflow=tf.__version__,seed=SEED,dropout_seed=DROPOUT_SEED,
        historical_initialization_preserved=True,historical_initial_hash=HISTORICAL_INITIAL_HASH,
        launch_sha256=digest(launch_path),
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arms={})
    common=None
    first_mask=None
    for arm in ARMS:
        model=build(arm)
        before=layer_hashes(model,omit_normalizer=True)
        if common is None:
            common=before
        require(before==common,'shared initial layers differ')
        if arm=='channel_norm':
            require(weight_hash(model)==HISTORICAL_INITIAL_HASH,'control initialization changed')
        normalizer=model.get_layer(NORMALIZER)
        pair=tf.constant([[[[3.,1.,0.],[6.,2.,0.]]]],tf.float32)
        normalized=normalizer(pair).numpy()
        if arm=='channel_norm':
            require(isinstance(normalizer,tf.keras.layers.LayerNormalization) and
                    tuple(normalizer.axis) in ((-1,),(3,)) and
                    normalizer.epsilon==0.001,'wrong control normalization')
        else:
            require(isinstance(normalizer,tf.keras.layers.Rescaling),'wrong fixed normalization')
            np.testing.assert_allclose(normalized,pair.numpy()/12.,rtol=1e-6,atol=1e-7)
        drop=model.get_layer('v240_cardinality_dropout')
        require(drop.seed==DROPOUT_SEED and drop.rate==0.08,'dropout treatment differs')
        mask=drop(tf.ones((2,192)),training=True).numpy()
        if first_mask is None:
            first_mask=mask
        np.testing.assert_array_equal(mask,first_mask)
        shared=shared_hash(model)
        initial=weight_hash(model)
        measures=compile_metrics(model)
        truth=np.array([0,1,2,3],np.int32)
        probs=np.eye(7,dtype=np.float32)[[0,2,2,3]]
        for metric in measures:
            metric.update_state(truth[:,None],probs)
            expected=0.75 if metric.name=='exact' else (1. if metric.name in ('poly_exact','k0_exact','k2_exact','k3_exact') else 0.)
            np.testing.assert_allclose(metric.result().numpy(),expected,atol=1e-7)
            metric.reset_state()
        rng=np.random.default_rng(90273)
        shapes={tensor.name.split(':')[0]:tuple(tensor.shape.as_list()) for tensor in model.inputs}
        require(set(shapes)=={'candidate_set','candidate_mask','cluster_stats','spectral_map'},'wrong input graph')
        inputs={key:rng.uniform(0.,1.,size=(4,*shape[1:])).astype(np.float32)
                for key,shape in shapes.items()}
        inputs['candidate_mask'].fill(1.)
        log=model.train_on_batch(inputs,truth,return_dict=True)
        require(all(np.isfinite(v) for v in log.values()),'nonfinite synthetic training')
        require(int(model.optimizer.iterations.numpy())==1,'synthetic update missing')
        require(shared_hash(model)!=shared,'synthetic update did not reach shared layers')
        p=model(inputs,training=False).numpy()
        require(p.shape==(4,7) and np.isfinite(p).all(),'invalid native outputs')
        np.testing.assert_allclose(p.sum(1),1.,atol=1e-6)
        result['arms'][arm]=dict(common_initial_sha256=shared,full_initial_sha256=initial,
            parameters=model.count_params(),normalizer_class=normalizer.__class__.__name__,
            normalization_example=normalized.tolist(),synthetic_training_passed=True,
            per_k_metric_semantics_verified=True)
        del model
    require(result['arms']['channel_norm']['parameters']-
            result['arms']['fixed_scale']['parameters']==6,'unexpected parameter difference')
    result.update(status='passed',common_layers_identical=True,initial_dropout_mask_identical=True,
                  script_sha256=digest(__file__))
    output.mkdir(parents=True)
    write_json(output/'report.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    preflight(p.parse_args().output)
