"""Paired native inputs: local eligibility versus exact neighboring ownership."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

from scripts import train_v250_count_only as v250
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.v273_native_protocol import load_config
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_window_experiment import batch_inputs, epoch_order, array_hash, require
from scripts.v273_spectral_normalization import (
    SEED, DROPOUT_SEED, EPOCHS, CHECKPOINTS, NORMALIZER, HISTORICAL_INITIAL_HASH,
    compile_metrics, layer_hashes, shared_hash, interpretation)

ARMS = ('local_only', 'with_neighbors')


def sorted_proposals(groups):
    positions = np.concatenate(groups).astype(np.int64)
    ids = np.repeat(np.arange(len(groups)), [len(g) for g in groups])
    order = np.lexsort((ids, positions))
    positions, ids = positions[order], ids[order]
    keep = np.r_[True, np.diff(positions) != 0]
    return positions[keep], ids[keep]


def lookup(positions, ids, queries):
    """Nearest candidate with the historical lower-group-ID tie rule."""
    require(len(positions) > 0, 'empty proposal prefix')
    j = np.searchsorted(positions, queries)
    left, right = np.clip(j-1, 0, len(positions)-1), np.clip(j, 0, len(positions)-1)
    dl, dr = np.abs(queries-positions[left]), np.abs(queries-positions[right])
    use_left = (dl < dr) | ((dl == dr) & (ids[left] <= ids[right]))
    chosen = np.where(use_left, left, right)
    distance = np.minimum(dl, dr)
    return np.where(distance <= 882, ids[chosen], -1), distance


def frame_fraction(mask):
    require(np.asarray(mask).shape == (4096,), 'wrong acoustic support')
    sums = np.r_[0, np.cumsum(mask, dtype=np.int32)]
    starts = np.arange(31)*128
    return ((sums[starts+256]-sums[starts])/256.).astype(np.float32)


def geometry_for_track(groups):
    """Only integer proposal positions enter this function; no labels/audio."""
    require(all(len(g) and np.asarray(g).dtype.kind in 'iu' for g in groups), 'invalid proposals')
    require(all(g[0] >= 0 and np.all(np.diff(g) >= 0) and g[-1]-g[0] <= 1764 for g in groups),
            'invalid original full group')
    flat, owners = sorted_proposals(groups)
    geometry = np.empty((len(groups),31,2),np.float32)
    losses = np.empty(len(groups),np.int32)
    watermarks = np.empty(len(groups),np.int64)
    decision_after = np.empty(len(groups),np.int64)
    for row, current in enumerate(groups):
        start = int(current[0]); q = start-1308+np.arange(4096,dtype=np.int64)
        _, local_distance = lookup(current, np.full(len(current),row), q)
        local = (q >= 0) & (local_distance <= 882)
        full_owner, _ = lookup(flat, owners, q)
        owned = (q >= 0) & (full_owner == row)
        watermark = int(current[-1])+1764
        end = np.searchsorted(flat,watermark,side='right')
        prefix_owner, _ = lookup(flat[:end],owners[:end],q)
        np.testing.assert_array_equal(owned,(q >= 0)&(prefix_owner == row))
        require(not np.any(owned & ~local),'owner outside local eligibility')
        geometry[row,:,0] = frame_fraction(local)
        geometry[row,:,1] = frame_fraction(owned)
        losses[row] = int(np.sum(local & ~owned))
        watermarks[row] = watermark
        # Current V8.6 contexts need max(current)+1024. Candidate peaks and
        # merge representatives need 4 further scores plus the peak's +1.
        # This is a sample-support requirement, not measured wall-clock latency.
        decision_after[row] = max(start+2788, watermark+5, int(current[-1])+1024)
    return geometry, losses, watermarks, decision_after


def load_geometry(root, cache, manifest, config):
    validate_original_inventory(root)
    r=json.loads((root/'report.json').read_text())
    require(r['status']=='verified' and r['config_sha256']==digest(config),'wrong geometry protocol')
    folds=load_config(config)['member_folds']
    require(r['source_cache_sha256']=={m:h for m,h in manifest['source_cache_sha256'].items()
            if folds[m]!=3},'geometry source caches differ')
    require(r['bundle_fields']==manifest['fields'],'geometry/bundle mismatch')
    g=np.load(root/'geometry.npy',mmap_mode='r',allow_pickle=False)
    with np.load(root/'rows.npz',allow_pickle=False) as z:
        rows={key:z[key] for key in z.files}
    require(g.shape==(len(cache['members']),31,2),'wrong geometry shape')
    np.testing.assert_array_equal(rows['member'],cache['members'])
    np.testing.assert_array_equal(rows['start'],cache['cluster_start_samples'])
    require(array_hash(g)==r['geometry_sha256'],'geometry changed')
    inner=np.asarray([folds[str(m)]!=3 for m in rows['member']])
    require(int(inner.sum())==59309 and r['rows']==59309 and
            r['outer_tracks_processed']==0 and r['outer_rows_evaluated']==0,'wrong geometry scope')
    require(np.isfinite(g[inner]).all() and (g[inner]>=0).all() and (g[inner]<=1).all(),
            'invalid inner geometry')
    require(np.isnan(g[~inner]).all() and (rows['lost_eligible_samples'][~inner]==-1).all(),
            'outer geometry was generated')
    np.testing.assert_array_equal(rows['lost_eligible_samples'][inner]>0,
        np.any(g[inner,:,0]!=g[inner,:,1],axis=1))
    return g,rows,r


def build(arm):
    require(arm in ARMS,'unknown ownership arm')
    return v250.build_model('categorical',SEED,time_frames=31,
        count_dropout_seed=DROPOUT_SEED,ownership_context=True)


def batches(cache, indices, frames, *, geometry, arm, seed=0, shuffle=False,
            k=None, weights=None, batch_size=128):
    import tensorflow as tf
    base=np.asarray(indices,np.int64)
    require(arm in ARMS and frames==31,'wrong treatment')
    require(len(base)>0 and len(np.unique(base))==len(base),'bad indices')
    column=ARMS.index(arm)
    class Batches(tf.keras.utils.Sequence):
        def __init__(self):
            self.epoch=0
            self.order=epoch_order(base,seed,0) if shuffle else base.copy()
        def __len__(self): return (len(base)+batch_size-1)//batch_size
        def __getitem__(self,batch):
            ids=self.order[batch*batch_size:(batch+1)*batch_size]
            x=batch_inputs(cache,ids,frames)
            x['ownership_map']=np.asarray(geometry[ids,:,column:column+1],np.float32)
            require(np.isfinite(x['ownership_map']).all(),'outer/missing ownership row')
            return x if k is None else (x,k[ids],weights[k[ids]])
        def on_epoch_end(self):
            self.epoch+=1
            if shuffle:self.order=epoch_order(base,seed,self.epoch)
    return Batches()


def preflight(output):
    import tensorflow as tf
    require(tf.__version__=='2.15.1','pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not output.exists(),'preflight exists')
    launch=Path('analysis/v273-ownership-launch.json')
    protocol=json.loads(launch.read_text())
    require(protocol['arms']==list(ARMS) and protocol['epochs']==EPOCHS and
            protocol['primary_epoch']==EPOCHS and protocol['snapshots']==list(CHECKPOINTS) and
            protocol['weighting']=='uniform' and protocol['seed']==SEED and
            protocol['dropout_seed']==DROPOUT_SEED and
            protocol['historical_initial_hash']==HISTORICAL_INITIAL_HASH,'protocol changed')
    require(protocol['config_sha256']==digest('analysis/v273-native-paired-config.json'),
            'partition configuration changed')
    historical=v250.build_model('categorical',SEED,time_frames=31)
    require(weight_hash(historical)==HISTORICAL_INITIAL_HASH,'historical initialization changed')
    old_params=historical.count_params(); del historical
    report=dict(status='running',common_layers_identical=True,source_sha=subprocess.check_output(
        ['git','rev-parse','HEAD'],text=True).strip(),launch_sha256=digest(launch),arms={})
    common=None; common_predictions=None; first_mask=None
    for arm in ARMS:
        model=build(arm); initial=weight_hash(model); layers=layer_hashes(model)
        if common is None: common=layers
        require(layers==common,'paired initialization differs')
        require(model.count_params()-old_params==288,'unexpected model size change')
        shapes={t.name.split(':')[0]:tuple(t.shape.as_list()) for t in model.inputs}
        require(set(shapes)=={'candidate_set','candidate_mask','cluster_stats','spectral_map','ownership_map'},
                'ownership input pruned')
        rng=np.random.default_rng(90273)
        inputs={key:rng.uniform(0,1,(4,*shape[1:])).astype(np.float32) for key,shape in shapes.items()}
        inputs['candidate_mask'].fill(1.)
        prediction=model(inputs,training=False).numpy()
        if common_predictions is None:common_predictions=prediction
        np.testing.assert_array_equal(prediction,common_predictions)
        tensor={key:tf.convert_to_tensor(value) for key,value in inputs.items()}
        with tf.GradientTape() as tape:
            tape.watch(tensor['ownership_map'])
            loss=tf.reduce_sum(model(tensor,training=False)[:,2])
        gradient=tape.gradient(loss,tensor['ownership_map'])
        require(gradient is not None and np.isfinite(gradient.numpy()).all() and
                np.any(gradient.numpy()!=0),'ownership input not connected to output')
        normalizer=model.get_layer(NORMALIZER)
        require(isinstance(normalizer,tf.keras.layers.LayerNormalization) and
                normalizer.epsilon==0.001,'normalization treatment differs')
        drop=model.get_layer('v240_cardinality_dropout')
        require(drop.seed==DROPOUT_SEED and drop.rate==0.08,'dropout treatment differs')
        mask=drop(tf.ones((2,192)),training=True).numpy()
        if first_mask is None:first_mask=mask
        np.testing.assert_array_equal(mask,first_mask)
        shared=shared_hash(model); compile_metrics(model)
        log=model.train_on_batch(inputs,np.asarray([0,1,2,3]),return_dict=True)
        require(all(np.isfinite(v) for v in log.values()) and int(model.optimizer.iterations.numpy())==1,
                'synthetic training failed')
        require(weight_hash(model)!=initial,'weights did not learn')
        report['arms'][arm]=dict(common_initial_sha256=shared,full_initial_sha256=initial,
            parameters=model.count_params(),normalizer_class=model.get_layer(NORMALIZER).__class__.__name__,
            geometry_gradient_nonzero=True,synthetic_training_passed=True)
    require(report['arms'][ARMS[0]]==report['arms'][ARMS[1]],'preflight branches differ')
    report.update(status='passed',historical_initialization_preserved=True,initial_dropout_mask_identical=True,
        same_input_same_prediction=True,identical_architecture_and_weights=True)
    output.mkdir(parents=True);write_json(output/'report.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    preflight(p.parse_args().output)
