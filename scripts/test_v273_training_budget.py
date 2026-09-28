"""Resume the exact inner epoch-8 checkpoint and Adam state through epoch 16.

Only the original inner training and validation partitions are evaluated.
The dropout stream restarts with a declared seed; this is audited resumption,
not a claim of bitwise identity to an uninterrupted 16-epoch run.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.audit_v273_learning_bottleneck import metrics
from scripts.rebuild_v273_sources import digest, write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.train_v273_window_pair import batches, weight_hash
from scripts.v273_window_experiment import array_hash, epoch_order, load_bundle, require


def optimizer_variables(model):
    values = model.optimizer.variables
    return list(values() if callable(values) else values)


def optimizer_hash(model):
    h=hashlib.sha256()
    for v in optimizer_variables(model):
        a=v.numpy()
        h.update(str((a.shape,str(a.dtype))).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def validate_optimizer_snapshot(values, expected_steps):
    require(len(values)>1, 'checkpoint has no complete Adam state')
    require(np.asarray(values[0]).shape==() and int(values[0])==expected_steps,
            'checkpoint optimizer iteration differs')
    require(all(np.isfinite(v).all() for v in values), 'nonfinite Adam state')


def interpret_budget(fit_delta, validation_delta):
    if fit_delta>0 and validation_delta>0:
        return 'more_training_improves_seen_and_internal_validation'
    if fit_delta>0 and validation_delta<=0:
        return 'better_fit_does_not_improve_internal_validation'
    if validation_delta>0:
        return 'internal_gain_without_better_aggregate_fit'
    return 'no_internal_gain_demonstrated'


def run(args):
    import h5py
    import tensorflow as tf
    require(tf.__version__=='2.15.1', 'pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(), 'refusing to overwrite results')
    for root in (args.bundle,args.model,args.baseline):
        validate_original_inventory(root)
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    k=np.minimum(np.asarray(cache['exact'],np.int32),6)
    fit,val=parts['inner_fit'],parts['inner_val']
    require(not np.intersect1d(np.r_[fit,val],parts['outer']).size, 'outer rows entered experiment')
    protocol=json.loads((args.model/'protocol.json').read_text())
    record=json.loads((args.model/'inner/training.json').read_text())
    baseline=json.loads((args.baseline/'report.json').read_text())
    require(protocol['frames']==31 and protocol['weighting']==args.weighting and
            protocol['outer_fold']==3 and record['epochs']==8, 'wrong frozen model')
    require(protocol['bundle_sha256']==digest(args.bundle/'bundle.json'), 'bundle identity differs')
    require(array_hash(fit)==record['fit_indices_sha256'] and
            array_hash(val)==record['predict_indices_sha256'], 'partitions differ')
    weight_path=args.model/'inner/latest.weights.h5'
    require(digest(weight_path)==record['weights_sha256']==
            baseline['phases']['inner']['weights_sha256'], 'checkpoint identity differs')
    table=v260.arm_weights(args.weighting,k[fit])
    np.testing.assert_array_equal(table,np.asarray(record['class_weights'],np.float32))
    batch_steps=(len(fit)+127)//128
    expected_steps=8*batch_steps
    with h5py.File(weight_path,'r') as h:
        require('optimizer/vars' in h, 'saved optimizer group absent')
        group=h['optimizer/vars']
        stored=[np.asarray(group[str(i)]) for i in range(len(group))]
    validate_optimizer_snapshot(stored,expected_steps)
    model=v260.build_model(args.weighting,record['seed'],time_frames=31)
    model.optimizer.build(model.trainable_variables)
    model.load_weights(weight_path)
    restored=optimizer_variables(model)
    require(len(restored)==len(stored), 'optimizer variable count differs')
    for current,saved in zip(restored,stored):
        np.testing.assert_array_equal(current.numpy(),saved)
    model_hash,opt_hash=weight_hash(model),optimizer_hash(model)
    args.output.mkdir(parents=True)
    result=dict(status='baseline_verification',weighting=args.weighting,frames=31,
        tensorflow=tf.__version__,numpy=np.__version__,
        outer_fold=3,outer_rows_evaluated=0,initial_epoch=8,final_epoch=16,
        original_seed=record['seed'],resume_dropout_seed=record['seed']+50000,
        dropout_stream='restarted once with declared seed; not uninterrupted replay',
        optimizer_state_restored_exactly=True,optimizer_variables=len(stored),
        original_optimizer_iterations=expected_steps,batches_per_epoch=batch_steps,
        original_weights_sha256=record['weights_sha256'],source_training_run=36351028493,
        source_baseline_audit_run=36375624830,fit_indices_sha256=array_hash(fit),
        val_indices_sha256=array_hash(val),bundle_sha256=protocol['bundle_sha256'],
        config_sha256=digest(args.config),script_sha256=digest(__file__),
        class_weights=table.tolist(),checkpoints={},history=[],epoch_orders=[],
        training_performed=True,output_corrector=False,automatic_promotion=False,
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())

    def prediction(ids):
        return np.asarray(model.predict(batches(cache,ids,31),workers=0,
                              max_queue_size=1,verbose=0),np.float32)

    def save_predictions(epoch,split,ids,p):
        np.savez_compressed(args.output/f'epoch-{epoch:02d}-{split}.npz',global_index=ids,
            member=cache['members'][ids],k=k[ids],probability=p,predicted=p.argmax(1))
        return metrics(k[ids],p,table)

    # Baseline fit predictions were independently replayed in the pinned audit.
    # The complete validation prediction is replayed again after restoring Adam.
    baseline_splits={}
    for split,ids,saved_split in (('fit',fit,'fit'),('validation',val,'heldout')):
        with np.load(args.baseline/f'inner-{saved_split}.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['global_index'],ids)
            np.testing.assert_array_equal(z['k'],k[ids])
            np.testing.assert_array_equal(z['member'],cache['members'][ids])
            saved=np.asarray(z['probability'])
        p=prediction(ids) if split=='validation' else saved
        if split=='validation':
            np.testing.assert_allclose(p,saved,rtol=2e-4,atol=2e-5)
            np.testing.assert_array_equal(p.argmax(1),saved.argmax(1))
            result['baseline_validation_max_probability_difference']=float(np.max(np.abs(p-saved)))
        baseline_splits[split]=save_predictions(8,split,ids,p)
    require(weight_hash(model)==model_hash and optimizer_hash(model)==opt_hash,
            'baseline inference mutated training state')
    shutil.copyfile(weight_path,args.output/'epoch-08.weights.h5')
    result['checkpoints']['8']=dict(optimizer_iterations=expected_steps,splits=baseline_splits,
        weights_sha256=digest(args.output/'epoch-08.weights.h5'))
    result['status']='training'
    write_json(args.output/'report.json',result)
    tf.keras.utils.set_random_seed(result['resume_dropout_seed'])
    train=batches(cache,fit,31,seed=record['seed'],shuffle=True,k=k,weights=table)
    train.epoch=8
    train.order=epoch_order(fit,record['seed'],8)

    class Preserve(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            np.testing.assert_array_equal(train.order,epoch_order(fit,record['seed'],epoch))
            result['epoch_orders'].append(dict(epoch=epoch+1,sha256=array_hash(train.order)))

        def on_epoch_end(self,epoch,logs=None):
            completed=epoch+1
            require(int(model.optimizer.iterations.numpy())==completed*batch_steps,
                    'wrong number of gradient updates')
            entry=dict(epoch=completed,**{key:float(v) for key,v in (logs or {}).items()})
            require(all(np.isfinite(v) for v in entry.values()),'nonfinite training')
            result['history'].append(entry)
            model.save_weights(args.output/'latest.weights.h5')
            if completed in (12,16):
                cp=args.output/f'epoch-{completed:02d}.weights.h5'
                shutil.copyfile(args.output/'latest.weights.h5',cp)
                state=(weight_hash(model),optimizer_hash(model))
                scores={split:save_predictions(completed,split,ids,prediction(ids))
                        for split,ids in (('fit',fit),('validation',val))}
                require(state==(weight_hash(model),optimizer_hash(model)), 'evaluation mutated checkpoint')
                result['checkpoints'][str(completed)]=dict(splits=scores,
                    optimizer_iterations=int(model.optimizer.iterations.numpy()),weights_sha256=digest(cp))
            write_json(args.output/'progress.json',dict(completed_epoch=completed,expected_epoch=16,
                weighting=args.weighting,last_loss=entry['loss'],last_val_loss=entry['val_loss']))
            write_json(args.output/'report.json',result)
            print(json.dumps(dict(completed_epoch=completed,weighting=args.weighting,
                snapshots=list(result['checkpoints']))),flush=True)
            if completed==12 and args.release_tag:
                # Save a usable intermediate package while later epochs continue.
                subprocess.run(['python','-B','scripts/archive_v273_native_stage.py',
                    '--root',str(args.output),'--name',f'budget-{args.weighting}-epoch12',
                    '--tag',args.release_tag],check=True)

    model.fit(train,initial_epoch=8,epochs=16,shuffle=False,workers=0,max_queue_size=1,
        validation_data=batches(cache,val,31,k=k,weights=table),verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),Preserve()])
    require([r['epoch'] for r in result['history']]==list(range(9,17)), 'incomplete continuation')
    require(set(result['checkpoints'])=={'8','12','16'}, 'missing endpoints')
    result['comparisons']={}
    for epoch in ('12','16'):
        a,b=result['checkpoints']['8']['splits'],result['checkpoints'][epoch]['splits']
        delta={split:b[split]['poly_correct']-a[split]['poly_correct'] for split in ('fit','validation')}
        result['comparisons'][epoch]=dict(poly_correct_delta=delta,
            interpretation=interpret_budget(delta['fit'],delta['validation']))
    result['status']='completed'
    result['limitations']=['One resumption seed and one internal validation split.',
        'More updates include a declared restarted dropout stream; uninterrupted 16-epoch identity is not claimed.',
        'No outer rows evaluated; no corrected output and no model promotion.',
        'Internal improvement does not establish generalization to the outer fold.']
    write_json(args.output/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','model','baseline','config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--weighting',choices=('uniform','weighted'),required=True)
    p.add_argument('--release-tag')
    run(p.parse_args())
