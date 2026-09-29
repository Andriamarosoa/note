"""Fresh paired ownership-context training on the original inner split of fold 3."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from scripts.audit_v273_learning_bottleneck import metrics
from scripts.rebuild_v273_sources import digest,write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.test_v273_training_budget import optimizer_hash
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_window_experiment import array_hash,epoch_order,load_bundle,require
from scripts.v273_ownership_experiment import (
    ARMS,SEED,DROPOUT_SEED,EPOCHS,CHECKPOINTS,NORMALIZER,
    build,batches,load_geometry,compile_metrics,layer_hashes,shared_hash)


def train(args):
    import tensorflow as tf
    require(tf.__version__=='2.15.1','pinned TensorFlow required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),'refusing to overwrite training')
    validate_original_inventory(args.bundle)
    validate_original_inventory(args.preflight)
    gate=json.loads((args.preflight/'report.json').read_text())
    require(gate['status']=='passed' and gate['common_layers_identical'],'preflight did not pass')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(gate['source_sha']==source,'preflight source differs')
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    geometry,geometry_rows,geometry_report=load_geometry(args.geometry,cache,manifest,args.config)
    require(geometry_report['source_sha']==source and
            geometry_report['launch_sha256']==gate['launch_sha256'],'geometry producer differs')
    fit,val=parts['inner_fit'],parts['inner_val']
    k=np.minimum(np.asarray(cache['exact'],np.int32),6)
    require(len(fit)==43357 and len(val)==15952,'wrong inner populations')
    require(not np.intersect1d(np.r_[fit,val],parts['outer']).size,'outer rows entered experiment')
    for begin in range(0,len(fit),512):
        x=cache['spectral'][fit[begin:begin+512]]
        require(np.isfinite(x).all() and (x>=0).all() and (x<=12).all(),'spectral scaling bounds differ')
    model=build(args.arm)
    initial=weight_hash(model)
    shared=shared_hash(model)
    require(shared==gate['arms'][args.arm]['common_initial_sha256'] and
            initial==gate['arms'][args.arm]['full_initial_sha256'],'model initialization differs from preflight')
    compile_metrics(model)
    require(int(model.optimizer.iterations.numpy())==0,'optimizer not fresh')
    args.output.mkdir(parents=True)
    model.save_weights(args.output/'initial.weights.h5')
    result=dict(status='training',experiment='native_ownership_context',arm=args.arm,
        seed=SEED,dropout_seed=DROPOUT_SEED,epochs=EPOCHS,primary_epoch=12,batch_size=128,
        weighting='uniform',class_weights=[1.]*7,learning_rate=0.0002,frames=31,
        outer_fold=3,inner_validation_fold=0,outer_rows_evaluated=0,
        config_sha256=digest(args.config),bundle_sha256=digest(args.bundle/'bundle.json'),
        preflight_sha256=digest(args.preflight/'report.json'),source_sha=source,
        launch_sha256=gate['launch_sha256'],
        script_sha256=digest(__file__),tensorflow=tf.__version__,numpy=np.__version__,
        geometry_sha256=geometry_report['geometry_sha256'],geometry_report_sha256=digest(args.geometry/'report.json'),
        geometry_column=ARMS.index(args.arm),same_sample_support_for_both_arms=True,
        common_initial_sha256=shared,full_initial_sha256=initial,
        fit_indices_sha256=array_hash(fit),val_indices_sha256=array_hash(val),
        parameters=model.count_params(),normalizer_class=model.get_layer(NORMALIZER).__class__.__name__,
        output_corrector=False,automatic_promotion=False,checkpoint_selection='fixed epoch 12',
        historical_resume=False,training_performed=True,checkpoints={},history=[],epoch_orders=[])
    write_json(args.output/'report.json',result)
    table=np.ones(7,np.float32)
    sequence=batches(cache,fit,31,geometry=geometry,arm=args.arm,seed=SEED,shuffle=True,k=k,weights=table)

    def snapshot(epoch):
        cp=args.output/f'epoch-{epoch:02d}.weights.h5'
        shutil.copyfile(args.output/'latest.weights.h5',cp)
        state=(layer_hashes(model),optimizer_hash(model))
        scores={}
        for split,ids in (('fit',fit),('validation',val)):
            p=np.asarray(model.predict(batches(cache,ids,31,geometry=geometry,arm=args.arm),workers=0,max_queue_size=1,verbose=0),np.float32)
            scores[split]=metrics(k[ids],p,table)
            np.savez_compressed(args.output/f'epoch-{epoch:02d}-{split}.npz',global_index=ids,
                member=cache['members'][ids],k=k[ids],probability=p,predicted=p.argmax(1),
                lost_eligible_samples=geometry_rows['lost_eligible_samples'][ids])
        require(state==(layer_hashes(model),optimizer_hash(model)),'inference changed training state')
        validation=scores['validation']
        log=result['history'][-1]
        np.testing.assert_allclose(validation['nll'],log['val_loss'],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(validation['poly_exact'],log['val_poly_exact'],atol=1e-6)
        for value in range(7):
            np.testing.assert_allclose(validation['by_true_k'][str(value)]['exact'],
                                       log[f'val_k{value}_exact'],atol=1e-6)
        result['checkpoints'][str(epoch)]=dict(splits=scores,weights_sha256=digest(cp),
            optimizer_iterations=int(model.optimizer.iterations.numpy()))

    class Preserve(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            np.testing.assert_array_equal(sequence.order,epoch_order(fit,SEED,epoch))
            result['epoch_orders'].append(dict(epoch=epoch+1,sha256=array_hash(sequence.order)))

        def on_epoch_end(self,epoch,logs=None):
            completed=epoch+1
            require(int(model.optimizer.iterations.numpy())==completed*339,'wrong gradient update count')
            entry=dict(epoch=completed,**{key:float(v) for key,v in (logs or {}).items()})
            require(all(np.isfinite(v) for v in entry.values()),'nonfinite training')
            result['history'].append(entry)
            model.save_weights(args.output/'latest.weights.h5')
            if completed in CHECKPOINTS:
                snapshot(completed)
            write_json(args.output/'report.json',result)
            write_json(args.output/'progress.json',dict(arm=args.arm,completed_epoch=completed,
                expected_epochs=EPOCHS,validation_poly_exact=entry['val_poly_exact'],last_epoch=entry))
            print(json.dumps(dict(arm=args.arm,completed_epoch=completed,
                val_poly_exact=entry['val_poly_exact'],snapshots=list(result['checkpoints']))),flush=True)
            if completed in (4,8) and args.release_tag:
                subprocess.run(['python','-B','scripts/archive_v273_native_stage.py',
                    '--root',str(args.output),'--name',f'ownership-{args.arm}-epoch{completed:02d}',
                    '--tag',args.release_tag],check=True)

    model.fit(sequence,epochs=EPOCHS,shuffle=False,workers=0,max_queue_size=1,
        validation_data=batches(cache,val,31,geometry=geometry,arm=args.arm,k=k,weights=table),verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),Preserve()])
    require([h['epoch'] for h in result['history']]==list(range(1,13)),'incomplete training')
    require(set(result['checkpoints'])==set(map(str,CHECKPOINTS)),'missing checkpoints')
    result['status']='completed'
    write_json(args.output/'report.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','config','preflight','geometry','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--release-tag')
    train(p.parse_args())
