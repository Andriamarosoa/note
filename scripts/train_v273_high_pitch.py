"""One frozen arm of the native high-pitch-input pair; no output correction."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from scripts.audit_v273_learning_bottleneck import metrics
from scripts.rebuild_v273_sources import digest,write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.test_v273_training_budget import optimizer_hash
from scripts.train_v273_window_pair import weight_hash
from scripts.v273_ownership_experiment import load_geometry
from scripts.v273_high_pitch_native import (
    ARMS,SEED,EPOCHS,CHECKPOINTS,build,batches,experiment_identity,compile_metrics,layer_hashes)
from scripts.v273_window_experiment import load_bundle,require,array_hash,epoch_order


def train(args):
    import tensorflow as tf
    require(tf.__version__=='2.15.1' and np.__version__=='1.26.4','pinned training environment required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),'refusing to overwrite training')
    identity=experiment_identity()
    gate=json.loads((args.preflight/'report.json').read_text())
    prepared=json.loads((args.maps/'report.json').read_text())
    require(gate['status']=='passed' and prepared['status']=='prepared','prerequisite gate failed')
    for source in (gate,prepared):
        for key in ('launch_sha256','protocol_sha256','config_sha256','feature_builder_sha256','register_builder_sha256'):
            require(source[key]==identity[key],'prerequisite identity differs: '+key)
    validate_original_inventory(args.bundle)
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    geometry,_,geometry_report=load_geometry(args.geometry,cache,manifest,args.config)
    require(prepared['bundle_sha256']==digest(args.bundle/'bundle.json') and
            prepared['geometry_report_sha256']==digest(args.geometry/'report.json'),'prepared inputs differ')
    require(digest(args.maps/'maps.npy')==prepared['maps_sha256'],'prepared maps changed')
    maps=np.load(args.maps/'maps.npy',mmap_mode='r',allow_pickle=False)
    fit,val=parts['inner_fit'],parts['inner_val']
    require(len(fit)==43357 and len(val)==15952 and prepared['outer_rows_evaluated']==0,'wrong scope')
    require(prepared['fit_indices_sha256']==array_hash(fit) and
            prepared['val_indices_sha256']==array_hash(val),'wrong row order')
    k=np.minimum(np.asarray(cache['exact'],np.int32),6)
    model=build(args.arm);initial=weight_hash(model)
    require(initial==gate['arms'][args.arm]['initial_sha256'],'preflight initialization differs')
    compile_metrics(model)
    require(int(model.optimizer.iterations.numpy())==0,'optimizer not fresh')
    args.output.mkdir(parents=True)
    model.save_weights(args.output/'initial.weights.h5')
    report=dict(status='training',experiment='native_high_pitch_pair',arm=args.arm,**identity,
        seed=SEED,epochs=EPOCHS,primary_epoch=12,snapshots=list(CHECKPOINTS),batch_size=128,
        tensorflow=tf.__version__,numpy=np.__version__,parameters=model.count_params(),
        initial_sha256=initial,initial_weights_sha256=digest(args.output/'initial.weights.h5'),
        preflight_sha256=digest(args.preflight/'report.json'),prepared_report_sha256=digest(args.maps/'report.json'),
        maps_sha256=prepared['maps_sha256'],geometry_report_sha256=prepared['geometry_report_sha256'],
        bundle_sha256=prepared['bundle_sha256'],fit_indices_sha256=array_hash(fit),val_indices_sha256=array_hash(val),
        training_performed=True,output_corrector=False,outer_rows_evaluated=0,automatic_promotion=False,
        decode='argmax P(K=0..6)',checkpoint_selection='fixed epoch 12',weighting='uniform',
        script_sha256=digest(__file__),model_builder_sha256=digest('scripts/v273_high_pitch_native.py'),
        history=[],epoch_orders=[],checkpoints={})
    write_json(args.output/'report.json',report)
    sequence=batches(cache,maps,geometry,fit,args.arm,labels=k,shuffle=True)
    started=time.monotonic()

    def snapshot(epoch):
        checkpoint=args.output/f'epoch-{epoch:02d}.weights.h5'
        shutil.copyfile(args.output/'latest.weights.h5',checkpoint)
        state=(layer_hashes(model),optimizer_hash(model))
        probability=np.asarray(model.predict(batches(cache,maps,geometry,val,args.arm),
            workers=0,max_queue_size=1,verbose=0),np.float32)
        require(state==(layer_hashes(model),optimizer_hash(model)),'inference changed model/optimizer')
        result=metrics(k[val],probability,np.ones(7,np.float32))
        log=report['history'][-1]
        np.testing.assert_allclose(result['nll'],log['val_loss'],rtol=2e-5,atol=2e-5)
        np.testing.assert_allclose(result['exact'],log['val_exact'],atol=1e-6)
        np.testing.assert_allclose(result['poly_exact'],log['val_poly_exact'],atol=1e-6)
        for value in range(7):
            np.testing.assert_allclose(result['by_true_k'][str(value)]['exact'],
                                       log[f'val_k{value}_exact'],atol=1e-6)
        path=args.output/f'epoch-{epoch:02d}-validation.npz'
        np.savez_compressed(path,global_index=val,member=cache['members'][val],k=k[val],
                            probability=probability,predicted=probability.argmax(1))
        report['checkpoints'][str(epoch)]=dict(validation=result,weights_sha256=digest(checkpoint),
            predictions_sha256=digest(path),optimizer_iterations=int(model.optimizer.iterations.numpy()))

    class Preserve(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            np.testing.assert_array_equal(sequence.order,epoch_order(fit,SEED,epoch))
            report['epoch_orders'].append(dict(epoch=epoch+1,sha256=array_hash(sequence.order)))
            self.epoch=epoch+1
        def on_train_batch_end(self,batch,logs=None):
            if (batch+1)%64==0:
                progress=dict(status='training',arm=args.arm,epoch=self.epoch,
                    completed_batches=batch+1,batches_per_epoch=339,expected_epochs=EPOCHS,
                    seconds=round(time.monotonic()-started,1))
                write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        def on_epoch_end(self,epoch,logs=None):
            completed=epoch+1
            require(int(model.optimizer.iterations.numpy())==completed*339,'wrong update count')
            entry=dict(epoch=completed,**{key:float(v) for key,v in (logs or {}).items()})
            require(all(np.isfinite(v) for v in entry.values()),'nonfinite training')
            report['history'].append(entry)
            model.save_weights(args.output/'latest.weights.h5')
            if completed in CHECKPOINTS:
                snapshot(completed)
            write_json(args.output/'report.json',report)
            progress=dict(status='training',arm=args.arm,completed_epoch=completed,expected_epochs=EPOCHS,
                val_exact=entry['val_exact'],val_poly_exact=entry['val_poly_exact'],
                seconds=round(time.monotonic()-started,1))
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
            if completed in (4,8) and args.release_tag:
                subprocess.run(['python','-B','scripts/archive_v273_native_stage.py',
                    '--root',str(args.output),'--name',f'high-pitch-{args.arm}-epoch{completed:02d}',
                    '--tag',args.release_tag],check=True)

    model.fit(sequence,epochs=EPOCHS,shuffle=False,workers=0,max_queue_size=1,verbose=2,
        validation_data=batches(cache,maps,geometry,val,args.arm,labels=k),
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),Preserve()])
    require(len(report['history'])==12 and set(report['checkpoints'])=={'4','8','12'},'incomplete training')
    report.update(status='completed',elapsed_seconds=time.monotonic()-started)
    write_json(args.output/'report.json',report)
    write_json(args.output/'progress.json',dict(status='completed',arm=args.arm,completed_epoch=12,expected_epochs=12))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','config','preflight','geometry','maps','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--release-tag')
    train(p.parse_args())
