"""Train the opt-in silence/count head only on verified, aligned target arrays."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np

from causal_note.count_states import encode_states,decode_probabilities,state_metrics
from scripts.rebuild_v273_sources import digest,write_json
from scripts.restore_v273_original_backup import validate_original_inventory
from scripts.v273_high_pitch_native import batches,ARMS,SEED
from scripts.v273_ownership_experiment import load_geometry
from scripts.v273_silence_state_model import build_model,require_training_population
from scripts.v273_window_experiment import load_bundle,require,array_hash
from scripts.train_v273_window_pair import weight_hash


def train(args):
    import tensorflow as tf
    require(tf.__version__=='2.15.1' and np.__version__=='1.26.4','pinned environment required')
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists() and args.epochs>0,'invalid output or epoch budget')
    validate_original_inventory(args.bundle)
    cache,parts,manifest=load_bundle(args.bundle,args.config)
    geometry,_,_=load_geometry(args.geometry,cache,manifest,args.config)
    prepared=json.loads((args.maps/'report.json').read_text())
    targets=json.loads((args.targets/'report.json').read_text())
    require(targets['status']=='prepared' and targets['training_population_ready'],
            'real silence and sounding-zero examples needed in both partitions')
    require(targets['state_order']==list(range(-1,7)),'wrong output state order')
    require(targets['state_builder_sha256']==digest('src/causal_note/count_states.py'),
            'target builder changed')
    for report in (prepared,targets):
        require(report['config_sha256']==digest(args.config) and
                report['geometry_report_sha256']==digest(args.geometry/'report.json') and
                report['outer_rows_evaluated']==0,'wrong source population')
    require(prepared['bundle_sha256']==digest(args.bundle/'bundle.json'),'wrong feature bundle')
    require(digest(args.maps/'maps.npy')==prepared['maps_sha256'],'feature maps changed')
    require(digest(args.targets/'targets.npz')==targets['target_sha256'],'target arrays changed')
    maps=np.load(args.maps/'maps.npy',mmap_mode='r',allow_pickle=False)
    with np.load(args.targets/'targets.npz',allow_pickle=False) as z:
        state,eligible=z['state'],z['trainable']
        np.testing.assert_array_equal(z['global_index'],np.arange(len(cache['exact'])))
        np.testing.assert_array_equal(z['member'],cache['members'])
        np.testing.assert_array_equal(z['start'],cache['cluster_start_samples'])
        inner=np.sort(np.r_[parts['inner_fit'],parts['inner_val']])
        np.testing.assert_array_equal(z['old_k'][inner],cache['exact'][inner])
        np.testing.assert_array_equal(z['fit_index'],parts['inner_fit'])
        np.testing.assert_array_equal(z['validation_index'],parts['inner_val'])
    fit,val=require_training_population(state,eligible,parts['inner_fit'],parts['inner_val'])
    labels=np.zeros(len(state),np.int64)
    labels[eligible]=encode_states(state[eligible])
    model=build_model();initial=weight_hash(model)
    args.output.mkdir(parents=True)
    report=dict(status='training',experiment='native_silence_count_states',state_order=list(range(-1,7)),
        arm=args.arm,seed=SEED,epochs=args.epochs,parameters=model.count_params(),
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        initial_sha256=initial,target_report_sha256=digest(args.targets/'report.json'),
        feature_report_sha256=digest(args.maps/'report.json'),policy=targets['policy'],
        model_builder_sha256=digest('scripts/v273_silence_state_model.py'),script_sha256=digest(__file__),
        fit_indices_sha256=array_hash(fit),validation_indices_sha256=array_hash(val),
        fit_rows=len(fit),validation_rows=len(val),outer_rows_evaluated=0,
        decode='argmax probabilities minus 1',output_corrector=False,
        historical_score_replacement=False,checkpoint_selection='last fixed epoch',history=[])
    write_json(args.output/'report.json',report)
    validation=batches(cache,maps,geometry,val,args.arm)
    class Preserve(tf.keras.callbacks.Callback):
        def on_epoch_end(self,epoch,logs=None):
            p=np.asarray(model.predict(validation,workers=0,verbose=0),np.float32)
            pred=decode_probabilities(p);metrics=state_metrics(state[val],pred)
            model.save_weights(args.output/'latest.weights.h5')
            np.savez_compressed(args.output/'latest-validation.npz',global_index=val,
                member=cache['members'][val],state=state[val],probability=p,predicted=pred)
            entry=dict(epoch=epoch+1,loss=float(logs['loss']),validation=metrics)
            require(np.isfinite(entry['loss']),'nonfinite loss')
            report['history'].append(entry);write_json(args.output/'report.json',report)
            print(json.dumps(entry),flush=True)
    model.fit(batches(cache,maps,geometry,fit,args.arm,labels=labels,shuffle=True),
        epochs=args.epochs,shuffle=False,workers=0,max_queue_size=1,verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),Preserve()])
    require(len(report['history'])==args.epochs,'incomplete training')
    report.update(status='completed',weights_sha256=digest(args.output/'latest.weights.h5'),
        predictions_sha256=digest(args.output/'latest-validation.npz'))
    write_json(args.output/'report.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','geometry','maps','targets','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path('analysis/v273-native-paired-config.json'))
    p.add_argument('--arm',choices=ARMS,default='observed_only')
    p.add_argument('--epochs',type=int,default=12)
    train(p.parse_args())
