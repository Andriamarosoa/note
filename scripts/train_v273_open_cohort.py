"""Four predeclared full-cohort neural producers, with strict outer-fold exclusion."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import erf, softmax
from sklearn.preprocessing import StandardScaler
from scripts.yourmt3_exactk_common import FOLDS, digest, require

SEED=27405
EPOCHS=(4,8,12)


def piece_names(members):
    return np.asarray(['_'.join(Path(str(m)).stem.split('_')[1:-1]) for m in members])


def id_hash(ids):
    return hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest()


def numpy_forward(x,weights,batch=256):
    out=[]
    for start in range(0,len(x),batch):
        a=np.asarray(x[start:start+batch],np.float64)
        for layer in range(3):
            a=a@weights[2*layer]+weights[2*layer+1]
            if layer<2:a=.5*a*(1+erf(a/np.sqrt(2)))
        out.append(softmax(a,axis=1))
    return np.concatenate(out)


def main(a):
    import tensorflow as tf
    require(a.fold in FOLDS and not a.output.exists(),'fold/output')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    arrays=[];sources=[]
    for fold in FOLDS:
        paths=list(a.features.rglob(f'features-fold-{fold}.npz'))
        require(len(paths)==1,'feature fold inventory')
        path=paths[0];report=json.loads(path.with_name('report.json').read_text())
        require(report['status']=='completed' and digest(path)==report['feature_sha256'],'feature checksum')
        with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k] for k in z.files})
        sources.append(dict(fold=fold,sha256=digest(path),path=path.name))
    ids=np.concatenate([z['global_index'] for z in arrays]);y=np.concatenate([z['k'] for z in arrays])
    folds=np.concatenate([z['fold'] for z in arrays]);members=np.concatenate([z['member'] for z in arrays])
    pieces=piece_names(members);fit=np.flatnonzero(folds!=a.fold);test=np.flatnonzero(folds==a.fold)
    require(len(ids)==len(set(ids))==59309,'native cohort drift')
    require(not set(pieces[fit]) & set(pieces[test]),'composition overlap')
    require(set(np.unique(y[fit]))==set(range(7)),'missing train classes')
    a.output.mkdir(parents=True);predictions={};records=[];t0=time.monotonic()
    for representation in ('summary58','trajectory'):
        blocks=[]
        for z in arrays:
            blocks.append(z['summary'] if representation=='summary58' else
                          np.column_stack([z['summary'],z['sequence'].reshape(len(z['k']),-1),z['geometry']]))
        raw=np.concatenate(blocks).astype(np.float32,copy=False);del blocks
        scaler=StandardScaler().fit(raw[fit])
        xfit=raw[fit].copy();xtest=raw[test].copy()
        for x in (xfit,xtest):
            x-=scaler.mean_.astype(np.float32);x/=scaler.scale_.astype(np.float32);np.clip(x,-6,6,out=x)
        del raw
        norm=a.output/f'{representation}-normalizer.npz'
        np.savez_compressed(norm,mean=scaler.mean_,scale=scaler.scale_)
        for poly_weight in (1.,1.5):
            tf.keras.utils.set_random_seed(SEED)
            model=tf.keras.Sequential([
                tf.keras.layers.Input(shape=(xfit.shape[1],)),
                tf.keras.layers.Dense(128,activation='gelu',kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
                tf.keras.layers.Dropout(.1),
                tf.keras.layers.Dense(64,activation='gelu',kernel_regularizer=tf.keras.regularizers.L2(1e-4)),
                tf.keras.layers.Dropout(.1),tf.keras.layers.Dense(7)])
            model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            weight=np.where(y[fit]>=2,poly_weight,1.).astype(np.float32);weight/=weight.mean()
            dataset=tf.data.Dataset.from_tensor_slices((xfit,y[fit].astype(np.int32),weight))
            dataset=dataset.shuffle(len(fit),seed=SEED,reshuffle_each_iteration=True).batch(256)
            history=[]
            for epoch in range(1,13):
                h=model.fit(dataset,epochs=1,verbose=0).history['loss'][0]
                require(np.isfinite(h),'nonfinite loss');history.append(float(h))
                print(json.dumps(dict(stage='train',fold=a.fold,representation=representation,
                    poly_weight=poly_weight,epoch=epoch,loss=float(h),elapsed_seconds=time.monotonic()-t0)),flush=True)
                if epoch in EPOCHS:
                    key=f'{representation}__poly{poly_weight:g}__epoch{epoch}'
                    prob=np.concatenate([tf.nn.softmax(model(xtest[i:i+256],training=False)).numpy()
                                         for i in range(0,len(test),256)])
                    weights=model.get_weights();replay=numpy_forward(xtest,weights)
                    maxdiff=float(np.max(np.abs(prob-replay)))
                    require(maxdiff<3e-5,'independent neural forward drift')
                    require(np.array_equal(prob.argmax(1),replay.argmax(1)),'independent argmax drift')
                    predictions[key]=prob
                    state=a.output/(key+'.weights.npz')
                    np.savez_compressed(state,**{f'w{i}':v for i,v in enumerate(weights)})
                    records.append(dict(id=key,held_fold=a.fold,train_folds=[f for f in FOLDS if f!=a.fold],
                        train_rows=len(fit),held_rows=len(test),train_id_sha256=id_hash(ids[fit]),
                        held_id_sha256=id_hash(ids[test]),train_pieces=sorted(set(pieces[fit])),
                        held_pieces=sorted(set(pieces[test])),normalizer=norm.name,
                        normalizer_sha256=digest(norm),weights=state.name,weights_sha256=digest(state),
                        parameters=model.count_params(),epochs=epoch,training_loss=list(history),
                        independent_forward_max_abs=maxdiff,independent_argmax_mismatches=0,
                        weighted_loss=poly_weight,probability_calibration_claimed=False))
            del model,dataset;tf.keras.backend.clear_session();gc.collect()
        del xfit,xtest;gc.collect()
    np.savez_compressed(a.output/'predictions.npz',global_index=ids[test],truth=y[test],fold=folds[test],
                        variant_ids=np.asarray(list(predictions)),probabilities=np.stack(list(predictions.values()),axis=1))
    np.savez_compressed(a.output/'partitions.npz',train_global_index=ids[fit],held_global_index=ids[test])
    report=dict(status='completed',held_fold=a.fold,records=records,feature_sources=sources,
        feature_label_input=False,baseline_input=False,yourmt3_input=False,
        independent_validation=False,automatic_promotion=False,
        prediction_sha256=digest(a.output/'predictions.npz'),elapsed_seconds=time.monotonic()-t0)
    (a.output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(stage='complete',fold=a.fold,models=len(records),elapsed_seconds=time.monotonic()-t0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--fold',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
