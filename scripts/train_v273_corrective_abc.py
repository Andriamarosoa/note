"""Matched trained corrective A/B/C for V27.3 learned-gate Exact-K.

Arms:
  control:
    historical learned gate; corrective parameters are present but zero-effect.
  pool_calib:
    learned gate + five learnable multiplicative scales (initially 1) on
    mean/max/attention/scaled_sum/stats before cluster_norm.
  pool_plus_spectral:
    pool_calib + a zero-initialized residual 3x3 kernel restricted to
    spectral input channels 0=log_power and 1=positive_pre feeding conv1
    output filter 27.

All arms:
  * same architecture/parameter count
  * same random initialization for shared parameters
  * same fold3 final-fit population, epoch order, seed, 31 frames
  * same 8-epoch uniform-loss protocol
  * outer fold never used for selection
  * no automatic promotion
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from scripts import train_v102_source_time_assignment as v102
from scripts import train_v100_spectral_string_slots as v100
from scripts import train_v240_categorical_k_candidate_subset as v240
from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v90_structured_cluster_cardinality import FROZEN_CANDIDATE_DIM
from scripts.train_v273_group_gate_ab import (
    BATCH_SIZE, FRAMES, batches, load_cluster_a, metrics, transitions, weight_hash,
)
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import (
    load_bundle, epoch_order, array_hash, require,
)

ARMS=("control","pool_calib","pool_plus_spectral")
POOL_ACTIVE={"control":False,"pool_calib":True,"pool_plus_spectral":True}
SPECTRAL_ACTIVE={"control":False,"pool_calib":False,"pool_plus_spectral":True}
POOL_NAMES=("mean","max","attention","scaled_sum","stats")
TARGET_FILTER=27
TARGET_CHANNELS=(0,1)


def _input_by_name(model,name):
    for x in model.inputs:
        if x.name.split(":")[0]==name:
            return x
    raise KeyError(name)


def _core_of(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model) and x.name=="v273_corrective_core"]
    require(len(xs)==1,"corrective core mismatch")
    return xs[0]


class PoolCalibration:
    """Factory to avoid importing TensorFlow at module import time."""
    @staticmethod
    def layer(tf, active):
        class _Layer(tf.keras.layers.Layer):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.active=bool(active)
            def build(self, input_shape):
                self.log_scales=self.add_weight(
                    "log_scales",shape=(5,),initializer="zeros",trainable=True
                )
            def call(self, parts):
                if self.active:
                    scales=tf.exp(tf.clip_by_value(self.log_scales,-1.5,1.5))
                else:
                    scales=tf.ones_like(self.log_scales)+0.0*self.log_scales
                return tf.concat(
                    [tf.cast(parts[i],tf.float32)*scales[i] for i in range(5)],
                    axis=-1,
                )
        return _Layer(name="v273_collective_calibration")


class SpectralResidual:
    @staticmethod
    def layer(tf, active):
        class _Layer(tf.keras.layers.Layer):
            def __init__(self, **kw):
                super().__init__(**kw)
                self.active=bool(active)
            def build(self, input_shape):
                self.delta=self.add_weight(
                    "delta_kernel",
                    shape=(2,3,3),
                    initializer="zeros",
                    trainable=True,
                )
            def call(self, inputs):
                dense_input, conv1_linear = inputs
                pieces=[]
                for local_idx,channel in enumerate(TARGET_CHANNELS):
                    x=dense_input[:,:,:,channel:channel+1]
                    kernel=self.delta[local_idx,:,:,None,None]
                    pieces.append(
                        tf.nn.conv2d(x,kernel,strides=[1,1,1,1],padding="SAME")
                    )
                corr=pieces[0]+pieces[1]
                onehot=tf.one_hot(TARGET_FILTER,tf.shape(conv1_linear)[-1],dtype=conv1_linear.dtype)
                corr=corr*onehot[None,None,None,:]
                if not self.active:
                    corr=0.0*corr
                return tf.nn.relu(conv1_linear+corr)
        return _Layer(name="v273_conv1_targeted_residual")


def build_core(arm:str, seed:int):
    import tensorflow as tf
    from tensorflow import keras
    if arm not in ARMS:
        raise ValueError(arm)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # Historical candidate scaffold and source-time model. We reuse its exact
    # pooling/cluster weights, but reconnect the collective representation
    # through an identity-initialized calibration layer.
    base,_,token_shape=v102._build_model(time_frames=FRAMES,spectral_channels=v100.SPECTRAL_CHANNELS)
    candidate_set=_input_by_name(base,"candidate_set")
    candidate_mask=_input_by_name(base,"candidate_mask")
    stats=_input_by_name(base,"cluster_stats")
    spectral=_input_by_name(base,"spectral_map")

    mean=base.get_layer("mean_pool").output
    maximum=base.get_layer("max_pool").output
    attended=base.get_layer("attention_pool").output
    scaled_sum=base.get_layer("scaled_sum_pool").output

    calibrator=PoolCalibration.layer(tf,POOL_ACTIVE[arm])
    pooled=calibrator([mean,maximum,attended,scaled_sum,stats])
    norm=base.get_layer("cluster_norm")(pooled)
    hidden1=base.get_layer("cluster_hidden1")(norm)
    hidden1=base.get_layer("cluster_dropout")(hidden1)
    hidden2=base.get_layer("cluster_hidden2")(hidden1)
    candidate_context=base.get_layer("candidate_context")(hidden2)

    # Recreate the native V24 count spectral path. Conv1 is split into linear
    # convolution + explicit ReLU so the targeted residual can be inserted
    # pre-activation. At zero residual this is algebraically identical.
    tc=v102.time_coordinates(FRAMES)[:,None]
    fc=np.linspace(-1.0,1.0,v100.SPECTRAL_BANDS,dtype=np.float32)[None,:]
    coord=np.stack([
        np.broadcast_to(tc,(FRAMES,v100.SPECTRAL_BANDS)),
        np.broadcast_to(fc,(FRAMES,v100.SPECTRAL_BANDS)),
    ],axis=-1).astype(np.float32)
    coord_const=tf.constant(coord,dtype=tf.float32)
    coords=keras.layers.Lambda(
        lambda s: tf.tile(coord_const[None,:,:,:],[tf.shape(s)[0],1,1,1]),
        name="v240_dense_coordinates",
    )(spectral)
    dense=keras.layers.LayerNormalization(axis=-1,name="v240_dense_channel_norm")(spectral)
    dense_in=keras.layers.Concatenate(axis=-1,name="v240_dense_plus_coordinates")([dense,coords])

    conv1_linear=keras.layers.Conv2D(
        32,(3,3),padding="same",activation=None,name="v240_dense_conv1"
    )(dense_in)
    spectral_residual=SpectralResidual.layer(tf,SPECTRAL_ACTIVE[arm])
    dense=spectral_residual([dense_in,conv1_linear])
    dense=keras.layers.Conv2D(
        64,(3,3),padding="same",activation="relu",name="v240_dense_conv2"
    )(dense)
    dense_features=keras.layers.Conv2D(
        v240.QUERY_DIM,(3,3),padding="same",activation="relu",name="v240_dense_conv3"
    )(dense)

    # Preserve historical layer construction/RNG order before cardinality head.
    center4=keras.layers.Conv2D(
        1,(1,1),padding="same",name="v240_birth_center_logits"
    )(dense_features)
    _=keras.layers.Reshape(
        (FRAMES*v100.SPECTRAL_BANDS,),name="v240_birth_center_logits_flat"
    )(center4)

    avg=keras.layers.GlobalAveragePooling2D(name="v240_dense_global_average")(dense_features)
    mx=keras.layers.GlobalMaxPooling2D(name="v240_dense_global_max")(dense_features)
    card_h=keras.layers.Concatenate(name="v240_cardinality_context")([avg,mx,candidate_context])
    card_h=keras.layers.Dense(192,activation="relu",name="v240_cardinality_hidden1")(card_h)
    card_h=keras.layers.Dropout(
        0.08,seed=None,name="v240_cardinality_dropout"
    )(card_h)
    card_h=keras.layers.Dense(96,activation="relu",name="v240_cardinality_hidden2")(card_h)
    cardinality=keras.layers.Dense(
        7,activation="softmax",name="cardinality"
    )(card_h)

    return keras.Model(
        base.inputs,cardinality,name="v273_corrective_core"
    )


def build_model(arm:str, seed:int):
    import tensorflow as tf
    if arm not in ARMS:
        raise ValueError(arm)
    core=build_core(arm,seed)
    if FROZEN_CANDIDATE_DIM != V88_FEATURE_DIM+8:
        raise RuntimeError("unexpected V8.8 feature layout")

    by_name={x.name.split(":")[0]:x for x in core.inputs}
    cand=by_name["candidate_set"]
    mask=by_name["candidate_mask"]
    stats=by_name["cluster_stats"]
    spectral=by_name["spectral_map"]

    # Same learned gate in every arm: this experiment starts from the
    # learned-gate treatment, not from the historical always-on control.
    gate_h=tf.keras.layers.Dense(
        12,activation="relu",name="v273_gate_hidden",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+71),
    )(stats[:,:4])
    gate=tf.keras.layers.Dense(
        1,activation="sigmoid",name="v273_gate_value",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed+72),
        bias_initializer=tf.keras.initializers.Zeros(),
    )(gate_h)
    g3=tf.keras.layers.Lambda(
        lambda g: tf.expand_dims(g,axis=1),name="v273_gate_expand"
    )(gate)

    left=cand[:,:,:V88_FEATURE_DIM]
    pseudo=cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5]
    right=cand[:,:,V88_FEATURE_DIM+5:]
    gated_pseudo=tf.keras.layers.Multiply(
        name="v273_gate_candidate_pseudocount"
    )([pseudo,g3])
    gated_cand=tf.keras.layers.Lambda(
        lambda z: tf.concat(z,axis=-1),name="v273_gate_candidate_concat"
    )([left,gated_pseudo,right])

    preserved=stats[:,:4]
    pseudo_stats=stats[:,4:8]
    gated_stats=tf.keras.layers.Multiply(
        name="v273_gate_stats_pseudocount"
    )([pseudo_stats,gate])
    merged_stats=tf.keras.layers.Lambda(
        lambda z: tf.concat(z,axis=-1),name="v273_gate_stats_concat"
    )([preserved,gated_stats])

    out=core({
        "candidate_set":gated_cand,
        "candidate_mask":mask,
        "cluster_stats":merged_stats,
        "spectral_map":spectral,
    })
    model=tf.keras.Model(core.inputs,out,name="v273_corrective_"+arm)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(2e-4),
        loss="sparse_categorical_crossentropy",
    )
    return model


def corrective_summary(model):
    core=_core_of(model)
    pool=core.get_layer("v273_collective_calibration")
    spectral=core.get_layer("v273_conv1_targeted_residual")
    log_scales=np.asarray(pool.get_weights()[0],np.float32)
    return {
        "pool_log_scales":log_scales.tolist(),
        "pool_scales":np.exp(np.clip(log_scales,-1.5,1.5)).tolist(),
        "spectral_delta_kernel":np.asarray(spectral.get_weights()[0],np.float32).tolist(),
        "spectral_delta_l2":float(np.linalg.norm(spectral.get_weights()[0])),
    }


def train(args):
    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing to overwrite")

    cfg=load_config(args.config)
    budget=cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget==8,"budget changed")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=np.asarray(parts["final_fit"],np.int64)
    outer=np.asarray(parts["outer"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    args.output.mkdir(parents=True)

    seed=v260.SEED+1003
    initial={}
    for arm in ARMS:
        m=build_model(arm,seed)
        initial[arm]={
            "hash":weight_hash(m),
            "params":int(np.sum([np.prod(v.shape) for v in m.trainable_variables])),
            "corrective":corrective_summary(m),
        }
        del m
        tf.keras.backend.clear_session()

    hashes={v["hash"] for v in initial.values()}
    params={v["params"] for v in initial.values()}
    require(len(hashes)==1,"initial weight mismatch")
    require(len(params)==1,"parameter count mismatch")

    order=hashlib.sha256()
    for e in range(budget):
        order.update(epoch_order(fit,seed,e).tobytes())

    cid,A=load_cluster_a(args.cluster_rows,outer)
    require(int(A.sum())==499,"Cluster A drift")

    report={
        "status":"training",
        "protocol":{
            "experiment":"v273_trained_corrective_abc",
            "arms":list(ARMS),
            "reference":"V27.3 learned_gate protocol",
            "frames":FRAMES,
            "weighting":"uniform",
            "epochs":budget,
            "seed":seed,
            "batch_size":BATCH_SIZE,
            "epoch_order_sha256":order.hexdigest(),
            "same_initial_weights":True,
            "same_trainable_parameter_count":True,
            "pool_parameters":5,
            "pool_initialization":"identity: exp(log_scale=0)=1",
            "pool_location":"mean/max/attention/scaled_sum/stats before cluster_norm",
            "spectral_parameters":18,
            "spectral_initialization":"zero residual",
            "spectral_location":"conv1 pre-ReLU, channels 0/1 to output filter 27 only",
            "spectral_channel_0":"log_power",
            "spectral_channel_1":"positive_pre",
            "outer_used_for_selection":False,
            "automatic_promotion":False,
        },
        "initial":initial,
        "cluster_A_id":cid,
        "cluster_A_rows":499,
        "arms":{},
        "paired_vs_control":{},
    }
    write_json(args.output/"protocol.json",report["protocol"])

    preds={
        "global_index":outer,
        "member":np.asarray(cache["members"][outer]),
        "cluster_start_samples":np.asarray(cache["cluster_start_samples"][outer]),
        "k":k[outer],
    }

    for arm in ARMS:
        tf.keras.backend.clear_session()
        model=build_model(arm,seed)
        require(weight_hash(model)==initial[arm]["hash"],"initialization drift")
        seq=batches(cache,fit,seed=seed,shuffle=True,k=k)
        hist_rows=[]; observed=[]

        class CB(tf.keras.callbacks.Callback):
            def on_epoch_begin(self,epoch,logs=None):
                expected=epoch_order(fit,seed,epoch)
                np.testing.assert_array_equal(seq.order,expected)
                observed.append(array_hash(seq.order))
            def on_epoch_end(self,epoch,logs=None):
                row={"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}}
                hist_rows.append(row)
                write_json(args.output/f"{arm}-progress.json",{
                    "arm":arm,"completed_epoch":int(epoch)+1,
                    "history":hist_rows,
                    "corrective":corrective_summary(model),
                })

        hist=model.fit(
            seq,epochs=budget,shuffle=False,workers=0,max_queue_size=1,verbose=2,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()]
        )
        require(
            len(hist_rows)==budget and all(np.isfinite(v).all() for v in hist.history.values()),
            "training failure",
        )
        prob=np.asarray(
            model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),
            np.float32,
        )
        require(
            prob.shape==(len(outer),7) and np.isfinite(prob).all()
            and np.allclose(prob.sum(1),1,atol=1e-5),
            "bad probabilities",
        )
        pred=prob.argmax(1).astype(np.int32)

        gate_model=tf.keras.Model(model.inputs,model.get_layer("v273_gate_value").output)
        gate=np.asarray(
            gate_model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0)
        ).reshape(-1)

        model.save_weights(args.output/f"{arm}.weights.h5")
        report["arms"][arm]={
            "metrics":metrics(k[outer],pred),
            "cluster_A_metrics":metrics(k[outer][A],pred[A]),
            "history":hist_rows,
            "corrective":corrective_summary(model),
            "gate_summary":{
                "mean":float(gate.mean()),
                "median":float(np.median(gate)),
                "p10":float(np.percentile(gate,10)),
                "p90":float(np.percentile(gate,90)),
                "by_k":{str(v):float(gate[k[outer]==v].mean()) for v in (2,3,4)},
                "cluster_A_mean":float(gate[A].mean()),
            },
            "observed_epoch_order_sha256":observed,
            "weights_sha256":digest(args.output/f"{arm}.weights.h5"),
        }
        preds[f"{arm}_probability"]=prob
        preds[f"{arm}_predicted"]=pred
        preds[f"{arm}_gate"]=gate.astype(np.float32)
        write_json(args.output/"report.json",report)
        np.savez_compressed(args.output/"predictions.npz",**preds)
        del model,gate_model,seq,hist
        tf.keras.backend.clear_session()

    control=preds["control_predicted"]
    for arm in ("pool_calib","pool_plus_spectral"):
        pred=preds[f"{arm}_predicted"]
        report["paired_vs_control"][arm]={
            "all":transitions(k[outer],control,pred),
            "poly":transitions(
                k[outer][k[outer]>=2],
                control[k[outer]>=2],
                pred[k[outer]>=2],
            ),
            "cluster_A":transitions(k[outer][A],control[A],pred[A]),
        }

    report["status"]="completed"
    write_json(args.output/"report.json",report)

    lines=[
        "# V27.3 trained corrective A/B/C","",
        "| Mesure | Control learned-gate | Pool calibration | Pool + spectral |",
        "|---|---:|---:|---:|",
    ]
    def val(arm,key):
        return report["arms"][arm]["metrics"][key]
    lines.append(
        f"| exact global | {100*val('control','exact'):.3f}% | "
        f"{100*val('pool_calib','exact'):.3f}% | "
        f"{100*val('pool_plus_spectral','exact'):.3f}% |"
    )
    lines.append(
        f"| exact poly | {100*val('control','poly_exact'):.3f}% | "
        f"{100*val('pool_calib','poly_exact'):.3f}% | "
        f"{100*val('pool_plus_spectral','poly_exact'):.3f}% |"
    )
    for v in (2,3,4):
        xs=[
            report["arms"][arm]["metrics"]["by_k"][str(v)]["exact"]
            for arm in ARMS
        ]
        lines.append(
            f"| K{v} exact | {100*xs[0]:.3f}% | {100*xs[1]:.3f}% | {100*xs[2]:.3f}% |"
        )
    lines.append(
        f"| under | {val('control','under')} | {val('pool_calib','under')} | {val('pool_plus_spectral','under')} |"
    )
    lines.append(
        f"| over | {val('control','over')} | {val('pool_calib','over')} | {val('pool_plus_spectral','over')} |"
    )
    lines += ["","## Paired net vs control",""]
    for arm in ("pool_calib","pool_plus_spectral"):
        p=report["paired_vs_control"][arm]
        lines.append(
            f"- {arm}: global {p['all']['net_correct']:+d}; "
            f"poly {p['poly']['net_correct']:+d}; "
            f"Cluster A {p['cluster_A']['net_correct']:+d}."
        )
        c=report["arms"][arm]["corrective"]
        lines.append(
            f"  Pool scales: "+", ".join(
                f"{n}={x:.4f}" for n,x in zip(POOL_NAMES,c["pool_scales"])
            )+f"; spectral delta L2={c['spectral_delta_l2']:.5f}."
        )
    lines += ["","Aucune promotion automatique."]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--cluster-rows",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    train(p.parse_args())
