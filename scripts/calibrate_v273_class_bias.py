"""Bias-only multiclass calibration for the best V27.3 weighted learned_gate.

Protocol:
- train the fixed best weighted learned_gate on inner_fit (folds 1/2/4),
- fit six free class-logit offsets on inner_val (fold 0) by maximum likelihood,
  with class 0 fixed to zero for identifiability,
- freeze the offsets,
- apply them to the already-trained final weighted model on outer fold 3.

The outer fold is evaluation-only and never used to fit or select calibration.
"""
from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import write_json
from scripts.train_v273_group_gate_ab import build_model, weight_hash, metrics, transitions
from scripts.train_v273_loss_weighting_ab import class_weight_table, batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, epoch_order, require

ARM="targeted_k23_half_k4_quarter"


def softmax_logits(logp, bias):
    z=np.asarray(logp,np.float64)+np.asarray(bias,np.float64)[None,:]
    z-=z.max(axis=1,keepdims=True)
    q=np.exp(z)
    q/=q.sum(axis=1,keepdims=True)
    return q


def nll(q,y):
    y=np.asarray(y,np.int64)
    return float(-np.log(np.clip(q[np.arange(len(y)),y],1e-12,1.0)).mean())


def fit_bias(prob,y,max_iter=100):
    """Convex MLE for class-logit biases with b0 fixed at 0."""
    y=np.asarray(y,np.int64)
    logp=np.log(np.clip(np.asarray(prob,np.float64),1e-12,1.0))
    theta=np.zeros(6,np.float64)
    history=[]
    eye_y=np.eye(7,dtype=np.float64)[y]
    for it in range(max_iter):
        bias=np.r_[0.0,theta]
        q=softmax_logits(logp,bias)
        loss=nll(q,y)
        g=(q-eye_y).mean(axis=0)[1:]
        # Hessian for the six free coordinates.
        H=np.zeros((6,6),np.float64)
        qr=q[:,1:]
        H=np.diag(qr.mean(axis=0))-(qr.T@qr)/len(qr)
        grad_norm=float(np.linalg.norm(g))
        history.append({"iteration":it,"nll":loss,"grad_norm":grad_norm})
        if grad_norm<1e-10:
            break
        step=np.linalg.pinv(H,rcond=1e-12)@g
        scale=1.0
        accepted=False
        for _ in range(30):
            cand=theta-scale*step
            cq=softmax_logits(logp,np.r_[0.0,cand])
            closs=nll(cq,y)
            if np.isfinite(closs) and closs<=loss:
                theta=cand
                accepted=True
                break
            scale*=0.5
        if not accepted:
            break
        if float(np.linalg.norm(scale*step))<1e-10:
            break
    bias=np.r_[0.0,theta]
    # Equivalent centered representation is easier to interpret; softmax is unchanged.
    centered=bias-bias.mean()
    q=softmax_logits(logp,centered)
    return centered,q,history


def load_outer(root):
    report=json.loads((root/"report.json").read_text())
    with np.load(root/"predictions.npz",allow_pickle=False) as z:
        pred={k:np.asarray(z[k]) for k in z.files}
    return report,pred


def main(args):
    import tensorflow as tf
    require(tf.__version__=="2.15.1","pinned TensorFlow required")
    tf.config.experimental.enable_op_determinism()
    require(not args.output.exists(),"refusing to overwrite")
    args.output.mkdir(parents=True)

    cfg=load_config(args.config)
    budget=cfg["folds"][3]["epochs"]["v260_uniform"]
    require(budget==8,"budget changed")
    cache,parts,_=load_bundle(args.bundle,args.config)
    k=np.minimum(cache["exact"].astype(np.int32),6)
    fit=np.asarray(parts["inner_fit"],np.int64)
    val=np.asarray(parts["inner_val"],np.int64)
    require(not np.intersect1d(fit,val).size,"inner leakage")
    counts=np.bincount(k[val],minlength=7)
    require(np.all(counts>0),"inner validation must contain all seven classes")

    seed=v260.SEED+103
    table=class_weight_table(ARM,k[fit])
    model=build_model("learned_gate",seed)
    initial_hash=weight_hash(model)
    seq=batches(cache,fit,seed=seed,shuffle=True,k=k,sample_table=table)

    order=hashlib.sha256()
    for e in range(budget):
        order.update(epoch_order(fit,seed,e).tobytes())

    hist=[]
    class CB(tf.keras.callbacks.Callback):
        def on_epoch_begin(self,epoch,logs=None):
            np.testing.assert_array_equal(seq.order,epoch_order(fit,seed,epoch))
        def on_epoch_end(self,epoch,logs=None):
            hist.append({"epoch":int(epoch)+1,**{x:float(y) for x,y in (logs or {}).items()}})
            write_json(args.output/"progress.json",{"completed_epoch":int(epoch)+1,"history":hist})

    result=model.fit(
        seq,epochs=budget,shuffle=False,workers=0,max_queue_size=1,verbose=2,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),CB()]
    )
    require(len(hist)==budget and all(np.isfinite(v).all() for v in result.history.values()),"inner training failure")

    prob=np.asarray(model.predict(batches(cache,val),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(prob.shape==(len(val),7) and np.allclose(prob.sum(1),1,atol=1e-5),"bad inner probabilities")
    y=k[val]
    raw_inner_pred=prob.argmax(1).astype(np.int32)
    bias,inner_q,opt_history=fit_bias(prob,y)
    inner_pred=inner_q.argmax(1).astype(np.int32)

    best_report,best_outer=load_outer(args.best_outer)
    _,uni_outer=load_outer(args.uniform_outer)
    require(np.array_equal(best_outer["global_index"],uni_outer["global_index"]),"outer index mismatch")
    require(np.array_equal(best_outer["k"],uni_outer["k"]),"outer target mismatch")
    ko=np.asarray(best_outer["k"],np.int32)
    weighted_prob=np.asarray(best_outer["probability"],np.float64)
    weighted_pred=np.asarray(best_outer["predicted"],np.int32)
    uniform_pred=np.asarray(uni_outer["predicted"],np.int32)

    outer_q=softmax_logits(np.log(np.clip(weighted_prob,1e-12,1.0)),bias)
    outer_pred=outer_q.argmax(1).astype(np.int32)

    report={
        "status":"completed",
        "protocol":{
            "experiment":"v273_class_bias_calibration",
            "base_arm":ARM,
            "calibration_formula":"softmax(log p_k + b_k)",
            "calibrator":"bias-only multiclass maximum likelihood; six free parameters; b0 fixed during fit then centered",
            "selection_metric":"inner_val NLL only",
            "inner_fit":"folds 1/2/4",
            "inner_val":"fold 0",
            "outer":"fold 3 evaluation only",
            "outer_used_for_selection":False,
            "automatic_promotion":False,
            "inner_seed":seed,
            "epochs":budget,
            "initial_hash":initial_hash,
            "epoch_order_sha256":order.hexdigest(),
            "inner_weight_table":table.tolist(),
            "inner_val_class_counts":counts.tolist(),
        },
        "class_bias":bias.tolist(),
        "inner":{
            "nll_raw":nll(prob,y),
            "nll_calibrated":nll(inner_q,y),
            "raw_metrics":metrics(y,raw_inner_pred),
            "calibrated_metrics":metrics(y,inner_pred),
            "optimization":opt_history,
        },
        "outer":{
            "uniform_metrics":metrics(ko,uniform_pred),
            "weighted_unadjusted_metrics":metrics(ko,weighted_pred),
            "calibrated_metrics":metrics(ko,outer_pred),
            "paired_vs_uniform":transitions(ko,uniform_pred,outer_pred),
            "paired_vs_unadjusted_weighted":transitions(ko,weighted_pred,outer_pred),
        },
        "history":hist,
    }
    write_json(args.output/"report.json",report)
    np.savez_compressed(
        args.output/"outer-class-bias-predictions.npz",
        global_index=best_outer["global_index"],k=ko,
        probability=outer_q.astype(np.float32),predicted=outer_pred,
        class_bias=bias.astype(np.float64)
    )

    u=report["outer"]["uniform_metrics"]
    w=report["outer"]["weighted_unadjusted_metrics"]
    c=report["outer"]["calibrated_metrics"]
    lines=[
        "# V27.3 class-bias calibration","",
        f"- inner NLL: **{report['inner']['nll_raw']:.6f} → {report['inner']['nll_calibrated']:.6f}**",
        "- learned centered class biases: "+", ".join(f"K{i}={x:+.4f}" for i,x in enumerate(bias)),"",
        "| Measure | Uniform | Weighted raw | Class-calibrated |",
        "|---|---:|---:|---:|",
        f"| exact global | {100*u['exact']:.3f}% | {100*w['exact']:.3f}% | {100*c['exact']:.3f}% |",
        f"| exact poly | {100*u['poly_exact']:.3f}% | {100*w['poly_exact']:.3f}% | {100*c['poly_exact']:.3f}% |",
    ]
    for v in range(7):
        lines.append(
            f"| K{v} exact | {100*u['by_k'][str(v)]['exact']:.3f}% | "
            f"{100*w['by_k'][str(v)]['exact']:.3f}% | {100*c['by_k'][str(v)]['exact']:.3f}% |"
        )
    lines += [
        "",
        f"Paired class-calibrated vs uniform net: {report['outer']['paired_vs_uniform']['net_correct']:+d}.",
        f"Paired class-calibrated vs raw weighted net: {report['outer']['paired_vs_unadjusted_weighted']['net_correct']:+d}.",
        "",
        "All calibration parameters were fit on inner fold 0 only. Outer fold was evaluation-only.",
        "No automatic promotion.",
    ]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--best-outer",type=Path,required=True)
    p.add_argument("--uniform-outer",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    main(p.parse_args())
