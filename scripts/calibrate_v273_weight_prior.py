"""Calibrate the class-prior distortion of the best V27.3 weighted learned_gate.

Selection is strictly internal:
- train the fixed best weighting profile on inner_fit (folds 1/2/4),
- choose one scalar alpha on inner_val (fold 0) by minimum NLL,
- freeze alpha,
- apply p'_k proportional to p_k / weight_k**alpha to the already-trained
  final weighted model on outer fold 3.

Outer fold is evaluation-only and never participates in alpha selection.
"""
from __future__ import annotations

import argparse, json, hashlib
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.rebuild_v273_sources import digest, write_json
from scripts.train_v273_group_gate_ab import build_model, weight_hash, metrics, transitions
from scripts.train_v273_loss_weighting_ab import class_weight_table, batches
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import load_bundle, epoch_order, require

ARM="targeted_k23_half_k4_quarter"
ALPHAS=np.linspace(0.0,1.5,151,dtype=np.float64)


def adjusted(prob, table, alpha):
    prob=np.asarray(prob,np.float64)
    table=np.asarray(table,np.float64)
    score=np.log(np.clip(prob,1e-12,1.0))-float(alpha)*np.log(table)[None,:]
    score-=score.max(axis=1,keepdims=True)
    q=np.exp(score)
    q/=q.sum(axis=1,keepdims=True)
    return q


def nll(prob, y):
    y=np.asarray(y,np.int64)
    return float(-np.log(np.clip(prob[np.arange(len(y)),y],1e-12,1.0)).mean())


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
    curve=[]
    for alpha in ALPHAS:
        q=adjusted(prob,table,alpha)
        pred=q.argmax(1).astype(np.int32)
        curve.append({
            "alpha":float(alpha),
            "nll":nll(q,y),
            "exact":float(np.mean(pred==y)),
            "poly_exact":float(np.mean(pred[y>=2]==y[y>=2])),
        })
    best=min(curve,key=lambda r:(r["nll"],r["alpha"]))
    alpha=float(best["alpha"])
    inner_adj=adjusted(prob,table,alpha)
    inner_pred=inner_adj.argmax(1).astype(np.int32)

    best_report,best_outer=load_outer(args.best_outer)
    uni_report,uni_outer=load_outer(args.uniform_outer)
    require(np.array_equal(best_outer["global_index"],uni_outer["global_index"]),"outer index mismatch")
    require(np.array_equal(best_outer["k"],uni_outer["k"]),"outer target mismatch")
    ko=np.asarray(best_outer["k"],np.int32)
    weighted_prob=np.asarray(best_outer["probability"],np.float64)
    weighted_pred=np.asarray(best_outer["predicted"],np.int32)
    uniform_pred=np.asarray(uni_outer["predicted"],np.int32)
    final_table=np.asarray(best_report["protocol"]["weight_table"],np.float64)

    outer_adj=adjusted(weighted_prob,final_table,alpha)
    outer_pred=outer_adj.argmax(1).astype(np.int32)

    report={
        "status":"completed",
        "protocol":{
            "experiment":"v273_weight_prior_calibration",
            "base_arm":ARM,
            "calibration_formula":"log p'_k = log p_k - alpha * log(weight_k), then renormalize",
            "alpha_grid":{"start":0.0,"stop":1.5,"step":0.01,"count":151},
            "selection_metric":"minimum NLL on inner_val only",
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
            "final_weight_table":final_table.tolist(),
        },
        "selected_alpha":alpha,
        "inner":{
            "selected":best,
            "unadjusted_metrics":metrics(y,prob.argmax(1)),
            "adjusted_metrics":metrics(y,inner_pred),
            "curve":curve,
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
        args.output/"outer-calibrated-predictions.npz",
        global_index=best_outer["global_index"],k=ko,
        probability=outer_adj.astype(np.float32),predicted=outer_pred
    )

    u=report["outer"]["uniform_metrics"]
    w=report["outer"]["weighted_unadjusted_metrics"]
    c=report["outer"]["calibrated_metrics"]
    lines=[
        "# V27.3 weighted-prior calibration","",
        f"- selected alpha on inner fold 0: **{alpha:.2f}**",
        f"- inner NLL selected: **{best['nll']:.6f}**","",
        "| Measure | Uniform | Weighted raw | Calibrated |",
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
        f"Paired calibrated vs uniform net: {report['outer']['paired_vs_uniform']['net_correct']:+d}.",
        f"Paired calibrated vs raw weighted net: {report['outer']['paired_vs_unadjusted_weighted']['net_correct']:+d}.",
        "",
        "Alpha was selected on inner validation only. Outer fold was evaluation-only.",
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
