"""Targeted no-training combinatorial transplant audit around promising layers.

Tests all combinations of a small predeclared layer set identified by the
previous full transplant audit. No training and no outer-label tuning.
"""
from __future__ import annotations
import argparse,json,itertools
from pathlib import Path
import numpy as np

from scripts.train_v273_group_gate_ab import build_model, metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_window_experiment import load_bundle, require

SEED=16061+1003
LAYERS=("candidate_hidden2","cardinality","v240_cardinality_hidden2","candidate_norm","cluster_norm")

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,"nested base mismatch")
    return xs[0]

def wd(model):
    return {x.name:[np.asarray(w).copy() for w in x.get_weights()] for x in model.layers if x.get_weights()}

def set_layers(model,src,names):
    for n in names:model.get_layer(n).set_weights(src[n])

def load_npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def predict(m,cache,outer):
    return np.asarray(m.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32).argmax(1).astype(np.int32)

def trans(k,u,w,x):
    reg=(u==k)&(w!=k); cor=(u!=k)&(w==k)
    return dict(recover=int(np.sum(reg&(x==k))),lose=int(np.sum(cor&(x!=k))),
                net_w=int(np.sum(x==k)-np.sum(w==k)),net_u=int(np.sum(x==k)-np.sum(u==k)))

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","weighted-weights","uniform-predictions","weighted-predictions","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"overwrite"); a.output.mkdir(parents=True)
    cache,parts,_=load_bundle(a.bundle,a.config); outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    up=load_npz(a.uniform_predictions); wp=load_npz(a.weighted_predictions)
    U=np.asarray(up["predicted"],np.int32); W=np.asarray(wp["predicted"],np.int32)

    mu=build_model("learned_gate",SEED); mu.load_weights(a.uniform_weights)
    mw=build_model("learned_gate",SEED); mw.load_weights(a.weighted_weights)
    bu=nested_base(mu); bw=nested_base(mw)
    Ubase=wd(bu); Wbase=wd(bw)
    require(set(LAYERS)<=set(Wbase),"missing target layers")

    rows=[]
    for r in range(len(LAYERS)+1):
        for comb in itertools.combinations(LAYERS,r):
            set_layers(bw,Wbase,set(Wbase))
            set_layers(bw,Ubase,set(comb))
            p=predict(mw,cache,outer)
            m=metrics(k,p); q=trans(k,U,W,p)
            rows.append({
                "layers":list(comb),"n":len(comb),
                "exact":m["exact"],"poly_exact":m["poly_exact"],
                "K0":m["by_k"]["0"]["exact"],"K1":m["by_k"]["1"]["exact"],
                "K2":m["by_k"]["2"]["exact"],"K3":m["by_k"]["3"]["exact"],
                "K4":m["by_k"]["4"]["exact"],"K5":m["by_k"]["5"]["exact"],
                **q,
            })
    rows.sort(key=lambda x:(x["net_w"],x["poly_exact"],-x["n"]),reverse=True)
    result={"status":"completed","training":False,"layers":list(LAYERS),"rows":rows,
            "reference":{"uniform_exact":float(np.mean(U==k)),"weighted_exact":float(np.mean(W==k)),
                         "uniform_poly":float(np.mean(U[k>=2]==k[k>=2])),
                         "weighted_poly":float(np.mean(W[k>=2]==k[k>=2]))}}
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=["# Targeted combinatorial transplant audit","",
           f"Uniform: {100*result['reference']['uniform_exact']:.3f}% global / {100*result['reference']['uniform_poly']:.3f}% poly.",
           f"Weighted: {100*result['reference']['weighted_exact']:.3f}% global / {100*result['reference']['weighted_poly']:.3f}% poly.","",
           "| layers | global | poly | K1 | K2 | K3 | K4 | K5 | recover | lose | net vs weighted |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in rows[:20]:
        label="+".join(x["layers"]) if x["layers"] else "none"
        lines.append(f"| {label} | {100*x['exact']:.3f}% | {100*x['poly_exact']:.3f}% | "
                     f"{100*x['K1']:.2f}% | {100*x['K2']:.2f}% | {100*x['K3']:.2f}% | {100*x['K4']:.2f}% | {100*x['K5']:.2f}% | "
                     f"{x['recover']} | {x['lose']} | {x['net_w']:+d} |")
    lines += ["","Diagnostic only. No training or promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
if __name__=="__main__":main()
