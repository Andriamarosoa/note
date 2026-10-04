"""No-training transplant audit: uniform anchor vs freeze_local_combo checkpoint.

Starting from freeze_local_combo, transplant one uniform layer/group at a time.
Measure low-K recovery and polyphonic preservation. Diagnostic only.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from scripts import train_v260_count_weighting as v260
from scripts.train_v273_group_gate_ab import build_model,metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_window_experiment import load_bundle,require

SEED=v260.SEED+1003

def nested_base(model):
    import tensorflow as tf
    xs=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(xs)==1,f"nested base mismatch: {[x.name for x in xs]}")
    return xs[0]

def wdict(model):
    return {x.name:[np.asarray(w).copy() for w in x.get_weights()]
            for x in model.layers if x.get_weights()}

def set_weights(model,source,names):
    for n in names:
        model.get_layer(n).set_weights(source[n])

def ancestor_weight_layers(model,layer_name):
    import tensorflow as tf
    out=model.get_layer(layer_name).output
    sub=tf.keras.Model(tf.keras.utils.get_source_inputs(out),out)
    return {x.name for x in sub.layers if x.get_weights()}

def pred(model,cache,outer):
    p=np.asarray(model.predict(batches(cache,outer),workers=0,max_queue_size=1,verbose=0),np.float32)
    require(p.shape==(len(outer),7) and np.allclose(p.sum(1),1,atol=1e-5),"bad probability")
    return p.argmax(1).astype(np.int32)

def load_npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.asarray(z[k]) for k in z.files}

def paired(k,u,f,x):
    reg=(u==k)&(f!=k)
    cor=(u!=k)&(f==k)
    low=k<=1
    poly=k>=2
    return {
      "recover_all_regressions":int(np.sum(reg&(x==k))),
      "lose_all_corrections":int(np.sum(cor&(x!=k))),
      "net_vs_freeze":int(np.sum(x==k)-np.sum(f==k)),
      "net_vs_uniform":int(np.sum(x==k)-np.sum(u==k)),
      "low_net_vs_freeze":int(np.sum((x==k)&low)-np.sum((f==k)&low)),
      "poly_net_vs_freeze":int(np.sum((x==k)&poly)-np.sum((f==k)&poly)),
      "recover_low_regressions":int(np.sum(reg&low&(x==k))),
      "lose_poly_corrections":int(np.sum(cor&poly&(x!=k))),
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights","uniform-predictions","freeze-predictions","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"overwrite"); a.output.mkdir(parents=True)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    up=load_npz(a.uniform_predictions); fp=load_npz(a.freeze_predictions)
    U=np.asarray(up["predicted"],np.int32)
    F=np.asarray(fp["freeze_local_combo_predicted"],np.int32)
    np.testing.assert_array_equal(up["global_index"],outer)
    np.testing.assert_array_equal(fp["global_index"],outer)
    np.testing.assert_array_equal(up["k"],k)
    np.testing.assert_array_equal(fp["k"],k)

    mu=build_model("learned_gate",SEED);mu.load_weights(a.uniform_weights)
    mf=build_model("learned_gate",SEED);mf.load_weights(a.freeze_weights)
    bu=nested_base(mu);bf=nested_base(mf)
    ub=wdict(bu);fb=wdict(bf);uo=wdict(mu);fo=wdict(mf)
    require(set(ub)==set(fb),"base inventory mismatch")
    require(set(uo)==set(fo),"outer inventory mismatch")

    candidate=ancestor_weight_layers(bf,"candidate_context")
    spectral=ancestor_weight_layers(bf,"v240_dense_conv3")
    head={"v240_cardinality_hidden1","v240_cardinality_hidden2","cardinality"}
    gate={x for x in fo if x.startswith("v273_gate_")}
    require(candidate<=set(fb) and spectral<=set(fb) and head<=set(fb),"group mismatch")

    frozen={"candidate_norm","candidate_hidden2","cardinality"}
    frozen_equal={}
    for name in sorted(frozen):
        frozen_equal[name]=all(np.array_equal(x,y) for x,y in zip(ub[name],fb[name]))
    require(all(frozen_equal.values()),"declared frozen layer differs from uniform")

    def reset():
        set_weights(bf,fb,set(fb));set_weights(mf,fo,set(fo))

    interventions={}
    preds={}
    def run(label,bnames=(),onames=()):
        reset();set_weights(bf,ub,set(bnames));set_weights(mf,uo,set(onames))
        x=pred(mf,cache,outer); preds[label]=x
        m=metrics(k,x);q=paired(k,U,F,x)
        interventions[label]={
          "base_layers":sorted(bnames),"outer_layers":sorted(onames),
          "metrics":m,"paired":q
        }

    run("none")
    # Whole groups.
    run("candidate",candidate)
    run("spectral",spectral)
    run("head",head)
    run("gate",(),gate)
    run("candidate_head",candidate|head)
    run("spectral_head",spectral|head)

    # Every individual changed base layer and gate layer.
    for name in sorted(fb):
        if name in frozen: continue
        run("base:"+name,{name},())
    for name in sorted(gate):
        run("outer:"+name,(),{name})

    require(np.array_equal(preds["none"],F),"freeze checkpoint replay mismatch")

    rows=[]
    for label,r in interventions.items():
        m=r["metrics"];q=r["paired"]
        rows.append({
          "label":label,
          "exact":m["exact"],"poly_exact":m["poly_exact"],
          "K0":m["by_k"]["0"]["exact"],"K1":m["by_k"]["1"]["exact"],
          "K2":m["by_k"]["2"]["exact"],"K3":m["by_k"]["3"]["exact"],
          "K4":m["by_k"]["4"]["exact"],"K5":m["by_k"]["5"]["exact"],
          **q
        })
    # Diagnostic ranking emphasizes low-K recovery, then poly preservation.
    rows.sort(key=lambda r:(r["low_net_vs_freeze"],r["poly_net_vs_freeze"],r["net_vs_freeze"]),reverse=True)

    ref={
      "uniform":metrics(k,U),"freeze":metrics(k,F),
      "paired":paired(k,U,F,F),
      "class_net":{str(v):int(np.sum((F==k)&(k==v))-np.sum((U==k)&(k==v))) for v in range(7)}
    }
    result={
      "status":"completed","training":False,
      "protocol":{"experiment":"v273_freeze_local_combo_transplant_audit",
                  "uniform_run":37208199201,"freeze_run":37233793945,
                  "outer_used_for_training":False,"automatic_promotion":False},
      "frozen_equal_to_uniform":frozen_equal,
      "groups":{"candidate":sorted(candidate),"spectral":sorted(spectral),"head":sorted(head),"gate":sorted(gate)},
      "reference":ref,"interventions":interventions,"ranking":rows
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
      "# Freeze-local-combo residual transplant audit","",
      f"Uniform: {100*ref['uniform']['exact']:.3f}% global / {100*ref['uniform']['poly_exact']:.3f}% poly.",
      f"Freeze local combo: {100*ref['freeze']['exact']:.3f}% global / {100*ref['freeze']['poly_exact']:.3f}% poly.",
      f"Class net vs uniform: {ref['class_net']}.","",
      "| intervention | global | poly | K0 | K1 | K2 | K3 | K4 | low net | poly net | total net | recover low reg | lose poly corr |",
      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for r in rows:
        lines.append(
          f"| {r['label']} | {100*r['exact']:.3f}% | {100*r['poly_exact']:.3f}% | "
          f"{100*r['K0']:.2f}% | {100*r['K1']:.2f}% | {100*r['K2']:.2f}% | {100*r['K3']:.2f}% | {100*r['K4']:.2f}% | "
          f"{r['low_net_vs_freeze']:+d} | {r['poly_net_vs_freeze']:+d} | {r['net_vs_freeze']:+d} | "
          f"{r['recover_low_regressions']} | {r['lose_poly_corrections']} |"
        )
    lines += ["","Diagnostic only. No training or promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
