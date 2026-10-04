"""No-training layer transplant audit: current frozen uniform vs best weighted learned_gate.

Starting from the best weighted checkpoint, transplant weights from the matched
uniform learned_gate checkpoint. Audit whole causal groups and every individual
weight-bearing layer. No model is trained and no outer label is used to select
or tune an intervention.

The purpose is diagnostic: identify which trained parameter groups carry
polyphonic gains and which groups cause the low-K/global regressions.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from scripts.train_v273_group_gate_ab import build_model, metrics
from scripts.train_v273_loss_weighting_ab import batches
from scripts.v273_window_experiment import load_bundle,require

SEED=16061+1003

def nested_base(model):
    import tensorflow as tf
    m=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(m)==1,f"expected one nested model, got {[x.name for x in m]}")
    return m[0]

def weights(model):
    return {x.name:[np.asarray(w).copy() for w in x.get_weights()]
            for x in model.layers if x.get_weights()}

def set_some(model, source, names):
    for name in names:
        model.get_layer(name).set_weights(source[name])

def ancestor_weight_layers(model,layer_name):
    import tensorflow as tf
    out=model.get_layer(layer_name).output
    sub=tf.keras.Model(tf.keras.utils.get_source_inputs(out),out)
    return {x.name for x in sub.layers if x.get_weights()}

def predict(model,cache,outer):
    return np.asarray(model.predict(
        batches(cache,outer),workers=0,max_queue_size=1,verbose=0
    ),np.float32)

def load_pred(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:np.asarray(z[k]) for k in z.files}

def load_old_clusters(path, global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={int(r["global_index"]):int(r["cluster"]) for r in rows}
    out=np.full(len(global_index),-1,np.int16)
    for i,g in enumerate(np.asarray(global_index,np.int64)):
        if int(g) in by: out[i]=by[int(g)]
    return out

def trans(k,u,w,x):
    k=np.asarray(k);u=np.asarray(u);w=np.asarray(w);x=np.asarray(x)
    reg=(u==k)&(w!=k)
    cor=(u!=k)&(w==k)
    return {
        "weighted_regressions_recovered":int(np.sum(reg&(x==k))),
        "weighted_corrections_lost":int(np.sum(cor&(x!=k))),
        "net_vs_weighted":int(np.sum(x==k)-np.sum(w==k)),
        "net_vs_uniform":int(np.sum(x==k)-np.sum(u==k)),
        "uniform_only_rows":int(reg.sum()),
        "weighted_only_rows":int(cor.sum()),
    }

def subset_metrics(k,p,mask):
    return metrics(np.asarray(k)[mask],np.asarray(p)[mask])

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","weighted-weights",
              "uniform-predictions","weighted-predictions","cluster-rows","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    a.output.mkdir(parents=True)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    up=load_pred(a.uniform_predictions)
    wp=load_pred(a.weighted_predictions)
    np.testing.assert_array_equal(up["global_index"],outer)
    np.testing.assert_array_equal(wp["global_index"],outer)
    np.testing.assert_array_equal(up["k"],k)
    np.testing.assert_array_equal(wp["k"],k)
    U=np.asarray(up["predicted"],np.int32)
    W=np.asarray(wp["predicted"],np.int32)
    require(int(np.sum((U==k)&(W!=k)))==358,"regression population drift")
    require(int(np.sum((U!=k)&(W==k)))==279,"correction population drift")

    mu=build_model("learned_gate",SEED);mu.load_weights(a.uniform_weights)
    mw=build_model("learned_gate",SEED);mw.load_weights(a.weighted_weights)
    bu=nested_base(mu);bw=nested_base(mw)

    Uouter=weights(mu); Wouter=weights(mw)
    Ubase=weights(bu); Wbase=weights(bw)
    require(set(Uouter)==set(Wouter),"outer weight inventory mismatch")
    require(set(Ubase)==set(Wbase),"base weight inventory mismatch")

    candidate=ancestor_weight_layers(bw,"candidate_context")
    spectral=ancestor_weight_layers(bw,"v240_dense_conv3")
    head={"v240_cardinality_hidden1","v240_cardinality_hidden2","cardinality"}
    require(candidate<=set(Wbase) and spectral<=set(Wbase) and head<=set(Wbase),"missing base group")
    gate={x for x in Wouter if x.startswith("v273_gate_")}
    require(gate=={"v273_gate_hidden","v273_gate_value"},f"unexpected gate layers {gate}")
    base_all=set(Wbase)

    old_cluster=load_old_clusters(a.cluster_rows,outer)
    scopes={
        "all":np.ones(len(k),bool),
        "low01":k<=1,
        "poly":k>=2,
        "K0":k==0,"K1":k==1,"K2":k==2,"K3":k==3,"K4":k==4,"K5":k==5,"K6":k==6,
        "old_cluster_B":old_cluster==0,
        "old_cluster_A":old_cluster==1,
    }

    # Reset helpers.
    def reset_weighted():
        set_some(mw,Wouter,set(Wouter))
        set_some(bw,Wbase,set(Wbase))

    def run(label, base_names=frozenset(), outer_names=frozenset()):
        reset_weighted()
        set_some(bw,Ubase,set(base_names))
        set_some(mw,Uouter,set(outer_names))
        prob=predict(mw,cache,outer)
        pred=prob.argmax(1).astype(np.int32)
        rec={
            "transplanted_base_layers":sorted(base_names),
            "transplanted_outer_layers":sorted(outer_names),
            "metrics":{s:subset_metrics(k,pred,m) for s,m in scopes.items()},
            "paired":trans(k,U,W,pred),
        }
        return rec,pred

    interventions={}
    predictions={}
    group_defs={
        "none":(set(),set()),
        "gate":(set(),gate),
        "candidate":(candidate,set()),
        "spectral":(spectral,set()),
        "head":(head,set()),
        "candidate_spectral":(candidate|spectral,set()),
        "candidate_head":(candidate|head,set()),
        "spectral_head":(spectral|head,set()),
        "gate_candidate":(candidate,gate),
        "gate_spectral":(spectral,gate),
        "gate_head":(head,gate),
        "all_base":(base_all,set()),
        "all_base_gate":(base_all,gate),
    }
    for label,(bn,on) in group_defs.items():
        rec,p=run(label,bn,on)
        interventions[label]=rec;predictions["group_"+label]=p
    require(np.array_equal(predictions["group_none"],W),"weighted checkpoint replay mismatch")
    require(np.array_equal(predictions["group_all_base_gate"],U),"full uniform transplant mismatch")

    # Individual layer audit: base and gate layers one at a time.
    individual={}
    for name in sorted(base_all):
        rec,p=run("base:"+name,{name},set())
        individual["base:"+name]=rec
        predictions["layer_base_"+name]=p
    for name in sorted(gate):
        rec,p=run("outer:"+name,set(),{name})
        individual["outer:"+name]=rec
        predictions["layer_outer_"+name]=p

    # Rank diagnostics, not selection.
    ranking=[]
    for label,rec in {**interventions,**individual}.items():
        m=rec["metrics"]
        q=rec["paired"]
        ranking.append({
            "label":label,
            "net_vs_weighted":q["net_vs_weighted"],
            "net_vs_uniform":q["net_vs_uniform"],
            "recover_regressions":q["weighted_regressions_recovered"],
            "lose_corrections":q["weighted_corrections_lost"],
            "global_exact":m["all"]["exact"],
            "poly_exact":m["poly"]["exact"],
            "K0_exact":m["K0"]["exact"],"K1_exact":m["K1"]["exact"],
            "K2_exact":m["K2"]["exact"],"K3_exact":m["K3"]["exact"],
            "K4_exact":m["K4"]["exact"],"K5_exact":m["K5"]["exact"],
            "old_A_exact":m["old_cluster_A"]["exact"],
        })
    ranking.sort(key=lambda r:(r["net_vs_weighted"],r["poly_exact"]),reverse=True)

    result={
        "status":"completed","training":False,
        "protocol":{
            "experiment":"v273_current_weighted_layer_transplant_audit",
            "uniform_reference_run":37208199201,
            "weighted_reference_run":37217693737,
            "outer_labels_used_for_training":False,
            "automatic_promotion":False,
        },
        "reference":{
            "uniform":{s:subset_metrics(k,U,m) for s,m in scopes.items()},
            "weighted":{s:subset_metrics(k,W,m) for s,m in scopes.items()},
            "paired":trans(k,U,W,W),
        },
        "groups":{
            "candidate_layers":sorted(candidate),"spectral_layers":sorted(spectral),
            "head_layers":sorted(head),"gate_layers":sorted(gate),
            "base_weight_layers":sorted(base_all),
        },
        "interventions":interventions,
        "individual_layers":individual,
        "ranking":ranking,
    }
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",global_index=outer,k=k,old_cluster=old_cluster,**predictions)

    u=result["reference"]["uniform"];w=result["reference"]["weighted"]
    lines=[
        "# Current weighted learned_gate — layer transplant audit","",
        "No training. Uniform weights are transplanted into the current best weighted checkpoint.","",
        f"Reference uniform: {100*u['all']['exact']:.3f}% global / {100*u['poly']['exact']:.3f}% poly.",
        f"Reference weighted: {100*w['all']['exact']:.3f}% global / {100*w['poly']['exact']:.3f}% poly.","",
        "| intervention | global | poly | K0 | K1 | K2 | K3 | K4 | K5 | A | recover 358 | lose 279 | net vs weighted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in group_defs:
        rec=interventions[label];m=rec["metrics"];q=rec["paired"]
        lines.append(
            f"| {label} | {100*m['all']['exact']:.3f}% | {100*m['poly']['exact']:.3f}% | "
            f"{100*m['K0']['exact']:.2f}% | {100*m['K1']['exact']:.2f}% | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | {100*m['K5']['exact']:.2f}% | "
            f"{100*m['old_cluster_A']['exact']:.2f}% | {q['weighted_regressions_recovered']} | "
            f"{q['weighted_corrections_lost']} | {q['net_vs_weighted']:+d} |"
        )
    lines += ["","## Top individual layers by net vs current weighted","",
              "| layer | net vs weighted | recover 358 | lose 279 | global | poly | K1 | K2 | K3 | K4 | A |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in [x for x in ranking if x["label"].startswith(("base:","outer:"))][:15]:
        lines.append(
            f"| {r['label']} | {r['net_vs_weighted']:+d} | {r['recover_regressions']} | {r['lose_corrections']} | "
            f"{100*r['global_exact']:.3f}% | {100*r['poly_exact']:.3f}% | {100*r['K1_exact']:.2f}% | "
            f"{100*r['K2_exact']:.2f}% | {100*r['K3_exact']:.2f}% | {100*r['K4_exact']:.2f}% | {100*r['old_A_exact']:.2f}% |"
        )
    lines += ["","Diagnostic only; no promotion and no training."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
