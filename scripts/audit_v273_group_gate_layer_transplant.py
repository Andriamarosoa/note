"""Layer-group transplant audit for V27.3 learned-gate K3 regressions.

No training. Starting from the learned-gate Exact-K checkpoint, replace selected
weight groups with the matched always-on checkpoint weights, while preserving
the learned gate values at inference. Measures which trained weight groups are
sufficient to recover K3 regressions.

Groups are derived from graph ancestry, not guessed layer-name sweeps:
  candidate: all weight-bearing ancestors of candidate_context
  spectral: all weight-bearing ancestors of v240_dense_conv3
  head: v240_cardinality_hidden1/hidden2/cardinality
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import load_bundle,batch_inputs,require

SEED=16061+1003
FRAMES=31

def nested_base(model):
    import tensorflow as tf
    m=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(m)==1,f"expected one nested model, got {[x.name for x in m]}")
    return m[0]

def layer_weight_dict(model):
    return {layer.name:[np.asarray(w).copy() for w in layer.get_weights()]
            for layer in model.layers if layer.get_weights()}

def set_layer_weights(model,weights,names):
    for name in names:
        model.get_layer(name).set_weights(weights[name])

def ancestor_weight_layers(model,layer_name):
    import tensorflow as tf
    out=model.get_layer(layer_name).output
    src=tf.keras.utils.get_source_inputs(out)
    sub=tf.keras.Model(src,out)
    return {x.name for x in sub.layers if x.get_weights()}

def apply_gate(x,g):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    gg=np.asarray(g,np.float32)
    if gg.ndim==0: gg=np.full(len(cand),float(gg),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5] *= gg[:,None,None]
    stats[:,4:8] *= gg[:,None]
    return {"candidate_set":cand,"candidate_mask":np.asarray(x["candidate_mask"],np.float32),
            "cluster_stats":stats,"spectral_map":np.asarray(x["spectral_map"],np.float32)}

def forward(base,cache,outer,g):
    out=[];g=np.asarray(g)
    for s in range(0,len(outer),128):
        ids=outer[s:s+128];x=batch_inputs(cache,ids,FRAMES)
        gg=g if g.ndim==0 else g[s:s+len(ids)]
        out.append(np.asarray(base(apply_gate(x,gg),training=False),np.float32))
    return np.concatenate(out)

def load_A(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")));by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0});d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(global_index,np.int64),list(ids))

def metric(k,p,sel):
    y=k[sel];q=p[sel]
    return {"rows":int(len(y)),"correct":int(np.sum(q==y)),
            "exact":float(np.mean(q==y)) if len(y) else None,
            "under":int(np.sum(q<y)),"over":int(np.sum(q>y))}

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","always-weights","learned-weights","saved-predictions","cluster-rows","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64);k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    with np.load(a.saved_predictions,allow_pickle=False) as z:saved={x:np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer);np.testing.assert_array_equal(saved["k"],k)
    g=np.asarray(saved["learned_gate_gate"],np.float32)

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested_base(mc); bl=nested_base(ml)

    wc=layer_weight_dict(bc); wl=layer_weight_dict(bl)
    require(set(wc)==set(wl),"base layer inventory differs")
    for name in wc:
        require(len(wc[name])==len(wl[name]) and all(a.shape==b.shape for a,b in zip(wc[name],wl[name])),
                "weight shape mismatch "+name)

    candidate=ancestor_weight_layers(bl,"candidate_context")
    spectral=ancestor_weight_layers(bl,"v240_dense_conv3")
    head={"v240_cardinality_hidden1","v240_cardinality_hidden2","cardinality"}
    for group in (candidate,spectral,head):
        require(group <= set(wl),"group contains missing layers")
    require(not candidate & spectral and not candidate & head and not spectral & head,"group overlap")
    ungrouped=set(wl)-candidate-spectral-head

    groups={
        "none":set(),
        "candidate":candidate,
        "spectral":spectral,
        "head":head,
        "candidate_head":candidate|head,
        "spectral_head":spectral|head,
        "candidate_spectral":candidate|spectral,
        "all_three":candidate|spectral|head,
        "all_base":set(wl),
    }

    # Reference matched A/B decisions.
    A=saved["always_on_predicted"].astype(np.int32)
    D=saved["learned_gate_predicted"].astype(np.int32)
    k3reg=(k==3)&(A==3)&(D!=3)
    require(int(k3reg.sum())==48,"K3 regression population drift")
    cid,clusterA=load_A(a.cluster_rows,outer)
    scopes={"all":np.ones(len(k),bool),"K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":clusterA}

    result={"status":"completed","training":False,"cluster_A_id":int(cid),
            "groups":{
                "candidate_layers":sorted(candidate),
                "spectral_layers":sorted(spectral),
                "head_layers":sorted(head),
                "ungrouped_weight_layers":sorted(ungrouped),
                "params_by_group":{},
            },"interventions":{}}

    # Count unique parameters by layer weights.
    for label,names in {"candidate":candidate,"spectral":spectral,"head":head,"ungrouped":ungrouped}.items():
        result["groups"]["params_by_group"][label]=int(sum(np.prod(w.shape) for n in names for w in wl[n]))

    predictions={}
    for label,names in groups.items():
        # Reset learned base, then transplant the requested control groups.
        set_layer_weights(bl,wl,set(wl))
        set_layer_weights(bl,wc,names)
        prob=forward(bl,cache,outer,g);pred=prob.argmax(1).astype(np.int32)
        predictions[label]=pred
        result["interventions"][label]={
            "metrics":{s:metric(k,pred,sel) for s,sel in scopes.items()},
            "k3_regressions_recovered":int(np.sum(k3reg&(pred==3))),
            "k3_regressions_remaining":int(np.sum(k3reg&(pred!=3))),
            "transplanted_layer_count":len(names),
        }

    require(np.array_equal(predictions["none"],D),"learned checkpoint replay mismatch")
    # all_base with learned gate must reproduce factorial B (control base + learned gate), not A.
    result["sanity"]={
        "none_matches_learned_gate_decisions":True,
        "all_base_k3_exact":result["interventions"]["all_base"]["metrics"]["K3"]["exact"],
    }

    np.savez_compressed(a.output/"transplant-predictions.npz",global_index=outer,k=k,learned_gate=g,**predictions)
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Audit transplantation de couches — regressions K3","",
           "Aucun entrainement. Checkpoints always-on et learned-gate figes.","",
           f"Population K3 regressees A->D: **{int(k3reg.sum())}**.","",
           "| Transplant controle -> learned | K2 | K3 | K4 | A | K3 regressions recuperees |",
           "|---|---:|---:|---:|---:|---:|"]
    for label in groups:
        x=result["interventions"][label];m=x["metrics"]
        lines.append(f"| {label} | {100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | {x['k3_regressions_recovered']}/48 |")
    lines += ["","## Inventaire causal des groupes","",
              f"- candidate: {len(candidate)} couches, {result['groups']['params_by_group']['candidate']} parametres",
              f"- spectral: {len(spectral)} couches, {result['groups']['params_by_group']['spectral']} parametres",
              f"- head: {len(head)} couches, {result['groups']['params_by_group']['head']} parametres",
              f"- non groupes: {len(ungrouped)} couches, {result['groups']['params_by_group']['ungrouped']} parametres",
              "","Les resultats mesurent des transplantations de poids sur checkpoint fige; aucune architecture n'est proposee ici."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines),flush=True)

if __name__=="__main__":main()
