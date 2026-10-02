"""Factorial audit of learned gate: training-weight shift vs inference gating.

No training. Replays the matched always_on and learned_gate models, extracts their
nested Exact-K networks, and crosses:
  weight state: always_on-trained vs learned_gate-trained
  input state: gate neutralized (1.0) vs learned gate values

This yields a 2x2 audit with identical held-out rows.
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

def load_A(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0}); d["n"]+=1; d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(global_index,np.int64),list(ids))

def apply_gate(x,g):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    gg=np.asarray(g,np.float32)
    if gg.ndim==0: gg=np.full(len(cand),float(gg),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5] *= gg[:,None,None]
    stats[:,4:8] *= gg[:,None]
    return {
        "candidate_set":cand,
        "candidate_mask":np.asarray(x["candidate_mask"],np.float32),
        "cluster_stats":stats,
        "spectral_map":np.asarray(x["spectral_map"],np.float32),
    }

def base_model(model):
    import tensorflow as tf
    nested=[x for x in model.layers if isinstance(x,tf.keras.Model)]
    require(len(nested)==1,f"expected one nested model, got {[x.name for x in nested]}")
    return nested[0]

def forward(base,cache,outer,gates):
    out=[]
    gates=np.asarray(gates)
    for s in range(0,len(outer),128):
        ids=outer[s:s+128]
        x=batch_inputs(cache,ids,FRAMES)
        g=gates if gates.ndim==0 else gates[s:s+len(ids)]
        out.append(np.asarray(base(apply_gate(x,g),training=False),np.float32))
    return np.concatenate(out)

def model_forward(model,cache,outer):
    out=[]
    for s in range(0,len(outer),128):
        ids=outer[s:s+128]
        x=batch_inputs(cache,ids,FRAMES)
        out.append(np.asarray(model(x,training=False),np.float32))
    return np.concatenate(out)

def gate_values(model,cache,outer):
    import tensorflow as tf
    gm=tf.keras.Model(model.inputs,model.get_layer("v273_gate_value").output)
    out=[]
    for s in range(0,len(outer),128):
        ids=outer[s:s+128]
        x=batch_inputs(cache,ids,FRAMES)
        out.append(np.asarray(gm(x,training=False),np.float32).reshape(-1))
    return np.concatenate(out)

def metric(k,p,sel):
    y=k[sel];q=p[sel]
    return {"rows":int(len(y)),"correct":int(np.sum(q==y)),
            "exact":float(np.mean(q==y)) if len(y) else None,
            "under":int(np.sum(q<y)),"over":int(np.sum(q>y))}

def transition(k,a,b,sel):
    y=k[sel];x=a[sel];z=b[sel]
    return {"rows":int(len(y)),
            "correct_before":int(np.sum(x==y)),"correct_after":int(np.sum(z==y)),
            "corrected":int(np.sum((x!=y)&(z==y))),
            "regressed":int(np.sum((x==y)&(z!=y))),
            "net_correct":int(np.sum(z==y)-np.sum(x==y)),
            "correct_to_under":int(np.sum((x==y)&(z<y))),
            "correct_to_over":int(np.sum((x==y)&(z>y))),
            "under_to_correct":int(np.sum((x<y)&(z==y))),
            "over_to_correct":int(np.sum((x>y)&(z==y)))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--always-weights",type=Path,required=True)
    ap.add_argument("--learned-weights",type=Path,required=True)
    ap.add_argument("--saved-predictions",type=Path,required=True)
    ap.add_argument("--cluster-rows",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    with np.load(a.saved_predictions,allow_pickle=False) as z:
        saved={x:np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k)

    always=build_model("always_on",SEED); always.load_weights(a.always_weights)
    learned=build_model("learned_gate",SEED); learned.load_weights(a.learned_weights)

    pa=model_forward(always,cache,outer); pd=model_forward(learned,cache,outer)
    A=pa.argmax(1).astype(np.int32); D=pd.argmax(1).astype(np.int32)
    np.testing.assert_array_equal(A,saved["always_on_predicted"])
    np.testing.assert_array_equal(D,saved["learned_gate_predicted"])

    g=gate_values(learned,cache,outer)
    require(float(np.max(np.abs(g-saved["learned_gate_gate"])))<1e-5,"gate replay mismatch")

    ba=base_model(always); bl=base_model(learned)
    # Factorial cells:
    # A: always-on weights, full pseudo-count signal
    # B: always-on weights, learned gate applied
    # C: learned-gate weights, full pseudo-count signal
    # D: learned-gate weights, learned gate applied
    pA=forward(ba,cache,outer,1.0)
    pB=forward(ba,cache,outer,g)
    pC=forward(bl,cache,outer,1.0)
    pD=forward(bl,cache,outer,g)
    preds={name:p.argmax(1).astype(np.int32) for name,p in
           (("A_control_weights_gate1",pA),("B_control_weights_learned_gate",pB),
            ("C_learned_weights_gate1",pC),("D_learned_weights_learned_gate",pD))}
    require(np.array_equal(preds["A_control_weights_gate1"],A),"A replay mismatch")
    require(np.array_equal(preds["D_learned_weights_learned_gate"],D),"D replay mismatch")

    cid,clusterA=load_A(a.cluster_rows,outer)
    scopes={"all":np.ones(len(k),bool),"K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":clusterA}
    result={"status":"completed","training":False,"cluster_A_id":int(cid),
            "factorial_cells":{
                "A":"always_on-trained Exact-K weights + gate=1",
                "B":"always_on-trained Exact-K weights + learned gate values",
                "C":"learned_gate-trained Exact-K weights + gate=1",
                "D":"learned_gate-trained Exact-K weights + learned gate values",
            },
            "cells":{},"contrasts":{},"k3_case_decomposition":{}}

    for name,p in preds.items():
        result["cells"][name]={scope:metric(k,p,sel) for scope,sel in scopes.items()}

    contrasts={
        "gate_effect_on_control_weights":("A_control_weights_gate1","B_control_weights_learned_gate"),
        "training_weight_effect_gate_neutral":("A_control_weights_gate1","C_learned_weights_gate1"),
        "gate_effect_on_learned_weights":("C_learned_weights_gate1","D_learned_weights_learned_gate"),
        "training_weight_effect_gate_active":("B_control_weights_learned_gate","D_learned_weights_learned_gate"),
        "total_matched_AB":("A_control_weights_gate1","D_learned_weights_learned_gate"),
    }
    for label,(x,y) in contrasts.items():
        result["contrasts"][label]={scope:transition(k,preds[x],preds[y],sel) for scope,sel in scopes.items()}

    # Exact decomposition of the 48 K3 A->D regressions.
    reg=(k==3)&(preds["A_control_weights_gate1"]==3)&(preds["D_learned_weights_learned_gate"]!=3)
    B=preds["B_control_weights_learned_gate"]; C=preds["C_learned_weights_gate1"]; DD=preds["D_learned_weights_learned_gate"]
    result["k3_case_decomposition"]={
        "A_correct_D_wrong":int(reg.sum()),
        "B_control_weights_with_gate_still_correct":int(np.sum(reg&(B==3))),
        "C_learned_weights_gate_neutral_still_correct":int(np.sum(reg&(C==3))),
        "both_B_and_C_correct":int(np.sum(reg&(B==3)&(C==3))),
        "B_wrong_C_correct":int(np.sum(reg&(B!=3)&(C==3))),
        "B_correct_C_wrong":int(np.sum(reg&(B==3)&(C!=3))),
        "B_wrong_C_wrong":int(np.sum(reg&(B!=3)&(C!=3))),
        "D_under":int(np.sum(reg&(DD<3))),
        "D_over":int(np.sum(reg&(DD>3))),
    }

    np.savez_compressed(a.output/"factorial-predictions.npz",global_index=outer,k=k,learned_gate=g,**preds)
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Audit factoriel learned gate: poids entraines vs gate inference","",
           "Aucun entrainement. Deux checkpoints figes, memes lignes externes.","",
           "| Cellule | K2 | K3 | K4 | A |",
           "|---|---:|---:|---:|---:|"]
    for name in preds:
        m=result["cells"][name]
        lines.append(f"| {name} | {100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% |")
    lines += ["","## Contrastes K3",""]
    for label in contrasts:
        q=result["contrasts"][label]["K3"]
        lines.append(f"- **{label}**: {q['correct_before']} -> {q['correct_after']} corrects; corriges {q['corrected']}, regressions {q['regressed']}, net {q['net_correct']:+d}.")
    d=result["k3_case_decomposition"]
    lines += ["","## Decomposition des regressions K3 A -> D","",
              f"- Total: {d['A_correct_D_wrong']}",
              f"- Avec poids controle + gate appris (B), encore corrects: {d['B_control_weights_with_gate_still_correct']}",
              f"- Avec poids learned + gate neutralise (C), encore corrects: {d['C_learned_weights_gate_neutral_still_correct']}",
              f"- B faux / C correct: {d['B_wrong_C_correct']}",
              f"- B correct / C faux: {d['B_correct_C_wrong']}",
              f"- B faux / C faux: {d['B_wrong_C_wrong']}",
              "","Ces contrastes separant poids et entree gate sont des interventions sur checkpoints figes; ils ne supposent aucune nouvelle architecture."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__": main()
