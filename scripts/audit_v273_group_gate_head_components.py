"""Fine-grained cardinality-head transplant audit.

No training. On the frozen learned-gate checkpoint, transplant control weights
at layer/component granularity for:
  v240_cardinality_hidden1
  v240_cardinality_hidden2
  cardinality (output)
including kernel-only and bias-only interventions.
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
HEAD=("v240_cardinality_hidden1","v240_cardinality_hidden2","cardinality")

def nested(model):
    import tensorflow as tf
    x=[l for l in model.layers if isinstance(l,tf.keras.Model)]
    require(len(x)==1,"nested base mismatch")
    return x[0]

def apply_gate(x,g):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    gg=np.asarray(g,np.float32)
    if gg.ndim==0: gg=np.full(len(cand),float(gg),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5]*=gg[:,None,None]
    stats[:,4:8]*=gg[:,None]
    return {"candidate_set":cand,"candidate_mask":np.asarray(x["candidate_mask"],np.float32),
            "cluster_stats":stats,"spectral_map":np.asarray(x["spectral_map"],np.float32)}

def forward(base,cache,outer,g):
    out=[];g=np.asarray(g)
    for s in range(0,len(outer),128):
        ids=outer[s:s+128];x=batch_inputs(cache,ids,FRAMES)
        gg=g[s:s+len(ids)] if g.ndim else g
        out.append(np.asarray(base(apply_gate(x,gg),training=False),np.float32))
    return np.concatenate(out)

def weights(base):
    return {n:[np.asarray(x).copy() for x in base.get_layer(n).get_weights()] for n in HEAD}

def reset(base,w):
    for n in HEAD: base.get_layer(n).set_weights(w[n])

def transplant(base,learned,control,spec):
    reset(base,learned)
    for layer,part in spec:
        cur=[x.copy() for x in learned[layer]]
        src=control[layer]
        if part=="all":
            cur=[x.copy() for x in src]
        elif part=="kernel":
            cur[0]=src[0].copy()
        elif part=="bias":
            require(len(cur)>=2,"bias missing "+layer);cur[1]=src[1].copy()
        else: raise ValueError(part)
        base.get_layer(layer).set_weights(cur)

def load_A(path,idx):
    rows=list(csv.DictReader(open(path,newline="")));by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"u":0});d["n"]+=1;d["u"]+=p<y
    cid=max(by,key=lambda c:(by[c]["u"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(idx,np.int64),list(ids))

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
    g=np.asarray(saved["learned_gate_gate"],np.float32)
    A0=saved["always_on_predicted"].astype(np.int32);D=saved["learned_gate_predicted"].astype(np.int32)

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested(mc);bl=nested(ml);wc=weights(bc);wl=weights(bl)

    reg=(k==3)&(A0==3)&(D!=3); learned_correct=(k==3)&(D==3)
    require(int(reg.sum())==48 and int(learned_correct.sum())==293,"K3 pinned populations changed")
    cid,A=load_A(a.cluster_rows,outer)
    scopes={"K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":A}

    specs={
        "none":[],
        "h1_all":[(HEAD[0],"all")],
        "h1_kernel":[(HEAD[0],"kernel")],
        "h1_bias":[(HEAD[0],"bias")],
        "h2_all":[(HEAD[1],"all")],
        "h2_kernel":[(HEAD[1],"kernel")],
        "h2_bias":[(HEAD[1],"bias")],
        "out_all":[(HEAD[2],"all")],
        "out_kernel":[(HEAD[2],"kernel")],
        "out_bias":[(HEAD[2],"bias")],
        "h1_h2":[(HEAD[0],"all"),(HEAD[1],"all")],
        "h1_out":[(HEAD[0],"all"),(HEAD[2],"all")],
        "h2_out":[(HEAD[1],"all"),(HEAD[2],"all")],
        "all_head":[(x,"all") for x in HEAD],
    }

    result={"status":"completed","training":False,"cluster_A_id":int(cid),
            "pinned":{"k3_regressions":48,"learned_correct_k3":293},
            "interventions":{}}
    preds={}
    for label,spec in specs.items():
        transplant(bl,wl,wc,spec)
        p=forward(bl,cache,outer,g).argmax(1).astype(np.int32);preds[label]=p
        result["interventions"][label]={
            "metrics":{s:metric(k,p,sel) for s,sel in scopes.items()},
            "original_48_recovered":int(np.sum(reg&(p==3))),
            "learned_correct_293_broken":int(np.sum(learned_correct&(p!=3))),
            "net_K3_vs_learned":int(np.sum((k==3)&(p==3))-293),
        }
    require(np.array_equal(preds["none"],D),"learned replay mismatch")

    np.savez_compressed(a.output/"head-component-predictions.npz",global_index=outer,k=k,gate=g,**preds)
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=["# Audit fin de la tete de cardinalite — K3","",
           "Aucun entrainement. Transplantations controle -> learned-gate.","",
           "| Intervention | K2 | K3 | K4 | A | 48 regressions recup. | 293 K3 learned corrects casses |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for label in specs:
        x=result["interventions"][label];m=x["metrics"]
        lines.append(f"| {label} | {100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']}/48 | {x['learned_correct_293_broken']}/293 |")
    lines += ["","Chaque ligne est une intervention de poids sur checkpoint fige; aucune modification d'architecture ni reentrainement."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines),flush=True)

if __name__=="__main__":main()
