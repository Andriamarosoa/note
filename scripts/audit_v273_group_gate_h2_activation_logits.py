"""Frozen activation/logit attribution audit for hidden2 neurons in V27.3 learned gate.

No training, no architecture change, no model selection.

For the published learned-gate and always-on checkpoints, measure:
- hidden2 activation means on K3 transition groups;
- final 7-class output weights per hidden2 neuron;
- per-neuron contribution to K3-vs-K2 and K3-vs-K4 logit margins;
- differences between learned and always-on checkpoints.

The audit focuses descriptively on all 96 neurons and reports the strongest
contributors. It does not promote any neuron or checkpoint.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import load_bundle, batch_inputs, require

SEED=16061+1003
FRAMES=31
H2="v240_cardinality_hidden2"
OUT="cardinality"

def nested(model):
    import tensorflow as tf
    xs=[l for l in model.layers if isinstance(l,tf.keras.Model)]
    require(len(xs)==1,"nested mismatch")
    return xs[0]

def apply_gate(x,g):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    gg=np.asarray(g,np.float32)
    if gg.ndim==0: gg=np.full(len(cand),float(gg),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5]*=gg[:,None,None]
    stats[:,4:8]*=gg[:,None]
    return {"candidate_set":cand,
            "candidate_mask":np.asarray(x["candidate_mask"],np.float32),
            "cluster_stats":stats,
            "spectral_map":np.asarray(x["spectral_map"],np.float32)}

def collect(base,cache,outer,gate):
    import tensorflow as tf
    h2m=tf.keras.Model(base.inputs,base.get_layer(H2).output)
    probs=[];acts=[]
    gate=np.asarray(gate)
    for s in range(0,len(outer),256):
        ids=outer[s:s+256]
        x=batch_inputs(cache,ids,FRAMES)
        gg=gate[s:s+len(ids)] if gate.ndim else gate
        xx=apply_gate(x,gg)
        acts.append(np.asarray(h2m(xx,training=False),np.float32))
        probs.append(np.asarray(base(xx,training=False),np.float32))
    return np.concatenate(acts),np.concatenate(probs)

def stats(x,sel):
    a=np.asarray(x)[sel]
    return {
      "n":int(len(a)),
      "mean":np.mean(a,axis=0).tolist(),
      "median":np.median(a,axis=0).tolist(),
      "std":np.std(a,axis=0).tolist(),
    }

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","always-weights","learned-weights","saved-predictions","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    with np.load(a.saved_predictions,allow_pickle=False) as z:
        saved={x:np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k)
    always_pred=saved["always_on_predicted"].astype(np.int32)
    learned_pred=saved["learned_gate_predicted"].astype(np.int32)
    gate=np.asarray(saved["learned_gate_gate"],np.float32)

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested(mc); bl=nested(ml)

    # Replay each model in its published condition.
    act_l,prob_l=collect(bl,cache,outer,gate)
    act_c,prob_c=collect(bc,cache,outer,1.0)
    np.testing.assert_array_equal(prob_l.argmax(1),learned_pred)
    np.testing.assert_array_equal(prob_c.argmax(1),always_pred)

    wl,blb=bl.get_layer(OUT).get_weights()
    wc,bcb=bc.get_layer(OUT).get_weights()
    require(wl.shape==(96,7) and wc.shape==(96,7),"unexpected output kernel")

    K3=k==3
    groups={
      "regressed":K3&(always_pred==3)&(learned_pred!=3),
      "corrected":K3&(always_pred!=3)&(learned_pred==3),
      "correct_both":K3&(always_pred==3)&(learned_pred==3),
      "wrong_both":K3&(always_pred!=3)&(learned_pred!=3),
      "learned_under":K3&(learned_pred<3),
      "learned_over":K3&(learned_pred>3),
    }
    require(int(groups["regressed"].sum())==48,"regression drift")

    # Per-neuron logit-margin contributions under each published checkpoint.
    margins={}
    for label,(j_other) in {"K3_vs_K2":2,"K3_vs_K4":4}.items():
        margins[label]={
          "learned_weight_margin":(wl[:,3]-wl[:,j_other]).tolist(),
          "control_weight_margin":(wc[:,3]-wc[:,j_other]).tolist(),
          "weight_margin_delta":((wl[:,3]-wl[:,j_other])-(wc[:,3]-wc[:,j_other])).tolist(),
        }

    result={
      "status":"completed",
      "training":False,
      "groups":{name:int(sel.sum()) for name,sel in groups.items()},
      "output_kernel":{
        "learned":wl.tolist(),
        "control":wc.tolist(),
        "learned_bias":blb.tolist(),
        "control_bias":bcb.tolist(),
      },
      "activation_stats":{
        "learned":{name:stats(act_l,sel) for name,sel in groups.items()},
        "control":{name:stats(act_c,sel) for name,sel in groups.items()},
      },
      "margins":margins,
      "neuron_summary":[],
      "limitations":[
        "This audit is descriptive attribution on frozen published checkpoints.",
        "Activation and output-weight contributions are not by themselves sufficient to establish a deployable correction.",
        "No neuron is selected or promoted from this analysis."
      ]
    }

    for j in range(96):
        row={"neuron":j}
        for name,sel in groups.items():
            al=act_l[sel,j]; ac=act_c[sel,j]
            row[name]={
              "learned_activation_mean":float(al.mean()) if len(al) else None,
              "control_activation_mean":float(ac.mean()) if len(ac) else None,
              "activation_delta":float(al.mean()-ac.mean()) if len(al) else None,
            }
        row["output"]={
          "learned_to_K2":float(wl[j,2]),"learned_to_K3":float(wl[j,3]),"learned_to_K4":float(wl[j,4]),
          "control_to_K2":float(wc[j,2]),"control_to_K3":float(wc[j,3]),"control_to_K4":float(wc[j,4]),
          "learned_margin_K3_K2":float(wl[j,3]-wl[j,2]),
          "learned_margin_K3_K4":float(wl[j,3]-wl[j,4]),
          "control_margin_K3_K2":float(wc[j,3]-wc[j,2]),
          "control_margin_K3_K4":float(wc[j,3]-wc[j,4]),
        }
        # Mean contribution on the 48 regressions.
        sel=groups["regressed"]
        row["regressed_contribution"]={
          "learned_K3_K2":float(np.mean(act_l[sel,j]*(wl[j,3]-wl[j,2]))),
          "learned_K3_K4":float(np.mean(act_l[sel,j]*(wl[j,3]-wl[j,4]))),
          "control_K3_K2":float(np.mean(act_c[sel,j]*(wc[j,3]-wc[j,2]))),
          "control_K3_K4":float(np.mean(act_c[sel,j]*(wc[j,3]-wc[j,4]))),
        }
        result["neuron_summary"].append(row)

    # Report known high-impact neurons from the independent per-neuron intervention audit,
    # plus top absolute changes in regression margin contribution.
    known=[42,2,68,16,5,37,14,32,56,90]
    def score(row):
        rc=row["regressed_contribution"]
        return abs(rc["learned_K3_K2"]-rc["control_K3_K2"])+abs(rc["learned_K3_K4"]-rc["control_K3_K4"])
    top=sorted(result["neuron_summary"],key=score,reverse=True)[:15]

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    lines=[
      "# Hidden2 activation / logit attribution audit","",
      "Aucun entrainement. Checkpoints publies figes.","",
      f"- K3 regressions: {result['groups']['regressed']}",
      f"- K3 corrections: {result['groups']['corrected']}","",
      "## Neurones deja identifies par intervention unitaire","",
      "| neuron | act delta regressed | learned margin K3-K2 | learned margin K3-K4 | contrib delta K3-K2 | contrib delta K3-K4 |",
      "|---:|---:|---:|---:|---:|---:|",
    ]
    for j in known:
        r=result["neuron_summary"][j]; rc=r["regressed_contribution"]; o=r["output"]
        lines.append(
          f"| {j} | {r['regressed']['activation_delta']:+.4f} | {o['learned_margin_K3_K2']:+.4f} | "
          f"{o['learned_margin_K3_K4']:+.4f} | "
          f"{rc['learned_K3_K2']-rc['control_K3_K2']:+.4f} | "
          f"{rc['learned_K3_K4']-rc['control_K3_K4']:+.4f} |"
        )
    lines+=["","## Top 15 par changement absolu de contribution sur les 48 regressions","",
            "| neuron | score contribution delta |","|---:|---:|"]
    for r in top:
        lines.append(f"| {r['neuron']} | {score(r):.5f} |")
    lines+=["","Descriptif uniquement; aucune promotion de neurone ou de poids."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
