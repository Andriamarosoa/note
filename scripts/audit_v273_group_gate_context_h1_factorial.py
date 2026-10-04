"""Frozen 2x2x2 attribution audit for v240_cardinality_hidden1 formation.

No training. No architecture change.

hidden1 = ReLU(cardinality_context @ W1 + b1)

Factors:
  C: cardinality_context source (learned-gate vs always-on)
  W: hidden1 kernel source (learned-gate vs always-on)
  B: hidden1 bias source (learned-gate vs always-on)

Everything downstream of hidden1 is held at the learned-gate checkpoint:
  learned hidden2 kernel+bias and learned final 7-class output kernel+bias.

The audit includes:
  1) global 2x2x2 reconstruction of all 192 hidden1 units;
  2) per-hidden1-unit substitutions while the other 191 units remain learned;
  3) direct measurement of how each intervention changes hidden2 neuron 37,
     the strongest H1-driven K3 regression path identified by the previous audit.

Frozen attribution only; no selection/promotion.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import batch_inputs, load_bundle, require

SEED = 16061 + 1003
FRAMES = 31
CTX = "v240_cardinality_context"
H1 = "v240_cardinality_hidden1"
H2 = "v240_cardinality_hidden2"
OUT = "cardinality"
TARGET_H2 = 37

def nested(model):
    import tensorflow as tf
    xs=[layer for layer in model.layers if isinstance(layer,tf.keras.Model)]
    require(len(xs)==1,"nested base mismatch")
    return xs[0]

def apply_gate(x, gate):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    g=np.asarray(gate,np.float32)
    if g.ndim==0:
        g=np.full(len(cand),float(g),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5]*=g[:,None,None]
    stats[:,4:8]*=g[:,None]
    return {
        "candidate_set":cand,
        "candidate_mask":np.asarray(x["candidate_mask"],np.float32),
        "cluster_stats":stats,
        "spectral_map":np.asarray(x["spectral_map"],np.float32),
    }

def collect(base, cache, outer, gate, batch_size=256):
    import tensorflow as tf
    probe=tf.keras.Model(
        base.inputs,
        [base.get_layer(CTX).output,
         base.get_layer(H1).output,
         base.get_layer(H2).output,
         base.output],
    )
    cs,h1s,h2s,ps=[],[],[],[]
    ga=np.asarray(gate)
    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=gate if ga.ndim==0 else ga[s:s+len(ids)]
        c,h1,h2,p=probe(apply_gate(x,g),training=False)
        cs.append(np.asarray(c,np.float32))
        h1s.append(np.asarray(h1,np.float32))
        h2s.append(np.asarray(h2,np.float32))
        ps.append(np.asarray(p,np.float32))
    return np.concatenate(cs),np.concatenate(h1s),np.concatenate(h2s),np.concatenate(ps)

def relu(x):
    return np.maximum(np.asarray(x,np.float32),0.0)

def metric(k,pred,sel):
    y=k[sel];p=pred[sel]
    return {
        "rows":int(len(y)),
        "correct":int(np.sum(p==y)),
        "exact":float(np.mean(p==y)) if len(y) else None,
        "under":int(np.sum(p<y)),
        "over":int(np.sum(p>y)),
    }

def load_cluster_a(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0})
        d["n"]+=1;d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(global_index,np.int64),list(ids))

def summarize(k,pred,learned_pred,always_pred,scopes):
    k3=k==3
    reg48=k3&(always_pred==3)&(learned_pred!=3)
    corr29=k3&(always_pred!=3)&(learned_pred==3)
    learned_correct=k3&(learned_pred==3)
    return {
        "metrics":{name:metric(k,pred,sel) for name,sel in scopes.items()},
        "original_48_recovered":int(np.sum(reg48&(pred==3))),
        "original_29_corrections_lost":int(np.sum(corr29&(pred!=3))),
        "learned_correct_293_broken":int(np.sum(learned_correct&(pred!=3))),
        "k3_net_correct_vs_learned":int(
            np.sum(k3&(pred==3))-np.sum(k3&(learned_pred==3))
        ),
        "changed_predictions_vs_learned":int(np.sum(pred!=learned_pred)),
    }

def source(flag):
    return "control" if flag else "learned"

def predict_from_h1(h1,w2,b2,out_w,out_b):
    h2=relu(np.asarray(h1,np.float32)@w2+b2)
    logits=h2@out_w+out_b
    return logits.argmax(1).astype(np.int32),h2

def main():
    ap=argparse.ArgumentParser()
    for name in (
        "bundle","config","always-weights","learned-weights",
        "saved-predictions","cluster-rows","output"
    ):
        ap.add_argument("--"+name,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)

    with np.load(a.saved_predictions,allow_pickle=False) as z:
        saved={key:np.asarray(z[key]) for key in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k)
    learned_pred=saved["learned_gate_predicted"].astype(np.int32)
    always_pred=saved["always_on_predicted"].astype(np.int32)
    gate=np.asarray(saved["learned_gate_gate"],np.float32)

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested(mc);bl=nested(ml)

    ctx_c,h1_c,h2_c,p_c=collect(bc,cache,outer,1.0)
    ctx_l,h1_l,h2_l,p_l=collect(bl,cache,outer,gate)
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    w1_c,b1_c=[np.asarray(x,np.float32) for x in bc.get_layer(H1).get_weights()]
    w1_l,b1_l=[np.asarray(x,np.float32) for x in bl.get_layer(H1).get_weights()]
    w2_l,b2_l=[np.asarray(x,np.float32) for x in bl.get_layer(H2).get_weights()]
    out_w_l,out_b_l=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    require(w1_l.shape==w1_c.shape,"hidden1 kernel mismatch")
    require(w1_l.shape[1]==192,f"hidden1 width changed {w1_l.shape}")
    require(b1_l.shape==(192,),"hidden1 bias mismatch")
    require(ctx_l.shape==ctx_c.shape==(len(outer),w1_l.shape[0]),f"context mismatch {ctx_l.shape}")

    h1_l_replay=relu(ctx_l@w1_l+b1_l)
    h1_c_replay=relu(ctx_c@w1_c+b1_c)
    e_l=float(np.max(np.abs(h1_l_replay-h1_l)))
    e_c=float(np.max(np.abs(h1_c_replay-h1_c)))
    require(e_l<1e-5,f"learned hidden1 replay mismatch {e_l}")
    require(e_c<1e-5,f"control hidden1 replay mismatch {e_c}")

    pred_replay,h2_replay=predict_from_h1(h1_l,w2_l,b2_l,out_w_l,out_b_l)
    np.testing.assert_array_equal(pred_replay,learned_pred)
    require(float(np.max(np.abs(h2_replay-h2_l)))<1e-5,"learned downstream replay mismatch")

    cid,A=load_cluster_a(a.cluster_rows,outer)
    scopes={
        "global":np.ones(len(k),bool),
        "K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":A
    }
    reg48=(k==3)&(always_pred==3)&(learned_pred!=3)
    corr29=(k==3)&(always_pred!=3)&(learned_pred==3)
    require(int(reg48.sum())==48,"48 population drift")
    require(int(corr29.sum())==29,"29 population drift")

    result={
        "status":"completed",
        "training":False,
        "cluster_A_id":int(cid),
        "target_hidden2_neuron":TARGET_H2,
        "shapes":{
            "context_dim":int(ctx_l.shape[1]),
            "hidden1":192,
            "hidden2":96,
        },
        "replay":{"learned_h1_max_abs":e_l,"control_h1_max_abs":e_c},
        "global_factorial_2x2x2":{},
        "per_hidden1_unit_factorial":[],
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Control cardinality_context is the published always-on context and includes upstream spectral/candidate-context differences.",
            "Everything downstream of hidden1 is fixed to learned-gate parameters.",
            "Per-unit interventions replace only one hidden1 scalar while the other 191 remain learned.",
            "Hybrid C/W/B combinations can be outside training distribution and are attribution tests only.",
            "No unit or hybrid is selected or promoted.",
        ],
    }

    # Global 2x2x2.
    for c_ctl in (False,True):
        ctx=ctx_c if c_ctl else ctx_l
        for w_ctl in (False,True):
            w1=w1_c if w_ctl else w1_l
            for b_ctl in (False,True):
                b1=b1_c if b_ctl else b1_l
                h1=relu(ctx@w1+b1)
                pred,h2=predict_from_h1(h1,w2_l,b2_l,out_w_l,out_b_l)
                label=f"C_{source(c_ctl)}__W_{source(w_ctl)}__B_{source(b_ctl)}"
                x=summarize(k,pred,learned_pred,always_pred,scopes)
                x["target_h2_37"]={
                    "regressed_mean":float(np.mean(h2[reg48,TARGET_H2])),
                    "delta_vs_learned_regressed":float(np.mean(h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2])),
                    "corrected_mean":float(np.mean(h2[corr29,TARGET_H2])),
                }
                result["global_factorial_2x2x2"][label]=x

    # Per hidden1 unit.
    combos=[
        (True,False,False),   # context only
        (False,True,False),   # W only
        (False,False,True),   # bias only
        (True,True,False),    # C+W
        (True,False,True),    # C+B
        (False,True,True),    # W+B
        (True,True,True),     # all
    ]

    # Learned baseline downstream logits and target-h2 preactivation decomposition.
    baseline_h2_pre=h1_l@w2_l+b2_l
    baseline_h2=relu(baseline_h2_pre)
    baseline_logits=baseline_h2@out_w_l+out_b_l

    for j in range(192):
        row={
            "unit":int(j),
            "context_to_unit_kernel_l2_delta":float(np.linalg.norm(w1_l[:,j]-w1_c[:,j])),
            "bias_delta":float(b1_l[j]-b1_c[j]),
            "learned_activation_regressed_mean":float(np.mean(h1_l[reg48,j])),
            "control_activation_regressed_mean":float(np.mean(h1_c[reg48,j])),
            "activation_delta_regressed_mean":float(np.mean(h1_l[reg48,j]-h1_c[reg48,j])),
            "downstream_weight_to_h2_37":float(w2_l[j,TARGET_H2]),
            "conditions":{},
        }
        for c_ctl,w_ctl,b_ctl in combos:
            ctx=ctx_c if c_ctl else ctx_l
            wcol=w1_c[:,j] if w_ctl else w1_l[:,j]
            bias=b1_c[j] if b_ctl else b1_l[j]
            new_h1_j=relu(ctx@wcol+bias)

            # Exact update from changing only hidden1 unit j:
            dh1=new_h1_j-h1_l[:,j]
            new_h2_pre=baseline_h2_pre+dh1[:,None]*w2_l[j,:][None,:]
            new_h2=relu(new_h2_pre)
            logits=baseline_logits+(new_h2-baseline_h2)@out_w_l
            pred=logits.argmax(1).astype(np.int32)

            label=f"C_{source(c_ctl)}__W_{source(w_ctl)}__B_{source(b_ctl)}"
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["target_h2_37"]={
                "regressed_activation_mean":float(np.mean(new_h2[reg48,TARGET_H2])),
                "delta_vs_learned_regressed":float(np.mean(new_h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2])),
            }
            row["conditions"][label]=x
        result["per_hidden1_unit_factorial"].append(row)

    # Descriptive rankings for main causal factors.
    labels={
        "context_only":"C_control__W_learned__B_learned",
        "kernel_only":"C_learned__W_control__B_learned",
        "bias_only":"C_learned__W_learned__B_control",
        "context_kernel":"C_control__W_control__B_learned",
        "all":"C_control__W_control__B_control",
    }
    result["descriptive_top"]={}
    for key,label in labels.items():
        ranked=sorted(
            result["per_hidden1_unit_factorial"],
            key=lambda r:(
                r["conditions"][label]["k3_net_correct_vs_learned"],
                r["conditions"][label]["original_48_recovered"],
                -r["conditions"][label]["learned_correct_293_broken"],
            ),
            reverse=True,
        )
        result["descriptive_top"][key]=[
            {
                "unit":r["unit"],
                "net_K3":r["conditions"][label]["k3_net_correct_vs_learned"],
                "recovered48":r["conditions"][label]["original_48_recovered"],
                "lost29":r["conditions"][label]["original_29_corrections_lost"],
                "K2":r["conditions"][label]["metrics"]["K2"]["exact"],
                "K3":r["conditions"][label]["metrics"]["K3"]["exact"],
                "K4":r["conditions"][label]["metrics"]["K4"]["exact"],
                "A":r["conditions"][label]["metrics"]["cluster_A"]["exact"],
                "h2_37_delta":r["conditions"][label]["target_h2_37"]["delta_vs_learned_regressed"],
                "w_to_h2_37":r["downstream_weight_to_h2_37"],
            }
            for r in ranked[:25]
        ]

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Cardinality-context x hidden1-kernel x hidden1-bias audit","",
        "Aucun entrainement. Tout l'aval de hidden1 reste learned-gate.","",
        "## Factoriel global 2x2x2","",
        "| context | W1 | b1 | K2 | K3 | K4 | A | recup48 | perd29 | delta h2[37] regressed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c_ctl in (False,True):
        for w_ctl in (False,True):
            for b_ctl in (False,True):
                label=f"C_{source(c_ctl)}__W_{source(w_ctl)}__B_{source(b_ctl)}"
                x=result["global_factorial_2x2x2"][label];m=x["metrics"]
                lines.append(
                    f"| {source(c_ctl)} | {source(w_ctl)} | {source(b_ctl)} | "
                    f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | "
                    f"{100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | "
                    f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} | "
                    f"{x['target_h2_37']['delta_vs_learned_regressed']:+.4f} |"
                )

    for key,title in (
        ("context_only","context control uniquement"),
        ("kernel_only","W1 colonne control uniquement"),
        ("bias_only","bias control uniquement"),
        ("context_kernel","context + W1 control"),
    ):
        lines += [
            "",f"## Top 12 hidden1 units — {title}","",
            "| unit | net K3 | recup48 | perd29 | K2 | K3 | K4 | A | delta h2[37] | W2->37 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in result["descriptive_top"][key][:12]:
            lines.append(
                f"| {r['unit']} | {r['net_K3']:+d} | {r['recovered48']} | {r['lost29']} | "
                f"{100*r['K2']:.2f}% | {100*r['K3']:.2f}% | {100*r['K4']:.2f}% | "
                f"{100*r['A']:.2f}% | {r['h2_37_delta']:+.4f} | {r['w_to_h2_37']:+.4f} |"
            )

    lines += ["","Attribution sur checkpoints figes uniquement; aucune promotion automatique."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
