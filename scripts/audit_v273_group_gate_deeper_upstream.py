"""Frozen deeper upstream attribution for V27.3 learned-gate K3 regressions.

No training. No architecture change.

A) Spectral path deeper audit
   conv1 representation source (learned/control)
   x conv2 kernel source (learned/control)
   x conv2 bias source (learned/control)
   x conv3 kernel source (learned/control)
   -> conv3 ReLU -> spectral_max
   -> learned spectral_avg + learned candidate_context
   -> learned hidden1/hidden2/output

   conv3 bias is held learned because the previous frozen audit measured ~zero
   effect from replacing it alone.

B) Candidate path deeper audit
   cluster_hidden1 representation source (learned/control)
   x cluster_hidden2 kernel source (learned/control)
   x cluster_hidden2 bias source (learned/control)
   -> cluster_hidden2
   -> learned candidate_context Dense
   -> learned spectral_avg + learned spectral_max
   -> learned hidden1/hidden2/output

Focused hidden1 units:
  spectral: 23, 158, 135
  candidate: 73, 104, 109

Frozen attribution only; no promotion.
"""
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

from scripts.train_v88_regime_moe import FEATURE_DIM as V88_FEATURE_DIM
from scripts.train_v273_group_gate_ab import build_model
from scripts.v273_window_experiment import batch_inputs, load_bundle, require

SEED=16061+1003
FRAMES=31

CONV1="v240_dense_conv1"
CONV2="v240_dense_conv2"
CONV3="v240_dense_conv3"
AVG="v240_dense_global_average"
MAX="v240_dense_global_max"

CH1="cluster_hidden1"
CH2="cluster_hidden2"
CAND="candidate_context"

H1="v240_cardinality_hidden1"
H2="v240_cardinality_hidden2"
OUT="cardinality"

SPECTRAL_UNITS=[23,158,135]
CANDIDATE_UNITS=[73,104,109]
TARGET_H2=37


def nested(model):
    import tensorflow as tf
    xs=[l for l in model.layers if isinstance(l,tf.keras.Model)]
    require(len(xs)==1,"nested base mismatch")
    return xs[0]


def apply_gate(x,g):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    gg=np.asarray(g,np.float32)
    if gg.ndim==0:
        gg=np.full(len(cand),float(gg),np.float32)
    cand[:,:,V88_FEATURE_DIM:V88_FEATURE_DIM+5]*=gg[:,None,None]
    stats[:,4:8]*=gg[:,None]
    return {
        "candidate_set":cand,
        "candidate_mask":np.asarray(x["candidate_mask"],np.float32),
        "cluster_stats":stats,
        "spectral_map":np.asarray(x["spectral_map"],np.float32),
    }


def collect_small(base,cache,outer,gate,batch_size=256):
    import tensorflow as tf
    probe=tf.keras.Model(
        base.inputs,
        [base.get_layer(AVG).output,
         base.get_layer(MAX).output,
         base.get_layer(CH1).output,
         base.get_layer(CH2).output,
         base.get_layer(CAND).output,
         base.output],
    )
    buckets=[[] for _ in range(6)]
    ga=np.asarray(gate)
    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=gate if ga.ndim==0 else ga[s:s+len(ids)]
        vals=probe(apply_gate(x,g),training=False)
        for b,v in zip(buckets,vals):
            b.append(np.asarray(v,np.float32))
    return tuple(np.concatenate(x) for x in buckets)


def spectral_label(c1_ctl,w2_ctl,b2_ctl,w3_ctl):
    return (
        f"conv1_{'control' if c1_ctl else 'learned'}__"
        f"conv2W_{'control' if w2_ctl else 'learned'}__"
        f"conv2b_{'control' if b2_ctl else 'learned'}__"
        f"conv3W_{'control' if w3_ctl else 'learned'}"
    )


def collect_spectral_hybrids(
    learned_base,control_base,cache,outer,gate,
    w2_l,b2_l,w2_c,b2_c,w3_l,b3_l,w3_c,
    batch_size=20,
):
    import tensorflow as tf
    p_l=tf.keras.Model(learned_base.inputs,learned_base.get_layer(CONV1).output)
    p_c=tf.keras.Model(control_base.inputs,control_base.get_layer(CONV1).output)

    labels=[
        spectral_label(c1,w2,b2,w3)
        for c1 in (False,True)
        for w2 in (False,True)
        for b2 in (False,True)
        for w3 in (False,True)
    ]
    out={k:[] for k in labels}
    ga=np.asarray(gate)

    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=ga[s:s+len(ids)]
        c1_l=np.asarray(p_l(apply_gate(x,g),training=False),np.float32)
        c1_c=np.asarray(p_c(apply_gate(x,1.0),training=False),np.float32)

        for c1_ctl in (False,True):
            c1=c1_c if c1_ctl else c1_l
            c1t=tf.convert_to_tensor(c1)
            for w2_ctl in (False,True):
                W2=w2_c if w2_ctl else w2_l
                z2=tf.nn.conv2d(c1t,tf.convert_to_tensor(W2),strides=[1,1,1,1],padding="SAME")
                for b2_ctl in (False,True):
                    B2=b2_c if b2_ctl else b2_l
                    a2=tf.nn.relu(tf.nn.bias_add(z2,tf.convert_to_tensor(B2)))
                    for w3_ctl in (False,True):
                        W3=w3_c if w3_ctl else w3_l
                        z3=tf.nn.conv2d(a2,tf.convert_to_tensor(W3),strides=[1,1,1,1],padding="SAME")
                        a3=tf.nn.relu(tf.nn.bias_add(z3,tf.convert_to_tensor(b3_l)))
                        mx=tf.reduce_max(a3,axis=[1,2])
                        out[spectral_label(c1_ctl,w2_ctl,b2_ctl,w3_ctl)].append(
                            np.asarray(mx,np.float32)
                        )
    return {k:np.concatenate(v) for k,v in out.items()}


def candidate_label(h1_ctl,w2_ctl,b2_ctl):
    return (
        f"clusterH1_{'control' if h1_ctl else 'learned'}__"
        f"clusterH2W_{'control' if w2_ctl else 'learned'}__"
        f"clusterH2b_{'control' if b2_ctl else 'learned'}"
    )


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


def load_A(path,idx):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]);y=int(r["true_k"]);p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"u":0})
        d["n"]+=1;d["u"]+=p<y
    cid=max(by,key=lambda c:(by[c]["u"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(idx,np.int64),list(ids))


def summarize(k,pred,learned_pred,always_pred,scopes):
    k3=k==3
    reg48=k3&(always_pred==3)&(learned_pred!=3)
    corr29=k3&(always_pred!=3)&(learned_pred==3)
    learned_correct=k3&(learned_pred==3)
    return {
        "metrics":{n:metric(k,pred,s) for n,s in scopes.items()},
        "original_48_recovered":int(np.sum(reg48&(pred==3))),
        "original_29_corrections_lost":int(np.sum(corr29&(pred!=3))),
        "learned_correct_293_broken":int(np.sum(learned_correct&(pred!=3))),
        "k3_net_correct_vs_learned":int(np.sum(k3&(pred==3))-np.sum(k3&(learned_pred==3))),
        "changed_predictions_vs_learned":int(np.sum(pred!=learned_pred)),
    }


def downstream_context(ctx,w1,b1,w2,b2,ow,ob):
    h1=relu(ctx@w1+b1)
    h2=relu(h1@w2+b2)
    pred=(h2@ow+ob).argmax(1).astype(np.int32)
    return pred,h1,h2


def one_unit(ctx,unit,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow):
    new_h1=relu(ctx@w1[:,unit]+b1[unit])
    dh1=new_h1-h1_l[:,unit]
    new_h2_pre=h2_pre_l+dh1[:,None]*w2[unit,:][None,:]
    new_h2=relu(new_h2_pre)
    logits=logits_l+(new_h2-h2_l)@ow
    return logits.argmax(1).astype(np.int32),new_h1,new_h2


def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","always-weights","learned-weights","saved-predictions","cluster-rows","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    with np.load(a.saved_predictions,allow_pickle=False) as z:
        saved={x:np.asarray(z[x]) for x in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k)
    learned_pred=saved["learned_gate_predicted"].astype(np.int32)
    always_pred=saved["always_on_predicted"].astype(np.int32)
    gate=np.asarray(saved["learned_gate_gate"],np.float32)

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested(mc);bl=nested(ml)

    avg_c,max_c,ch1_c,ch2_c,cand_c,p_c=collect_small(bc,cache,outer,1.0)
    avg_l,max_l,ch1_l,ch2_l,cand_l,p_l=collect_small(bl,cache,outer,gate)
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    conv2W_c,conv2b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV2).get_weights()]
    conv2W_l,conv2b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV2).get_weights()]
    conv3W_c,conv3b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV3).get_weights()]
    conv3W_l,conv3b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV3).get_weights()]

    ch2W_c,ch2b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CH2).get_weights()]
    ch2W_l,ch2b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CH2).get_weights()]
    candW_l,candb_l=[np.asarray(x,np.float32) for x in bl.get_layer(CAND).get_weights()]

    w1,b1=[np.asarray(x,np.float32) for x in bl.get_layer(H1).get_weights()]
    w2,b2=[np.asarray(x,np.float32) for x in bl.get_layer(H2).get_weights()]
    ow,ob=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    spectral=collect_spectral_hybrids(
        bl,bc,cache,outer,gate,
        conv2W_l,conv2b_l,conv2W_c,conv2b_c,
        conv3W_l,conv3b_l,conv3W_c
    )
    spectral_base=spectral[spectral_label(False,False,False,False)]
    spec_err=float(np.max(np.abs(spectral_base-max_l)))
    require(spec_err<1e-5,f"spectral replay mismatch {spec_err}")

    candidate={}
    for h1_ctl in (False,True):
        h1=ch1_c if h1_ctl else ch1_l
        for w2_ctl in (False,True):
            W=ch2W_c if w2_ctl else ch2W_l
            for b2_ctl in (False,True):
                B=ch2b_c if b2_ctl else ch2b_l
                h2=relu(h1@W+B)
                candidate[candidate_label(h1_ctl,w2_ctl,b2_ctl)]=relu(h2@candW_l+candb_l)
    cand_err=float(np.max(np.abs(candidate[candidate_label(False,False,False)]-cand_l)))
    require(cand_err<1e-5,f"candidate replay mismatch {cand_err}")

    ctx_l=np.concatenate([avg_l,max_l,cand_l],axis=1)
    h1_l=relu(ctx_l@w1+b1)
    h2_pre_l=h1_l@w2+b2
    h2_l=relu(h2_pre_l)
    logits_l=h2_l@ow+ob
    np.testing.assert_array_equal(logits_l.argmax(1),learned_pred)

    cid,A=load_A(a.cluster_rows,outer)
    scopes={"global":np.ones(len(k),bool),"K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":A}
    reg48=(k==3)&(always_pred==3)&(learned_pred!=3)
    corr29=(k==3)&(always_pred!=3)&(learned_pred==3)
    require(int(reg48.sum())==48 and int(corr29.sum())==29,"pinned populations changed")

    result={
        "status":"completed","training":False,"cluster_A_id":int(cid),
        "replay":{"spectral_max":spec_err,"candidate_context":cand_err},
        "spectral_deep":{
            "factors":["conv1 representation","conv2 kernel","conv2 bias","conv3 kernel"],
            "conv3_bias":"held learned",
            "global_factorial":{},"focused_units":{}
        },
        "candidate_deep":{
            "factors":["cluster_hidden1 representation","cluster_hidden2 kernel","cluster_hidden2 bias"],
            "candidate_context_dense":"held learned",
            "global_factorial":{},"focused_units":{}
        },
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Spectral audit changes only spectral_max; spectral_avg and candidate_context stay learned.",
            "Candidate audit changes only candidate_context; spectral_avg and spectral_max stay learned.",
            "All cardinality hidden1/hidden2/output parameters stay learned-gate.",
            "Hybrid factor combinations may be out of distribution.",
            "No intervention is promoted."
        ]
    }

    # Spectral: 16 global hybrids.
    for label,mx in spectral.items():
        ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-h1_l[reg48,u])) for u in SPECTRAL_UNITS
        }
        result["spectral_deep"]["global_factorial"][label]=x

    for u in SPECTRAL_UNITS:
        result["spectral_deep"]["focused_units"][str(u)]={}
        for label,mx in spectral.items():
            ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
            pred,nh1,nh2=one_unit(ctx,u,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow)
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(nh1[reg48]-h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
            result["spectral_deep"]["focused_units"][str(u)][label]=x

    # Candidate: 8 global hybrids.
    for label,cand in candidate.items():
        ctx=np.concatenate([avg_l,max_l,cand],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-h1_l[reg48,u])) for u in CANDIDATE_UNITS
        }
        result["candidate_deep"]["global_factorial"][label]=x

    for u in CANDIDATE_UNITS:
        result["candidate_deep"]["focused_units"][str(u)]={}
        for label,cand in candidate.items():
            ctx=np.concatenate([avg_l,max_l,cand],axis=1)
            pred,nh1,nh2=one_unit(ctx,u,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow)
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(nh1[reg48]-h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
            result["candidate_deep"]["focused_units"][str(u)][label]=x

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Deeper upstream audit — conv1/conv2/conv3 and cluster_hidden1/2","",
        "Aucun entrainement.","",
        "## Spectral path: key global interventions","",
        "| conv1 | W2 | b2 | W3 | K2 | K3 | K4 | A | recup48 | perd29 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c1,w2c,b2c,w3c in [
        (False,False,False,False),
        (True,False,False,False),
        (False,True,False,False),
        (False,False,True,False),
        (False,False,False,True),
        (True,True,False,False),
        (True,False,False,True),
        (False,True,False,True),
        (True,True,False,True),
        (True,True,True,True),
    ]:
        label=spectral_label(c1,w2c,b2c,w3c)
        x=result["spectral_deep"]["global_factorial"][label];m=x["metrics"]
        lines.append(
            f"| {'C' if c1 else 'L'} | {'C' if w2c else 'L'} | {'C' if b2c else 'L'} | {'C' if w3c else 'L'} | "
            f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
        )

    lines += ["","### Spectral focused units","",
              "| unit | conv1 C recup/net | W2 C recup/net | W3 C recup/net | conv1+W2 recup/net | W2+W3 recup/net | all W/rep C recup/net |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for u in SPECTRAL_UNITS:
        rows=result["spectral_deep"]["focused_units"][str(u)]
        labels=[
            spectral_label(True,False,False,False),
            spectral_label(False,True,False,False),
            spectral_label(False,False,False,True),
            spectral_label(True,True,False,False),
            spectral_label(False,True,False,True),
            spectral_label(True,True,False,True),
        ]
        xs=[rows[l] for l in labels]
        vals=[f"{x['original_48_recovered']}/{x['k3_net_correct_vs_learned']:+d}" for x in xs]
        lines.append(f"| {u} | "+" | ".join(vals)+" |")

    lines += ["","## Candidate path: cluster_hidden1 x cluster_hidden2 W/b","",
              "| clusterH1 | W2 | b2 | K2 | K3 | K4 | A | recup48 | perd29 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for h,w,b in [
        (False,False,False),(True,False,False),(False,True,False),
        (False,False,True),(True,True,False),(True,True,True)
    ]:
        label=candidate_label(h,w,b)
        x=result["candidate_deep"]["global_factorial"][label];m=x["metrics"]
        lines.append(
            f"| {'C' if h else 'L'} | {'C' if w else 'L'} | {'C' if b else 'L'} | "
            f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
        )

    lines += ["","### Candidate focused units","",
              "| unit | clusterH1 C recup/net | H2W C recup/net | H1+H2W recup/net |",
              "|---:|---:|---:|---:|"]
    for u in CANDIDATE_UNITS:
        rows=result["candidate_deep"]["focused_units"][str(u)]
        a1=rows[candidate_label(True,False,False)]
        a2=rows[candidate_label(False,True,False)]
        a3=rows[candidate_label(True,True,False)]
        lines.append(
            f"| {u} | {a1['original_48_recovered']}/{a1['k3_net_correct_vs_learned']:+d} | "
            f"{a2['original_48_recovered']}/{a2['k3_net_correct_vs_learned']:+d} | "
            f"{a3['original_48_recovered']}/{a3['k3_net_correct_vs_learned']:+d} |"
        )

    lines += ["","Attribution sur checkpoints figes uniquement; aucune promotion automatique."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
