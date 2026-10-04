"""Frozen attribution of conv1 formation and cluster_hidden1 formation.

No training. No architecture change.

A) Spectral conv1 formation
   dense_plus_coordinates representation source (learned/control)
   x conv1 kernel source (learned/control)
   x conv1 bias source (learned/control)
   -> conv1 ReLU
   -> learned conv2 -> learned conv3 -> spectral_max
   -> learned spectral_avg + learned candidate_context
   -> learned cardinality hidden1/hidden2/output

B) Candidate cluster_hidden1 formation
   cluster_norm representation source (learned/control)
   x cluster_hidden1 kernel source (learned/control)
   x cluster_hidden1 bias source (learned/control)
   -> cluster_hidden1 ReLU
   -> learned cluster_hidden2 -> learned candidate_context Dense
   -> learned spectral branches
   -> learned cardinality hidden1/hidden2/output

Focused cardinality hidden1 units:
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

DENSE_IN="v240_dense_plus_coordinates"
CONV1="v240_dense_conv1"
CONV2="v240_dense_conv2"
CONV3="v240_dense_conv3"
AVG="v240_dense_global_average"
MAX="v240_dense_global_max"

CLUSTER_NORM="cluster_norm"
CLUSTER_H1="cluster_hidden1"
CLUSTER_H2="cluster_hidden2"
CAND="candidate_context"

CARD_H1="v240_cardinality_hidden1"
CARD_H2="v240_cardinality_hidden2"
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
         base.get_layer(CLUSTER_NORM).output,
         base.get_layer(CLUSTER_H1).output,
         base.get_layer(CLUSTER_H2).output,
         base.get_layer(CAND).output,
         base.output],
    )
    buckets=[[] for _ in range(7)]
    ga=np.asarray(gate)
    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=gate if ga.ndim==0 else ga[s:s+len(ids)]
        vals=probe(apply_gate(x,g),training=False)
        for b,v in zip(buckets,vals):
            b.append(np.asarray(v,np.float32))
    return tuple(np.concatenate(x) for x in buckets)


def spectral_label(src_ctl,w_ctl,b_ctl):
    return (
        f"densein_{'control' if src_ctl else 'learned'}__"
        f"conv1W_{'control' if w_ctl else 'learned'}__"
        f"conv1b_{'control' if b_ctl else 'learned'}"
    )


def candidate_label(src_ctl,w_ctl,b_ctl):
    return (
        f"clusterNorm_{'control' if src_ctl else 'learned'}__"
        f"clusterH1W_{'control' if w_ctl else 'learned'}__"
        f"clusterH1b_{'control' if b_ctl else 'learned'}"
    )


def relu(x):
    return np.maximum(np.asarray(x,np.float32),0.0)


def metric(k,pred,sel):
    y=k[sel]; p=pred[sel]
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
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"u":0})
        d["n"]+=1; d["u"]+=p<y
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


def one_card_unit(ctx,unit,h1_l,h2pre_l,h2_l,logits_l,w1,b1,w2,ow):
    new_h1=relu(ctx@w1[:,unit]+b1[unit])
    dh1=new_h1-h1_l[:,unit]
    new_h2pre=h2pre_l+dh1[:,None]*w2[unit,:][None,:]
    new_h2=relu(new_h2pre)
    logits=logits_l+(new_h2-h2_l)@ow
    return logits.argmax(1).astype(np.int32),new_h1,new_h2


def collect_spectral_max_hybrids(
    learned_base,control_base,cache,outer,gate,
    c1W_l,c1b_l,c1W_c,c1b_c,
    c2W_l,c2b_l,c3W_l,c3b_l,
    batch_size=20,
):
    import tensorflow as tf
    p_l=tf.keras.Model(learned_base.inputs,learned_base.get_layer(DENSE_IN).output)
    p_c=tf.keras.Model(control_base.inputs,control_base.get_layer(DENSE_IN).output)

    out={
        spectral_label(src,w,b):[]
        for src in (False,True)
        for w in (False,True)
        for b in (False,True)
    }
    ga=np.asarray(gate)

    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=ga[s:s+len(ids)]
        din_l=np.asarray(p_l(apply_gate(x,g),training=False),np.float32)
        din_c=np.asarray(p_c(apply_gate(x,1.0),training=False),np.float32)

        for src_ctl in (False,True):
            din=tf.convert_to_tensor(din_c if src_ctl else din_l)
            for w_ctl in (False,True):
                W1=c1W_c if w_ctl else c1W_l
                z1=tf.nn.conv2d(din,tf.convert_to_tensor(W1),strides=[1,1,1,1],padding="SAME")
                for b_ctl in (False,True):
                    B1=c1b_c if b_ctl else c1b_l
                    a1=tf.nn.relu(tf.nn.bias_add(z1,tf.convert_to_tensor(B1)))
                    z2=tf.nn.conv2d(a1,tf.convert_to_tensor(c2W_l),strides=[1,1,1,1],padding="SAME")
                    a2=tf.nn.relu(tf.nn.bias_add(z2,tf.convert_to_tensor(c2b_l)))
                    z3=tf.nn.conv2d(a2,tf.convert_to_tensor(c3W_l),strides=[1,1,1,1],padding="SAME")
                    a3=tf.nn.relu(tf.nn.bias_add(z3,tf.convert_to_tensor(c3b_l)))
                    mx=tf.reduce_max(a3,axis=[1,2])
                    out[spectral_label(src_ctl,w_ctl,b_ctl)].append(np.asarray(mx,np.float32))
    return {k:np.concatenate(v) for k,v in out.items()}


def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","always-weights","learned-weights","saved-predictions","cluster-rows","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)

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

    mc=build_model("always_on",SEED); mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED); ml.load_weights(a.learned_weights)
    bc=nested(mc); bl=nested(ml)

    avg_c,max_c,cn_c,ch1_c,ch2_c,cand_c,p_c=collect_small(bc,cache,outer,1.0)
    avg_l,max_l,cn_l,ch1_l,ch2_l,cand_l,p_l=collect_small(bl,cache,outer,gate)
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    c1W_c,c1b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV1).get_weights()]
    c1W_l,c1b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV1).get_weights()]
    c2W_l,c2b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV2).get_weights()]
    c3W_l,c3b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV3).get_weights()]

    cl1W_c,cl1b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CLUSTER_H1).get_weights()]
    cl1W_l,cl1b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H1).get_weights()]
    cl2W_l,cl2b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H2).get_weights()]
    candW_l,candb_l=[np.asarray(x,np.float32) for x in bl.get_layer(CAND).get_weights()]

    w1,b1=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H1).get_weights()]
    w2,b2=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H2).get_weights()]
    ow,ob=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    spectral=collect_spectral_max_hybrids(
        bl,bc,cache,outer,gate,
        c1W_l,c1b_l,c1W_c,c1b_c,
        c2W_l,c2b_l,c3W_l,c3b_l
    )
    spec_err=float(np.max(np.abs(
        spectral[spectral_label(False,False,False)]-max_l
    )))
    require(spec_err<1e-5,f"spectral max replay mismatch {spec_err}")

    candidate={}
    for src_ctl in (False,True):
        src=cn_c if src_ctl else cn_l
        for w_ctl in (False,True):
            W=cl1W_c if w_ctl else cl1W_l
            for b_ctl in (False,True):
                B=cl1b_c if b_ctl else cl1b_l
                h1=relu(src@W+B)
                h2=relu(h1@cl2W_l+cl2b_l)
                cand=relu(h2@candW_l+candb_l)
                candidate[candidate_label(src_ctl,w_ctl,b_ctl)]=cand
    cand_err=float(np.max(np.abs(
        candidate[candidate_label(False,False,False)]-cand_l
    )))
    require(cand_err<1e-5,f"candidate replay mismatch {cand_err}")

    ctx_l=np.concatenate([avg_l,max_l,cand_l],axis=1)
    card_h1_l=relu(ctx_l@w1+b1)
    card_h2pre_l=card_h1_l@w2+b2
    card_h2_l=relu(card_h2pre_l)
    logits_l=card_h2_l@ow+ob
    np.testing.assert_array_equal(logits_l.argmax(1),learned_pred)

    cid,A=load_A(a.cluster_rows,outer)
    scopes={"global":np.ones(len(k),bool),"K2":k==2,"K3":k==3,"K4":k==4,"cluster_A":A}
    reg48=(k==3)&(always_pred==3)&(learned_pred!=3)
    corr29=(k==3)&(always_pred!=3)&(learned_pred==3)
    require(int(reg48.sum())==48 and int(corr29.sum())==29,"pinned populations changed")

    result={
        "status":"completed","training":False,"cluster_A_id":int(cid),
        "replay":{"spectral_max":spec_err,"candidate_context":cand_err},
        "spectral_conv1_formation":{
            "factors":["dense_plus_coordinates source","conv1 kernel","conv1 bias"],
            "downstream":"conv2/conv3/cardinality all learned",
            "global_factorial":{},"focused_units":{}
        },
        "candidate_clusterH1_formation":{
            "factors":["cluster_norm source","cluster_hidden1 kernel","cluster_hidden1 bias"],
            "downstream":"cluster_hidden2/candidate_context/cardinality all learned",
            "global_factorial":{},"focused_units":{}
        },
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Spectral interventions alter only spectral_max; spectral_avg and candidate_context remain learned.",
            "Candidate interventions alter only candidate_context; spectral branches remain learned.",
            "Hybrid source/kernel/bias combinations can be out of training distribution.",
            "No intervention is promoted."
        ]
    }

    for label,mx in spectral.items():
        ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u])) for u in SPECTRAL_UNITS
        }
        result["spectral_conv1_formation"]["global_factorial"][label]=x

    for u in SPECTRAL_UNITS:
        result["spectral_conv1_formation"]["focused_units"][str(u)]={}
        for label,mx in spectral.items():
            ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
            pred,nh1,nh2=one_card_unit(
                ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
            )
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
            result["spectral_conv1_formation"]["focused_units"][str(u)][label]=x

    for label,cand in candidate.items():
        ctx=np.concatenate([avg_l,max_l,cand],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u])) for u in CANDIDATE_UNITS
        }
        result["candidate_clusterH1_formation"]["global_factorial"][label]=x

    for u in CANDIDATE_UNITS:
        result["candidate_clusterH1_formation"]["focused_units"][str(u)]={}
        for label,cand in candidate.items():
            ctx=np.concatenate([avg_l,max_l,cand],axis=1)
            pred,nh1,nh2=one_card_unit(
                ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
            )
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
            result["candidate_clusterH1_formation"]["focused_units"][str(u)][label]=x

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Conv1 and cluster_hidden1 formation audit","",
        "Aucun entrainement.","",
        "## Spectral: dense_plus_coordinates x conv1 W/b","",
        "| dense input | W1 | b1 | K2 | K3 | K4 | A | recup48 | perd29 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for src,w,b in [
        (False,False,False),(True,False,False),(False,True,False),
        (False,False,True),(True,True,False),(True,True,True)
    ]:
        label=spectral_label(src,w,b)
        x=result["spectral_conv1_formation"]["global_factorial"][label];m=x["metrics"]
        lines.append(
            f"| {'C' if src else 'L'} | {'C' if w else 'L'} | {'C' if b else 'L'} | "
            f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | "
            f"{100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | "
            f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
        )

    lines += ["","### Spectral focused units","",
              "| unit | dense input C recup/net | conv1W C recup/net | input+W recup/net |",
              "|---:|---:|---:|---:|"]
    for u in SPECTRAL_UNITS:
        rows=result["spectral_conv1_formation"]["focused_units"][str(u)]
        a1=rows[spectral_label(True,False,False)]
        a2=rows[spectral_label(False,True,False)]
        a3=rows[spectral_label(True,True,False)]
        lines.append(
            f"| {u} | {a1['original_48_recovered']}/{a1['k3_net_correct_vs_learned']:+d} | "
            f"{a2['original_48_recovered']}/{a2['k3_net_correct_vs_learned']:+d} | "
            f"{a3['original_48_recovered']}/{a3['k3_net_correct_vs_learned']:+d} |"
        )

    lines += ["","## Candidate: cluster_norm x cluster_hidden1 W/b","",
              "| cluster_norm | W1 | b1 | K2 | K3 | K4 | A | recup48 | perd29 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for src,w,b in [
        (False,False,False),(True,False,False),(False,True,False),
        (False,False,True),(True,True,False),(True,True,True)
    ]:
        label=candidate_label(src,w,b)
        x=result["candidate_clusterH1_formation"]["global_factorial"][label];m=x["metrics"]
        lines.append(
            f"| {'C' if src else 'L'} | {'C' if w else 'L'} | {'C' if b else 'L'} | "
            f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | "
            f"{100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | "
            f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
        )

    lines += ["","### Candidate focused units","",
              "| unit | cluster_norm C recup/net | clusterH1W C recup/net | norm+W recup/net |",
              "|---:|---:|---:|---:|"]
    for u in CANDIDATE_UNITS:
        rows=result["candidate_clusterH1_formation"]["focused_units"][str(u)]
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
