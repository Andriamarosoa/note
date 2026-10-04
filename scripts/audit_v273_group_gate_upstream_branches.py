"""Frozen upstream attribution for the two harmful cardinality_context branches.

No training. No architecture change.

A) spectral_max path:
   conv2 representation source (learned/control)
   x conv3 kernel source (learned/control)
   x conv3 bias source (learned/control)
   -> ReLU(conv3)
   -> GlobalMaxPooling only
   -> cardinality_context with learned spectral_avg + learned candidate_context

B) candidate_context path:
   cluster_hidden2 representation source (learned/control)
   x candidate_context Dense kernel source (learned/control)
   x candidate_context bias source (learned/control)
   -> candidate_context
   -> cardinality_context with learned spectral_avg + learned spectral_max

Everything downstream of cardinality_context is held learned-gate.
Focused hidden1 units:
  spectral path: 23, 158, 135
  candidate path: 73, 104, 109
Also reports hidden2[37].

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

AVG="v240_dense_global_average"
MAX="v240_dense_global_max"
CONV2="v240_dense_conv2"
CONV3="v240_dense_conv3"
CAND_H="cluster_hidden2"
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
         base.get_layer(CAND_H).output,
         base.get_layer(CAND).output,
         base.output],
    )
    buckets=[[] for _ in range(5)]
    ga=np.asarray(gate)
    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=gate if ga.ndim==0 else ga[s:s+len(ids)]
        vals=probe(apply_gate(x,g),training=False)
        for b,v in zip(buckets,vals):
            b.append(np.asarray(v,np.float32))
    return tuple(np.concatenate(x) for x in buckets)


def collect_hybrid_spectral_max(
    learned_base, control_base, cache, outer, learned_gate,
    k3_kernel_l, k3_bias_l, k3_kernel_c, k3_bias_c,
    batch_size=24,
):
    """Return 8 arrays [N,96] keyed by conv2/kernel/bias source."""
    import tensorflow as tf
    probe_l=tf.keras.Model(learned_base.inputs,learned_base.get_layer(CONV2).output)
    probe_c=tf.keras.Model(control_base.inputs,control_base.get_layer(CONV2).output)

    out={label:[] for label in spectral_labels()}
    ga=np.asarray(learned_gate)

    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=ga[s:s+len(ids)]
        c2_l=np.asarray(probe_l(apply_gate(x,g),training=False),np.float32)
        c2_c=np.asarray(probe_c(apply_gate(x,1.0),training=False),np.float32)

        for h_ctl in (False,True):
            c2=c2_c if h_ctl else c2_l
            c2t=tf.convert_to_tensor(c2)
            for w_ctl in (False,True):
                ker=k3_kernel_c if w_ctl else k3_kernel_l
                conv=tf.nn.conv2d(c2t,tf.convert_to_tensor(ker),strides=[1,1,1,1],padding="SAME")
                for b_ctl in (False,True):
                    bias=k3_bias_c if b_ctl else k3_bias_l
                    z=tf.nn.relu(tf.nn.bias_add(conv,tf.convert_to_tensor(bias)))
                    mx=tf.reduce_max(z,axis=[1,2])
                    out[spectral_label(h_ctl,w_ctl,b_ctl)].append(np.asarray(mx,np.float32))

    return {k:np.concatenate(v) for k,v in out.items()}


def spectral_label(h_ctl,w_ctl,b_ctl):
    return (
        f"conv2_{'control' if h_ctl else 'learned'}__"
        f"conv3W_{'control' if w_ctl else 'learned'}__"
        f"conv3b_{'control' if b_ctl else 'learned'}"
    )


def spectral_labels():
    return [
        spectral_label(h,w,b)
        for h in (False,True)
        for w in (False,True)
        for b in (False,True)
    ]


def candidate_label(h_ctl,w_ctl,b_ctl):
    return (
        f"clusterH_{'control' if h_ctl else 'learned'}__"
        f"candW_{'control' if w_ctl else 'learned'}__"
        f"candb_{'control' if b_ctl else 'learned'}"
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


def one_unit_from_context(ctx,unit,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow):
    new_h1=relu(ctx@w1[:,unit]+b1[unit])
    dh1=new_h1-h1_l[:,unit]
    new_h2_pre=h2_pre_l+dh1[:,None]*w2[unit,:][None,:]
    new_h2=relu(new_h2_pre)
    logits=logits_l+(new_h2-h2_l)@ow
    pred=logits.argmax(1).astype(np.int32)
    return pred,new_h1,new_h2


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

    avg_c,max_c,ch_c,cand_c,p_c=collect_small(bc,cache,outer,1.0)
    avg_l,max_l,ch_l,cand_l,p_l=collect_small(bl,cache,outer,gate)
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    conv3W_c,conv3b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV3).get_weights()]
    conv3W_l,conv3b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV3).get_weights()]
    candW_c,candb_c=[np.asarray(x,np.float32) for x in bc.get_layer(CAND).get_weights()]
    candW_l,candb_l=[np.asarray(x,np.float32) for x in bl.get_layer(CAND).get_weights()]
    w1,b1=[np.asarray(x,np.float32) for x in bl.get_layer(H1).get_weights()]
    w2,b2=[np.asarray(x,np.float32) for x in bl.get_layer(H2).get_weights()]
    ow,ob=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    require(conv3W_l.shape==conv3W_c.shape,"conv3 shape mismatch")
    require(candW_l.shape==candW_c.shape==(96,96),f"candidate_context kernel {candW_l.shape}")
    require(ch_l.shape==ch_c.shape==(len(outer),96),f"cluster_hidden2 {ch_l.shape}")

    spectral_max=collect_hybrid_spectral_max(
        bl,bc,cache,outer,gate,
        conv3W_l,conv3b_l,conv3W_c,conv3b_c
    )
    spectral_replay=float(np.max(np.abs(spectral_max[spectral_label(False,False,False)]-max_l)))
    require(spectral_replay<1e-5,f"spectral max replay mismatch {spectral_replay}")

    candidate_hybrids={}
    for h_ctl in (False,True):
        h=ch_c if h_ctl else ch_l
        for w_ctl in (False,True):
            W=candW_c if w_ctl else candW_l
            for b_ctl in (False,True):
                bb=candb_c if b_ctl else candb_l
                candidate_hybrids[candidate_label(h_ctl,w_ctl,b_ctl)]=relu(h@W+bb)
    candidate_replay=float(np.max(np.abs(candidate_hybrids[candidate_label(False,False,False)]-cand_l)))
    require(candidate_replay<1e-5,f"candidate context replay mismatch {candidate_replay}")

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
        "replay":{"spectral_max":spectral_replay,"candidate_context":candidate_replay},
        "spectral_path":{
            "factors":["conv2 representation","conv3 kernel","conv3 bias"],
            "global_factorial":{},"focused_units":{}
        },
        "candidate_path":{
            "factors":["cluster_hidden2 representation","candidate_context kernel","candidate_context bias"],
            "global_factorial":{},"focused_units":{}
        },
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Spectral audit changes only spectral_max; learned spectral_avg and learned candidate_context are held fixed.",
            "Candidate audit changes only candidate_context; learned spectral_avg and learned spectral_max are held fixed.",
            "All downstream hidden1/hidden2/output parameters are held learned-gate.",
            "Hybrid conditions may be out of distribution and are attribution tests only.",
            "No intervention is promoted."
        ]
    }

    # Spectral global factorial: learned avg, hybrid max, learned candidate.
    for label,mx in spectral_max.items():
        ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-h1_l[reg48,u])) for u in SPECTRAL_UNITS
        }
        result["spectral_path"]["global_factorial"][label]=x

    # Spectral focused hidden1 units: alter one H1 activation only.
    for u in SPECTRAL_UNITS:
        result["spectral_path"]["focused_units"][str(u)]={}
        for label,mx in spectral_max.items():
            ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
            pred,new_h1,new_h2=one_unit_from_context(
                ctx,u,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow
            )
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(new_h1[reg48]-h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(new_h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
            result["spectral_path"]["focused_units"][str(u)][label]=x

    # Candidate global factorial: learned avg/max, hybrid candidate.
    for label,cand in candidate_hybrids.items():
        ctx=np.concatenate([avg_l,max_l,cand],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-h1_l[reg48,u])) for u in CANDIDATE_UNITS
        }
        result["candidate_path"]["global_factorial"][label]=x

    for u in CANDIDATE_UNITS:
        result["candidate_path"]["focused_units"][str(u)]={}
        for label,cand in candidate_hybrids.items():
            ctx=np.concatenate([avg_l,max_l,cand],axis=1)
            pred,new_h1,new_h2=one_unit_from_context(
                ctx,u,h1_l,h2_pre_l,h2_l,logits_l,w1,b1,w2,ow
            )
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x["unit_delta_regressed"]=float(np.mean(new_h1[reg48]-h1_l[reg48,u]))
            x["h2_37_delta_regressed"]=float(np.mean(new_h2[reg48,TARGET_H2]-h2_l[reg48,TARGET_H2]))
            result["candidate_path"]["focused_units"][str(u)][label]=x

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Upstream audit: spectral_max / candidate_context","",
        "Aucun entrainement. Aval de cardinality_context fige en learned-gate.","",
        "## Spectral_max: conv2 x conv3 kernel x conv3 bias","",
        "| conv2 | W3 | b3 | K2 | K3 | K4 | A | recup48 | perd29 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for h in (False,True):
        for w in (False,True):
            for b in (False,True):
                label=spectral_label(h,w,b)
                x=result["spectral_path"]["global_factorial"][label];m=x["metrics"]
                lines.append(
                    f"| {'control' if h else 'learned'} | {'control' if w else 'learned'} | "
                    f"{'control' if b else 'learned'} | {100*m['K2']['exact']:.2f}% | "
                    f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
                    f"{100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']} | "
                    f"{x['original_29_corrections_lost']} |"
                )

    lines += ["","### Unites spectral ciblees","",
              "| unit | conv2 control recup/net | W3 control recup/net | conv2+W3 recup/net |",
              "|---:|---:|---:|---:|"]
    for u in SPECTRAL_UNITS:
        rows=result["spectral_path"]["focused_units"][str(u)]
        a1=rows[spectral_label(True,False,False)]
        a2=rows[spectral_label(False,True,False)]
        a3=rows[spectral_label(True,True,False)]
        lines.append(
            f"| {u} | {a1['original_48_recovered']}/{a1['k3_net_correct_vs_learned']:+d} | "
            f"{a2['original_48_recovered']}/{a2['k3_net_correct_vs_learned']:+d} | "
            f"{a3['original_48_recovered']}/{a3['k3_net_correct_vs_learned']:+d} |"
        )

    lines += ["","## candidate_context: cluster_hidden2 x candidate kernel x bias","",
              "| cluster_hidden2 | Wcand | bcand | K2 | K3 | K4 | A | recup48 | perd29 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for h in (False,True):
        for w in (False,True):
            for b in (False,True):
                label=candidate_label(h,w,b)
                x=result["candidate_path"]["global_factorial"][label];m=x["metrics"]
                lines.append(
                    f"| {'control' if h else 'learned'} | {'control' if w else 'learned'} | "
                    f"{'control' if b else 'learned'} | {100*m['K2']['exact']:.2f}% | "
                    f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
                    f"{100*m['cluster_A']['exact']:.2f}% | {x['original_48_recovered']} | "
                    f"{x['original_29_corrections_lost']} |"
                )

    lines += ["","### Unites candidate ciblees","",
              "| unit | clusterH control recup/net | Wcand control recup/net | clusterH+Wcand recup/net |",
              "|---:|---:|---:|---:|"]
    for u in CANDIDATE_UNITS:
        rows=result["candidate_path"]["focused_units"][str(u)]
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
