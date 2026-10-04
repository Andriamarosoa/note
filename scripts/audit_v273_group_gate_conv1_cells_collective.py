"""Frozen fine attribution: conv1 acoustic channel-filter cells and collective pool branches.

No training. No architecture change.

A) Spectral:
  Conv1 input channels are fixed by the source code:
    0 = log_power
    1 = positive_pre
    2 = flux
    3 = time coordinate
    4 = frequency coordinate
  Previous audit found channels 0/1 and several conv1 output filters most involved.
  Here, for EVERY output filter 0..31 and acoustic input channel 0/1, replace
  only kernel[:, :, channel, filter] from always_on into learned_gate.
  All other conv1 weights/biases and all downstream conv2/conv3/cardinality
  parameters remain learned_gate.

B) Candidate collective representation:
  collective_cluster_representation = concat([
      mean_pool (64),
      max_pool (64),
      attention_pool (64),
      scaled_sum_pool (64),
      cluster_stats (8),
  ])
  Evaluate all 2^5 learned/control source combinations while keeping
  cluster_norm gamma/beta, cluster_hidden1/2, candidate_context and final
  cardinality path learned_gate.

Frozen attribution only; no promotion.
"""
from __future__ import annotations

import argparse, csv, itertools, json
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

MEAN="mean_pool"
POOL_MAX="max_pool"
ATTN="attention_pool"
SCALED="scaled_sum_pool"
STATS="cluster_stats"
CLUSTER_NORM="cluster_norm"
CLUSTER_H1="cluster_hidden1"
CLUSTER_H2="cluster_hidden2"
CAND="candidate_context"

CARD_H1="v240_cardinality_hidden1"
CARD_H2="v240_cardinality_hidden2"
OUT="cardinality"

CHANNEL_NAMES={
    0:"log_power",
    1:"positive_pre",
    2:"flux",
    3:"time_coord",
    4:"frequency_coord",
}
AUDIT_CHANNELS=[0,1]
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


def collect(base,cache,outer,gate,batch_size=256):
    import tensorflow as tf
    probe=tf.keras.Model(
        base.inputs,
        [
            base.get_layer(AVG).output,
            base.get_layer(MAX).output,
            base.get_layer(MEAN).output,
            base.get_layer(POOL_MAX).output,
            base.get_layer(ATTN).output,
            base.get_layer(SCALED).output,
            base.get_layer(STATS).output,
            base.get_layer(CAND).output,
            base.output,
        ],
    )
    buckets=[[] for _ in range(9)]
    ga=np.asarray(gate)
    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=gate if ga.ndim==0 else ga[s:s+len(ids)]
        vals=probe(apply_gate(x,g),training=False)
        for b,v in zip(buckets,vals):
            b.append(np.asarray(v,np.float32))
    return tuple(np.concatenate(x) for x in buckets)


def relu(x):
    return np.maximum(np.asarray(x,np.float32),0.0)


def layer_norm_np(x,gamma,beta,epsilon):
    x=np.asarray(x,np.float32)
    mean=np.mean(x,axis=-1,keepdims=True)
    var=np.mean(np.square(x-mean),axis=-1,keepdims=True)
    return (x-mean)/np.sqrt(var+float(epsilon))*gamma+beta


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
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
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


def one_card_unit(ctx,unit,h1_l,h2pre_l,h2_l,logits_l,w1,b1,w2,ow):
    new_h1=relu(ctx@w1[:,unit]+b1[unit])
    dh1=new_h1-h1_l[:,unit]
    new_h2pre=h2pre_l+dh1[:,None]*w2[unit,:][None,:]
    new_h2=relu(new_h2pre)
    logits=logits_l+(new_h2-h2_l)@ow
    return logits.argmax(1).astype(np.int32),new_h1,new_h2


def spectral_cell_maxima(
    learned_base,cache,outer,gate,
    k_l,k_c,b_l,c2w,c2b,c3w,c3b,
    batch_size=24,
):
    import tensorflow as tf
    probe=tf.keras.Model(learned_base.inputs,learned_base.get_layer(DENSE_IN).output)
    filters=range(int(k_l.shape[3]))
    labels=["baseline"]+[f"c{c}_f{f}" for c in AUDIT_CHANNELS for f in filters]
    out={k:[] for k in labels}

    kernels={"baseline":k_l}
    for c in AUDIT_CHANNELS:
        for f in filters:
            kk=k_l.copy()
            kk[:,:,c,f]=k_c[:,:,c,f]
            kernels[f"c{c}_f{f}"]=kk

    kt={name:tf.convert_to_tensor(val) for name,val in kernels.items()}
    b1t=tf.convert_to_tensor(b_l)
    c2wt=tf.convert_to_tensor(c2w); c2bt=tf.convert_to_tensor(c2b)
    c3wt=tf.convert_to_tensor(c3w); c3bt=tf.convert_to_tensor(c3b)
    ga=np.asarray(gate)

    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=ga[s:s+len(ids)]
        din=np.asarray(probe(apply_gate(x,g),training=False),np.float32)
        dint=tf.convert_to_tensor(din)
        for name in labels:
            a1=tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(dint,kt[name],strides=[1,1,1,1],padding="SAME"),b1t))
            a2=tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(a1,c2wt,strides=[1,1,1,1],padding="SAME"),c2bt))
            a3=tf.nn.relu(tf.nn.bias_add(
                tf.nn.conv2d(a2,c3wt,strides=[1,1,1,1],padding="SAME"),c3bt))
            mx=tf.reduce_max(a3,axis=[1,2])
            out[name].append(np.asarray(mx,np.float32))
    return {k:np.concatenate(v) for k,v in out.items()}


def collective_label(bits):
    names=["mean","max","attention","scaled_sum","stats"]
    return "__".join(f"{n}_{'C' if b else 'L'}" for n,b in zip(names,bits))


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

    mc=build_model("always_on",SEED);mc.load_weights(a.always_weights)
    ml=build_model("learned_gate",SEED);ml.load_weights(a.learned_weights)
    bc=nested(mc);bl=nested(ml)

    vals_c=collect(bc,cache,outer,1.0)
    vals_l=collect(bl,cache,outer,gate)
    avg_c,max_c,mean_c,pmax_c,attn_c,scaled_c,stats_c,cand_c,p_c=vals_c
    avg_l,max_l,mean_l,pmax_l,attn_l,scaled_l,stats_l,cand_l,p_l=vals_l
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    c1k_c,c1b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV1).get_weights()]
    c1k_l,c1b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV1).get_weights()]
    c2w,c2b=[np.asarray(x,np.float32) for x in bl.get_layer(CONV2).get_weights()]
    c3w,c3b=[np.asarray(x,np.float32) for x in bl.get_layer(CONV3).get_weights()]

    ln=bl.get_layer(CLUSTER_NORM)
    gamma,beta=[np.asarray(x,np.float32) for x in ln.get_weights()]
    eps=float(ln.epsilon)
    cl1w,cl1b=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H1).get_weights()]
    cl2w,cl2b=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H2).get_weights()]
    candw,candb=[np.asarray(x,np.float32) for x in bl.get_layer(CAND).get_weights()]

    w1,b1=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H1).get_weights()]
    w2,b2=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H2).get_weights()]
    ow,ob=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    require(c1k_l.shape==c1k_c.shape==(3,3,5,32),f"unexpected conv1 kernel {c1k_l.shape}")
    require(mean_l.shape[1]==64 and pmax_l.shape[1]==64 and attn_l.shape[1]==64 and scaled_l.shape[1]==64,
            "unexpected pool dims")
    require(stats_l.shape[1]==8,"unexpected stats dim")

    spectral=spectral_cell_maxima(bl,cache,outer,gate,c1k_l,c1k_c,c1b_l,c2w,c2b,c3w,c3b)
    spec_err=float(np.max(np.abs(spectral["baseline"]-max_l)))
    require(spec_err<1e-5,f"spectral replay mismatch {spec_err}")

    # Learned downstream baselines.
    collective_l=np.concatenate([mean_l,pmax_l,attn_l,scaled_l,stats_l],axis=1)
    norm_l=layer_norm_np(collective_l,gamma,beta,eps)
    ch1_l=relu(norm_l@cl1w+cl1b)
    ch2_l=relu(ch1_l@cl2w+cl2b)
    cand_replay=relu(ch2_l@candw+candb)
    cand_err=float(np.max(np.abs(cand_replay-cand_l)))
    require(cand_err<2e-5,f"candidate replay mismatch {cand_err}")

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
        "spectral_channel_semantics":CHANNEL_NAMES,
        "replay":{"spectral_max":spec_err,"candidate_context":cand_err},
        "conv1_channel_filter_cells":[],
        "collective_factorial":{},
        "focused_units":{
            "spectral":{str(u):{} for u in SPECTRAL_UNITS},
            "candidate":{str(u):{} for u in CANDIDATE_UNITS},
        },
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Conv1 cell swap replaces only kernel[:, :, channel, filter] for acoustic channels 0/1.",
            "Collective factorial swaps source values of mean/max/attention/scaled_sum/stats; all downstream weights remain learned-gate.",
            "Hybrid conditions can be outside the training distribution.",
            "No slice or collective component combination is promoted."
        ]
    }

    # Spectral channel-filter cell interventions.
    for c in AUDIT_CHANNELS:
        for f in range(32):
            mx=spectral[f"c{c}_f{f}"]
            ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
            pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x.update({
                "channel":c,
                "channel_name":CHANNEL_NAMES[c],
                "filter":f,
                "weight_delta_l2":float(np.linalg.norm(c1k_l[:,:,c,f]-c1k_c[:,:,c,f])),
                "focused_hidden1_delta_regressed":{
                    str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u])) for u in SPECTRAL_UNITS
                },
                "h2_37_delta_regressed":float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2])),
            })
            result["conv1_channel_filter_cells"].append(x)
            for u in SPECTRAL_UNITS:
                pp,nh1,nh2=one_card_unit(
                    ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
                )
                sx=summarize(k,pp,learned_pred,always_pred,scopes)
                sx["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
                result["focused_units"]["spectral"][str(u)][f"c{c}_f{f}"]=sx

    # All 32 collective source combinations.
    learned_parts=[mean_l,pmax_l,attn_l,scaled_l,stats_l]
    control_parts=[mean_c,pmax_c,attn_c,scaled_c,stats_c]
    for bits in itertools.product((False,True), repeat=5):
        parts=[control_parts[i] if bits[i] else learned_parts[i] for i in range(5)]
        coll=np.concatenate(parts,axis=1)
        norm=layer_norm_np(coll,gamma,beta,eps)
        ch1=relu(norm@cl1w+cl1b)
        ch2=relu(ch1@cl2w+cl2b)
        cand=relu(ch2@candw+candb)
        ctx=np.concatenate([avg_l,max_l,cand],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        label=collective_label(bits)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["bits_control"]=[bool(v) for v in bits]
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u])) for u in CANDIDATE_UNITS
        }
        result["collective_factorial"][label]=x

        for u in CANDIDATE_UNITS:
            pp,nh1,nh2=one_card_unit(
                ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
            )
            sx=summarize(k,pp,learned_pred,always_pred,scopes)
            sx["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
            result["focused_units"]["candidate"][str(u)][label]=sx

    # Descriptive rankings.
    result["top_conv1_cells_by_k3_net"]=sorted(
        result["conv1_channel_filter_cells"],
        key=lambda x:(x["k3_net_correct_vs_learned"],x["original_48_recovered"],-x["original_29_corrections_lost"]),
        reverse=True
    )
    result["top_collective_combinations_by_k3_net"]=sorted(
        [
            {"label":label,**x}
            for label,x in result["collective_factorial"].items()
        ],
        key=lambda x:(x["k3_net_correct_vs_learned"],x["original_48_recovered"],-x["original_29_corrections_lost"]),
        reverse=True
    )

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Conv1 acoustic cell + collective pooling attribution audit","",
        "Aucun entrainement.","",
        "Spectral channel semantics:",
        "- 0 = log_power",
        "- 1 = positive_pre",
        "- 2 = flux",
        "- 3 = time_coord",
        "- 4 = frequency_coord","",
        "## Top 20 conv1 channel x filter cells","",
        "| channel | filter | net K3 | recup48 | perd29 | K2 | K3 | K4 | A | ||dW|| |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in result["top_conv1_cells_by_k3_net"][:20]:
        m=x["metrics"]
        lines.append(
            f"| {x['channel']} {x['channel_name']} | {x['filter']} | "
            f"{x['k3_net_correct_vs_learned']:+d} | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['weight_delta_l2']:.5f} |"
        )

    single_labels={
        "mean":collective_label((True,False,False,False,False)),
        "max":collective_label((False,True,False,False,False)),
        "attention":collective_label((False,False,True,False,False)),
        "scaled_sum":collective_label((False,False,False,True,False)),
        "stats":collective_label((False,False,False,False,True)),
    }
    lines += ["","## Collective components — single swaps","",
              "| component control | net K3 | recup48 | perd29 | K2 | K3 | K4 | A |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,label in single_labels.items():
        x=result["collective_factorial"][label];m=x["metrics"]
        lines.append(
            f"| {name} | {x['k3_net_correct_vs_learned']:+d} | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% |"
        )

    lines += ["","## Top 12 collective combinations descriptives","",
              "| control components | net K3 | recup48 | perd29 | K2 | K3 | K4 | A |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names=["mean","max","attention","scaled_sum","stats"]
    for x in result["top_collective_combinations_by_k3_net"][:12]:
        controls=[names[i] for i,v in enumerate(x["bits_control"]) if v]
        label="+".join(controls) if controls else "none"
        m=x["metrics"]
        lines.append(
            f"| {label} | {x['k3_net_correct_vs_learned']:+d} | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% |"
        )

    lines += ["","Attribution sur checkpoints figes uniquement; aucune promotion automatique."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
