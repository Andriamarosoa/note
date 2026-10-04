"""Frozen fine-grained attribution audit for conv1.kernel and cluster_norm.

No training. No architecture change.

A) Spectral conv1.kernel:
   Starting from the learned-gate checkpoint and learned dense_plus_coordinates,
   swap control weights into the learned conv1 kernel:
     * one input-channel slice at a time: kernel[:, :, input_channel, :]
     * one output-filter slice at a time: kernel[:, :, :, output_filter]
   conv1 bias and all downstream conv2/conv3/cardinality parameters remain learned.
   Measures Exact-K effects globally and on focused hidden1 units 23/158/135.

B) Candidate cluster_norm:
   Reconstruct LayerNorm exactly from:
     * collective_cluster_representation source (learned/control)
     * gamma source (learned/control)
     * beta source (learned/control)
   Everything after cluster_norm remains learned-gate.
   Measures Exact-K effects globally and focused hidden1 units 73/104/109.

Frozen attribution only. No model selection or promotion.
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

COLLECTIVE="collective_cluster_representation"
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
        [
            base.get_layer(AVG).output,
            base.get_layer(MAX).output,
            base.get_layer(COLLECTIVE).output,
            base.get_layer(CLUSTER_NORM).output,
            base.get_layer(CAND).output,
            base.output,
        ],
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


def layer_norm_np(x,gamma,beta,epsilon):
    x=np.asarray(x,np.float32)
    mean=np.mean(x,axis=-1,keepdims=True)
    var=np.mean(np.square(x-mean),axis=-1,keepdims=True)
    y=(x-mean)/np.sqrt(var+float(epsilon))
    return y*np.asarray(gamma,np.float32)+np.asarray(beta,np.float32)


def conv1_swap_maxima(
    learned_base,cache,outer,gate,
    k_l,k_c,b_l,c2w,c2b,c3w,c3b,
    input_channels,output_filters,batch_size=24,
):
    """Compute spectral_max for baseline and each single input/output slice swap."""
    import tensorflow as tf
    probe=tf.keras.Model(learned_base.inputs,learned_base.get_layer(DENSE_IN).output)
    labels=["baseline"]+[f"in_{i}" for i in input_channels]+[f"out_{j}" for j in output_filters]
    out={k:[] for k in labels}
    ga=np.asarray(gate)

    kernels={"baseline":k_l}
    for i in input_channels:
        kk=k_l.copy(); kk[:,:,i,:]=k_c[:,:,i,:]; kernels[f"in_{i}"]=kk
    for j in output_filters:
        kk=k_l.copy(); kk[:,:,:,j]=k_c[:,:,:,j]; kernels[f"out_{j}"]=kk

    # Tensor constants once.
    kt={name:tf.convert_to_tensor(val) for name,val in kernels.items()}
    b1t=tf.convert_to_tensor(b_l)
    c2wt=tf.convert_to_tensor(c2w); c2bt=tf.convert_to_tensor(c2b)
    c3wt=tf.convert_to_tensor(c3w); c3bt=tf.convert_to_tensor(c3b)

    for s in range(0,len(outer),batch_size):
        ids=outer[s:s+batch_size]
        x=batch_inputs(cache,ids,FRAMES)
        g=ga[s:s+len(ids)]
        din=np.asarray(probe(apply_gate(x,g),training=False),np.float32)
        dint=tf.convert_to_tensor(din)
        for name in labels:
            z1=tf.nn.conv2d(dint,kt[name],strides=[1,1,1,1],padding="SAME")
            a1=tf.nn.relu(tf.nn.bias_add(z1,b1t))
            z2=tf.nn.conv2d(a1,c2wt,strides=[1,1,1,1],padding="SAME")
            a2=tf.nn.relu(tf.nn.bias_add(z2,c2bt))
            z3=tf.nn.conv2d(a2,c3wt,strides=[1,1,1,1],padding="SAME")
            a3=tf.nn.relu(tf.nn.bias_add(z3,c3bt))
            mx=tf.reduce_max(a3,axis=[1,2])
            out[name].append(np.asarray(mx,np.float32))
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

    avg_c,max_c,coll_c,norm_c,cand_c,p_c=collect_small(bc,cache,outer,1.0)
    avg_l,max_l,coll_l,norm_l,cand_l,p_l=collect_small(bl,cache,outer,gate)
    np.testing.assert_array_equal(p_c.argmax(1),always_pred)
    np.testing.assert_array_equal(p_l.argmax(1),learned_pred)

    c1k_c,c1b_c=[np.asarray(x,np.float32) for x in bc.get_layer(CONV1).get_weights()]
    c1k_l,c1b_l=[np.asarray(x,np.float32) for x in bl.get_layer(CONV1).get_weights()]
    c2w,c2b=[np.asarray(x,np.float32) for x in bl.get_layer(CONV2).get_weights()]
    c3w,c3b=[np.asarray(x,np.float32) for x in bl.get_layer(CONV3).get_weights()]

    ln_c=bc.get_layer(CLUSTER_NORM); ln_l=bl.get_layer(CLUSTER_NORM)
    lnw_c=[np.asarray(x,np.float32) for x in ln_c.get_weights()]
    lnw_l=[np.asarray(x,np.float32) for x in ln_l.get_weights()]
    require(len(lnw_c)==2 and len(lnw_l)==2,"cluster_norm expected gamma,beta")
    gamma_c,beta_c=lnw_c; gamma_l,beta_l=lnw_l
    eps=float(ln_l.epsilon)

    cl1w,cl1b=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H1).get_weights()]
    cl2w,cl2b=[np.asarray(x,np.float32) for x in bl.get_layer(CLUSTER_H2).get_weights()]
    candw,candb=[np.asarray(x,np.float32) for x in bl.get_layer(CAND).get_weights()]

    w1,b1=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H1).get_weights()]
    w2,b2=[np.asarray(x,np.float32) for x in bl.get_layer(CARD_H2).get_weights()]
    ow,ob=[np.asarray(x,np.float32) for x in bl.get_layer(OUT).get_weights()]

    input_channels=list(range(int(c1k_l.shape[2])))
    output_filters=list(range(int(c1k_l.shape[3])))
    spectral=conv1_swap_maxima(
        bl,cache,outer,gate,
        c1k_l,c1k_c,c1b_l,c2w,c2b,c3w,c3b,
        input_channels,output_filters
    )
    spec_err=float(np.max(np.abs(spectral["baseline"]-max_l)))
    require(spec_err<1e-5,f"spectral replay mismatch {spec_err}")

    norm_replay=layer_norm_np(coll_l,gamma_l,beta_l,eps)
    norm_err=float(np.max(np.abs(norm_replay-norm_l)))
    require(norm_err<2e-5,f"cluster_norm replay mismatch {norm_err}")

    # 2x2x2 cluster_norm reconstruction.
    norm_hybrids={}
    for src_ctl in (False,True):
        src=coll_c if src_ctl else coll_l
        for g_ctl in (False,True):
            gamma=gamma_c if g_ctl else gamma_l
            for b_ctl in (False,True):
                beta=beta_c if b_ctl else beta_l
                label=(
                    f"input_{'control' if src_ctl else 'learned'}__"
                    f"gamma_{'control' if g_ctl else 'learned'}__"
                    f"beta_{'control' if b_ctl else 'learned'}"
                )
                norm_hybrids[label]=layer_norm_np(src,gamma,beta,eps)

    # Candidate contexts from reconstructed cluster_norm, downstream learned.
    candidate={}
    for label,norm in norm_hybrids.items():
        ch1=relu(norm@cl1w+cl1b)
        ch2=relu(ch1@cl2w+cl2b)
        candidate[label]=relu(ch2@candw+candb)
    cand_err=float(np.max(np.abs(
        candidate["input_learned__gamma_learned__beta_learned"]-cand_l
    )))
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
        "replay":{"spectral_max":spec_err,"cluster_norm":norm_err,"candidate_context":cand_err},
        "conv1_kernel":{
            "kernel_shape":list(c1k_l.shape),
            "input_channel_count":len(input_channels),
            "output_filter_count":len(output_filters),
            "input_channel_swaps":[],
            "output_filter_swaps":[],
            "focused_units":{str(u):{} for u in SPECTRAL_UNITS},
        },
        "cluster_norm":{
            "dim":int(gamma_l.shape[0]),
            "epsilon":eps,
            "global_factorial":{},
            "focused_units":{str(u):{} for u in CANDIDATE_UNITS},
        },
        "limitations":[
            "Frozen attribution only; no retraining or architecture change.",
            "Conv1 slice swaps use learned dense_plus_coordinates input, learned conv1 bias, and learned downstream conv2/conv3/cardinality.",
            "Input-channel indices are reported exactly from the trained kernel shape; semantic channel names are not inferred here.",
            "Cluster_norm is reconstructed exactly from collective representation, gamma, beta and layer epsilon.",
            "Hybrid swaps may be out of training distribution.",
            "No channel, filter, LayerNorm parameter set or hybrid is promoted."
        ]
    }

    # Conv1 input-channel and output-filter interventions.
    for kind,indices in (("in",input_channels),("out",output_filters)):
        for idx in indices:
            mx=spectral[f"{kind}_{idx}"]
            ctx=np.concatenate([avg_l,mx,cand_l],axis=1)
            pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
            x=summarize(k,pred,learned_pred,always_pred,scopes)
            x.update({
                "index":int(idx),
                "weight_delta_l2":float(
                    np.linalg.norm(c1k_l[:,:,idx,:]-c1k_c[:,:,idx,:])
                    if kind=="in"
                    else np.linalg.norm(c1k_l[:,:,:,idx]-c1k_c[:,:,:,idx])
                ),
                "h2_37_delta_regressed":float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2])),
                "focused_hidden1_delta_regressed":{
                    str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u]))
                    for u in SPECTRAL_UNITS
                }
            })
            key="input_channel_swaps" if kind=="in" else "output_filter_swaps"
            result["conv1_kernel"][key].append(x)

            # Focused one-cardinality-hidden1-unit decision effect.
            for u in SPECTRAL_UNITS:
                pp,nh1,nh2=one_card_unit(
                    ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
                )
                sx=summarize(k,pp,learned_pred,always_pred,scopes)
                sx["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
                sx["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
                result["conv1_kernel"]["focused_units"][str(u)][f"{kind}_{idx}"]=sx

    # ClusterNorm 2x2x2.
    for label,cand in candidate.items():
        ctx=np.concatenate([avg_l,max_l,cand],axis=1)
        pred,h1,h2=downstream_context(ctx,w1,b1,w2,b2,ow,ob)
        x=summarize(k,pred,learned_pred,always_pred,scopes)
        x["h2_37_delta_regressed"]=float(np.mean(h2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
        x["focused_hidden1_delta_regressed"]={
            str(u):float(np.mean(h1[reg48,u]-card_h1_l[reg48,u]))
            for u in CANDIDATE_UNITS
        }
        result["cluster_norm"]["global_factorial"][label]=x

        for u in CANDIDATE_UNITS:
            pp,nh1,nh2=one_card_unit(
                ctx,u,card_h1_l,card_h2pre_l,card_h2_l,logits_l,w1,b1,w2,ow
            )
            sx=summarize(k,pp,learned_pred,always_pred,scopes)
            sx["unit_delta_regressed"]=float(np.mean(nh1[reg48]-card_h1_l[reg48,u]))
            sx["h2_37_delta_regressed"]=float(np.mean(nh2[reg48,TARGET_H2]-card_h2_l[reg48,TARGET_H2]))
            result["cluster_norm"]["focused_units"][str(u)][label]=sx

    # Descriptive rankings by net K3, not used for selection/promotion.
    result["conv1_kernel"]["top_input_channels_by_k3_net"]=sorted(
        result["conv1_kernel"]["input_channel_swaps"],
        key=lambda x:(x["k3_net_correct_vs_learned"],x["original_48_recovered"],-x["original_29_corrections_lost"]),
        reverse=True
    )
    result["conv1_kernel"]["top_output_filters_by_k3_net"]=sorted(
        result["conv1_kernel"]["output_filter_swaps"],
        key=lambda x:(x["k3_net_correct_vs_learned"],x["original_48_recovered"],-x["original_29_corrections_lost"]),
        reverse=True
    )

    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    lines=[
        "# Conv1 kernel slices + cluster_norm attribution audit","",
        "Aucun entrainement.","",
        f"Conv1 kernel shape: {tuple(c1k_l.shape)}",
        f"cluster_norm dim: {len(gamma_l)}, epsilon={eps}","",
        "## Conv1 input-channel swaps — tous les canaux","",
        "| channel | net K3 | recup48 | perd29 | K2 | K3 | K4 | A | ||dW|| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for x in result["conv1_kernel"]["top_input_channels_by_k3_net"]:
        m=x["metrics"]
        lines.append(
            f"| {x['index']} | {x['k3_net_correct_vs_learned']:+d} | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['weight_delta_l2']:.4f} |"
        )

    lines += ["","## Conv1 output-filter swaps — top 16 descriptifs","",
              "| filter | net K3 | recup48 | perd29 | K2 | K3 | K4 | A | ||dW|| |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in result["conv1_kernel"]["top_output_filters_by_k3_net"][:16]:
        m=x["metrics"]
        lines.append(
            f"| {x['index']} | {x['k3_net_correct_vs_learned']:+d} | {x['original_48_recovered']} | "
            f"{x['original_29_corrections_lost']} | {100*m['K2']['exact']:.2f}% | "
            f"{100*m['K3']['exact']:.2f}% | {100*m['K4']['exact']:.2f}% | "
            f"{100*m['cluster_A']['exact']:.2f}% | {x['weight_delta_l2']:.4f} |"
        )

    lines += ["","## cluster_norm: input x gamma x beta","",
              "| input | gamma | beta | K2 | K3 | K4 | A | recup48 | perd29 |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for src in (False,True):
        for g in (False,True):
            for b in (False,True):
                label=(
                    f"input_{'control' if src else 'learned'}__"
                    f"gamma_{'control' if g else 'learned'}__"
                    f"beta_{'control' if b else 'learned'}"
                )
                x=result["cluster_norm"]["global_factorial"][label];m=x["metrics"]
                lines.append(
                    f"| {'C' if src else 'L'} | {'C' if g else 'L'} | {'C' if b else 'L'} | "
                    f"{100*m['K2']['exact']:.2f}% | {100*m['K3']['exact']:.2f}% | "
                    f"{100*m['K4']['exact']:.2f}% | {100*m['cluster_A']['exact']:.2f}% | "
                    f"{x['original_48_recovered']} | {x['original_29_corrections_lost']} |"
                )

    lines += ["","### cluster_norm focused units","",
              "| unit | input C recup/net | gamma C recup/net | beta C recup/net | input+gamma recup/net |",
              "|---:|---:|---:|---:|---:|"]
    for u in CANDIDATE_UNITS:
        rows=result["cluster_norm"]["focused_units"][str(u)]
        labels=[
            "input_control__gamma_learned__beta_learned",
            "input_learned__gamma_control__beta_learned",
            "input_learned__gamma_learned__beta_control",
            "input_control__gamma_control__beta_learned",
        ]
        xs=[rows[l] for l in labels]
        vals=[f"{x['original_48_recovered']}/{x['k3_net_correct_vs_learned']:+d}" for x in xs]
        lines.append(f"| {u} | "+" | ".join(vals)+" |")

    lines += ["","Attribution sur checkpoints figes uniquement; aucune promotion automatique."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
