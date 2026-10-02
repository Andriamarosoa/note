"""Frozen learned-gate intervention audit for V27.3 Exact-K.

No training. Uses the trained learned-gate arm and its nested Exact-K network.
Interventions modify only the V8.8 pseudo-count inputs at inference:
  - force the shared scalar gate to fixed values;
  - gate router only;
  - gate local-cardinality only;
  - switch router/cardinality off separately.

This is a sensitivity/intervention audit, not a deployable correction.
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

def load_cluster_a(path,global_index):
    rows=list(csv.DictReader(open(path,newline="")))
    by={}
    for r in rows:
        c=int(r["cluster"]); y=int(r["true_k"]); p=int(r["predicted_k"])
        d=by.setdefault(c,{"n":0,"under":0})
        d["n"]+=1; d["under"]+=p<y
    cid=max(by,key=lambda c:(by[c]["under"]/by[c]["n"],by[c]["n"]))
    ids={int(r["global_index"]) for r in rows if int(r["cluster"])==cid}
    return cid,np.isin(np.asarray(global_index,np.int64),list(ids))

def metric(k,p,sel):
    y=np.asarray(k)[sel]; q=np.asarray(p)[sel]
    return {
        "rows":int(len(y)),
        "exact":float(np.mean(q==y)) if len(y) else None,
        "under":int(np.sum(q<y)),
        "over":int(np.sum(q>y)),
        "pred_hist":np.bincount(q,minlength=7).tolist(),
    }

def transition(k,a,b,sel):
    y=np.asarray(k)[sel]; x=np.asarray(a)[sel]; z=np.asarray(b)[sel]
    return {
        "corrected":int(np.sum((x!=y)&(z==y))),
        "regressed":int(np.sum((x==y)&(z!=y))),
        "correct_to_under":int(np.sum((x==y)&(z<y))),
        "correct_to_over":int(np.sum((x==y)&(z>y))),
        "under_to_correct":int(np.sum((x<y)&(z==y))),
        "over_to_correct":int(np.sum((x>y)&(z==y))),
        "net_correct":int(np.sum(z==y)-np.sum(x==y)),
    }

def gate_summary(g,sel):
    x=np.asarray(g,float)[sel]
    return {
        "n":int(len(x)),
        "mean":float(np.mean(x)),
        "median":float(np.median(x)),
        "p10":float(np.percentile(x,10)),
        "p90":float(np.percentile(x,90)),
    }

def batches(cache,idx,batch=128):
    idx=np.asarray(idx,np.int64)
    for s in range(0,len(idx),batch):
        ids=idx[s:s+batch]
        yield ids,batch_inputs(cache,ids,FRAMES)

def apply_gates(x,router_gate,card_gate):
    cand=np.asarray(x["candidate_set"],np.float32).copy()
    stats=np.asarray(x["cluster_stats"],np.float32).copy()
    rg=np.asarray(router_gate,np.float32)
    cg=np.asarray(card_gate,np.float32)
    if rg.ndim==0: rg=np.full(len(cand),float(rg),np.float32)
    if cg.ndim==0: cg=np.full(len(cand),float(cg),np.float32)
    cand[:,:,V88_FEATURE_DIM] *= rg[:,None]
    cand[:,:,V88_FEATURE_DIM+1:V88_FEATURE_DIM+5] *= cg[:,None,None]
    stats[:,4] *= rg
    stats[:,5:8] *= cg[:,None]
    return {
        "candidate_set":cand,
        "candidate_mask":np.asarray(x["candidate_mask"],np.float32),
        "cluster_stats":stats,
        "spectral_map":np.asarray(x["spectral_map"],np.float32),
    }

def predict_nested(base,cache,outer,router_gate,card_gate):
    probs=[]
    rg=np.asarray(router_gate)
    cg=np.asarray(card_gate)
    for start in range(0,len(outer),128):
        ids=outer[start:start+128]
        x=batch_inputs(cache,ids,FRAMES)
        r=router_gate if rg.ndim==0 else rg[start:start+len(ids)]
        c=card_gate if cg.ndim==0 else cg[start:start+len(ids)]
        xx=apply_gates(x,r,c)
        probs.append(np.asarray(base(xx,training=False),np.float32))
    return np.concatenate(probs,axis=0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--bundle",type=Path,required=True)
    ap.add_argument("--config",type=Path,required=True)
    ap.add_argument("--weights",type=Path,required=True)
    ap.add_argument("--saved-predictions",type=Path,required=True)
    ap.add_argument("--cluster-rows",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)

    cache,parts,_=load_bundle(a.bundle,a.config)
    outer=np.asarray(parts["outer"],np.int64)
    k=np.minimum(cache["exact"][outer].astype(np.int32),6)
    with np.load(a.saved_predictions,allow_pickle=False) as z:
        saved={key:np.asarray(z[key]) for key in z.files}
    np.testing.assert_array_equal(saved["global_index"],outer)
    np.testing.assert_array_equal(saved["k"],k)

    model=build_model("learned_gate",SEED)
    model.load_weights(a.weights)
    import tensorflow as tf
    nested=[layer for layer in model.layers if isinstance(layer,tf.keras.Model)]
    require(len(nested)==1,f"expected one nested Exact-K model, got {[x.name for x in nested]}")
    base=nested[0]
    gate_model=tf.keras.Model(model.inputs,model.get_layer("v273_gate_value").output)

    actual_prob=[]; actual_gate=[]
    for _,x in batches(cache,outer):
        actual_prob.append(np.asarray(model(x,training=False),np.float32))
        actual_gate.append(np.asarray(gate_model(x,training=False),np.float32).reshape(-1))
    actual_prob=np.concatenate(actual_prob); actual_gate=np.concatenate(actual_gate)
    actual_pred=actual_prob.argmax(1).astype(np.int32)
    np.testing.assert_array_equal(actual_pred,saved["learned_gate_predicted"])
    max_gate_err=float(np.max(np.abs(actual_gate-saved["learned_gate_gate"])))
    require(max_gate_err<1e-5,f"gate replay mismatch {max_gate_err}")

    # External actual-gate replay proves that the intervention path is equivalent.
    replay_prob=predict_nested(base,cache,outer,actual_gate,actual_gate)
    replay_pred=replay_prob.argmax(1).astype(np.int32)
    require(np.array_equal(replay_pred,actual_pred),"external gate replay changes decisions")
    max_prob_err=float(np.max(np.abs(replay_prob-actual_prob)))
    require(max_prob_err<1e-5,f"external gate replay mismatch {max_prob_err}")

    always=saved["always_on_predicted"].astype(np.int32)
    cid,A=load_cluster_a(a.cluster_rows,outer)
    K3=k==3
    corrected=K3&(always!=3)&(actual_pred==3)
    regressed=K3&(always==3)&(actual_pred!=3)
    correct_both=K3&(always==3)&(actual_pred==3)
    wrong_both=K3&(always!=3)&(actual_pred!=3)

    interventions={
        "actual_gate":(actual_gate,actual_gate),
        "router_actual_card_on":(actual_gate,1.0),
        "router_on_card_actual":(1.0,actual_gate),
        "router_off_card_on":(0.0,1.0),
        "router_on_card_off":(1.0,0.0),
        "both_off":(0.0,0.0),
    }
    for v in (0.25,0.50,0.75,1.00):
        interventions[f"both_fixed_{v:.2f}"]=(v,v)

    prediction_bank={"always_on":always,"actual_gate":actual_pred}
    result={
        "status":"completed",
        "training":False,
        "model":"frozen learned_gate arm",
        "cluster_A_id":int(cid),
        "replay":{"gate_max_abs_error":max_gate_err,"external_probability_max_abs_error":max_prob_err},
        "k3_transition_from_always_on":{
            "baseline_correct":int(np.sum(K3&(always==3))),
            "learned_correct":int(np.sum(K3&(actual_pred==3))),
            "corrected":int(corrected.sum()),
            "regressed":int(regressed.sum()),
            "under_to_correct":int(np.sum(K3&(always<3)&(actual_pred==3))),
            "over_to_correct":int(np.sum(K3&(always>3)&(actual_pred==3))),
            "correct_to_under":int(np.sum(K3&(always==3)&(actual_pred<3))),
            "correct_to_over":int(np.sum(K3&(always==3)&(actual_pred>3))),
        },
        "k3_gate_by_transition":{
            "corrected":gate_summary(actual_gate,corrected),
            "regressed":gate_summary(actual_gate,regressed),
            "correct_both":gate_summary(actual_gate,correct_both),
            "wrong_both":gate_summary(actual_gate,wrong_both),
        },
        "interventions":{},
        "limitations":[
            "Fixed/mixed gates are inference interventions on a network trained with the learned scalar gate.",
            "They measure sensitivity of the frozen learned-gate model; they do not estimate retrained multi-gate performance.",
            "Mixed router/card interventions can be out of the learned training distribution.",
        ],
    }

    for name,(rg,cg) in interventions.items():
        prob=actual_prob if name=="actual_gate" else predict_nested(base,cache,outer,rg,cg)
        pred=prob.argmax(1).astype(np.int32)
        prediction_bank[name]=pred
        result["interventions"][name]={
            "all":metric(k,pred,np.ones(len(k),bool)),
            "K2":metric(k,pred,k==2),
            "K3":metric(k,pred,K3),
            "K4":metric(k,pred,k==4),
            "cluster_A":metric(k,pred,A),
            "K3_vs_actual":transition(k,actual_pred,pred,K3),
            "K3_vs_always_on":transition(k,always,pred,K3),
        }

    # Exact case-level attribution for the 48 K3 regressions caused by learned gating.
    reg = K3 & (always == 3) & (actual_pred != 3)
    card_on = prediction_bank["router_actual_card_on"]
    router_on = prediction_bank["router_on_card_actual"]
    both_on = prediction_bank["both_fixed_1.00"]

    card_rec = reg & (card_on == 3)
    router_rec = reg & (router_on == 3)
    both_rec = reg & (both_on == 3)

    attribution = {
        "regressed_rows": int(reg.sum()),
        "actual_under": int(np.sum(reg & (actual_pred < 3))),
        "actual_over": int(np.sum(reg & (actual_pred > 3))),
        "cardinality_on_recovers": int(card_rec.sum()),
        "router_on_recovers": int(router_rec.sum()),
        "both_on_recovers": int(both_rec.sum()),
        "cardinality_only_sufficient": int(np.sum(card_rec & ~router_rec)),
        "router_only_sufficient": int(np.sum(router_rec & ~card_rec)),
        "either_single_sufficient_both": int(np.sum(card_rec & router_rec)),
        "neither_single_but_both_on_recovers": int(np.sum(~card_rec & ~router_rec & both_rec)),
        "not_recovered_even_both_on": int(np.sum(reg & ~both_rec)),
        "by_actual_direction": {}
    }
    for label, direction in (("under", actual_pred < 3), ("over", actual_pred > 3)):
        rr = reg & direction
        attribution["by_actual_direction"][label] = {
            "rows": int(rr.sum()),
            "cardinality_on_recovers": int(np.sum(rr & (card_on == 3))),
            "router_on_recovers": int(np.sum(rr & (router_on == 3))),
            "both_on_recovers": int(np.sum(rr & (both_on == 3))),
        }

    # Also record whether the same interventions damage the 29 K3 rows that learned gating corrected.
    corr = K3 & (always != 3) & (actual_pred == 3)
    attribution["learned_gate_corrected_rows"] = int(corr.sum())
    attribution["corrected_rows_broken_by_cardinality_on"] = int(np.sum(corr & (card_on != 3)))
    attribution["corrected_rows_broken_by_router_on"] = int(np.sum(corr & (router_on != 3)))
    attribution["corrected_rows_broken_by_both_on"] = int(np.sum(corr & (both_on != 3)))

    result["k3_case_attribution"] = attribution
    np.savez_compressed(
        a.output/"intervention-predictions.npz",
        global_index=outer,k=k,actual_gate_value=actual_gate,
        **{name:pred for name,pred in prediction_bank.items()}
    )
    (a.output/"report.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")

    t=result["k3_transition_from_always_on"]; gs=result["k3_gate_by_transition"]
    lines=[
        "# Audit d'intervention du learned gate — K3",
        "",
        "Aucun entrainement. Modele learned-gate fige.",
        "",
        "## Transition K3 always-on -> learned gate",
        "",
        f"- Corrects always-on: {t['baseline_correct']}",
        f"- Corrects learned gate: {t['learned_correct']}",
        f"- Corriges: {t['corrected']} (under->correct {t['under_to_correct']}, over->correct {t['over_to_correct']})",
        f"- Regresses: {t['regressed']} (correct->under {t['correct_to_under']}, correct->over {t['correct_to_over']})",
        "",
        "## Gate observe",
        "",
        f"- Gate moyen K3 corriges: {gs['corrected']['mean']:.3f}",
        f"- Gate moyen K3 regressés: {gs['regressed']['mean']:.3f}",
        f"- Gate moyen K3 corrects dans les deux: {gs['correct_both']['mean']:.3f}",
        f"- Gate moyen K3 faux dans les deux: {gs['wrong_both']['mean']:.3f}",
        "",
        "## Attribution exacte des 48 regressions K3",
        "",
        f"- local_cardinality remis a 100% recupere: {attribution['cardinality_on_recovers']}/48",
        f"- router remis a 100% recupere: {attribution['router_on_recovers']}/48",
        f"- les deux remis a 100% recuperent: {attribution['both_on_recovers']}/48",
        f"- cardinality seule suffisante: {attribution['cardinality_only_sufficient']}",
        f"- router seul suffisant: {attribution['router_only_sufficient']}",
        f"- les deux interventions individuelles recuperent le meme cas: {attribution['either_single_sufficient_both']}",
        f"- aucune intervention seule mais les deux ensemble recuperent: {attribution['neither_single_but_both_on_recovers']}",
        f"- non recuperes meme avec les deux a 100%: {attribution['not_recovered_even_both_on']}",
        "",
        "## Interventions",
        "",
        "| Intervention | K2 exact | K3 exact | K4 exact | A exact | K3 under | K3 over |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name,x in result["interventions"].items():
        lines.append(
            f"| {name} | {100*x['K2']['exact']:.2f}% | {100*x['K3']['exact']:.2f}% | "
            f"{100*x['K4']['exact']:.2f}% | {100*x['cluster_A']['exact']:.2f}% | "
            f"{x['K3']['under']} | {x['K3']['over']} |"
        )
    lines += [
        "",
        "Les interventions mixtes/fixes sont des tests de sensibilite du modele fige, pas une estimation d'une architecture retrainée.",
    ]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
