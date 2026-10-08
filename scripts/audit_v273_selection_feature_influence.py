"""Forensic: do learned Exact-K subset selections actually use acoustic features / OOF audits?

Replicate ORIGINAL transition combination gate without any training change,
then intervene on input features at inference with the SAME learned weights:
  - audit_flatten: set per-subset OOF audit to the mean valid-subset audit,
    retaining scenario evidence but removing subset-specific reliability.
  - audit_reverse: permute OOF audit profiles across eligible subsets of
    the same source/candidate pair, retaining values and scale but
    breaking their alignment to candidate head combinations.
  - acoustic_shuffle: permute acoustic context among heldout events,
    retaining marginal feature distributions while losing event coupling.
  - direct_votes_flatten: erase subset-specific vote evidence by scenario.
No true K enters interventions; it is used only after prediction for
descriptive scoring.

A sensitivity result does NOT prove better generalization. It measures
whether audit and acoustic features influence this trained network.
"""
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.summarize_v273_harmonic_global import load_cohort,load_features,get_matrix
from scripts.evaluate_v273_dynamic_heads import EXPERTS,oof_experts,outer_experts
from scripts.evaluate_v273_transition_combo_risk import inference_inputs
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options,attach_oof_subset_audits,train_model,COMBOS)
from scripts.yourmt3_exactk_common import require,metrics,paired,FOLDS


def interventions(batch,seed=27402):
    """Matched, label-blind input counterfactuals."""
    combo=batch["subset_features"]
    support=batch["subset_mask"]
    options={}
    for variant in ("audit_flatten","audit_reverse","acoustic_shuffle",
                    "direct_votes_flatten"):
        sample={k:np.asarray(v).copy() for k,v in batch.items()}
        if variant=="acoustic_shuffle":
            rng=np.random.default_rng(seed)
            perm=rng.permutation(len(batch["context"]))
            # Permutation within each frozen predicted K prevents trivially
            # confounding acoustic context with baseline class differences.
            base=np.argmax(batch["baseline"],axis=1)
            for k in (2,3,4):
                ids=np.flatnonzero(base==k)
                sample["context"][ids]=batch["context"][rng.permutation(ids)]
        else:
            target=sample["subset_features"]
            span=slice(6,10) if variant.startswith("audit_") else slice(0,6)
            for i in range(len(combo)):
                for target_K in range(5):
                    eligible=np.flatnonzero(support[i,target_K])
                    if len(eligible)<2:continue
                    original=combo[i,target_K,eligible,span].copy()
                    if variant in ("audit_flatten","direct_votes_flatten"):
                        value=original.mean(axis=0,keepdims=True)
                        target[i,target_K,eligible,span]=value
                    else:
                        target[i,target_K,eligible,span]=original[::-1]
        options[variant]=sample
    return options


def predicted(model,data):
    out=model(data,training=False)
    classes=np.array([2,3,4,5,6,7],int)
    action=classes[np.argmax(out["action_logits"].numpy(),axis=1)]
    original=np.argmax(data["baseline"],axis=1)
    p=np.where(action==7,original,action)
    return p,out["subset_attention"].numpy(),out["action_logits"].numpy()


def compare(pred0,pred,attention0,attention,y,base):
    prev_good=pred0==y
    now_good=pred==y
    totalvariation=np.sum(np.abs(attention-attention0),axis=2)*0.5
    # Includes unavailable targets, where attention zero.
    eligible=np.any(attention0>0.,axis=2)
    result={
        "n":int(len(y)),
        "prediction_changes":int(np.sum(pred0!=pred)),
        "rescued_errors":int(np.sum(~prev_good&now_good)),
        "new_errors":int(np.sum(prev_good&~now_good)),
        "net_correct_delta":int(np.sum(now_good)-np.sum(prev_good)),
        "attention_mean_TVD":float(np.mean(totalvariation[eligible])),
        "attention_fraction_changed_1e_4":float(np.mean(
            totalvariation[eligible]>1e-4)),
        "by_true_K":{
            str(k):{
                "n":int(np.sum(y==k)),
                "delta_correct":int(np.sum((y==k)&now_good)-
                                    np.sum((y==k)&prev_good)),
                "decision_changes":int(np.sum((y==k)&(pred0!=pred))),
            } for k in range(7)
        },
        "by_frozen_K":{
            str(k):{
                "n":int(np.sum(base==k)),
                "delta_correct":int(np.sum((base==k)&now_good)-
                                    np.sum((base==k)&prev_good)),
                "decision_changes":int(np.sum((base==k)&(pred0!=pred))),
            } for k in (2,3,4)
        }
    }
    require(np.isfinite(result["attention_mean_TVD"]),"bad TVD")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cohort",type=Path,required=True)
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refuse overwrite")

    with np.load(a.reference,allow_pickle=False) as z:
        old={k:z[k] for k in z.files}
    y,base,idx,fold,member,onset=load_cohort(a.cohort)
    elig,rows,names,_=load_features(a.features,y,base,idx,fold,member,onset)
    require(len(y)==59309 and len(elig)==7493,"cohort drift")
    # NOTE reference uses different key naming than older neural runs.
    for key,exp in (
        ("global_index",idx),("true_K",y),
        ("frozen_baseline_K",base),("fold",fold),
        ("eligible_global_index",idx[elig])
    ):
        require(key in old and np.array_equal(old[key],exp),
                "frozen provenance mismatch: "+key)
    yc=y[elig];bc=base[elig];fc=fold[elig]
    x={n:get_matrix(rows,names,field,prefix)
       for n,(field,prefix) in EXPERTS.items()}
    context_names=[n for n in names if n.startswith(
        ("spectral__","birth__","persistence__","damping__"))][:26]
    raw=np.array([[row["features"][n] for n in context_names]
                  for row in rows],dtype=np.float64)
    require(np.isfinite(raw).all(),"nonfinite acoustic input")
    results={}
    original_output=base.copy()
    variant_output={k:base.copy() for k in
                    ("audit_flatten","audit_reverse","acoustic_shuffle",
                     "direct_votes_flatten")}
    for outer in FOLDS:
        tr=fc!=outer;va=fc==outer
        ptr=oof_experts({k:v[tr] for k,v in x.items()},
                        yc[tr],bc[tr],fc[tr])
        pval=outer_experts({k:{"train":v[tr],"test":v[va]}
                            for k,v in x.items()},yc[tr],bc[va])
        tr_direct,tr_mask=compute_direct_options(ptr,bc[tr])
        va_direct,va_mask=compute_direct_options(pval,bc[va])
        tr_opt,va_opt,_=attach_oof_subset_audits(
            tr_direct,tr_mask,bc[tr],yc[tr],fc[tr],
            va_direct,va_mask,bc[va])
        standard=StandardScaler().fit(raw[tr])
        inputs_tr=inference_inputs(
            ptr,bc[tr],np.clip(standard.transform(raw[tr]),-6,6),
            tr_opt,tr_mask)
        inputs_val=inference_inputs(
            pval,bc[va],np.clip(standard.transform(raw[va]),-6,6),
            va_opt,va_mask)
        decision,out,_,model=train_model(
            inputs_tr,yc[tr],bc[tr],inputs_val,return_model=True)
        baseline_pred,weights,_=predicted(model,inputs_val)
        require(np.array_equal(np.where(decision==7,bc[va],decision),
                               baseline_pred),"same-checkpoint main infer mismatch")
        original_output[elig[va]]=baseline_pred
        view={}
        for name,counterfactual in interventions(inputs_val,seed=27402+int(outer)).items():
            modified,weight,logits=predicted(model,counterfactual)
            variant_output[name][elig[va]]=modified
            view[name]=compare(
                baseline_pred,modified,weights,weight,yc[va],bc[va])
        results[str(outer)]=view
        print(json.dumps({"fold":int(outer),"original_correct":
            int(np.sum(baseline_pred==yc[va])),
            "ablations":{k:dict(decision_changes=v["prediction_changes"],
                               attention_TVD=round(v["attention_mean_TVD"],5),
                               net=v["net_correct_delta"])
                         for k,v in view.items()}}),flush=True)

    require(np.array_equal(original_output,old["predicted_K"]),
            "replica drift vs reference: cannot attribute changes to features")
    report=dict(status="completed",
                run_original=37755994598,
                tested_native=59309,tested_eligible=7493,
                **{"exact_replica_disagreements":int(np.sum(
                    original_output!=old["predicted_K"]))},
                original_metrics=metrics(y,original_output),
                original_paired=paired(y,base,original_output),
                interventions={},
                per_fold=results,
                feature_names={
                    "acoustic_context":context_names,
                    "per_subset_direct_vote":[
                        "target_minus_keep_margin","keep_prob","target_prob",
                        "direct_vote_target_gt_keep","subset_head_fraction",
                        "has_transition_corrector"],
                    "OOF_audit_prior":[
                        "log_proposal_count","smoothed_correction_rate",
                        "smoothed_regression_rate","smoothed_neutral_rate"
                    ],
                    "historical_archived_audits_integrated":False,
                },
                caveats=[
                    "Each intervention uses exactly the same trained per-fold model; no refitting.",
                    "Flattening/permuting audit features creates counterfactual off-distribution combinations; sensitivity is not a claim of deployable improvement.",
                    "Acoustic shuffle changes event-to-context assignment only within the original baseline class.",
                    "Earlier historical cluster A/B, pitch-shift, morphological and other audit results are not directly connected as executable features.",
                    "OOF audit priors exclude direct validation-fold labels, but full representation-level independence of inner source experts has not been established.",
                    "Outer folds have been inspected; no genuinely new-player validation.",
                    "Frozen reference is untouched; no model promotion."
                ])
    for variant,pred in variant_output.items():
        # Global decisions are measured from paired predictions.
        # Attention was measured on real unmodified/modified model
        # forward outputs per fold. Zero placeholder attention would
        # produce an invalid empty denominator.
        origgood=(original_output[elig]==yc)
        newgood=(pred[elig]==yc)
        summary=dict(
            n=int(len(elig)),
            prediction_changes=int(np.sum(original_output[elig]!=pred[elig])),
            rescued_errors=int(np.sum(~origgood&newgood)),
            new_errors=int(np.sum(origgood&~newgood)),
            net_correct_delta=int(np.sum(newgood)-np.sum(origgood)),
        )
        # attention_TVD across folds, weighted by number of active candidates
        numer=0.;denom=0.
        for f in FOLDS:
            m=fc==f
            numer+=results[str(f)][variant]["attention_mean_TVD"]*int(m.sum())
            denom+=int(m.sum())
        summary["attention_mean_TVD"] = float(numer/denom)
        summary["full_native_metrics"]=metrics(y,pred)
        summary["paired_vs_frozen"]=paired(y,base,pred)["global"]
        report["interventions"][variant]=summary
    a.output.mkdir(parents=True)
    (a.output/"audit.json").write_text(json.dumps(
        report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    np.savez_compressed(a.output/"feature-intervention-decisions.npz",
        global_index=idx,baseline=base,truth=y,fold=fold,
        main_neural_pred=original_output,
        **{f"counterfactual_{k}":v for k,v in variant_output.items()})
    lines=[
        "# Impact mesuré des audits et des caractéristiques sur la sélection neuronale",
        "",
        "Ablations des **entrées du même réseau entraîné** par fold.",
        "**Ce n'est pas une modification promue du système.**",
        "",
        "| Variable modifiée | Décisions différentes | Δ corrects | Variation moyenne des poids (TVD) |",
        "|---|---:|---:|---:|"
    ]
    for k,r in report["interventions"].items():
        lines.append(f"| {k} | {r['prediction_changes']} | "
                     f"{r['net_correct_delta']:+d} | "
                     f"{r['attention_mean_TVD']:.5f} |")
    lines+=["","## Interprétation : limites des anciens audits"]
    lines+=["- "+v for v in report["caveats"]]
    (a.output/"verdict.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
