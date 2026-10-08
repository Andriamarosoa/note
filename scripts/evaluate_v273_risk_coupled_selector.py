"""Measure a learned decision-coupled, class-specific risk selector on native K0-K6.

PREVIOUS EXPERIMENTS ARE NOT OVERWRITTEN.
This is a research ablation, not baseline promotion. All reporting includes
full 59,309 K0..K6 events, 7,385 true poly events, 7,493 eligible rows,
every true-K correction/regression, baseline-pred-K transition and per-fold
audit. No post-hoc threshold is chosen on outer-fold labels.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_v273_dynamic_heads import (
    EXPERTS, oof_experts, outer_experts
)
from scripts.evaluate_v273_neural_history_mix import (
    build_candidate_logits,cross_audits,full_train_audits,
    inputs,HEADS,H,KEEP
)
from scripts.learn_v273_risk_coupled_selector import fit_risk_selector
from scripts.summarize_v273_harmonic_global import (
    load_cohort,load_features,get_matrix
)
from scripts.yourmt3_exactk_common import FOLDS,paired,metrics,require

N=59309
N_GATED=7493

def transitions(y,b,p,fold):
    out=[]
    for f in (*FOLDS,"ALL"):
        where=(np.ones(len(y),bool) if f=="ALL" else fold==f)
        for old in (2,3,4):
            for new in range(7):
                if old==new:continue
                mask=where&(b==old)&(p==new)
                if mask.sum()==0:continue
                corr=int(np.sum(mask&(y==new)))
                regress=int(np.sum(mask&(y==old)))
                total=int(mask.sum())
                out.append({
                    "fold":str(f),"baseline_K":old,"predicted_K":new,
                    "actions":total,"corrections":corr,"regressions":regress,
                    "neutral":total-corr-regress,"net":corr-regress,
                    "by_true_K":{str(k):int(np.sum(mask&(y==k)))
                                 for k in range(7)},
                })
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",required=True,type=Path)
    parser.add_argument("--features",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    require(not args.output.exists(),"refusing result overwrite")
    y,b,idx,fold,member,onsets=load_cohort(args.cohort)
    elig,rows,names,_=load_features(
        args.features,y,b,idx,fold,member,onsets)
    require(len(y)==N and len(elig)==N_GATED,"cohort changed")
    require(sum(y>=2)==7385 and np.sum((y>=2)&(y==b))==2530,
            "baseline poly exact drift")
    yc=y[elig];bc=b[elig];fc=fold[elig]
    x={name:get_matrix(rows,names,field,prefix)
       for name,(field,prefix) in EXPERTS.items()}
    context_names=[k for k in names if
                   k.startswith(("spectral__","birth__","persistence__",
                                 "damping__"))][:26]
    raw=np.asarray([[row["features"][k] for k in context_names]
                    for row in rows],dtype=np.float64)
    require(np.isfinite(raw).all(),"invalid context features")
    prediction=b.copy()
    conditional_weights=np.full((len(elig),H,7),np.nan,np.float32)
    correct_probs=np.full((len(elig),7),np.nan,np.float32)
    keep_probs=np.full(len(elig),np.nan,np.float32)
    risk_weights=np.full((len(elig),H,2),np.nan,np.float32)
    logs={}
    for vf in FOLDS:
        tr=fc!=vf;va=fc==vf
        ptr=oof_experts(
            {name:arr[tr] for name,arr in x.items()},
            yc[tr],bc[tr],fc[tr]
        )
        testP=outer_experts(
            {name:{"train":arr[tr],"test":arr[va]}
             for name,arr in x.items()},
            yc[tr],bc[va])
        tr_logits,tr_mask,tr_types,tr_support=build_candidate_logits(ptr,bc[tr])
        va_logits,va_mask,va_types,va_support=build_candidate_logits(testP,bc[va])
        tr_audits=cross_audits(yc[tr],bc[tr],fc[tr],tr_logits)
        va_audits=np.broadcast_to(
            full_train_audits(yc[tr],bc[tr],fc[tr],tr_logits),
            (int(va.sum()),H,21)
        ).copy()
        scaler=StandardScaler().fit(raw[tr])
        xtr=inputs(
            tr_logits,tr_mask,tr_types,tr_support,tr_audits,
            np.clip(scaler.transform(raw[tr]),-6,6),bc[tr])
        xva=inputs(
            va_logits,va_mask,va_types,va_support,va_audits,
            np.clip(scaler.transform(raw[va]),-6,6),bc[va])
        action,result,learning_logs=fit_risk_selector(
            xtr,yc[tr],bc[tr],xva
        )
        require(np.isin(action,[2,3,4,5,6,KEEP]).all(),
                "unsupported output K0/K1")
        candidate=np.where(action==KEEP,bc[va],action)
        prediction[elig[va]]=candidate
        conditional_weights[va]=result["class_selection_weights"].numpy()
        correct_probs[va]=result["class_correct_prob"].numpy()
        keep_probs[va]=result["keep_preferred_prob"].numpy()
        risk_weights[va]=result["head_risk"].numpy()
        idx_rows=elig[va]
        perf=paired(y[idx_rows],b[idx_rows],prediction[idx_rows])["global"]
        logs[str(vf)]={"training_history":learning_logs,
                       "evaluation":perf}
        print(json.dumps({"fold":vf,
                          "corrections":perf["corrections"],
                          "regressions":perf["regressions"],
                          "net":perf["net"]}),flush=True)

    require(np.isfinite(conditional_weights).all() and
            np.isfinite(correct_probs).all() and
            np.isfinite(keep_probs).all() and
            np.isfinite(risk_weights).all(),"missing risk output")
    require(not np.any(np.isin(prediction[elig],(0,1))),
            "wrongly predicted unsupported K0 or K1")
    unchanged=np.ones(len(y),bool);unchanged[elig]=False
    require(np.array_equal(prediction[unchanged],b[unchanged]),
            "changed frozen excluded events")
    baseline=metrics(y,b)
    final=metrics(y,prediction)
    pair=paired(y,b,prediction)
    require(final["correct"]==baseline["correct"]+pair["global"]["net"],
            "global net mismatch")
    require(pair["global"]["net"]==pair["poly"]["net"],
            "nonpoly corrections despite support mask")

    k2to1=next((d for d in transitions(y,b,prediction,fold)
                 if d["fold"]=="ALL" and
                    d["baseline_K"]==2 and d["predicted_K"]==1),None)
    require(k2to1 is None,
            "unsupported K1 transition occurred")
    actions_by_K={}
    for k in (2,3,4):
        mask=b[elig]==k
        idx3=elig[mask]
        actions_by_K[str(k)]={"eligible_events":int(mask.sum()),
            "correct_baseline_edited":int(np.sum((y[idx3]==k)&(prediction[idx3]!=k))),
            "true_poly_events":int(np.sum(y[idx3]>=2)),
            "corrected_events":int(np.sum((y[idx3]!=k)&
                                              (prediction[idx3]==y[idx3]))),
        }
    notes=[
        "Same 59309 native events and 7493 eligible, folds 0/1/2/4; fold3/player05 excluded",
        "Only H0-H5 plus 4 transition and 4 KEEP adapters; history not all executable yet",
        "Audio features extracted up to +160ms: not equivalent to causal inference",
        "Separate fit per outer fold; inner OOF audits NEVER include row's own validation fold",
        "K0/K1 outputs structurally unsupported by acoustic heads: suppressed, therefore no improvements on true K0/K1 possible here",
        "Wrong baseline K2/K3/K4 when true K0/K1 cannot be corrected until a true nonpoly specialist is added",
        "Risk estimates are used directly in final action advantage, not merely auxiliary training",
        "Comparison uses previously exposed outer folds: not proof of generalization to truly new players",
        "Loss coefficients are training hyperparameters, not static inference head weights",
        "Freeze_local_combo untouched and promotion disabled"
    ]
    report=dict(
        experiment="v273_risk_coupled_neural_class_conditioned_selector",
        status="completed",automatic_promotion=False,
        baseline=baseline,modified=final,paired=pair,by_fold=logs,
        by_frozen_class=actions_by_K,
        transitions=transitions(y,b,prediction,fold),
        head_names=list(HEADS),comments=notes,
        architecture="learned class conditional head interaction plus risk-coupled candidate K2..K6 versus KEEP decision",
        constraints=dict(full_rows=N,eligible=N_GATED,folds=list(FOLDS),
                         no_fold3=True,no_player05=True,
                         nonpoly_proposal_allowed=False),
    )
    args.output.mkdir(parents=True)
    (args.output/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    np.savez_compressed(
        args.output/"risk-coupled-predictions.npz",
        global_index=idx,fold=fold,true_k=y,baseline=b,corrected=prediction,
        eligible_indices=idx[elig],class_selection_weights=conditional_weights,
        per_K_correct_probability=correct_probs,keep_preferred_probability=keep_probs,
        neural_per_head_risk=risk_weights
    )
    lines=[
        "# Class-conditional learned risk coupled to final KEEP / Exact-K decision",
        "",
        f"Complete {N} native events, {N_GATED} eligible, "
        "no fold3/player05. Reference unchanged.",
        "",
        "| System | Global K0–K6 | Poly K2–K6 | Corrections | Regressions | Net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for title,m,c in [
        ("Reference",baseline,None),("Risk-coupled NN",final,pair["global"])]:
        net_text="—" if c is None else "%+d" % int(c["net"])
        lines.append(f"| {title} | {100*m['exact']:.4f}% | "
                     f"{100*m['poly']['exact']:.4f}% | "
                     f"{'—' if c is None else c['corrections']} | "
                     f"{'—' if c is None else c['regressions']} | "
                     f"{net_text} |")
    lines+=["","## True K corrections vs regressions","",
            "| True K | Corrections | Regressions | Net |",
            "|---|---:|---:|---:|"]
    for k in range(7):
        by=pair["by_k"][str(k)]
        lines.append(f"| K{k} | {by['corrections']} | {by['regressions']} | {by['net']:+d} |")
    lines+=["","## Outer fold stats","",
            "| Fold | Corrections | Regressions | Net |",
            "|---|---:|---:|---:|"]
    for f in FOLDS:
        z=logs[str(f)]["evaluation"]
        lines.append(f"| {f} | {z['corrections']} | {z['regressions']} | {z['net']:+d} |")
    lines+=["","## Costliest observed transitions","",
            "| Base K → New K | Actions | Corrections | Regressions | Net |",
            "|---|---:|---:|---:|---:|"]
    for r in sorted((x for x in report["transitions"] if x["fold"]=="ALL"),
                    key=lambda x:x["net"])[:14]:
        lines.append(f"| {r['baseline_K']}→{r['predicted_K']} | "
                     f"{r['actions']} | {r['corrections']} | {r['regressions']} | "
                     f"{r['net']:+d} |")
    lines+=["","## Scientific caveats"]+["- "+v for v in notes]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    main()
