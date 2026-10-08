"""Native K0..K6 train OOF neural source-target subset/risk experiment.

Selection audits for each of the 64 H0-anchored subsets are computed
ONLY from the outer training fold; each inner fold's row receives its
audit descriptor calculated from the other two inner folds.
The first heldout fold never sees its true-K in features or selection.

This experiment DOES NOT use independent player05/fold3 data; do not
claim independent validation. Full exposed native 59,309 cohort.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scripts.summarize_v273_harmonic_global import load_cohort,load_features,get_matrix
from scripts.evaluate_v273_dynamic_heads import EXPERTS,oof_experts,outer_experts
from scripts.evaluate_v273_neural_history_mix import build_candidate_logits,HEADS,KEEP
from scripts.learn_v273_transition_combo_risk import (
    compute_direct_options,attach_oof_subset_audits,train_model,COMBOS
)
from scripts.yourmt3_exactk_common import metrics,paired,require,FOLDS

EXPECTED_N=59309
EXPECTED_REVISED=7493
CANDIDATES=np.array([2,3,4,5,6,7],int)


def inference_inputs(P,b,ctx,opt,mask):
    head_logits,active,typ,support=build_candidate_logits(P,b)
    head_prob=tf.nn.softmax(head_logits,axis=-1).numpy()
    keep=head_prob[:,:,KEEP].astype(np.float32)*active.astype(np.float32)
    return dict(subset_features=opt.astype(np.float32),
                subset_mask=mask.astype(bool),
                baseline=np.eye(7,dtype=np.float32)[b],
                context=ctx.astype(np.float32),keep_heads=keep)


def by_true_class(y,b,p):
    result={}
    for k in range(7):
        row=(y==k)
        fixes=int(np.sum(row&(b!=y)&(p==y)))
        reg=int(np.sum(row&(b==y)&(p!=y)))
        result[str(k)]={"events":int(np.sum(row)),
                         "corrections":fixes,"regressions":reg,
                         "net":fixes-reg}
    return result


def transitions(y,b,p):
    result=[]
    for a in (2,3,4):
        for k in range(7):
            if a==k:continue
            mask=(b==a)&(p==k)
            total=int(np.sum(mask))
            if total==0:continue
            cor=int(np.sum(mask&(y==k)))
            reg=int(np.sum(mask&(y==a)))
            result.append(dict(frozen_K=a,target_K=k,actions=total,
                corrections=cor,regressions=reg,net=cor-reg,
                neutral=total-cor-reg,
                by_true_K={str(true):int(np.sum(mask&(y==true)))
                           for true in range(7)}))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cohort",type=Path,required=True)
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    require(not args.output.exists(),"no overwriting diagnostics")
    y,base,idx,fold,members,starts=load_cohort(args.cohort)
    eligible,rows,names,_=load_features(args.features,y,base,idx,fold,members,starts)
    require(len(y)==EXPECTED_N and len(eligible)==EXPECTED_REVISED,
            "native cohort mismatch")
    require(tuple(sorted(np.unique(fold)))==tuple(FOLDS),"unexpected fold3")
    yc=y[eligible];bc=base[eligible];fc=fold[eligible]
    x={k:get_matrix(rows,names,source,prefixes)
       for k,(source,prefixes) in EXPERTS.items()}
    ctx_names=[n for n in names if
              n.startswith(("spectral__","birth__","persistence__",
                            "damping__"))][:26]
    raw=np.asarray([[r["features"][z] for z in ctx_names] for r in rows],
                   dtype=np.float64)
    require(np.isfinite(raw).all(),"bad audio feature matrix")
    pred=base.copy()
    weights=np.zeros((len(eligible),5,COMBOS),np.float32)
    predicted_candidate_risk=np.zeros((len(eligible),5,2),np.float32)
    estimated_keep=np.zeros(len(eligible),np.float32)
    fold_stats={}
    train_audit_rows=[]
    for outer in FOLDS:
        tr=fc!=outer;val=fc==outer
        fit_prob=oof_experts(
            {k:v[tr] for k,v in x.items()},
            yc[tr],bc[tr],fc[tr])
        test_prob=outer_experts(
            {k:{"train":v[tr],"test":v[val]} for k,v in x.items()},
            yc[tr],bc[val])
        tr_opt,tr_mask=compute_direct_options(fit_prob,bc[tr])
        va_opt,va_mask=compute_direct_options(test_prob,bc[val])
        tr_feat,va_feat,train_audits=attach_oof_subset_audits(
            tr_opt,tr_mask,bc[tr],yc[tr],fc[tr],
            va_opt,va_mask,bc[val])
        for entry in train_audits:
            entry["outer_fold"]=int(outer)
        train_audit_rows+=train_audits
        scale=StandardScaler().fit(raw[tr])
        tr_input=inference_inputs(
            fit_prob,bc[tr],np.clip(scale.transform(raw[tr]),-6,6),
            tr_feat,tr_mask)
        va_input=inference_inputs(
            test_prob,bc[val],np.clip(scale.transform(raw[val]),-6,6),
            va_feat,va_mask)
        actions,out,history=train_model(tr_input,yc[tr],bc[tr],va_input)
        require(np.isin(actions,[2,3,4,5,6,7]).all(),"invalid action")
        mapped=np.where(actions==KEEP,bc[val],actions)
        pred[eligible[val]]=mapped
        weights[val]=out["subset_attention"].numpy()
        predicted_candidate_risk[val,:,0]=out["candidate_correct"].numpy()
        predicted_candidate_risk[val,:,1]=out["candidate_regress"].numpy()
        estimated_keep[val]=out["keep_correct"].numpy()
        ix=eligible[val]
        per=paired(y[ix],base[ix],pred[ix])["global"]
        fold_stats[str(outer)]=dict(heldout_rows=int(np.sum(val)),
                                    corrections=per["corrections"],
                                    regressions=per["regressions"],
                                    net=per["net"],
                                    by_true_K=by_true_class(y[ix],base[ix],pred[ix]),
                                    training=history)
        print(json.dumps({"outer_fold":outer,"net":per["net"],
            "corrections":per["corrections"],"regressions":per["regressions"]}),
            flush=True)
    orig=metrics(y,base);mod=metrics(y,pred)
    pair=paired(y,base,pred)
    require(orig["correct"]+pair["global"]["net"]==mod["correct"],
            "native paired total mismatch")
    require(not np.any(np.isin(pred[eligible],(0,1))),
            "K0/K1 proposed without trained acoustic specialist")
    others=np.ones(len(y),bool);others[eligible]=False
    require(np.array_equal(pred[others],base[others]),
            "not allowed to edit outside frozen K2 K3 K4 cohort")
    require(np.isfinite(weights).all() and
            np.isfinite(predicted_candidate_risk).all(),"nonfinite estimates")
    report=dict(
        experiment="v273_source_target_64_subset_risk_arbiter",
        status="completed",promotion=False,
        freeze_reference=orig,neural_candidate=mod,
        native_paired=pair,
        source_target_transitions=transitions(y,base,pred),
        per_true_K=by_true_class(y,base,pred),
        folds=fold_stats,train_only_audited_subsets=len(train_audit_rows),
        configuration={
            "full_cohort":EXPECTED_N,"eligible":EXPECTED_REVISED,
            "head_total":14,"K2K3K4_directed_available":64,
            "other_K2to6_target_selections":32,
            "candidate_K":[2,3,4,5,6],
            "K0K1_specialist_support":False,
            "no_test_true_K_in_routing":True,
            "OOF_audit_excludes_current_inner_fold":True,
            "no_fold3_no_player05":True,
        },
        caveats=[
            "Outer evaluation folds have ALREADY been inspected in prior experiments, so these outcomes are exploratory and NOT independent confirmation.",
            "Historical player05/fold3 not included or available in these fixed cohort artifacts.",
            "Inner-OOF expert predictions of other folds may themselves have been fit using rows from the current inner fold; strict representation-level leave-fold-out audit independence is NOT established.",
            "The 64 explicit selections contain H0 by architectural contract; all 127 theoretical sets are NOT viable without changing architecture.",
            "All 14 heads retain action/KEEP interfaces, but only six specialists and applicable correction adapter are part of each target-directed subset.",
            "This experiment does not connect every historical fix or hypothesis.",
            "Selected combinations and action decisions are trainable by neural gradients; no test-fold tuned vetoes.",
            "No promotion to freeze_local_combo. +160ms future audio window remains.",
        ]
    )
    args.output.mkdir(parents=True)
    (args.output/"report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False,sort_keys=True)+"\n")
    fields=list(train_audit_rows[0])
    with (args.output/"per-selection-train-only-OOF-audits.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_audit_rows)
    np.savez_compressed(args.output/"predictions-and-selected-combinations.npz",
        global_index=idx,true_K=y,frozen_baseline_K=base,predicted_K=pred,
        fold=fold,eligible_global_index=idx[eligible],
        selected_subset_probabilities=weights,
        candidate_correction_regression_risk=predicted_candidate_risk,
        keep_probability=estimated_keep)
    lines=[
        "# Arbitre neuronal (K initial, K candidat, combinaison, risque, KEEP)",
        "",
        "Evaluation expérimentale sur folds 0,1,2,4 DEJA INSPECTES. "
        "**Ce n'est PAS une validation indépendante**.", "",
        "| Système | Exact-K global K0–K6 | Exact-K poly K2–K6 | Corrections | Régressions | Net |",
        "|---|---:|---:|---:|---:|---:|",
        f"| freeze_local_combo | {orig['exact']*100:.4f}% | "
        f"{orig['poly']['exact']*100:.4f}% | — | — | — |",
        f"| NN combinaisons et risques | {mod['exact']*100:.4f}% | "
        f"{mod['poly']['exact']*100:.4f}% | "
        f"{pair['global']['corrections']} | {pair['global']['regressions']} | "
        f"{pair['global']['net']:+d} |",
        "",
        "## Corrections et régressions par vrai K",
        "",
        "| Vrai K | Corrections | Régressions | Net |",
        "|---|---:|---:|---:|"
    ]
    for k in range(7):
        d=report["per_true_K"][str(k)]
        lines.append(f"| K{k} | {d['corrections']} | {d['regressions']} | {d['net']:+d} |")
    lines+=["","## Foldwise generalization (exposed folds)","",
            "| Fold | Fixes | Regressions | Net |","|---|---:|---:|---:|"]
    for f in FOLDS:
        d=fold_stats[str(f)]
        lines.append(f"| {f} | {d['corrections']} | {d['regressions']} | {d['net']:+d} |")
    lines+=["","## Plus mauvaises transitions","",
            "| Frozen → NN | Actions | Corrections | Régressions | Net |",
            "|---|---:|---:|---:|---:|"]
    for d in sorted(report["source_target_transitions"],key=lambda z:z["net"])[:14]:
        lines.append(f"| {d['frozen_K']}→{d['target_K']} | {d['actions']} | "
                     f"{d['corrections']} | {d['regressions']} | {d['net']:+d} |")
    lines+=["","## Contraintes et limites"]
    lines+=["- "+v for v in report["caveats"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
