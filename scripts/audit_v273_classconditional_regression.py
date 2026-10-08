"""Forensic root-cause audit of frozen K0..K6 neural head-selection regressions.

Compares the ORIGINAL 14-head global router to the NEW K-conditional
router on the same 59309 native events, and inspects all 7493 eligible
events. Does not train, change thresholds, or promote any new model.

Outputs reproducible per-true-K, per original-K, per action transition,
per fold, and paired old/new regression tables, plus neural selection
and risk-head diagnostics. Pure diagnostic "revert" counterfactuals are
labeled as post-hoc, NEVER as independently validated improvements.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

N_NATIVE=59309
N_ELIGIBLE=7493
HEADS=("H0_baseline","H1_spectral","H2_lifecycle","H3_harmonics",
       "H4_fundamentals","H5_sources",
       "C23","C32","C34","C43",
       "F_keep2","F_keep3","F_keep4","F_keep_any")
FOLDS=(0,1,2,4)

def require(expr,msg):
    if not expr:raise AssertionError(msg)
def load(p):
    with np.load(p,allow_pickle=False) as f:
        return {k:f[k] for k in f.files}
def count(y,b,p,m):
    m=np.asarray(m,bool)
    change=m & (p!=b)
    fixes=change & (p==y)
    regressions=change & (b==y)
    n=int(change.sum());cor=int(fixes.sum());reg=int(regressions.sum())
    require(cor+reg<=n,"double counted repair")
    return dict(events=int(m.sum()),actions=n,corrections=cor,
                regressions=reg,neutral=n-cor-reg,net=cor-reg)
def fullmetrics(y,b,p):
    poly=y>=2
    return dict(global_exact_pct=100*float(np.mean(y==p)),
                poly_exact_pct=100*float(np.mean(y[poly]==p[poly])),
                global_correct=int(np.sum(y==p)),
                poly_correct=int(np.sum(y[poly]==p[poly])),
                paired=count(y,b,p,np.ones(len(y),bool)))
def table_by_k(y,b,p):
    return {str(k):count(y,b,p,y==k) for k in range(7)}

def transition_table(y,b,p,fold):
    details=[]
    for f in (*FOLDS,"ALL"):
        byfold=np.ones(len(y),bool) if f=="ALL" else fold==f
        for source in (2,3,4):
            for target in range(7):
                if target==source:continue
                m=byfold & (b==source)&(p==target)
                if m.sum()==0:continue
                stats=count(y,b,p,m)
                stats.update(fold=str(f),original_K=source,new_K=target,
                             trueK_counts={str(k):int(np.sum(m&(y==k)))
                                           for k in range(7)})
                details.append(stats)
    return sorted(details,key=lambda x:(x["fold"]!="ALL",x["fold"],-x["regressions"]))

def mask_extract(data):
    k=("eligible_indices" if "eligible_indices" in data else "eligible_global_indices")
    return np.asarray(data[k],dtype=np.int64)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--conditional",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refuse audit overwrite")
    old=load(a.original)
    new=load(a.conditional)
    for key in ("global_index","true_k","baseline","fold"):
        require(key in old and key in new and
                np.array_equal(old[key],new[key]),"cohort drift "+key)
    for d in (old,new):require("corrected" in d,"missing predictions")
    y=old["true_k"];b=old["baseline"];fold=old["fold"]
    po=old["corrected"];pn=new["corrected"]
    n=len(y)
    require(n==N_NATIVE and np.isin(y,range(7)).all() and
            np.isin(pn,range(7)).all() and np.isin(po,range(7)).all(),
            "full native sample drift")
    require(set(np.unique(fold))==set(FOLDS),"unexpected fold3")
    eligible_o=mask_extract(old);eligible_n=mask_extract(new)
    require(len(eligible_o)==len(eligible_n)==N_ELIGIBLE and
            np.array_equal(eligible_o,eligible_n),"candidate drift")
    lookup={int(k):i for i,k in enumerate(old["global_index"])}
    ii=np.array([lookup[int(k)] for k in eligible_n],int)
    require(np.isin(b[ii],(2,3,4)).all(),"eligibility not based on frozen baseline")
    only=np.ones(n,bool);only[ii]=False
    require(np.array_equal(po[only],b[only]) and
            np.array_equal(pn[only],b[only]),"modified outside eligible")
    require(int(np.sum(b==y))==48454,"baseline score drift")
    new_W=new["router_head_weights_by_hypothetical_k"]
    new_risk=new["router_head_risk"]
    new_w=new["router_head_weights"]
    require(new_W.shape==(N_ELIGIBLE,14,7),"unexpected per K router tensor")
    require(new_risk.shape==(N_ELIGIBLE,14,2),"unexpected risk tensor")
    require(new_w.shape==(N_ELIGIBLE,14),"unexpected marginal weights")
    require(np.isfinite(new_W).all() and np.isfinite(new_risk).all(),
            "nonfinite neural output")
    require(np.allclose(new_W.sum(axis=1),1,atol=1e-5),"per K weight sums drift")

    base_stats=fullmetrics(y,b,b)
    old_stats=fullmetrics(y,b,po)
    new_stats=fullmetrics(y,b,pn)
    by_k=table_by_k(y,b,pn)
    by_orig={str(k):count(y,b,pn,b==k) for k in range(7)}
    by_fold={str(f):count(y,b,pn,fold==f) for f in FOLDS}
    transitions=transition_table(y,b,pn,fold)
    good_base=b==y
    edited=pn!=b
    oldgood=po==y
    newgood=pn==y
    paired_versions=dict(
        old_correct_new_wrong=int(np.sum(oldgood & ~newgood)),
        old_wrong_new_correct=int(np.sum(~oldgood & newgood)),
        old_and_new_correct=int(np.sum(oldgood & newgood)),
        both_wrong=int(np.sum(~oldgood & ~newgood)),
        changed_predictions_between_versions=int(np.sum(po!=pn)),
        old_vs_new_delta_correct=int(np.sum(newgood)-np.sum(oldgood)),
        new_regressions_from_frozen_base=int(np.sum(good_base&edited)),
        new_corrections_from_frozen_base=int(np.sum(~good_base&newgood&edited)),
        new_neutral_edits=int(np.sum(~good_base&~newgood&edited)),
    )
    violations=[]
    for k in (0,1):
        # For K0 and K1, the *only allowed class-specific voter is H0*.
        # But H0's adapter has zero mass on both classes (clipped eps).
        eligible_k=(pn[ii]==k)
        violations.append(dict(target_K=k,
            selected_actions=int(eligible_k.sum()),
            originating_K={str(bk):int(np.sum(eligible_k&(b[ii]==bk)))
                           for bk in (2,3,4)},
            only_k_support_is_head_H0=bool(np.allclose(new_W[:,0,k],1,atol=1e-6)),
            zero_mass_from_all_other_heads=bool(np.all(new_W[:,1:,k]<1e-6)),
        ))
    # Distinguish losses caused by wrong downward K0/K1 from all other errors.
    down=(pn<=1)&edited
    down_details=count(y,b,pn,down)
    downward_poly_loss=count(y,b,pn,down&(y>=2))
    polychanged=count(y,b,pn,edited&(y>=2))
    # No new model or feature learned in these ablations: descriptive only.
    counterfactual={}
    tests={
        "revert_all_K0_K1_outputs":np.where(down,b,pn),
        "revert_edits_of_base_pred_K3":np.where((b==3)&edited,b,pn),
        "revert_edits_of_base_pred_K4":np.where((b==4)&edited,b,pn),
        "revert_edits_of_base_pred_K3_K4":np.where(np.isin(b,(3,4))&edited,b,pn)
    }
    for key,pred in tests.items():
        counterfactual[key]=fullmetrics(y,b,pred)
    # Calibration diagnostic: risk values are per head, not actual final
    # correction probability, and DO NOT feed back to the decoder.
    riskby={}
    for typ,subset in {
        "changed_and_fixed":edited[ii]&(pn[ii]==y[ii]),
        "changed_and_regressed":edited[ii]&(b[ii]==y[ii]),
        "changed_neutral":edited[ii]&(pn[ii]!=y[ii])&(b[ii]!=y[ii]),
        "kept":pn[ii]==b[ii]
    }.items():
        riskby[typ]=dict(rows=int(subset.sum()),
            mean_head_correction_risk=[float(v) for v in new_risk[subset,:,0].mean(axis=0)],
            mean_head_regression_risk=[float(v) for v in new_risk[subset,:,1].mean(axis=0)])
    # Per-head class weights for K2-K4 on actions that destroy valid accords.
    selection_insight={}
    for k in (2,3,4):
        correct_baseline_edited=(b[ii]==k)&(y[ii]==k)&(pn[ii]!=k)
        subset=correct_baseline_edited
        others=(b[ii]==k)&(y[ii]==k)&(pn[ii]==k)
        selection_insight[str(k)]=dict(
            original_correct_but_changed=int(subset.sum()),
            originally_correct_and_preserved=int(others.sum()),
            mean_head_weights_for_true_k_if_regressed=
              {HEADS[h]:float(np.mean(new_W[subset,h,k])) if subset.any() else None
               for h in range(14)},
            mean_head_weights_for_true_k_if_preserved=
              {HEADS[h]:float(np.mean(new_W[others,h,k])) if others.any() else None
               for h in range(14)}
        )
    actions_by_truth=[]
    for orig in (2,3,4):
        for k in range(7):
            stats=count(y,b,pn,(b==orig)&(y==k))
            if stats["events"]>0:
                stats.update(original_K=orig,true_K=k)
                actions_by_truth.append(stats)
    report=dict(
        status="completed",
        experiment="v273_class_conditional_forensic_regression",
        original_run=37745469151,
        conditional_run=37747861591,
        native_rows=n,eligible_rows=len(ii),
        baseline=base_stats,
        original_neural=old_stats,
        class_conditional=new_stats,
        effects_by_true_K=by_k,
        effects_by_original_prediction=by_orig,
        effects_by_fold=by_fold,
        by_original_K_and_true_K=actions_by_truth,
        transitions=transitions,
        paired_model_comparison=paired_versions,
        unsupported_K0_K1_evidence=violations,
        changes_to_K0_K1=down_details,
        losses_in_K2plus_due_to_K0_K1_edits=downward_poly_loss,
        all_edits_in_true_poly=polychanged,
        per_head_risk_diagnostic=riskby,
        originally_correct_poly_lost_selection=selection_insight,
        posthoc_revert_counterfactuals_not_validated=counterfactual,
        code_verified_causes=[
            "Loss uses unweighted per-example action categorical cross-entropy; no per-K poly regression cost.",
            "Auxiliary per-head risk is trained but never used as decoder input or action veto.",
            "True-K-correct baseline becomes KEEP target; every wrong baseline becomes true K target including K0/K1.",
            "For K0/K1, structural per-class eligibility allows only H0, whose adapter places no material probability on K0/K1; nevertheless a learned class-logit decoder can output these classes.",
            "Correction adapters each use KEEP probability 1-0.95*q and target class 0.95*q; their argmax typically stays KEEP when q<=1/1.9.",
            "Existing fix heads are newly engineered KEEP placeholders, not audited historical repair implementations.",
            "Actual final argmax uses action logits alone and ignores predicted head-risk values and baseline-correctness probabilities."
        ],
        caveats=[
            "All 59309 positions are included in Exact-K and K0..K6 confusion; 7493 eligible based only on baseline predictions.",
            "Both versions are on the same exposed outer folds; this is a posthoc forensic audit, NOT fresh validation.",
            "Counterfactual revert policies shown only to localize loss mechanisms; do not promote or tune on heldout labels.",
            "Head risk predictions are per individual head, not validated end-to-end decision risk.",
            "Observed output classes and weights do not uniquely identify causal contributions of each hidden layer.",
            "Reference freeze_local_combo has not been modified."
        ],automatic_promotion=False
    )
    a.output.mkdir(parents=True)
    (a.output/"audit.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    with (a.output/"transitions.csv").open("w",newline="") as o:
        w=csv.DictWriter(o,fieldnames=["fold","original_K","new_K","events","actions",
           "corrections","regressions","neutral","net"]+[f"trueK{k}" for k in range(7)])
        w.writeheader()
        for d in transitions:
            z={k:d[k] for k in ("fold","original_K","new_K","events","actions",
                "corrections","regressions","neutral","net")}
            z.update({f"trueK{k}":d["trueK_counts"][str(k)] for k in range(7)})
            w.writerow(z)
    with (a.output/"decision-details.csv").open("w",newline="") as o:
        w=csv.writer(o)
        w.writerow(["global_index","fold","true_K","freeze_K","old_neural_K",
                    "conditional_neural_K","baseline_correct","conditional_correct",
                    "conditional_regression","conditional_fix","changed_between_models"])
        for i in ii:
            w.writerow([int(old["global_index"][i]),int(fold[i]),int(y[i]),int(b[i]),
                        int(po[i]),int(pn[i]),int(y[i]==b[i]),int(y[i]==pn[i]),
                        int(y[i]==b[i] and pn[i]!=b[i]),
                        int(y[i]==pn[i] and b[i]!=pn[i]),int(po[i]!=pn[i])])
    lines=[
      "# Forensic audit — true causes of K-dependent head regressions",
      "",
      "**This is a failure audit, not a new correction.**",
      f"Native {N_NATIVE} events, {N_ELIGIBLE} modifiable, fold3/player05 excluded.",
      "",
      "| Model | Exact K global | Exact K poly | Corrections | Regressions | Net |",
      "|---|---:|---:|---:|---:|---:|",
    ]
    for key,z in (("Frozen reference",base_stats),("Original neural",old_stats),
                  ("K-conditional neural",new_stats)):
        u=z["paired"]
        lines.append(f"| {key} | {z['global_exact_pct']:.4f}% | "
                     f"{z['poly_exact_pct']:.4f}% | {u['corrections']} | "
                     f"{u['regressions']} | {u['net']:+d} |")
    lines+=["","## Full true K breakdown","",
           "| True K | Fix | Regress | Net |","|---|---:|---:|---:|"]
    for k,v in by_k.items():
        lines.append(f"| K{k} | {v['corrections']} | {v['regressions']} | {v['net']:+d} |")
    lines+=["","## Most costly original -> new transitions","",
            "| Frozen K -> Network K | Actions | Fix | Regression | Neutral | Net |",
            "|---|---:|---:|---:|---:|---:|"]
    for v in sorted((x for x in transitions if x["fold"]=="ALL"),
                    key=lambda x:x["net"])[:20]:
        lines.append(f"| K{v['original_K']} -> K{v['new_K']} | {v['actions']} | "
                     f"{v['corrections']} | {v['regressions']} | {v['neutral']} | {v['net']:+d} |")
    lines+=["","## K0/K1 decoder without class evidence",""]
    for v in violations:
        lines.append(f"- Network outputs K{v['target_K']} on {v['selected_actions']} "
                     f"eligible events; only H0 is class-eligible: {v['only_k_support_is_head_H0']}. "
                     f"All other heads weight zero: {v['zero_mass_from_all_other_heads']}.")
    lines+=["","## Original neural versus class-conditional neural",""]
    for k,v in paired_versions.items():
        lines.append(f"- {k}: {v}")
    lines+=["","## Diagnostic ablations (POSTHOC; not deployable evidence)","",
            "| Counterfactual | Exact global | Exact poly | Net vs frozen |",
            "|---|---:|---:|---:|"]
    for k,v in counterfactual.items():
        lines.append(f"| {k} | {v['global_exact_pct']:.4f}% | "
                     f"{v['poly_exact_pct']:.4f}% | {v['paired']['net']:+d} |")
    lines+=["","## Verified code-level defects"]+["- "+x for x in report["code_verified_causes"]]
    lines+=["","## Limits"]+["- "+x for x in report["caveats"]]
    (a.output/"audit.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
if __name__=="__main__":
    main()
