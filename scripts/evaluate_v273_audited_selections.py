"""Train-only per-K audited specialist selections; native all-event Exact-K.

Every one of 63 nonempty subsets of H0..H5 gets its *own* audit on
CROSS-FITTED outer-train specialist predictions. Audits include true K0..K6,
corrections, regressions, and decision cells (baseline-predicted K, proposed
K). At inference true K is unknown; the routing prior reads ONLY the
training-audited (baseline K, proposed K) correction risk and signal-dependent
neural-router probabilities. Rare cells shrink toward zero net benefit.

Never tune to labels of outer fold; no player05/fold3. No model promotion.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_v273_dynamic_heads import (
    EXPERTS, CLASSES, fixed_head, oof_experts, outer_experts,
    gate_inputs, SoftHeadGate, get_matrix, load_cohort, load_features,
    policy_predictions
)
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require


N_HEADS = len(EXPERTS) + 1
HEAD_NAMES = ("H0", "H1", "H2", "H3", "H4", "H5")
SUBSETS = tuple(tuple(i for i in range(N_HEADS) if mask & (1 << i))
                for mask in range(1, 1 << N_HEADS))
MASKS = np.asarray(
    [[float(i in ids) for i in range(N_HEADS)] for ids in SUBSETS],float
)
SIZES = MASKS.sum(axis=1)
assert len(SUBSETS) == 63 and len({tuple(s) for s in SUBSETS}) == 63
CELL_SHRINK = 30.0
GLOBAL_SHRINK = 65.0
FEATURE_NAMES = ("spectral__", "birth__", "persistence__", "damping__")
POLICIES = ("audit_soft", "audit_conservative", "audit_subset")


def combo_probabilities(P):
    """All 63 subset means; (rows,63,5), no true K dependency."""
    require(P.ndim == 3 and P.shape[1:] == (N_HEADS,5), "bad expert shapes")
    combo = np.einsum("she,ch->sce",P,MASKS, optimize=True)
    combo /= SIZES[None,:,None]
    require(np.isfinite(combo).all() and np.allclose(combo.sum(axis=2),1), "bad combination normalization")
    return combo


def audit_one(y,base,proposed):
    """Independent audit against unchanged baseline, by real K and transition.

    At predicted K2, a proposal K3 corrected if true K3, regressed if true K2;
    other true K values are neutral changes (both predictions incorrect).
    """
    y=np.asarray(y,int); base=np.asarray(base,int); proposed=np.asarray(proposed,int)
    require(len(y)==len(base)==len(proposed),"audit arrays")
    require(np.isin(y,np.arange(7)).all(),"bad true K")
    changed=proposed!=base
    fix=changed&(proposed==y)
    regression=changed&(base==y)
    require(not np.any(fix&regression),"cannot both correct and regress")
    def rec(mask):
        c=int(np.sum(fix&mask))
        r=int(np.sum(regression&mask))
        n=int(np.sum(changed&mask))
        return {"events":int(np.sum(mask)),"actions":n,
                "corrections":c,"regressions":r,"neutral":n-c-r,"net":c-r}
    by_true={str(k):rec(y==k) for k in range(7)}
    cells={}
    for b in (2,3,4):
        for p in CLASSES:
            if p==b: continue
            cmask=(base==b)&(proposed==p)
            cells[f"{b}->{p}"]=rec(cmask)
    return {"total":rec(np.ones(len(y),bool)),
            "by_true_k":by_true,"by_predicted_transition":cells}


def audit_all(y,base,p):
    require(len(y)==p.shape[0],"audit tensor")
    proposals=CLASSES[np.argmax(combo_probabilities(p),axis=2)]
    return [audit_one(y,base,proposals[:,j]) for j in range(len(SUBSETS))]


def shrink_score(count, shrink):
    """Expected net gain per action with symmetric zero-effect prior."""
    n=int(count["actions"])
    return (float(count["corrections"])-float(count["regressions"]))/(n+shrink)


def audited_priors(audits,base,proposed):
    """Reliability weights use TRAIN-OFF-FOLD audits, never real test K.

    Combine local transition payoff, global net payoff, and mean singleton
    reliability of the selected heads. The latter implements explicit head
    audits even inside combinations.
    """
    base=np.asarray(base,int);proposed=np.asarray(proposed,int)
    n,c=proposed.shape
    require(len(base)==n and c==len(SUBSETS) and len(audits)==c,"audit tensor dimensions")
    singleton_index={v[0]:i for i,v in enumerate(SUBSETS) if len(v)==1}
    global_scores=np.asarray([shrink_score(r["total"],GLOBAL_SHRINK) for r in audits])
    util=np.zeros((n,c),float)
    for j,s in enumerate(SUBSETS):
        # An audit of a combination remains independent of its constituent
        # singleton audits; both contribute and are explicitly retained.
        component=float(np.mean([global_scores[singleton_index[h]] for h in s]))
        base_global=float(global_scores[j])
        prior=.35*base_global+.20*component
        for b in (2,3,4):
            ix=base==b
            if not np.any(ix):continue
            for k in CLASSES:
                cell=(ix&(proposed[:,j]==k))
                if k==b or not np.any(cell):continue
                specific=audits[j]["by_predicted_transition"][f"{b}->{k}"]
                local=shrink_score(specific,CELL_SHRINK)
                util[cell,j]=.45*local+prior
    return util


def weight_subsets(router_weights,utility):
    """Reweight the neural gate using measured individual audit reliability.

    Outputs convex normalized per-event weights for 63 candidate selections.
    Negative reliability reduces a subset's contribution; positive raises it.
    """
    w=np.asarray(router_weights,float)
    require(w.shape[1]==N_HEADS and utility.shape==(len(w),len(SUBSETS)),"shape")
    # Penalize huge subsets so an ensemble of all six heads does not
    # overwhelm singletons/pairs simply through sum of included weights.
    routing=(w@MASKS.T)/np.sqrt(SIZES[None,:])
    logits=np.log(np.maximum(routing,1e-12))+6.0*utility
    logits-=np.max(logits,axis=1,keepdims=True)
    z=np.exp(logits);alpha=z/z.sum(axis=1,keepdims=True)
    require(np.isfinite(alpha).all() and np.allclose(alpha.sum(axis=1),1),"invalid audited weights")
    return alpha


def propose(combos,alphas,utility,base,head_probs):
    """Three frozen abstention policies and rich per-event diagnostics."""
    fused=np.einsum("sc,sck->sk",alphas,combos,optimize=True)
    proposal=CLASSES[np.argmax(fused,axis=1)]
    margin=fused[np.arange(len(base)),proposal-2]-fused[np.arange(len(base)),base-2]
    expected=np.sum(alphas*utility,axis=1)
    # Look up the inferred transition utility for the resulting proposal,
    # not a K supplied by the ground truth.
    n,s=alphas.shape
    util_proposed=np.zeros((n,s),float)
    for j in range(s):
        pp=CLASSES[np.argmax(combos[:,j,:],axis=1)]
        util_proposed[:,j]=utility[:,j]*np.where(pp==proposal,1.,.0)
    risk=np.sum(alphas*util_proposed,axis=1)
    raw_votes=CLASSES[np.argmax(head_probs[:,1:,:],axis=2)]
    support=np.sum(raw_votes==proposal[:,None],axis=1)
    chosen=np.argmax(alphas,axis=1)
    strongest=CLASSES[np.argmax(combos[np.arange(n),chosen,:],axis=1)]
    strongest_margin=combos[np.arange(n),chosen,strongest-2]-combos[
        np.arange(n),chosen,base-2
    ]
    out={}
    allowed=(proposal!=base)&(risk>0)
    out["audit_soft"]=np.where(allowed&(margin>=.07),proposal,base)
    out["audit_conservative"]=np.where(
        allowed&(margin>=.15)&(risk>=.006)&(support>=2),proposal,base
    )
    out["audit_subset"]=np.where(
        (strongest!=base)&(strongest_margin>=.12)&(utility[np.arange(n),chosen]>.008),
        strongest,base
    )
    require(all(np.isin(p,CLASSES).all() for p in out.values()),"invalid proposals")
    diagnostic={
        "weights":alphas,
        "winning_subset":chosen,
        "suggested_class":proposal,
        "confidence_margin":margin,
        "expected_risk_adjustment":expected,
        "proposal_audit_benefit":risk,
        "expert_agreement":support,
    }
    return out,diagnostic


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),"refuse overwrite")
    y,baseline,indices,folds,members,starts=load_cohort(args.cohort)
    eligible,rows,names,_=load_features(
        args.features,y,baseline,indices,folds,members,starts
    )
    require(len(y)==59309 and len(eligible)==7493,"full sample drift")
    yc=y[eligible];bc=baseline[eligible];fc=folds[eligible]
    X={name:get_matrix(rows,names,field,prefix)
       for name,(field,prefix) in EXPERTS.items()}
    context_names=[n for n in names if n.startswith(FEATURE_NAMES)][:26]
    raw=np.asarray([[r["features"][name] for name in context_names]
                    for r in rows],float)
    require(np.isfinite(raw).all(),"nonfinite contexts")

    saved={"true_k":y,"baseline":baseline,"global_index":indices,"fold":folds}
    results={key:baseline.copy() for key in POLICIES}
    raw_router=baseline.copy()
    winners=np.full(len(eligible),-1,int)
    weights=np.full((len(eligible),len(SUBSETS)),np.nan,float)
    expected=np.full(len(eligible),np.nan,float)
    agreement=np.full(len(eligible),-1,int)
    audits_by_fold={}
    no_audit_by_fold={}
    for val_fold in FOLDS:
        tr=fc!=val_fold;val=fc==val_fold
        train_y=yc[tr]; train_b=bc[tr]; train_f=fc[tr]
        ptr=oof_experts({k:v[tr] for k,v in X.items()},
                        train_y,train_b,train_f)
        oof_audits=audit_all(train_y,train_b,ptr)
        ztrain=gate_inputs(ptr,train_b,raw[tr])
        router=SoftHeadGate().fit(ztrain,ptr,train_y)
        pv=outer_experts(
            {k:{"train":v[tr],"test":v[val]} for k,v in X.items()},
            train_y,bc[val]
        )
        ztest=gate_inputs(pv,bc[val],raw[val])
        neural_probs,neural_weights=router.predict(ztest,pv)
        # Existing router is comparison arm (ungated), not chosen by audit.
        raw_router[eligible[val]]=policy_predictions(
            neural_probs,pv,bc[val]
        )[0]["ungated"]
        cv=combo_probabilities(pv)
        proposal=CLASSES[np.argmax(cv,axis=2)]
        utilities=audited_priors(oof_audits,bc[val],proposal)
        audited_weights=weight_subsets(neural_weights,utilities)
        pred,diagnostics=propose(cv,audited_weights,utilities,bc[val],pv)
        for key,v in pred.items():results[key][eligible[val]]=v
        winners[val]=diagnostics["winning_subset"]
        weights[val]=diagnostics["weights"]
        expected[val]=diagnostics["proposal_audit_benefit"]
        agreement[val]=diagnostics["expert_agreement"]
        audits_by_fold[str(val_fold)]={
            "audit_sample_rows":int(np.sum(tr)),
            "selection_audits":[{
                "combination_id":f"S{i+1:02}",
                "heads":[HEAD_NAMES[h] for h in subset],
                **a
            } for i,(subset,a) in enumerate(zip(SUBSETS,oof_audits))]
        }
        no_audit_by_fold[str(val_fold)]={
            "outer_rows":int(val.sum()),
            "winning_counts":{f"S{i+1:02}":int(np.sum(winners[val]==i))
                              for i in range(len(SUBSETS))},
            "change_counts":{key:int(np.sum(v!=bc[val])) for key,v in pred.items()}
        }
        print(json.dumps({"fold":int(val_fold),
                          "selection_audits":len(oof_audits),
                          "actions":no_audit_by_fold[str(val_fold)]["change_counts"]}),
              flush=True)

    require(np.all(winners>=0) and np.isfinite(weights).all() and
            np.isfinite(expected).all() and np.all(agreement>=0),
            "missing predictions")
    ref=metrics(y,baseline)
    report={
        "status":"completed",
        "experiment":"v273_audit_conditioned_dynamic_selection",
        "baseline":ref,
        "cohort":{"native_rows":len(y),"eligible_rows":len(eligible),
                  "folds":list(FOLDS),"excluded_player05":True,
                  "excluded_fold3":True},
        "head_names":HEAD_NAMES,
        "selection_catalogue":[{
            "combination_id":f"S{i+1:02}",
            "heads":[HEAD_NAMES[k] for k in combo],
            "head_count":len(combo)
        } for i,combo in enumerate(SUBSETS)],
        "audit_by_heldout_fold":audits_by_fold,
        "per_fold_descriptive":no_audit_by_fold,
        "variants":{},
        "constraints":[
            "All 63 selection audits use cross-fitted train-fold predictions",
            "Each audit records separate corrections/regressions for true K0..K6",
            "At inference only baseline predicted K and head proposals index conditional reliability",
            "No true K is used in inference audit weight calculation",
            "Rare decision cells are shrunk toward zero expected improvement",
            "Neural router receives audit priors only from outer training folds",
            "Outer test labels are used strictly to compute final evaluation",
            "Audit utility estimates and policies were defined before these outer outcomes",
            "All original 59309 native events scored; only base predictions K2 K3 K4 change",
            "Poly-only proposals, no player05/fold3, no baseline promotion",
            "Exploratory folds previously inspected; not independent validation",
            "Future audio to +160ms; not equivalent to causal original runtime"
        ],
        "automatic_promotion":False
    }
    for key,pred in {**results,"previous_neural_ungated":raw_router}.items():
        m=metrics(y,pred);p=paired(y,baseline,pred)
        require(m["correct"]==ref["correct"]+p["global"]["net"],"metric mismatch")
        require(p["global"]["net"]==p["poly"]["net"],"nonpoly moved")
        score_by_fold={}
        for f in FOLDS:
            mask=folds==f
            score_by_fold[str(f)]=paired(y[mask],baseline[mask],pred[mask])
        report["variants"][key]={"metrics":m,"paired":p,"by_fold":score_by_fold}
        saved[key]=pred
    saved.update({
        "eligible_global_indices":indices[eligible],
        "selection_index":winners+1,
        "audit_weight_on_each_subset":weights,
        "estimated_audit_benefit":expected,
        "expert_agreement":agreement
    })
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"all-audited-selections.npz",**saved)
    # Complete 63 x 4 individual audits exported as machine-readable JSON.
    (args.output/"audits-all-selections.json").write_text(json.dumps(
        {"selection_catalogue":report["selection_catalogue"],
         "per_outer_fold":audits_by_fold},indent=2,sort_keys=True)+"\n")
    light={k:v for k,v in report.items() if k!="audit_by_heldout_fold"}
    (args.output/"report.json").write_text(
        json.dumps(light,indent=2,sort_keys=True,allow_nan=False)+"\n"
    )
    lines=["# Audited dynamic selections — exact K0..K6 native score","",
           "63 nonempty H0–H5 subsets; every selection has individual training-only OOF audit "
           "with corrections/regressions per TRUE K0..K6 and per allowed predicted transition.","",
           f"Baseline {100*ref['exact']:.4f}% global, "
           f"{100*ref['poly']['exact']:.4f}% poly. "
           "59,309 full events, 7,493 eligible.","",
           "| Policy | Global | Δ global pp | Poly | Δ poly pp | Corrections | Regressions | Net | Fold2 net |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,item in report["variants"].items():
        m=item["metrics"];p=item["paired"]["global"]
        lines.append(
            f"| {key} | {m['exact']*100:.4f}% | {(m['exact']-ref['exact'])*100:+.4f}"
            f" | {m['poly']['exact']*100:.4f}% | "
            f"{(m['poly']['exact']-ref['poly']['exact'])*100:+.4f}"
            f" | {p['corrections']} | {p['regressions']} | {p['net']:+d}"
            f" | {item['by_fold']['2']['global']['net']:+d} |"
        )
    lines+=["","## Breakdown by true K","",
            "| Policy | "+" | ".join(f"K{k}" for k in range(7))+" |",
            "|---|"+"---:|"*7]
    for key,item in report["variants"].items():
        p=item["paired"]["by_k"]
        lines.append("| "+key+" | "+" | ".join(
            f"{p[str(k)]['corrections']:+d}/-{p[str(k)]['regressions']}"
            for k in range(7))+" |")
    lines+=["","## Cautions"]+["- "+x for x in report["constraints"]]
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
