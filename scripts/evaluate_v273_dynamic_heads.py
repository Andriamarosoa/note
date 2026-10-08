"""Dynamic mixture of specialist heads for native polyphonic Exact-K.

Research prototype; freeze_local_combo weights remain untouched.

Outer-fold rule: train on 3 recording-disjoint folds, test fourth (0,1,2,4).
Inside outer train, every head supplies OOF predictions to the gate (head
training excludes the rows whose outputs feed the gate). Final heads are
refitted on outer training, never test labels. Gate learns per-event convex
weights over head K2..K6 probabilities with a neural hidden layer.

All predictions are reconstructed over the ORIGINAL 59,309 K0..K6 events.
The gate is eligible ONLY where frozen base predicted K2/K3/K4.
It may return a K2..K6 proposal or abstain, never force K0/K1 changes.
No player05/fold3 used. Note 160ms future context: not online causal.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.summarize_v273_harmonic_global import (
    load_cohort, load_features, get_matrix
)
from scripts.yourmt3_exactk_common import FOLDS, metrics, paired, require

CLASSES = np.arange(2,7, dtype=int)
BASE_PROB = .84
EXPERTS = {
    "spectral": ("features", ("spectral__",)),
    "lifecycle": ("features", ("birth__", "persistence__", "damping__")),
    "harmonic_full": ("features", tuple()),
    "fundamentals": ("fundamental", tuple()),
    "sources": ("features", ("source__", "coherence__")),
}
SEED = 27402
HIDDEN = 24
STEPS = 150
KEEP_BASE = 0.12
POLICIES = ("ungated", "confidence_only", "agreement_guard")


def sigmoid_stable(x):
    # Softmax is stable independently per sample.
    v = np.asarray(x, np.float64)
    z = np.exp(v - np.max(v,axis=1,keepdims=True))
    return z / np.sum(z,axis=1,keepdims=True)


def fixed_head(p):
    # Smoothed baseline head; never uses test label.
    p = np.asarray(p, int)
    require(np.isin(p,(2,3,4)).all(), "bad frozen candidate")
    out=np.full((len(p),5),(1.0-BASE_PROB)/4,float)
    out[np.arange(len(p)),p-2]=BASE_PROB
    return out


def head_estimator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=.1, max_iter=2500, random_state=SEED)
    )


def train_head(x,y):
    mask=np.isin(y,CLASSES)
    require(mask.sum()>100 and np.array_equal(np.unique(y[mask]), CLASSES),
            "missing poly target K")
    return head_estimator().fit(x[mask],y[mask])


def head_proba(model,x):
    require(np.array_equal(model[-1].classes_,CLASSES), "head class order")
    prob=model.predict_proba(x)
    require(prob.shape==(len(x),5) and np.isfinite(prob).all(),"bad head output")
    return np.maximum(prob,1e-10)/np.maximum(prob.sum(axis=1,keepdims=True),1e-10)


def oof_experts(X,y,base,folds):
    """Cross-fitted specialist outputs on outer TRAIN: no in-sample expert."""
    out=np.full((len(y), len(EXPERTS)+1, 5),np.nan)
    out[:,0,:]=fixed_head(base)
    for fold in np.unique(folds):
        val=folds==fold
        tr=~val
        require(np.any(val),"OOF group empty")
        for j,key in enumerate(EXPERTS,1):
            model=train_head(X[key][tr],y[tr])
            out[val,j,:]=head_proba(model,X[key][val])
    require(np.isfinite(out).all(),"incomplete OOF")
    return out


def outer_experts(X,y_train,base_val):
    out=np.full((len(base_val),len(EXPERTS)+1,5),np.nan)
    out[:,0,:]=fixed_head(base_val)
    for j,key in enumerate(EXPERTS,1):
        clf=train_head(X[key]["train"],y_train)
        out[:,j,:]=head_proba(clf,X[key]["test"])
    require(np.isfinite(out).all(),"incomplete holdout experts")
    return out


def gate_inputs(P,base,raw):
    """Gate sees head confidence, disagreements and observed acoustic context."""
    require(P.shape==(len(base),len(EXPERTS)+1,5),"head tensor wrong")
    on=np.eye(3)[base-2]
    # Mixture disagreement and confidence are observable without true K.
    maxp=np.max(P,axis=2)
    entropy=-np.sum(P*np.log(P+1e-10),axis=2)/np.log(5)
    consensus=np.mean(np.argmax(P[:,1:,:],axis=2)==base[:,None],axis=1)
    spread=np.std(P[:,1:,:],axis=1)
    return np.column_stack([P.reshape(len(base),-1),maxp,entropy,
                            consensus,spread,raw,on])


class SoftHeadGate:
    """One-hidden-layer neural softmax router trained on OOF expert probabilities."""

    def __init__(self):
        self.scaler=StandardScaler()
        self.fitted=False

    def fit(self,z,p,y):
        z=np.asarray(z,float);p=np.asarray(p,float);y=np.asarray(y,int)
        require(z.shape[0]==len(y) and p.shape==(len(y),len(EXPERTS)+1,5),
                "router training shape")
        mask=np.isin(y,CLASSES)
        require(mask.sum()>200,"router needs poly examples")
        z=z[mask];p=p[mask]; y=y[mask]-2
        vals=self.scaler.fit_transform(z)
        vals=np.clip(vals,-6,6)
        n,d=vals.shape; h=HIDDEN; e=p.shape[1]
        rnd=np.random.default_rng(SEED)
        W1=rnd.normal(0,.09/np.sqrt(max(1,d/24)),size=(d,h))
        b1=np.zeros(h)
        W2=rnd.normal(0,.04,size=(h,e))
        b2=np.zeros(e)
        b2[0]=.65  # trust the preserved baseline until evidence accumulates
        weights={k:np.zeros_like(v) for k,v in
                 (("W1",W1),("b1",b1),("W2",W2),("b2",b2))}
        sq={k:np.zeros_like(v) for k,v in weights.items()}
        par={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
        counts=np.bincount(y,minlength=5)
        require(np.all(counts>0),"router lacks K2-K6 classes")
        classwt=np.sqrt(counts.sum()/np.maximum(counts,1)/5)
        classwt=np.clip(classwt,0.35,5.0)
        samplewt=classwt[y]
        sw=np.sum(samplewt)
        at=p[np.arange(n),:,y]
        for step in range(1,STEPS+1):
            hidden=np.tanh(vals@W1+b1)
            w=sigmoid_stable(hidden@W2+b2)
            soft=np.sum(w*at,axis=1)
            pred=KEEP_BASE*at[:,0]+(1-KEEP_BASE)*soft
            # analytic d(-log P(y))/d(router logits)
            dz=(1-KEEP_BASE)*w*(soft[:,None]-at)/(pred[:,None]+1e-10)
            dz*=samplewt[:,None]/sw
            dW2=hidden.T@dz+1e-4*W2
            db2=dz.sum(axis=0)
            dh=(dz@W2.T)*(1-hidden**2)
            dW1=vals.T@dh+1e-4*W1
            db1=dh.sum(axis=0)
            grads={"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}
            lr=.024
            for key,g in grads.items():
                weights[key]=.9*weights[key]+.1*g
                sq[key]=.999*sq[key]+.001*g*g
                par[key]-=lr*(weights[key]/(1-.9**step))/(
                    np.sqrt(sq[key]/(1-.999**step))+1e-8
                )
        self.params=par
        self.fitted=True
        return self

    def predict(self,z,p):
        require(self.fitted,"router untrained")
        v=np.clip(self.scaler.transform(z),-6,6)
        W1=self.params["W1"];b1=self.params["b1"]
        W2=self.params["W2"];b2=self.params["b2"]
        weights=sigmoid_stable(np.tanh(v@W1+b1)@W2+b2)
        probs=KEEP_BASE*p[:,0]+(1-KEEP_BASE)*np.einsum("ne,nec->nc",weights,p)
        require(np.max(np.abs(probs.sum(axis=1)-1))<1e-7,"invalid convex fusion")
        return probs,weights


def policy_predictions(probs,p,base):
    proposal=CLASSES[np.argmax(probs,axis=1)]
    margin=probs[np.arange(len(base)),proposal-2]-probs[np.arange(len(base)),base-2]
    support=np.sum(np.argmax(p[:,1:,:],axis=2)==(proposal-2)[:,None],axis=1)
    output={}
    for policy in POLICIES:
        if policy=="ungated":
            active=(proposal!=base)
        elif policy=="confidence_only":
            active=(proposal!=base)&(margin>=.15)
        else:
            # Hard-coded BEFORE evaluation; no data-dependent threshold tuning.
            active=(proposal!=base)&(margin>=.25)&(support>=3)
        output[policy]=np.where(active,proposal,base)
    return output,proposal,margin,support


def summarize_weights(weights,orig,proposal):
    summaries={}
    for base in (2,3,4):
        m=orig==base
        if np.any(m):
            summaries[str(base)]={k:float(np.mean(weights[m,i]))
                                  for i,k in enumerate(("baseline",*EXPERTS))}
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cohort",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    require(not a.output.exists(),"refusing to overwrite")
    y,base,idx,folds,mem,start=load_cohort(a.cohort)
    eligible,rows,names,_=load_features(a.features,y,base,idx,folds,mem,start)
    require(len(y)==59309 and len(eligible)==7493,"frozen full cohort drift")
    yc=y[eligible];bc=base[eligible];fc=folds[eligible]
    X={}
    for k,(field,prefixes) in EXPERTS.items():
        X[k]=get_matrix(rows,names,field,prefixes)
    # Acoustic descriptors for dynamic routing, with no label leakage.
    context_names=[n for n in names if
                   n.startswith(("spectral__","birth__","persistence__","damping__"))]
    # Limit dimension to avoid overfitting while retaining musical states.
    context_names=context_names[:26]
    raw=get_matrix(rows,names,"features",tuple(context_names)) if False else (
        np.asarray([[row["features"][name] for name in context_names]
                    for row in rows],float)
    )
    require(np.isfinite(raw).all(),"nonfinite acoustic router context")
    oof_weights=np.full((len(yc),len(EXPERTS)+1),np.nan)
    variants={key:base.copy() for key in POLICIES}
    mean_equal=base.copy()
    per_fold={}
    head_names=("baseline",*EXPERTS)
    for vf in FOLDS:
        tr=fc!=vf;val=fc==vf
        yy=yc[tr]; bb=bc[tr]; ff=fc[tr]
        ptr=oof_experts({key:value[tr] for key,value in X.items()},yy,bb,ff)
        ztr=gate_inputs(ptr,bb,raw[tr])
        clf=SoftHeadGate().fit(ztr,ptr,yy)
        pv=outer_experts({key:{"train":value[tr],"test":value[val]}
                           for key,value in X.items()},yy,bc[val])
        zv=gate_inputs(pv,bc[val],raw[val])
        p_mix,w=clf.predict(zv,pv)
        oof_weights[val]=w
        pred,proposal,margin,support=policy_predictions(p_mix,pv,bc[val])
        for policy,kk in pred.items():
            variants[policy][eligible[val]]=kk
        # A *nonlearned* baseline, to establish whether dynamic routing matters.
        equal_proba=np.mean(pv,axis=1)
        equal_proposal=CLASSES[np.argmax(equal_proba,axis=1)]
        equal_margin=equal_proba[np.arange(int(val.sum())),equal_proposal-2]-equal_proba[
            np.arange(int(val.sum())),bc[val]-2]
        eqsupport=np.sum(np.argmax(pv[:,1:,:],axis=2)==(equal_proposal-2)[:,None],axis=1)
        equal_active=(equal_proposal!=bc[val])&(equal_margin>=.25)&(eqsupport>=3)
        mean_equal[eligible[val]]=np.where(equal_active,equal_proposal,bc[val])
        per_fold[str(vf)]=dict(
            rows=int(val.sum()),
            true_poly_events=int(np.isin(yc[val],CLASSES).sum()),
            avg_weights=summarize_weights(w,bc[val],proposal),
            policy_actions={policy:int(np.sum(pk!=bc[val])) for policy,pk in pred.items()},
        )
        print(json.dumps({"fold":vf,"candidate_rows":int(val.sum()),
                          "policy_actions":per_fold[str(vf)]["policy_actions"]}),
              flush=True)
    require(np.isfinite(oof_weights).all(),"missing router weights")
    baseline=metrics(y,base)
    report=dict(
        status="completed",
        experiment="v273_dynamic_head_mixture_native_exactk",
        head_names=head_names,
        gating_type="one-hidden-layer neural softmax convex mixture over per-event expert posteriors",
        training="nested: expert OOF within each outer training split, train router on OOF, refit experts on outer train",
        policies={"confidence_only":"margin >= .15",
                  "agreement_guard":"margin >= .25 and at least three specialist heads agree",
                  "ungated":"router argmax no abstention"},
        cohort={"full_rows":len(y),"eligible_rows":len(eligible),
                "unmodified_outside_eligibility":len(y)-len(eligible),
                "folds":list(FOLDS),"player05_used":False,"fold3_used":False},
        baseline=baseline,by_fold=per_fold,variants={},
        constraints=[
            "True K never selects the set of candidate events; baseline prediction only",
            "Only K2+ corrections permitted, so K0/K1 baseline correct counts cannot change",
            "Outer-fold test labels never enter expert or router training",
            "Fixed abstention policies; NO threshold adjustment to outer fold labels",
            "Already-inspected compositions; experimental and NOT independent",
            "Feature extraction looks ahead +160ms; no causal-runtime equivalence",
            "Some heads share feature spaces and are not independent modalities",
            "Full native 59309-event metrics not inferred from 3-class evaluation",
            "No automatic baseline promotion, no player05/fold3",
        ],promote=False
    )
    for key,pp in list(variants.items())+[("equal_weights_guard",mean_equal)]:
        m=metrics(y,pp);c=paired(y,base,pp)
        require(m["rows"]==len(y) and
                m["correct"]==baseline["correct"]+c["global"]["net"],"metric drift")
        require(c["global"]["net"]==c["poly"]["net"],"nonpoly modified")
        folds_summary={}
        for f in FOLDS:
            mask=folds==f
            cp=paired(y[mask],base[mask],pp[mask])
            folds_summary[str(f)]=dict(rows=int(mask.sum()),paired=cp,
                        original=metrics(y[mask],base[mask]),
                        corrected=metrics(y[mask],pp[mask]))
        report["variants"][key]=dict(metrics=m,paired=c,by_fold=folds_summary)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/"oof-dynamic-predictions.npz",
                        global_index=idx,true_k=y,baseline=base,fold=folds,
                        eligible_global_indices=idx[eligible],
                        router_head_weights=oof_weights,
                        **variants,equal_weights_guard=mean_equal)
    (a.output/"report.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    lines=[
        "# Dynamic mixture of heads — full native Exact-K K0..K6",
        "",
        f"Frozen baseline: **{baseline['correct']}/{len(y)} ({100*baseline['exact']:.4f}%) global**, "
        f"**{baseline['poly']['correct']}/{baseline['poly']['rows']} ({100*baseline['poly']['exact']:.4f}%) poly**.",
        f"Model changes possible only on {len(eligible)} baseline-predicted K2/K3/K4 examples.",
        "",
        "| Variant | Exact-K global | Δ global pp | Exact-K poly | Δ poly pp | Fixes | Regressions | Net | Fold2 net |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key,r in report["variants"].items():
        m=r["metrics"];p=r["paired"]["global"]
        f2=r["by_fold"]["2"]["paired"]["global"]
        lines.append(f"| {key} | {m['exact']*100:.4f}% | {(m['exact']-baseline['exact'])*100:+.4f} "
                     f"| {m['poly']['exact']*100:.4f}% | {(m['poly']['exact']-baseline['poly']['exact'])*100:+.4f} "
                     f"| {p['corrections']} | {p['regressions']} | {p['net']:+d} | {f2['net']:+d} |")
    lines += ["","## Head contribution by original predicted K","",
              "Mean router weights are descriptive, not proof of source causality."]
    for f,v in per_fold.items():
        lines.append(f"\nFold {f}:")
        for c,w in v["avg_weights"].items():
            lines.append(f"- Predicted K{c}: "+", ".join(
                f"{key}={value:.3f}" for key,value in w.items()
            ))
    lines += ["","## Per-K true-class recall (all 59,309 events)","",
              "| Variant | " + " | ".join(f"K{k}" for k in range(7))+" |",
              "|---|"+ "---:|"*7]
    for key,r in report["variants"].items():
        m=r["metrics"]
        lines.append("| "+key+" | "+" | ".join(
            f"{m['by_k'][str(k)]['exact']*100:.3f}%"
            for k in range(7)
        )+" |")
    lines += ["","## Interpretation limits"] + ["- "+t for t in report["constraints"]]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)


if __name__=="__main__":
    main()
