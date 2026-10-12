"""S70 research-only PR16 probability meta-selector on S69 poly parent.

Outer 0/1/2/4 recording folds: fit on three other folds. For threshold,
use inner OOF on the three training folds, never held-fold labels.
The underlying PR16 producers are not nested with the meta learner; NOT an
independent estimate nor a promotion candidate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

FOLDS=(0,1,2,4)
CUTS=(0,.02,.04,.06,.08,.1,.15,.2,.25,.3,.4,.5,.6,.75)
POLICIES={"proposed_poly":8,"downward_poly":1}

def require(test,reason):
    if not test:
        raise ValueError(reason)

def read(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}

def clf_score(x,truth,baseline,tr,te):
    mdl=make_pipeline(StandardScaler(),
        LogisticRegression(C=.01,max_iter=400,solver="lbfgs"))
    mdl.fit(x[tr],truth[tr])
    p=np.zeros((len(te),7))
    raw=mdl.predict_proba(x[te])
    for j,k in enumerate(mdl[-1].classes_):
        p[:,int(k)]=raw[:,j]
    return p

def propose(prob,base):
    idx=np.arange(len(base));candidate=prob.argmax(1)
    return candidate,prob[idx,candidate]-prob[idx,base]

def permitted(candidate,base,domain):
    if domain=="proposed_poly":
        return (candidate>=2)&(candidate!=base)
    if domain=="downward_poly":
        return (candidate<base)&(candidate>=2)
    raise ValueError(domain)

def paired(truth,parent,predicted):
    prev=parent==truth
    nxt=predicted==truth
    return dict(global_correct=int(nxt.sum()),
        poly_correct=int((nxt&(truth>=2)).sum()),
        corrections=int((nxt&~prev).sum()),
        regressions=int((~nxt&prev).sum()),
        net=int(nxt.sum()-prev.sum()),
        poly_net=int(((nxt.astype(int)-prev.astype(int))*(truth>=2)).sum()),
        actions=int((predicted!=parent).sum()))

def choose_threshold(score,proposal,base,truth,mask,weight,domain):
    valid=permitted(proposal,base,domain)&mask
    best=(0,0,0,999.)
    for threshold in CUTS:
        chosen=valid&(score>=threshold)
        if not chosen.any():continue
        parent=base[chosen];new=proposal[chosen];actual=truth[chosen]
        effect=(new==actual).astype(int)-(parent==actual).astype(int)
        poly_gain=int((effect*(actual>=2)).sum());global_gain=int(effect.sum())
        if poly_gain<0 or global_gain<0:continue
        utility=global_gain+weight*poly_gain
        if utility<5:continue
        reg=int((effect==-1).sum());candidate=(utility,-reg,-int(chosen.sum()),threshold)
        if candidate>best:best=candidate
    return best[3]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ("s69","votes","probabilities","output"):
        parser.add_argument("--"+field,required=True,type=Path)
    args=parser.parse_args()
    require(not args.output.exists(),"experiment cannot overwrite evidence")
    s=read(args.s69);v=read(args.votes);q=read(args.probabilities)
    ids=s["global_index"];truth=s["true_K"];fold=s["fold"]
    base=s["polymax_protected_K"];s18=s["s18_K"];balanced=s["balanced_protected_K"]
    require(len(ids)==59309 and len(set(ids.tolist()))==59309,"native count/IDs drift")
    require(set(fold.tolist())==set(FOLDS),"unexpected folds")
    for z in (v,q):
        require(np.array_equal(ids,z["global_index"]),"PR16 native identity mismatch")
    require(np.array_equal(truth,v["true_K"]) and
            np.array_equal(fold,v["fold"]),"PR16 target/fold drift")
    require(v["predictions"].shape==(59309,36) and
            q["probabilities"].shape==(59309,12,7),"PR16 head inventory")
    require(np.isfinite(q["probabilities"]).all(),"nonfinite neural probabilities")
    vote=v["predictions"]
    counts=np.stack([(vote==k).sum(1) for k in range(7)],axis=1)
    # All selection inputs are label-blind. No test truth in region selection.
    chosen_rows=np.flatnonzero((base>=2)|(vote>=2).any(1))
    b=base[chosen_rows];a=balanced[chosen_rows];s0=s18[chosen_rows]
    yt=truth[chosen_rows];ft=fold[chosen_rows];n=len(chosen_rows)
    prob=q["probabilities"][chosen_rows]
    x=np.column_stack((prob.reshape(n,-1),prob.mean(1),prob.std(1),
        counts[chosen_rows]/36.,np.eye(7)[b],np.eye(7)[s0],np.eye(7)[a],
        np.column_stack([b,s0,b-s0,b>=2,s0>=2]))).astype(np.float32)
    out_prob=np.full((n,7),np.nan);train_oof={}
    for held in FOLDS:
        tr=np.flatnonzero(ft!=held);te=np.flatnonzero(ft==held)
        out_prob[te]=clf_score(x,yt,b,tr,te)
        oof=np.full((n,7),np.nan)
        for inner_held in FOLDS:
            if inner_held==held:continue
            fi=np.flatnonzero((ft!=held)&(ft!=inner_held))
            vi=np.flatnonzero(ft==inner_held)
            oof[vi]=clf_score(x,yt,b,fi,vi)
        require(np.isfinite(oof[tr]).all(),"inner probability coverage")
        train_oof[held]=oof
    require(np.isfinite(out_prob).all(),"outer probability coverage")
    proposals,margins=propose(out_prob,b)
    predictions={"global_index":ids,"fold":fold,"true_K":truth,"s69_poly_K":base}
    results={}
    for domain,cost in POLICIES.items():
        accepted=np.zeros(n,bool);thresholds={}
        for held in FOLDS:
            own_p,own_margin=propose(train_oof[held],b)
            threshold=choose_threshold(own_margin,own_p,b,yt,ft!=held,cost,domain)
            thresholds[str(held)]=None if threshold==999 else threshold
            accepted[(ft==held)&permitted(proposals,b,domain)&(margins>=threshold)]=True
        result=base.copy()
        result[chosen_rows[accepted]]=proposals[accepted]
        require(np.array_equal(result[np.setdiff1d(np.arange(len(base)),chosen_rows)],base[np.setdiff1d(np.arange(len(base)),chosen_rows)]),"noneligible changed")
        outcome=paired(truth,base,result)
        results[domain]=dict(cost=cost,**outcome,thresholds=thresholds,
            by_fold={str(held):paired(truth[fold==held],base[fold==held],result[fold==held]) for held in FOLDS})
        predictions[domain+"_K"]=result
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"predictions.npz",**predictions)
    report=dict(status="completed",n=len(ids),eligible=n,validation="historically exposed outer folds; inner OOF threshold",
        training_source_labels_only=True,never_used_yourmt3_input=True,
        independent_validation=False,automatic_promotion=False,results=results,
        limitations=["Upstream PR16 producer outputs are OOF individually but not fully nested with a second-stage meta learner",
                     "Two regularisation values and several gate domains were compared on these development folds before fixing this replay",
                     "These are correlated network producers; no claim of 36 independent neural networks"])
    (args.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for k,v in results.items():
        print(k,"correct",v["global_correct"],"poly",v["poly_correct"],
              "fixes",v["corrections"],"regressions",v["regressions"],
              "net",v["net"],"poly_net",v["poly_net"],flush=True)

if __name__=="__main__":
    main()
