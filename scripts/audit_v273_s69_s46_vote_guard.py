"""S69: reproducible 4-fold historical development audit (not promoted)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
EDGES=((3,2),(2,3),(4,3),(3,4),(1,0),(0,1))
POLICIES={
 "balanced":("series46__historic_acoustic__polyLambda2__cut0__votes1__neighbor_and_no_drop_to_K1",1.5),
 "polymax":("series46__all16__polyLambda4__cut0.1__votes1__neighbor_and_no_drop_to_K1",1.2),
}

def require(ok,reason):
    if not ok: raise ValueError(reason)

def read(path):
    with np.load(path,allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}

def paired(y,b,p):
    correct=p==y; original=b==y
    return dict(correct=int(correct.sum()),poly_correct=int((correct&(y>=2)).sum()),
        corrections=int((~original&correct).sum()),
        regressions=int((original&~correct).sum()),
        gain=int(correct.sum()-original.sum()),errors=int((~correct).sum()))

def signals(ct,base,proposal):
    n=len(base)
    cb=ct[np.arange(n),base];cp=ct[np.arange(n),proposal]
    other=np.max(np.where(np.arange(7)[None,:]==base[:,None],-1,ct),axis=1)
    return dict(base_votes=cb.astype(float),advantage=(cb-cp).astype(float),
        candidate_negative=(-cp).astype(float),
        lead_over_other=(cb-other).astype(float),
        votes_total_negative=np.where(base==proposal,0,-cp).astype(float))

def choose(train,subset,y,b,p,signalbank,cost):
    region=train&subset
    N=int(region.sum())
    if N<25: return None
    regress=(b==y)&(p!=y)
    corrections=(b!=y)&(p==y)
    options=[]
    for key,signal in signalbank.items():
        for cut in np.unique(np.percentile(signal[region],[50,65,75,80,85,90,95,97,99])):
            mask=region&(signal>=cut)
            saved=int((regress&mask).sum());lost=int((corrections&mask).sum())
            utility=saved-cost*lost
            if utility>max(2,.004*N):
                options.append((utility,-int(mask.sum()),key,float(cut),saved,lost))
    return max(options) if options else None

def run(s46,votes,output):
    require(not output.exists(),"never overwrite previous evidence")
    a=read(s46);v=read(votes)
    for key in ("global_index","true_K","fold"):
        require(np.array_equal(a[key],v[key]),"source alignment: "+key)
    require(len(a["global_index"])==len(set(a["global_index"].tolist()))==59309,"native cohort")
    require(a["predictions"].shape==(59309,193),"S46 source policy inventory")
    require(v["predictions"].shape==(59309,36),"PR16 vote policy inventory")
    require(set(a["fold"].tolist())==set(FOLDS),"unexpected source fold")
    names=list(map(str,a["variant_ids"]))
    require(names[0]=="series18_parent","reference drift")
    y=a["true_K"];base=a["predictions"][:,0];fold=a["fold"]
    n=len(y)
    ct=np.stack([(v["predictions"]==k).sum(1) for k in range(7)],axis=1)
    results={};predictions={"global_index":a["global_index"],"true_K":y,
        "fold":fold,"s18_K":base}
    for label,(policy,cost) in POLICIES.items():
        require(policy in names,"missing fixed source policy: "+policy)
        parent=a["predictions"][:,names.index(policy)]
        sb=signals(ct,base,parent)
        proposed=parent.copy()
        rules=[]
        for outer in FOLDS:
            tr=fold!=outer;te=fold==outer
            for src,dst in EDGES:
                sub=(base==src)&(parent==dst)
                selected=choose(tr,sub,y,base,parent,sb,cost)
                if selected is None:continue
                utility,_,name,threshold,saved,lost=selected
                accept=te&sub&(sb[name]>=threshold)
                proposed[accept]=base[accept]
                rules.append(dict(held_fold=outer,source=src,target=dst,
                    score=name,threshold=threshold,training_utility=utility,
                    train_saved=saved,train_lost=lost,held_actions=int(accept.sum())))
        require(np.all((proposed==parent)|(proposed==base)),"forbidden new proposal")
        before=paired(y,base,parent);after=paired(y,base,proposed)
        results[label]=dict(policy=policy,cost=cost,before=before,after=after,
            delta_global=after["correct"]-before["correct"],
            delta_poly=after["poly_correct"]-before["poly_correct"],
            folds={str(f):dict(before=paired(y[fold==f],base[fold==f],parent[fold==f]),
                after=paired(y[fold==f],base[fold==f],proposed[fold==f])) for f in FOLDS},
            rules=rules)
        predictions[label+"_s46_K"]=parent
        predictions[label+"_protected_K"]=proposed
    output.mkdir(parents=True)
    np.savez_compressed(output/"predictions.npz",**predictions)
    report=dict(status="completed",independent_validation=False,promotion=False,
        count=len(y),policies=results,
        limitations=["Training-fold OOF gates, historically exposed evaluation folds",
        "Policies and costs compared on this same research cohort",
        "Correlated PR16 votes are not independent neural heads",
        "Upstream S46 producer provenance not fully nested"])
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for label,r in results.items():print(label,r["before"],r["after"],flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s46",type=Path,required=True)
    parser.add_argument("--votes",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    run(a.s46,a.votes,a.output)

if __name__=="__main__":main()
