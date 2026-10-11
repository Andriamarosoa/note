"""S71: acoustic transition-specific abstention trained without held-fold labels.

Freeze S70 producer (49,202 global / 3,166 poly) and test a NEW veto only
on 1,263 changed events. No new notes, no automatic promotion.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FOLDS=(0,1,2,4)
EDGES=((2,3),(3,2),(3,4),(4,3),(4,5),(5,4))
PERCENTILES=(40,50,60,70,75,80,85,90,92,94,96,98,99,99.5)

def require(x,msg):
    if not x:raise ValueError(msg)

def read(root):
    with np.load(root,allow_pickle=False) as a:
        return {k:a[k].copy() for k in a.files}

def features(root,ids,positions,truth,fold):
    lookup={int(ids[i]):j for j,i in enumerate(positions)}
    x=np.full((len(positions),104),np.nan,np.float32)
    names=np.empty(len(positions),dtype="<U96")
    found=np.zeros(len(positions),bool)
    original=set()
    for outer in FOLDS:
        paths=list(root.rglob(f"features-fold-{outer}.npz"))
        require(len(paths)==1,"missing/duplicate fold feature set")
        file=paths[0]
        report=json.loads(file.with_name("report.json").read_text())
        require(report["status"]=="completed" and report["fold"]==outer
                and report["labels_used_in_features"] is False
                and report["baseline_used_in_features"] is False
                and report["yourmt3_used_in_features"] is False,
                "feature provenance incorrectly marked")
        with np.load(file,allow_pickle=False) as z:
            all_ids=z["global_index"]
            require(len(set(all_ids.tolist()))==len(all_ids),"repeated native ID")
            require(not (set(map(int,all_ids.tolist())) & original),"cross-fold duplicated IDs")
            original.update(map(int,all_ids.tolist()))
            subset=np.flatnonzero(np.isin(all_ids,ids[positions]))
            local=np.asarray([lookup[int(all_ids[i])] for i in subset],int)
            require(not found[local].any(),"duplicate target fragment")
            require(np.array_equal(z["fold"][subset],fold[positions[local]])
                    and np.array_equal(z["k"][subset],truth[positions[local]]),
                    "native fold/truth audit mismatch")
            x[local]=np.column_stack([z["summary"][subset],z["geometry"][subset]])
            names[local]=z["member"][subset]
            found[local]=True
    require(len(original)==len(ids)==59309,"native source coverage mismatch")
    require(found.all() and np.isfinite(x).all(),"acoustic features incomplete")
    return x,names

def fit(x,y,weights,fit_indices,held_indices):
    model=make_pipeline(StandardScaler(),
        LogisticRegression(C=.05,max_iter=500,solver="lbfgs"))
    model.fit(x[fit_indices],y[fit_indices],
              logisticregression__sample_weight=weights[fit_indices])
    return model.predict_proba(x[held_indices])[:,1]

def threshold(score,mask,regression,correction):
    valid=np.flatnonzero(np.isfinite(score)&mask)
    if len(valid)<4:return 1.01
    cuts=np.unique(np.percentile(score[valid],PERCENTILES))
    best=(0,0,-1000000,1.01)
    for t in cuts:
        selected=valid[score[valid]>=t]
        count=len(selected)
        if count<4:continue
        saved=int(regression[selected].sum())
        lost=int(correction[selected].sum())
        utility=saved-lost
        if utility<1:continue
        candidate=(utility,-lost,-count,float(t))
        if candidate>best:best=candidate
    return best[3]

def paired(y,b,p):
    old=b==y
    new=p==y
    return dict(correct=int(new.sum()),poly_correct=int((new&(y>=2)).sum()),
                corrections=int((new&~old).sum()),
                regressions=int((~new&old).sum()),
                net=int(new.sum()-old.sum()))

def run(parent_path,feature_root,output):
    require(not output.exists(),"no overwrites allowed")
    d=read(parent_path)
    ids=d["global_index"];truth=d["true_K"];fold=d["fold"]
    b=d["s69_poly_K"];p=d["proposed_poly_K"]
    require(len(ids)==len(set(ids.tolist()))==59309 and set(fold)==set(FOLDS),
            "S70 native IDs and folds drift")
    require(paired(truth,b,p)["correct"]==49202 and
            paired(truth,b,p)["poly_correct"]==3166,"S70 reference metrics drift")
    pos=np.flatnonzero(b!=p)
    require(len(pos)==1263,"S70 proposed change inventory drift")
    audio,member=features(feature_root,ids,pos,truth,fold)
    old=b[pos];proposed=p[pos];yp=truth[pos];fp=fold[pos]
    x=np.column_stack([audio,np.eye(7)[old],np.eye(7)[proposed],
                       proposed-old]).astype(np.float32)
    correct_old=old==yp;correct_new=proposed==yp
    regression=correct_old&~correct_new
    correction=~correct_old&correct_new
    neutral=~regression&~correction
    weights=np.where(neutral,.2,1.).astype(float)
    assert (int(correction.sum()),int(regression.sum()))==(451,397)
    veto=np.zeros(len(pos),bool)
    rules=[]
    for outer in FOLDS:
        fit_ix=np.flatnonzero(fp!=outer)
        held_ix=np.flatnonzero(fp==outer)
        held_score=np.full(len(pos),np.nan)
        held_score[held_ix]=fit(x,regression.astype(int),weights,fit_ix,held_ix)
        inner_score=np.full(len(pos),np.nan)
        for inner in FOLDS:
            if inner==outer:continue
            ins_fit=np.flatnonzero((fp!=outer)&(fp!=inner))
            ins_held=np.flatnonzero(fp==inner)
            inner_score[ins_held]=fit(x,regression.astype(int),weights,
                                     ins_fit,ins_held)
        require(np.isfinite(inner_score[fit_ix]).all(),"inner folds not fully scored")
        for source,destination in EDGES:
            subset=(old==source)&(proposed==destination)
            cut=threshold(inner_score,(fp!=outer)&subset,regression,correction)
            decisions=(fp==outer)&subset&(held_score>=cut)
            veto|=decisions
            rules.append(dict(held_fold=outer,from_k=source,to_k=destination,
                threshold=cut,held_actions=int(decisions.sum())))
    safer=p.copy()
    safer[pos[veto]]=b[pos[veto]]
    # Alternative: exclude K3->K2 veto (exploratory ablation only).
    veto_poly=veto&~((old==3)&(proposed==2))
    poly=p.copy()
    poly[pos[veto_poly]]=b[pos[veto_poly]]
    versions={"risk":safer,"poly":poly}
    output.mkdir(parents=True)
    np.savez_compressed(output/"predictions.npz",
        global_index=ids,true_K=truth,fold=fold,s69_poly_K=b,s70_K=p,
        s71_risk_K=safer,s71_poly_K=poly,
        veto_global_index=ids[pos[veto]],
        member_changed=member)
    report=dict(status="completed",independent_validation=False,
      promotion=False,folds=list(FOLDS),n=len(ids),changed_from_s69=len(pos),
      baseline=paired(truth,b,p),rules=rules,results={})
    for label,v in versions.items():
        require(np.all((v==p)|(v==b)),"veto may only return to parent S69")
        modified=pos[(v[pos]!=p[pos])]
        require(len(modified)<len(pos),"unexpected complete intervention")
        result=paired(truth,b,v)
        result["change_from_s70"]=int((v==truth).sum()-(p==truth).sum())
        result["poly_gain_from_s70"]=int(((v==truth)&(truth>=2)).sum()-
                                          ((p==truth)&(truth>=2)).sum())
        result["abstentions"]=len(modified)
        result["by_fold"]={str(f):dict(parent=paired(truth[fold==f],b[fold==f],p[fold==f]),
                       candidate=paired(truth[fold==f],b[fold==f],v[fold==f]))
                      for f in FOLDS}
        result["by_true_k"]={str(k):int(((v==truth)&(truth==k)).sum()-
                                         ((p==truth)&(truth==k)).sum()) for k in range(7)}
        report["results"][label]=result
    report["limitations"]=[
      "Exposed historical 19 GuitarSet compositions; 144 architectures/gates scanned before this replay",
      "Features from PR16 extracted without labels; upstream S70 meta-producers not completely nested",
      "Variants chosen after development-fold comparisons; no independent model accuracy claim",
      "19-piece clustered uncertainty is descriptive; no automatic promotion"]
    (output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    for label,r in report["results"].items():
        print(label,"global",r["correct"],"poly",r["poly_correct"],
              "fixes",r["corrections"],"regressions",r["regressions"],
              "gain",r["change_from_s70"],flush=True)
    return report

if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("--s70",required=True,type=Path)
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    args=a.parse_args()
    run(args.s70,args.features,args.output)
