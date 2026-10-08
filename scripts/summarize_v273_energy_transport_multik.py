"""Locked cross-fold three-class (K2/K3/K4) source-transport study.

Evaluates an intentionally restricted 3-way diagnostic, not a replacement
for 7-way global Exact-K. Fixed class balance, fixed logistic settings,
no fold-level hyperparameter search, no heldout retuning.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.yourmt3_exactk_common import FOLDS,require

CLASSES=(2,3,4)
GROUPS={
  "static_only":("static__",),
  "birth_only":("birth__",),
  "transport_only":("transport__",),
  "persistence_only":("persistence__",),
  "extinction_only":("extinction__",),
  "source_plus_sink":("birth__","extinction__"),
  "source_survival_sink":("birth__","persistence__","extinction__"),
  "full_dynamics":("birth__","transport__","persistence__","extinction__"),
  "full_static_plus_dynamics":("static__","birth__","transport__","persistence__","extinction__"),
}
def metrics(y,p,proba=None):
    conf=confusion_matrix(y,p,labels=CLASSES)
    report=dict(rows=len(y),accuracy=float(accuracy_score(y,p)),
                balanced_accuracy=float(balanced_accuracy_score(y,p)),
                macro_F1=float(f1_score(y,p,labels=CLASSES,average="macro",zero_division=0)),
                recall_by_K={str(k):float(np.mean(p[y==k]==k)) for k in CLASSES},
                confusion=conf.astype(int).tolist())
    if proba is not None:
        report["macro_ovr_auc"]=float(roc_auc_score(y,proba,labels=CLASSES,multi_class="ovr",average="macro"))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    seen=set();rows=[]
    for file in sorted(a.input_root.rglob("report.json")):
        rep=json.loads(file.read_text())
        if rep.get("experiment")!="v273_energy_transport_multik_extract":continue
        f=int(rep["fold"]);require(f in FOLDS and f not in seen,"duplicate/bad fold")
        seen.add(f)
        chunk=[json.loads(line) for line in (file.parent/"rows.jsonl").read_text().splitlines() if line.strip()]
        require(len(chunk)==450 and all(r["fold"]==f and r["true_k"] in CLASSES for r in chunk),"row drift")
        require(all(sum(r["true_k"]==k for r in chunk)==150 for k in CLASSES),"class sampling drift")
        rows+=chunk
    require(set(FOLDS)==seen and len(rows)==1800,"fold coverage drift")
    rows.sort(key=lambda r:(r["fold"],r["global_index"]))
    require(len({r["global_index"] for r in rows})==1800,"global id reused")
    groups={}
    for r in rows:
        m=r["member"]
        if m in groups:require(groups[m]==r["fold"],"RECORDING LEAKAGE")
        else:groups[m]=r["fold"]
    y=np.array([r["true_k"] for r in rows],int)
    folds=np.array([r["fold"] for r in rows],int)
    baseline=np.array([r["base_pred"] for r in rows],int)
    names=sorted(rows[0]["features"])
    require(all(sorted(r[k])==names for r in rows for k in ("features","scrambled","permuted")),"schema inconsistency")
    arms={**GROUPS,"negative_time_scramble":tuple(), "negative_band_permute":tuple()}
    summary={}
    for name,prefixes in arms.items():
        neg="scrambled" if name=="negative_time_scramble" else ("permuted" if name=="negative_band_permute" else "features")
        cols=names if neg!="features" else [x for x in names if x.startswith(prefixes)]
        require(cols,"empty columns")
        X=np.asarray([[r[neg][c] for c in cols] for r in rows],float)
        require(np.isfinite(X).all(),"nonfinite matrix")
        pred=np.full(len(y),-1,int)
        proba=np.full((len(y),3),np.nan,float)
        folds_report={}
        for f in FOLDS:
            tr=folds!=f;val=folds==f
            clf=make_pipeline(StandardScaler(),LogisticRegression(
                C=.1,max_iter=4000,solver="lbfgs",class_weight="balanced",random_state=27402))
            clf.fit(X[tr],y[tr])
            pv=clf.predict(X[val]).astype(int)
            probs=clf.predict_proba(X[val])
            require(np.array_equal(clf[-1].classes_,CLASSES),"classifier class drift")
            pred[val]=pv;proba[val]=probs
            folds_report[str(f)]=metrics(y[val],pv,probs)
        require(np.isfinite(proba).all() and np.isin(pred,CLASSES).all(),"incomplete prediction")
        summary[name]=dict(dimensions=len(cols),aggregate=metrics(y,pred,proba),folds=folds_report)
    baseline_result=metrics(y,baseline)
    report=dict(experiment="v273_energy_transport_k234_crossfold",status="completed",
                cohort=dict(rows=len(rows),per_class=600,folds=list(FOLDS),
                            recordings=len(groups),player05_used=False,fold3_used=False,
                            label_stratified_sampling=True),
                task="balanced restricted 3-class K2/K3/K4 diagnostic, not global Exact-K",
                baseline_frozen_full_7way_on_selected_events=baseline_result,
                arms=summary,automatic_promotion=False,
                interpretation_limits=[
                   "Crossfold is exploratory: data come from previously exposed compositions",
                   "Balanced class distributions and 3-way training differ from full 7-way baseline",
                   "Spectral-band movements are descriptive, not measured acoustic fluid velocity",
                   "Full 160ms post-event future context used: not causal realtime",
                   "Time scrambling alters transition boundaries, not solely abstract causality",
                   "No use of withheld player05 or fold3"
                ])
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    lines=["# Source / transport / survival — K2 K3 K4","",
           "**1,800 preselected label-balanced groups. Three-way diagnostic only; not comparable to 7-class headline Exact-K.**","",
           "| Arm | dims | Balanced accuracy | Macro F1 | Macro AUC | Recall K2 | K3 | K4 | Fold2 balanced |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key,rec in summary.items():
        m=rec["aggregate"];f2=rec["folds"]["2"]
        lines.append(f"| {key} | {rec['dimensions']} | {100*m['balanced_accuracy']:.2f}% | {100*m['macro_F1']:.2f}% | {m['macro_ovr_auc']:.4f} | {100*m['recall_by_K']['2']:.1f}% | {100*m['recall_by_K']['3']:.1f}% | {100*m['recall_by_K']['4']:.1f}% | {100*f2['balanced_accuracy']:.2f}% |")
    lines.extend(["","Frozen 7-way baseline on same balanced events: "+
                  f"{100*baseline_result['accuracy']:.2f}% Exact-K; non-comparable training class set.",
                  "","Limitations:"]+["- "+s for s in report["interpretation_limits"]]+
                 ["","No promotion."])
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)
if __name__=="__main__":main()
