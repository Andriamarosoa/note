"""Cross-fold evaluation of symmetric pitch TTA."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.v273_residual_audit import FOLDS,require,write_json

HORIZONS=(1,2,3,4,6)
FAMILIES=("mean_margin","lcb_margin","median_margin","mean_argmax")

def discover(root):
    out={}
    for p in root.rglob("report.json"):
        try:r=json.loads(p.read_text())
        except Exception:continue
        if r.get("experiment")!="v273_symmetric_pitch_tta_fold":continue
        f=int(r["validation_fold"]); npz=p.parent/"symmetric.npz"
        require(f in FOLDS and f not in out and npz.exists(),"bad source")
        out[f]=npz
    require(set(out)==set(FOLDS),"missing folds")
    return out

def load_rows(found):
    rows=[]
    for f,p in found.items():
        with np.load(p,allow_pickle=False) as z:
            steps=np.asarray(z["steps"],np.int32)
            require(np.array_equal(steps,np.arange(-6,7)),"step drift")
            for i,rid in enumerate(z["row_id"]):
                rows.append({"row_id":int(rid),"fold":f,"true_k":int(z["true_k"][i]),
                             "p":np.asarray(z["probability"][i],np.float64),
                             "pred":np.asarray(z["predicted"][i],np.int32),"steps":steps})
    require(len(rows)==488 and len({r["row_id"] for r in rows})==488,"cohort drift")
    return rows

def decision(row,family,h):
    take=np.abs(row["steps"])<=h
    p=row["p"][take]
    margin=p[:,2]-p[:,3]
    if family=="mean_margin": return float(np.mean(margin))>0
    if family=="median_margin": return float(np.median(margin))>0
    if family=="mean_argmax": return int(np.argmax(np.mean(p,axis=0)))==2
    if family=="lcb_margin":
        return float(np.mean(margin)-np.std(margin)/np.sqrt(len(margin)))>0
    raise ValueError(family)

def metrics(rows,family,h):
    action=np.asarray([decision(r,family,h) for r in rows],bool)
    y=np.asarray([r["true_k"] for r in rows],np.int32)
    corr=int(np.sum(action&(y==2))); reg=int(np.sum(action&(y==3)))
    return {"actions":int(action.sum()),"corrections":corr,"regressions":reg,"net":corr-reg}

def select(fit):
    cand=[]
    order={x:i for i,x in enumerate(FAMILIES)}
    for fam in FAMILIES:
        for h in HORIZONS:
            q=metrics(fit,fam,h); cand.append({"family":fam,"h":h,**q})
    best=min(cand,key=lambda q:(-q["net"],q["regressions"],q["actions"],q["h"],order[q["family"]]))
    return (None if best["net"]<=0 else best),cand

def diagnostics(rows):
    vals=[]
    for r in rows:
        m=r["p"][:,2]-r["p"][:,3]
        diffs=[]
        for s in range(1,7):
            plus=m[np.where(r["steps"]==s)[0][0]]
            minus=m[np.where(r["steps"]==-s)[0][0]]
            diffs.append(abs(float(plus-minus)))
        vals.append(float(np.mean(diffs)))
    return {"mean_abs_plus_minus_margin_difference":float(np.mean(vals)),
            "median_abs_plus_minus_margin_difference":float(np.median(vals))}

def run(a):
    require(not a.output.exists(),"refusing overwrite")
    rows=load_rows(discover(a.input_root))
    rotations=[]
    for f in FOLDS:
        fit=[r for r in rows if r["fold"]!=f]; val=[r for r in rows if r["fold"]==f]
        best,cand=select(fit)
        q={"actions":0,"corrections":0,"regressions":0,"net":0} if best is None else metrics(val,best["family"],best["h"])
        rotations.append({"fold":f,"selected":best,"val":q,
                          "top5_fit":sorted(cand,key=lambda x:(-x["net"],x["regressions"],x["actions"],x["h"]))[:5]})
    total={k:int(sum(r["val"][k] for r in rotations)) for k in ("actions","corrections","regressions","net")}
    fixed={}
    for fam in FAMILIES:
        for h in HORIZONS:
            key=f"{fam}|H={h}"
            per={str(f):metrics([r for r in rows if r["fold"]==f],fam,h) for f in FOLDS}
            fixed[key]={"folds":per,"total":{k:int(sum(per[str(f)][k] for f in FOLDS))
                                                for k in ("actions","corrections","regressions","net")}}
    nonneg=sum(r["val"]["net"]>=0 for r in rotations); worst=min(r["val"]["net"] for r in rotations)
    criteria={"beats_previous_plus1":total["net"]>1,"at_least_3_nonnegative_folds":nonneg>=3,
              "worst_fold_gte_minus5":worst>=-5,"at_least_25_corrections":total["corrections"]>=25}
    criteria["all_met"]=all(criteria.values())
    report={"status":"completed","experiment":"v273_symmetric_pitch_tta_summary",
            "rows":len(rows),"rotations":rotations,"selected_policy_total":total,
            "fixed_configs":fixed,"diagnostics":diagnostics(rows),"criteria":criteria,
            "outer_fold_3_used":False,"automatic_promotion":False}
    a.output.mkdir(parents=True);write_json(a.output/"report.json",report)
    lines=["# Symmetric pitch TTA","",
           "| fold | FIT choice | corr | reg | net |","|---:|---|---:|---:|---:|"]
    for r in rotations:
        s=r["selected"]; label="abstain" if s is None else f"{s['family']} H={s['h']}"
        q=r["val"];lines.append(f"| {r['fold']} | {label} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["",f"Selected policy total: **{total['corrections']} / {total['regressions']} = {total['net']:+d}**.",
             f"Criteria met: **{criteria['all_met']}**.","","Fold 3 excluded."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);run(p.parse_args())
