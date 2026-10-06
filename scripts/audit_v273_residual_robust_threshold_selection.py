"""Robust global threshold selection for the raw residual LR.

Searches only a predeclared threshold grid on frozen internal OOF probabilities.
No new features and no refit.

A threshold is considered robust only if:
- total Exact-K net > 0,
- every internal fold net >= 0,
- every GuitarSet player net >= 0,
- every style net >= 0.

Among feasible thresholds choose max total net, then more corrections, then the
higher threshold (conservative tie break).

Part (comp/solo), variant and recording are diagnostic only, not selection
constraints. No outer fold 3.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np

FOLDS=(0,1,2,4)
GRID=tuple(round(x,2) for x in np.arange(.50,.751,.01))

def require(c,m):
    if not c: raise RuntimeError(m)

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    if not m:return {"player":"?","style":"?","part":"?","variant":"?","piece":stem}
    player,style,variant,rest,part=m.groups()
    return {"player":player,"style":style,"part":part,"variant":style+variant,"piece":f"{style}{variant}-{rest}"}

def load(root):
    rows=[]
    for f in FOLDS:
        p=Path(root)/f"fold-{f}"/"replay.npz"
        with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
        action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
        valid=a["val_valid"].astype(bool)
        y=a["val_true_k"].astype(int)[action][valid]
        pr=a["val_probability"].astype(float)
        rec=a["val_recording"].astype(str)[action][valid]
        require(len(y)==len(pr)==len(rec),"length drift")
        for yy,pp,rr in zip(y,pr,rec):
            q={"fold":int(f),"true_k":int(yy),"p":float(pp),"recording":str(rr)}
            q.update(meta(str(rr)));rows.append(q)
    return rows

def net(rows,th):
    aa=[r for r in rows if r["p"]>=th]
    c=sum(r["true_k"]==2 for r in aa);g=sum(r["true_k"]==3 for r in aa);o=sum(r["true_k"] not in (2,3) for r in aa)
    return {"actions":len(aa),"corrections":c,"regressions":g,"other_k":o,"net":c-g}

def group(rows,key,th):
    return {str(g):net([r for r in rows if r[key]==g],th) for g in sorted({r[key] for r in rows},key=str)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--internal-exports",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    rows=load(a.internal_exports)
    candidates=[]
    for th in GRID:
        total=net(rows,th)
        folds=group(rows,"fold",th);players=group(rows,"player",th);styles=group(rows,"style",th)
        feasible=(total["net"]>0 and all(q["net"]>=0 for q in folds.values())
                  and all(q["net"]>=0 for q in players.values())
                  and all(q["net"]>=0 for q in styles.values()))
        candidates.append({"threshold":float(th),"total":total,"folds":folds,"players":players,"styles":styles,
                           "parts":group(rows,"part",th),"feasible":bool(feasible)})
    feasible=[q for q in candidates if q["feasible"]]
    best=None if not feasible else max(feasible,key=lambda q:(q["total"]["net"],q["total"]["corrections"],q["threshold"]))
    rep={"status":"completed","experiment":"v273_residual_robust_threshold_selection",
         "grid":list(GRID),"constraints":["total net > 0","all folds net >=0","all players net >=0","all styles net >=0"],
         "best":best,"candidates":candidates,"outer_fold_3_used":False,"prediction_changes":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual robust global threshold selection","",
           "Constraints: every fold, player and style must be non-negative; total net must be positive.",""]
    if best is None:
        lines+=["**No feasible threshold on the frozen grid.**",""]
    else:
        lines+=[f"Selected threshold: **{best['threshold']:.2f}**.",
                f"Total: **{best['total']['corrections']} corrections / {best['total']['regressions']} regressions = {best['total']['net']:+d}**.",""]
    lines+=["| threshold | total net | min fold | min player | min style | feasible |",
            "|---:|---:|---:|---:|---:|---|"]
    for q in candidates:
        lines.append(f"| {q['threshold']:.2f} | {q['total']['net']:+d} | {min(x['net'] for x in q['folds'].values()):+d} | {min(x['net'] for x in q['players'].values()):+d} | {min(x['net'] for x in q['styles'].values()):+d} | {q['feasible']} |")
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
