"""Audit the remaining internal fold-2 actions of the raw residual LR at p>=0.60.

Uses only frozen OOF replay exports from run 37356100423.
No refit, no audio, no outer labels, no threshold search.

Goal: explain the sole negative internal fold at p=.60 before considering any
additional rule.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.v273_residual_audit import require

TH=.60

def summarize(y,p,mask):
    c=int(np.sum(mask&(y==2)));r=int(np.sum(mask&(y==3)));o=int(np.sum(mask&~np.isin(y,(2,3))))
    return {"actions":int(mask.sum()),"corrections":c,"regressions":r,"other_k":o,"net":c-r}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--internal-exports",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    allrows=[];per={}
    for fold in (0,1,2,4):
        pth=a.internal_exports/f"fold-{fold}"/"replay.npz"
        with np.load(pth,allow_pickle=False) as z:d={k:np.asarray(z[k]) for k in z.files}
        action=d["val_b_low"].astype(bool)&(d["val_base_k"].astype(int)==3)
        valid=d["val_valid"].astype(bool)
        ids=d["val_action_ids"][valid].astype(int)
        y=d["val_true_k"].astype(int)[action][valid]
        prob=d["val_probability"].astype(float)
        X=d["val_X"].astype(float)[valid]
        f0=d["val_f0"].astype(float)[valid]
        reg=np.asarray(d["val_register"])[...].astype(str)
        rec=d["val_recording"].astype(str)[action][valid]
        require(len(ids)==len(y)==len(prob)==len(X)==len(f0)==len(reg)==len(rec),"length drift")
        mask=prob>=TH
        per[str(fold)]=summarize(y,prob,mask)
        for i in np.flatnonzero(mask):
            allrows.append({"fold":fold,"row_id":int(ids[i]),"recording":str(rec[i]),"true_k":int(y[i]),
                            "probability":float(prob[i]),"pair":float(X[i,0]),"triplet":float(X[i,1]),
                            "pair_minus_triplet":float(X[i,0]-X[i,1]),"median_triplet_f0":float(f0[i]),
                            "register":str(reg[i])})

    f2=[r for r in allrows if r["fold"]==2 and r["true_k"] in (2,3)]
    require(sum(r["true_k"]==2 for r in f2)-sum(r["true_k"]==3 for r in f2)==-1,"fold2 p60 net drift")
    by_reg={}
    for name in ("low","mid","high","unassigned"):
        rr=[r for r in f2 if r["register"]==name]
        by_reg[name]={"K2":sum(r["true_k"]==2 for r in rr),"K3":sum(r["true_k"]==3 for r in rr),
                      "net":sum(r["true_k"]==2 for r in rr)-sum(r["true_k"]==3 for r in rr),"rows":len(rr)}
    def cls(k,key):
        z=np.asarray([r[key] for r in f2 if r["true_k"]==k],float)
        return None if not len(z) else {"n":len(z),"min":float(z.min()),"median":float(np.median(z)),"max":float(z.max()),"mean":float(z.mean())}
    compare={key:{"K2":cls(2,key),"K3":cls(3,key)} for key in ("probability","pair","triplet","pair_minus_triplet","median_triplet_f0")}

    # Identify precisely what changes between .60 and .65 in fold2.
    band=[r for r in f2 if .60<=r["probability"]<.65]
    report={"status":"completed","experiment":"v273_residual_p060_fold2_cases","threshold":TH,
            "per_fold":per,"fold2":{"rows":f2,"by_register":by_reg,"class_compare":compare,
                                    "p060_to_p065_band":band,
                                    "band_K2":sum(r["true_k"]==2 for r in band),
                                    "band_K3":sum(r["true_k"]==3 for r in band)},
            "prediction_changes":False,"outer_fold_3_used":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# Residual p0.60 — fold2 action audit","",
           "| fold | corrections | regressions | other-K | net |","|---:|---:|---:|---:|---:|"]
    for f in (0,1,2,4):
        q=per[str(f)];lines.append(f"| {f} | {q['corrections']} | {q['regressions']} | {q['other_k']} | {q['net']:+d} |")
    lines+=["","## Fold2 by estimated register","","| register | K2 | K3 | net |","|---|---:|---:|---:|"]
    for name,q in by_reg.items():lines.append(f"| {name} | {q['K2']} | {q['K3']} | {q['net']:+d} |")
    lines+=["",f"Rows in probability band [0.60,0.65): **{len(band)}** = {report['fold2']['band_K2']} K2 / {report['fold2']['band_K3']} K3.",
            "","## Fold2 p>=.60 K2/K3 rows","","| row | K | p | pair | triplet | pair-triplet | f0 | register |",
            "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(f2,key=lambda z:z["probability"],reverse=True):
        lines.append(f"| {r['row_id']} | {r['true_k']} | {r['probability']:.4f} | {r['pair']:.4f} | {r['triplet']:.4f} | {r['pair_minus_triplet']:.4f} | {r['median_triplet_f0']:.2f} | {r['register']} |")
    lines+=["","Frozen replay audit only; no new rule selected."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
