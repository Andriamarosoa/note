"""Stability audit for the frozen raw residual policy p(K2)>=0.60.

No tuning. Uses only frozen OOF replay exports from run 37356100423.
Reports concentration by GuitarSet player, style, comp/solo part and recording.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import numpy as np
def require(cond,msg):
    if not cond: raise RuntimeError(msg)

TH=.60
FOLDS=(0,1,2,4)

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    if not m:return {"player":"?","style":"?","part":"?","variant":"?","piece":stem}
    player,style,variant,rest,part=m.groups()
    return {"player":player,"style":style,"part":part,"variant":style+variant,"piece":f"{style}{variant}-{rest}"}

def rows(root):
    out=[]
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
            q={"fold":f,"true_k":int(yy),"p":float(pp),"recording":str(rr),"apply":bool(pp>=TH)}
            q.update(meta(str(rr)));out.append(q)
    return out

def acc(rr):
    aa=[r for r in rr if r["apply"]]
    c=sum(r["true_k"]==2 for r in aa);g=sum(r["true_k"]==3 for r in aa);o=sum(r["true_k"] not in (2,3) for r in aa)
    return {"rows":len(rr),"actions":len(aa),"corrections":c,"regressions":g,"other_k":o,"net":c-g}

def breakdown(rows,key):
    out=[]
    for g in sorted({r[key] for r in rows}):
        q=acc([r for r in rows if r[key]==g]);q[key]=g;out.append(q)
    return out

def concentration(groups):
    gains=np.asarray([q["net"] for q in groups],int)
    positive=np.asarray([max(0,int(x)) for x in gains],int)
    total=int(gains.sum())
    return {"groups":len(groups),"positive":int(np.sum(gains>0)),"zero":int(np.sum(gains==0)),
            "negative":int(np.sum(gains<0)),"total_net":total,
            "max_positive":int(positive.max()) if len(positive) else 0,
            "largest_positive_share_of_total":None if total<=0 else float(positive.max()/total),
            "min_group_net":int(gains.min()) if len(gains) else 0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--internal-exports",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    r=rows(a.internal_exports);overall=acc(r)
    require(overall["corrections"]-overall["regressions"]==7,"p060 net drift")
    bd={}
    for key in ("fold","player","style","part","variant","piece","recording"):
        gg=breakdown(r,key);bd[key]={"groups":gg,"concentration":concentration(gg)}
    rep={"status":"completed","experiment":"v273_residual_p060_stability",
         "threshold":TH,"overall":overall,"breakdowns":bd,
         "prediction_changes":False,"threshold_search":False,"outer_fold_3_used":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual p0.60 stability audit","",
           f"Overall: **{overall['corrections']} corrections / {overall['regressions']} regressions = {overall['net']:+d}**.","",
           "## By player","","| player | corr | reg | net |","|---|---:|---:|---:|"]
    for q in bd["player"]["groups"]:lines.append(f"| {q['player']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["","## By style","","| style | corr | reg | net |","|---|---:|---:|---:|"]
    for q in bd["style"]["groups"]:lines.append(f"| {q['style']} | {q['corrections']} | {q['regressions']} | {q['net']:+d} |")
    lines+=["","## Concentration summary","",json.dumps({k:v["concentration"] for k,v in bd.items()},indent=2,sort_keys=True)]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
