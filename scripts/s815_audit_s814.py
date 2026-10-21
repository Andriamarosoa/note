"""Forensic review of S814 completed signed 80-track report; no retuning.

Show which K classes and compositions drive excess overcount. No model
training, no new candidate scores. Work exclusively from authenticated
archived S814 output, NOT from synthetic fixtures or imagined metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

def audit(path:Path):
    r=json.loads(path.read_text())
    if (r.get("schema")!="S814_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1" or
        not r.get("complete") or len(r.get("tracks",[]))!=80 or
        len(r.get("folds",{}))!=5 or
        r["original_fit80_active_k"]["correct"]!=142213 or
        r["original_fit80_active_k"]["frames"]!=249411 or
        r["teacher_active_exact_k"]["correct"]!=136360 or
        r["teacher_active_exact_k"]["frames"]!=249411 or
        r["unification_gate"]["allow_unification"]):
        raise ValueError("S814 authentic output or S704 counts mismatch")
    by_k={}
    for key in range(7):
        c={"correct":0,"frames":0,"undercount":0,"overcount":0}
        for row in r["tracks"]:
            d=row["by_k"][str(key)]
            for field in c: c[field]+=int(d[field])
        if sum(c[f] for f in ("correct","undercount","overcount"))!=c["frames"]:
            raise AssertionError("K class accounting mismatch")
        c["exact_rate"]=c["correct"]/max(c["frames"],1)
        c["under_rate"]=c["undercount"]/max(c["frames"],1)
        c["over_rate"]=c["overcount"]/max(c["frames"],1)
        by_k[str(key)]=c
    comps={}
    for comp, fold in r["folds"].items():
        rows=[x["active_k"] for x in r["tracks"] if x["composition"]==comp]
        frames=sum(x["frames"] for x in rows)
        if frames<=0: raise AssertionError("empty evaluation composition")
        comps[comp]={"selected_selector":fold["heads"]["selector"],
                     "selected_shift":fold["heads"]["shift"],
                     "accuracy":sum(x["correct"] for x in rows)/frames,
                     "overcount":sum(x["overcount"] for x in rows),
                     "undercount":sum(x["undercount"] for x in rows),
                     "calibration_results":fold["heads"]["calibration_results"]}
    z={"schema":"S815_INDEPENDENT_FORENSIC_S814_V1",
       "source_report_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
       "source_fit80":r["original_fit80_active_k"],
       "rejected_s814_fit80":r["teacher_active_exact_k"],
       "by_true_k":by_k,
       "by_composition":comps,
       "only_research_fit80":True,"new_validation_score":False,
       "changed_model_or_midi":False}
    return z

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.report)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("S815_S814_FORENSIC_BASE",json.dumps(result["source_fit80"]))
    print("S815_S814_FORENSIC_REJECTED",json.dumps(result["rejected_s814_fit80"]))
    print("S815_S814_FORENSIC_BY_K",json.dumps(result["by_true_k"]))
    for key,v in result["by_composition"].items():
        print("S815_S814_FORENSIC_COMP",key,
              json.dumps({x:v[x] for x in ("selected_selector","selected_shift","accuracy","overcount","undercount")}))
    print("S815_FORENSIC_NO_RETUNING",True)
