"""Read-only S815 genuine-score forensics, never feeds any selector."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path

def inspect(report_path:Path):
    data=json.loads(report_path.read_text())
    if (data.get("schema")!="S815_ACTIVE_EXACT_K_REAL_FRESH256_FIT80_V1"
        or not data.get("complete") or len(data.get("tracks",[]))!=80
        or data["teacher_active_exact_k"]["correct"]!=141946
        or data["teacher_active_exact_k"]["frames"]!=249411
        or data["original_fit80_active_k"]["correct"]!=142213):
        raise ValueError("invalid signed S815 scientific report")
    by_k={}
    for k in range(7):
        subset=[r["by_k"][str(k)] for r in data["tracks"]]
        values={name:sum(r[name] for r in subset) for name in
                ("correct","frames","undercount","overcount")}
        if values["correct"]+values["undercount"]+values["overcount"]!=values["frames"]:
            raise AssertionError("real K confusion accounting failed")
        values["accuracy"]=values["correct"]/max(values["frames"],1)
        by_k[str(k)]=values
    policies={}
    for fold,meta in data["folds"].items():
        choice=meta["heads"]
        policies[fold]=dict(selector=choice["selector"],shift=choice["shift"],
            calibration=[{"selector":v["selector"],"shift":v["shift"],
             "correct":v["correct"],"frames":v["frames"],"overcount":v["overcount"]}
                        for v in choice["calibration_results"]])
    return dict(schema="S816_READONLY_S815_FORENSIC_V1",
                source_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                by_true_k=by_k,calibration_policy_per_fold=policies,
                source_untouched=True,no_new_candidate_score=True,
                FIT80_exploratory_only=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    r=inspect(args.report)
    args.out.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print("S816_S815_REAL_TRUE_K",json.dumps(r["by_true_k"]))
    for k,fold in r["calibration_policy_per_fold"].items():
        print("S816_S815_FOLD_FROZEN_POLICY",k,fold["selector"],fold["shift"])
