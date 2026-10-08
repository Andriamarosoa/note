"""Cross-fold audit of label-free energy source/transport/survival features.

Exploratory failure cohort: folds 0/1/2/4 only; player05 and fold3 are out
of scope. Test fixed feature ablations and negative controls identically.
No winner may be selected for deployment from the held-out outcomes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scripts.summarize_v273_energy_flux_combined import FOLDS,require,evaluate
from scripts.extract_v273_energy_transport import FAMILIES

GROUPS={
 "static_only":("static__",),
 "birth_only":("birth__",),
 "transport_only":("transport__",),
 "persistence_only":("persistence__",),
 "extinction_only":("extinction__",),
 "birth_plus_transport":("birth__","transport__"),
 "birth_plus_persistence_plus_extinction":("birth__","persistence__","extinction__"),
 "all_dynamics":("birth__","transport__","persistence__","extinction__"),
 "static_plus_dynamics":("static__","birth__","transport__","persistence__","extinction__"),
}
NEGATIVES=("scrambled","permuted")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    rows=[];seen=set()
    for path in sorted(a.input_root.rglob("report.json")):
        rep=json.loads(path.read_text())
        if rep.get("experiment")!="v273_energy_transport_extract":continue
        f=int(rep["fold"]);require(f in FOLDS and f not in seen,"duplicate fold")
        seen.add(f)
        chunk=[json.loads(line) for line in (path.parent/"rows.jsonl").read_text().splitlines() if line.strip()]
        require(len(chunk)==rep["rows"],"bad cohort count")
        require(all(r["fold"]==f and r["true_k"] in (2,3) for r in chunk),"bad fold labels")
        rows.extend(chunk)
    require(seen==set(FOLDS) and len(rows)==488,"missing input")
    rows.sort(key=lambda r:(int(r["fold"]),int(r["row_id"])))
    ids=[r["row_id"] for r in rows];require(len(set(ids))==488,"duplicate row")
    y=np.asarray([int(r["true_k"]==2) for r in rows],int)
    require(int(y.sum())==216,"K2 count drift")
    fold=np.asarray([r["fold"] for r in rows],int)
    recording_folds={}
    for row in rows:
        key=row["recording_id"]
        if key in recording_folds:require(recording_folds[key]==row["fold"],"RECORDING LEAK")
        else:recording_folds[key]=row["fold"]
    names=sorted(rows[0]["features"])
    require(all(sorted(r[k])==names for r in rows for k in ("features",*NEGATIVES)),"feature schema drift")
    require(all(sum(n.startswith(f+"__") for f in FAMILIES)==1 for n in names),"unexpected names")
    fam={}
    for key,prefixes in GROUPS.items():
        cols=[n for n in names if n.startswith(prefixes)]
        require(cols,"empty group")
        X=np.asarray([[float(r["features"][n]) for n in cols] for r in rows],float)
        require(np.isfinite(X).all(),"nonfinite group")
        stats=evaluate(X,y,fold)
        fam[key]=dict(dimensions=len(cols),**stats)
    for neg in NEGATIVES:
        cols=names
        X=np.asarray([[float(r[neg][n]) for n in cols] for r in rows],float)
        require(np.isfinite(X).all(),"nonfinite negative control")
        fam["negative_"+neg]=dict(dimensions=len(cols),**evaluate(X,y,fold))
    report={
      "status":"completed","experiment":"v273_energy_transport_crossfold",
      "cohort":{"rows":488,"k2":216,"k3":272,"recordings":len(recording_folds),
                "folds":list(FOLDS),"fold3_used":False,"player05_used":False},
      "hypothesis":"observable spectral transfer + residual source/sink + temporal persistence vs static energy",
      "caveats":[
        "K2/K3 attack-owned error cohort, NOT all polyphonic Exact-K and not sustained simultaneous pitch count",
        "Derived STFT mass balance is a spectrotemporal proxy, not a Navier-Stokes velocity/pressure field",
        "Adjacent-band transport is model-assumed and not uniquely identified from mono audio",
        "Observation uses 160ms of post-onset samples, not a causal online prediction",
        "Negative controls test dependence on band adjacency and time order, not physical source identification",
        "Already-inspected internal cohort: hypothesis generation only, not an independent test"
      ],
      "automatic_promotion":False,"families":fam}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    lines=["# V27.3 source / transport / persistence / extinction audit","",
           "Exploratory K2/K3 onset-ownership cohort of 488 internal events; neither fold 3 nor player05 used.","",
           "| Family | Dimensions | OOF AUC | Fold 2 AUC | Correct | Regress | Net | Fold 2 net |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,rec in fam.items():
        f2=next(x for x in rec["rotations"] if x["val_fold"]==2)
        tot=rec["selected_policy_total"]
        lines.append(f"| {name} | {rec['dimensions']} | {rec['oof_auc']:.4f} | {f2['val_auc']:.4f} | {tot['corrections']} | {tot['regressions']} | {tot['net']:+d} | {f2['val_policy']['net']:+d} |")
    lines+=["","## Interpretation constraints",""]
    lines+=["- "+s for s in report["caveats"]]
    lines+=["","Results cannot justify selecting one arm on the already exposed internal cohort. No promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
