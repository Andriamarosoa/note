"""Controlled harmonic-source trajectory ablations on the identical 1,800 native events.

Joins exact fixed-seed K2/K3/K4 rows of the prior 24-band energy audit;
the control feature matrices are themselves label-free. Four held-out
composition folds and fixed, untuned 3-class logistic regression.
THIS IS NOT full seven-class Exact-K or an unseen-composition validation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scripts.yourmt3_exactk_common import FOLDS,require
from scripts.summarize_v273_energy_transport_multik import metrics

ARMS={
 "band_static_control":(("band","features",("static__",)),),
 "band_dynamics_control":(("band","features",("birth__","transport__","persistence__","extinction__")),),
 "band_full_control":(("band","features",("static__","birth__","transport__","persistence__","extinction__")),),
 "harmonic_spectral_only":(("harmonic","features",("spectral__",)),),
 "harmonic_source_only":(("harmonic","features",("source__",)),),
 "harmonic_lifecycle_only":(("harmonic","features",("birth__","persistence__","damping__")),),
 "harmonic_coherence_only":(("harmonic","features",("coherence__",)),),
 "harmonic_full":(("harmonic","features",tuple()),),
 "harmonic_detuned_control":(("harmonic","detuned",tuple()),),
 "harmonic_time_scrambled_control":(("harmonic","scrambled",tuple()),),
 "band_static_plus_harmonic_lifecycle":(
    ("band","features",("static__",)),
    ("harmonic","features",("source__","birth__","persistence__","damping__","coherence__")),
 ),
 "band_full_plus_harmonic_lifecycle":(
    ("band","features",("static__","birth__","transport__","persistence__","extinction__")),
    ("harmonic","features",("source__","birth__","persistence__","damping__","coherence__")),
 ),
}

def load(root, experiment):
    rows={}; seen=set()
    for path in sorted(root.rglob("report.json")):
        report=json.loads(path.read_text())
        if report.get("experiment")!=experiment:continue
        fold=int(report["fold"])
        require(fold in FOLDS and fold not in seen,"fold duplicate/missing")
        seen.add(fold)
        chunk=[json.loads(s) for s in (path.parent/"rows.jsonl").read_text().splitlines() if s.strip()]
        require(len(chunk)==450,"cohort rows mismatch")
        for r in chunk:
            key=int(r["global_index"])
            require(key not in rows and int(r["fold"])==fold,"duplicate or alien row")
            rows[key]=r
    require(seen==set(FOLDS) and len(rows)==1800,"missing cohort")
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--harmonic-root",type=Path,required=True)
    p.add_argument("--band-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),"refusing overwrite")
    h=load(a.harmonic_root,"v273_harmonic_trajectory_extract")
    b=load(a.band_root,"v273_energy_transport_multik_extract")
    require(set(h)==set(b),"cohort mismatch")
    ids=sorted(h)
    for i in ids:
        require(all(h[i][z]==b[i][z] for z in ("member","fold","true_k","base_pred")),"metadata mismatch")
    y=np.asarray([h[i]["true_k"] for i in ids],int)
    fold=np.asarray([h[i]["fold"] for i in ids],int)
    baseline=np.asarray([h[i]["base_pred"] for i in ids],int)
    recordings={}
    for i in ids:
        m=h[i]["member"]
        if m in recordings:require(recordings[m]==h[i]["fold"],"recording overlap")
        else:recordings[m]=h[i]["fold"]
        require(m[:2] in ("00","01","02","03","04"),"withheld player")
    require(all(np.sum(y[fold==f]==k)==150 for f in FOLDS for k in (2,3,4)),"K composition drift")
    basem=metrics(y,baseline)
    outputs={}
    predictions={}
    for arm,parts in ARMS.items():
        columns=[]
        arrays=[]
        for origin,field,prefixes in parts:
            source=h if origin=="harmonic" else b
            names=sorted(source[ids[0]][field])
            if prefixes:names=[n for n in names if n.startswith(prefixes)]
            require(len(names)>0,"empty arm")
            require(not set(origin+"__"+x for x in names).intersection(columns),"duplicate cols")
            columns.extend(origin+"__"+x for x in names)
            X=np.asarray([[source[i][field][n] for n in names] for i in ids],float)
            require(np.isfinite(X).all(),"nonfinite")
            arrays.append(X)
        X=np.column_stack(arrays)
        result=np.full(len(y),-1,int)
        proba=np.full((len(y),3),np.nan)
        per_fold={}
        for f in FOLDS:
            fit=fold!=f;val=fold==f
            model=make_pipeline(StandardScaler(),LogisticRegression(
                C=.1,solver="lbfgs",max_iter=4000,class_weight="balanced",
                random_state=27402))
            model.fit(X[fit],y[fit])
            require(np.array_equal(model[-1].classes_,(2,3,4)),"model class drift")
            result[val]=model.predict(X[val]).astype(int)
            proba[val]=model.predict_proba(X[val])
            per_fold[str(f)]=metrics(y[val],result[val],proba[val])
        require(np.isfinite(proba).all() and np.isin(result,[2,3,4]).all(),"predictions missing")
        outputs[arm]=dict(dimensions=X.shape[1],metrics=metrics(y,result,proba),folds=per_fold,
                          paired_v_band_static=None)
        predictions[arm]=result
    ctrl=predictions["band_static_control"]
    for name,rec in outputs.items():
        p0=predictions[name]
        corrected=int(np.sum((p0==y)&(ctrl!=y)))
        regressed=int(np.sum((p0!=y)&(ctrl==y)))
        rec["paired_v_band_static"]={"corrections":corrected,"regressions":regressed,
                                    "net":corrected-regressed}
    report=dict(status="completed",experiment="v273_harmonic_trajectory_K234_ablation",
                cohort=dict(rows=len(y),per_class=600,recordings=len(recordings),
                            folds=list(FOLDS),fold3_used=False,player05_used=False),
                baseline_original_7class_on_balanced_rows=basem,
                task="Restricted equal-prior three-class K2/K3/K4 held-out-fold diagnostic, NOT global Exact-K",
                arms=outputs,
                hypothesis_notes=[
                  "Harmonic-source tracks are overlapping-template reconstructions, not isolated notes",
                  "NNLS-like activation uses 160ms future audio, so it is not an online causal estimator",
                  "Each fold has 150 observations per class, unlike natural class frequencies",
                  "Existing native GuitarSet folds are development-exposed, not independent",
                  "No player 05 or fold 3 used, no promotion into freeze_local_combo",
                  "Inharmonic templates and time scramble are descriptive negative controls"
                ],automatic_promotion=False)
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",
        global_index=np.asarray(ids,np.int64),fold=fold,true_k=y,
        **predictions)
    lines=["# V27.3 harmonic-source trajectories — fixed K2/K3/K4 ablations","",
           "Same exact 1,800 events and folds as the prior 24-band experiment. No player05/fold3.",
           "",
           "| Arm | dims | Balanced accuracy | Macro AUC | K2 recall | K3 recall | K4 recall | Fold2 balanced | Net vs band static |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,r in outputs.items():
        m=r["metrics"];f2=r["folds"]["2"];p=r["paired_v_band_static"]
        lines.append(f"| {name} | {r['dimensions']} | {100*m['balanced_accuracy']:.2f}% | {m['macro_ovr_auc']:.4f} | {100*m['recall_by_K']['2']:.1f}% | {100*m['recall_by_K']['3']:.1f}% | {100*m['recall_by_K']['4']:.1f}% | {100*f2['balanced_accuracy']:.2f}% | {p['net']:+d} |")
    lines+=["","## Interpretation limits"]+["- "+x for x in report["hypothesis_notes"]]+
           ["","No automatic promotion; these are already exposed compositions."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines),flush=True)

if __name__=="__main__":main()
