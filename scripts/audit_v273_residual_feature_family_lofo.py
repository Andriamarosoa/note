"""LOFO feature-family comparison for the V27.3 residual K2-vs-K3 classifier.

For each validation fold 0/1/2/4, this audit reuses the exact frozen FIT/VAL
B_low+baseK3 residual population exported by run 37356100423.

Variants (all StandardScaler + balanced LogisticRegression C=1):
  base              = pair residual + triplet residual
  base_geom         = base + 3 selected harmonic/geometry features
  base_attack       = base + 5 selected attack/exclusivity features
  base_geom_attack  = base + all 8 extra features

Extra features are inference-safe: frozen row timestamp + raw PCM + selected
triplet F0 decomposition. No annotation F0, no fold3, no threshold tuning on
fold3.

After LOFO predictions are frozen, each variant gets the same internal robust
threshold sweep. A threshold is feasible only if every fold, GuitarSet player
and style has non-negative Exact-K net and the total net is positive.
"""
from __future__ import annotations
import argparse,json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_low_band_fold2_vs_fold4 import inference_features
from scripts.v273_window_experiment import load_bundle

FOLDS=(0,1,2,4)
BASE=("best_pair_residual_ratio","best_triplet_residual_ratio")
GEOM=("geom_harmonic_relation_min_error","geom_span_cents","amp3_over_amp2")
ATTACK=("nov_onset_contrast_norm_median","exc_unique_attack_fraction_min",
        "joint_unique_x_post_median","weak_unique_fraction","weak_unique_post1_norm")
VARIANTS={
  "base":BASE,
  "base_geom":BASE+GEOM,
  "base_attack":BASE+ATTACK,
  "base_geom_attack":BASE+GEOM+ATTACK,
}
GRID=tuple(round(x,2) for x in np.arange(.30,.801,.01))

def require(c,m):
    if not c: raise RuntimeError(m)

def classifier():
    return Pipeline([("scale",StandardScaler()),
                     ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                                               class_weight="balanced",random_state=28431))])

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    if not m:return {"player":"?","style":"?","part":"?"}
    player,style,variant,rest,part=m.groups()
    return {"player":player,"style":style,"part":part}

def load_split(root,fold,split):
    p=Path(root)/f"fold-{fold}"/"replay.npz"
    with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
    action=a[split+"_b_low"].astype(bool)&(a[split+"_base_k"].astype(int)==3)
    valid=a[split+"_valid"].astype(bool)
    ids=a[split+"_action_ids"][valid].astype(int)
    y=a[split+"_true_k"].astype(int)[action][valid]
    X=a[split+"_X"].astype(float)[valid]
    rec=a[split+"_recording"].astype(str)[action][valid]
    f0=a[split+"_f0"].astype(float)[valid]
    prob=np.asarray(a[split+"_probability"],float)
    require(len(ids)==len(y)==len(X)==len(rec)==len(f0)==len(prob),"split length drift")
    rows=[]
    for i in range(len(y)):
        rows.append({"row_id":int(ids[i]),"fold":int(fold),"true_k":int(y[i]),
                     "recording_id":str(rec[i]),"median_triplet_f0":float(f0[i]),
                     "best_pair_residual_ratio":float(X[i,0]),
                     "best_triplet_residual_ratio":float(X[i,1]),
                     "exported_probability":float(prob[i]),**meta(str(rec[i]))})
    return rows

def reconstruct_extras(all_rows,bundle,config,dataset):
    cache,_,_=load_bundle(bundle,config)
    unique={}
    for r in all_rows:
        rid=r["row_id"]
        if rid in unique:
            require(unique[rid]["recording_id"]==r["recording_id"],"row recording drift")
        else:unique[rid]={"recording_id":r["recording_id"]}
    wanted={q["recording_id"] for q in unique.values()}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for rid,q in unique.items():
        require(str(cache["members"][rid])==q["recording_id"],"bundle member mismatch")
        by[q["recording_id"]].append(rid)
    extras={}
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid in sorted(by[member]):
            start=int(cache["cluster_start_samples"][rid])
            q=inference_features(samples,start)
            extras[rid]={k:float(q[k]) for k in GEOM+ATTACK}
        print(json.dumps({"recording":member,"features_done":len(extras)}),flush=True)
    require(len(extras)==len(unique),"extra feature coverage")
    return extras

def matrix(rows,names,extras):
    out=[]
    for r in rows:
        vals=[]
        for n in names:
            vals.append(float(r[n] if n in r else extras[r["row_id"]][n]))
        out.append(vals)
    X=np.asarray(out,float)
    require(np.isfinite(X).all(),"nonfinite feature")
    return X

def counts(rows,p,th):
    y=np.asarray([r["true_k"] for r in rows],int);m=np.asarray(p)>=th
    c=int(np.sum(m&(y==2)));g=int(np.sum(m&(y==3)));o=int(np.sum(m&~np.isin(y,(2,3))))
    return {"actions":int(m.sum()),"corrections":c,"regressions":g,"other_k":o,"net":c-g}

def grouped(rows,p,th,key):
    vals={}
    arr=np.asarray(p)
    for g in sorted({r[key] for r in rows},key=str):
        idx=np.asarray([r[key]==g for r in rows],bool)
        vals[str(g)]=counts([r for r,k in zip(rows,idx) if k],arr[idx],th)
    return vals

def robust_threshold(rows,p):
    candidates=[]
    for th in GRID:
        total=counts(rows,p,th)
        folds=grouped(rows,p,th,"fold");players=grouped(rows,p,th,"player");styles=grouped(rows,p,th,"style")
        feasible=(total["net"]>0 and all(q["net"]>=0 for q in folds.values())
                  and all(q["net"]>=0 for q in players.values())
                  and all(q["net"]>=0 for q in styles.values()))
        candidates.append({"threshold":float(th),"total":total,"folds":folds,"players":players,
                           "styles":styles,"feasible":bool(feasible)})
    fs=[q for q in candidates if q["feasible"]]
    best=None if not fs else max(fs,key=lambda q:(q["total"]["net"],q["total"]["corrections"],q["threshold"]))
    return best,candidates

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","dataset","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    folds={}
    all_rows=[]
    for f in FOLDS:
        fit=load_split(a.internal_exports,f,"fit");val=load_split(a.internal_exports,f,"val")
        folds[f]={"fit":fit,"val":val};all_rows+=fit+val
    extras=reconstruct_extras(all_rows,a.bundle,a.config,a.dataset)

    pooled={name:[] for name in VARIANTS};pooled_rows=[]
    per_variant={name:{"per_fold":[]} for name in VARIANTS}
    baseline_replay_max_abs=0.0

    for f in FOLDS:
        fit=folds[f]["fit"];val=folds[f]["val"]
        train=np.asarray([r["true_k"] in (2,3) for r in fit],bool)
        ytr=np.asarray([r["true_k"]==2 for r in fit],int)[train]
        require(len(np.unique(ytr))==2,"fit class collapse")
        pooled_rows+=val
        for name,names in VARIANTS.items():
            Xf=matrix(fit,names,extras);Xv=matrix(val,names,extras)
            m=classifier();m.fit(Xf[train],ytr)
            pv=m.predict_proba(Xv)[:,1]
            pooled[name].extend(map(float,pv))
            yv=np.asarray([r["true_k"] for r in val],int)
            k23=np.isin(yv,(2,3))
            auc=float(roc_auc_score((yv[k23]==2).astype(int),pv[k23]))
            per_variant[name]["per_fold"].append({"fold":int(f),"auc":auc,
                                                   "p050":counts(val,pv,.5)})
            if name=="base":
                exp=np.asarray([r["exported_probability"] for r in val],float)
                baseline_replay_max_abs=max(baseline_replay_max_abs,float(np.max(np.abs(exp-pv))))
                require(np.allclose(exp,pv,rtol=0,atol=1e-12),"baseline probability replay mismatch")

    require(len({r["row_id"] for r in pooled_rows})==len(pooled_rows),"VAL row duplicated across folds")
    yall=np.asarray([r["true_k"] for r in pooled_rows],int);k23=np.isin(yall,(2,3))
    results={}
    for name in VARIANTS:
        p=np.asarray(pooled[name],float)
        auc=float(roc_auc_score((yall[k23]==2).astype(int),p[k23]))
        best,cands=robust_threshold(pooled_rows,p)
        results[name]={"features":list(VARIANTS[name]),"oof_auc_K2":auc,
                       "p050":counts(pooled_rows,p,.5),
                       "robust_best":best,"threshold_candidates":cands,
                       "per_fold":per_variant[name]["per_fold"]}

    base_best=results["base"]["robust_best"]
    require(base_best is not None and abs(base_best["threshold"]-.64)<1e-12
            and base_best["total"]["net"]==8,"historical robust baseline drift")
    enriched=[(n,q) for n,q in results.items() if n!="base" and q["robust_best"] is not None]
    candidate=None
    if enriched:
        n,q=max(enriched,key=lambda x:(x[1]["robust_best"]["total"]["net"],
                                       x[1]["oof_auc_K2"],
                                       -len(x[1]["features"])))
        if q["robust_best"]["total"]["net"]>base_best["total"]["net"]:
            candidate={"variant":n,"threshold":q["robust_best"]["threshold"],
                       "net":q["robust_best"]["total"]["net"],
                       "auc":q["oof_auc_K2"]}

    rep={"status":"completed","experiment":"v273_residual_feature_family_lofo",
         "baseline_replay_max_abs_error":baseline_replay_max_abs,
         "variants":results,"selected_candidate_if_strictly_better_than_base":candidate,
         "selection_rule":"robust threshold: all folds/players/styles nonnegative; maximize total net then corrections then higher threshold",
         "outer_fold_3_used":False,"annotation_f0_used":False,"prediction_changes":False,
         "automatic_promotion":False}
    a.output.mkdir(parents=True);(a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Residual feature-family LOFO audit","",
           f"Baseline probability replay max abs error: **{baseline_replay_max_abs:.3g}**.","",
           "| variant | features | OOF AUC(K2) | robust th | corr/reg | net | min fold | min player | min style |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n,q in results.items():
        b=q["robust_best"]
        if b is None:
            lines.append(f"| {n} | {len(q['features'])} | {q['oof_auc_K2']:.3f} | none | - | - | - | - | - |")
        else:
            lines.append(f"| {n} | {len(q['features'])} | {q['oof_auc_K2']:.3f} | {b['threshold']:.2f} | {b['total']['corrections']}/{b['total']['regressions']} | {b['total']['net']:+d} | {min(x['net'] for x in b['folds'].values()):+d} | {min(x['net'] for x in b['players'].values()):+d} | {min(x['net'] for x in b['styles'].values()):+d} |")
    lines+=["",f"Strictly better enriched candidate: **{candidate}**.","",
            "No fold3, no annotation F0, no automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
