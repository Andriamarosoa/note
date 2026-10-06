"""Inference-safe proxy audit for correction-vs-regression signatures.

Uses only frozen residual decomposition triplet F0/amplitudes, source scalars, and
raw PCM. Annotation-derived features are forbidden. Fixed feature families proxy
the descriptive signatures found in run 37529351802.

No threshold tuning, no Exact-K prediction changes, no fold 3.
"""
from __future__ import annotations
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_attack_novelty import raw_powers
from scripts.audit_v273_selected_f0_attack_exclusive import row_features as selected_f0_features
from scripts.v273_residual_audit import FOLDS,require

GROUPS=("K2_corrected","K3_regressed")
EPS=1e-12

def cents(a,b): return abs(1200.0*math.log2((a+EPS)/(b+EPS)))

def harmonic_relation_error_cents(a,b):
    # Symmetric closeness to ratios n:m for small guitar-relevant partial orders.
    best=1e9
    for n in range(1,7):
        for m in range(1,7):
            target=n/m
            e=abs(1200.0*math.log2(((a+EPS)/(b+EPS))/target))
            best=min(best,e)
    return best

def base_geometry(row):
    d=row["decomposition"]
    f0=np.asarray(d["triplet_f0"],float); amp=np.asarray(d["triplet_amplitudes"],float)
    require(len(f0)==3 and len(amp)==3,"triplet drift")
    pair_c=[cents(f0[i],f0[j]) for i in range(3) for j in range(i+1,3)]
    rel=[harmonic_relation_error_cents(f0[i],f0[j]) for i in range(3) for j in range(i+1,3)]
    s=np.sort(amp)[::-1]
    return {
      "geom_min_pair_cents":float(min(pair_c)),
      "geom_span_cents":float(max(pair_c)),
      "geom_harmonic_relation_min_error":float(min(rel)),
      "geom_harmonic_relation_median_error":float(np.median(rel)),
      "amp_range":float(np.ptp(amp)),
      "amp_cv":float(amp.std()/(abs(amp.mean())+EPS)),
      "amp3_over_amp1":float(s[2]/(s[0]+EPS)),
      "amp3_over_amp2":float(s[2]/(s[1]+EPS)),
      "candidate_span_ms":float(row.get("candidate_span_ms",0.0)),
      "pair_minus_triplet":float(row["best_pair_residual_ratio"]-row["best_triplet_residual_ratio"]),
      "probability_K2_weighted":float(row.get("probability_K2_weighted",0.0)),
    }

def safe_features(row,samples):
    freq,powers,_=raw_powers(samples,int(row["start_sample"]))
    sf=selected_f0_features(row,freq,powers)
    f=base_geometry(row)
    keep=(
      "nov_onset_contrast_norm_median","nov_onset_contrast_norm_min","nov_post1_norm_median",
      "nov_pre_norm_median","nov_positive_retained_fraction_median",
      "exc_unique_attack_fraction_median","exc_unique_attack_fraction_min",
      "exc_shared_attack_fraction_median","exc_attack_energy_total_median",
      "joint_unique_x_post_median","joint_unique_x_contrast_median",
      "weak_unique_fraction","weak_unique_contrast","weak_unique_post1_norm",
    )
    for k in keep: f[k]=float(sf[k])
    return f

FEATURE_SETS={
 "onset":("nov_onset_contrast_norm_median","nov_post1_norm_median","nov_pre_norm_median"),
 "geometry":("geom_min_pair_cents","geom_span_cents","geom_harmonic_relation_min_error",
             "amp_range","amp_cv","amp3_over_amp1","candidate_span_ms"),
 "exclusive":("exc_unique_attack_fraction_median","exc_unique_attack_fraction_min",
              "exc_shared_attack_fraction_median","exc_attack_energy_total_median"),
 "joint":("nov_onset_contrast_norm_median","nov_post1_norm_median",
          "exc_unique_attack_fraction_median","exc_shared_attack_fraction_median",
          "geom_min_pair_cents","geom_span_cents","geom_harmonic_relation_min_error",
          "amp_range","amp_cv","amp3_over_amp1","candidate_span_ms"),
 "joint_plus_source":("nov_onset_contrast_norm_median","nov_post1_norm_median",
          "exc_unique_attack_fraction_median","exc_shared_attack_fraction_median",
          "geom_min_pair_cents","geom_span_cents","geom_harmonic_relation_min_error",
          "amp_range","amp_cv","amp3_over_amp1","candidate_span_ms",
          "pair_minus_triplet","probability_K2_weighted"),
}

def eval_set(rows,names):
    X=np.asarray([[r["features"][n] for n in names] for r in rows],float)
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int)
    folds=np.asarray([r["fold"] for r in rows],int)
    p=np.zeros(len(y)); per=[]
    for f in FOLDS:
        fit=folds!=f; val=folds==f
        m=make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=4000,class_weight="balanced"))
        m.fit(X[fit],y[fit]); p[val]=m.predict_proba(X[val])[:,1]
        per.append({"fold":int(f),"auc":float(roc_auc_score(y[val],p[val]))})
    return {"auc":float(roc_auc_score(y,p)),"per_fold":per,
            "mean_fold_auc":float(np.mean([q["auc"] for q in per])),
            "min_fold_auc":float(np.min([q["auc"] for q in per]))}

def main():
    ap=argparse.ArgumentParser()
    for n in ("cases","dataset","output"): ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args(); require(not a.output.exists(),"refusing overwrite")
    cases=[json.loads(x) for x in a.cases.read_text().splitlines()]
    cases=[r for r in cases if r["group"] in GROUPS]
    require(sum(r["group"]=="K2_corrected" for r in cases)==108,"K2 drift")
    require(sum(r["group"]=="K3_regressed" for r in cases)==125,"K3 drift")
    require(all(r["fold"] in FOLDS and r["fold"]!=3 for r in cases),"fold leak")
    wanted={r["recording_id"] for r in cases}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in cases: by[r["recording_id"]].append(r)
    rows=[]
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in sorted(by[member],key=lambda z:z["row_id"]):
            rows.append({"row_id":r["row_id"],"fold":r["fold"],"group":r["group"],
                         "features":safe_features(r,samples)})
        print(json.dumps({"recording":member,"done":len(rows)}),flush=True)
    results={k:eval_set(rows,v) for k,v in FEATURE_SETS.items()}
    # single-feature direction consistency
    y=np.asarray([r["group"]=="K3_regressed" for r in rows],int); folds=np.asarray([r["fold"] for r in rows],int)
    uni=[]
    for n in rows[0]["features"]:
        x=np.asarray([r["features"][n] for r in rows],float); raw=float(roc_auc_score(y,x)); orient=1 if raw>=.5 else -1
        per=[]
        for f in FOLDS:
            m=folds==f; fold_auc=float(roc_auc_score(y[m],x[m])); per.append(fold_auc if orient==1 else 1-fold_auc)
        uni.append({"feature":n,"mean_auc":float(np.mean(per)),"min_auc":float(np.min(per)),
                    "global_oriented_auc":raw if orient==1 else 1-raw,
                    "direction":"higher_in_regressions" if orient==1 else "lower_in_regressions"})
    uni.sort(key=lambda q:(q["min_auc"],q["mean_auc"]),reverse=True)
    rep={"status":"completed","experiment":"v273_inference_safe_signature_proxy",
         "cases":233,"outer_fold_3_used":False,"prediction_changes":False,"threshold_tuning":False,
         "feature_sets":FEATURE_SETS,"results":results,"univariate":uni,
         "progression_criterion":{"mean_fold_auc":0.65,"min_fold_auc":0.55},
         "passes":{k:(v["mean_fold_auc"]>=.65 and v["min_fold_auc"]>=.55) for k,v in results.items()}}
    a.output.mkdir(parents=True); (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Inference-safe correction/regression signature proxy","",
           "Only selected residual F0s, frozen source scalars and raw PCM are used. No annotations. Fold 3 excluded.","",
           "| feature set | global AUC | mean fold AUC | min fold AUC | f0 | f1 | f2 | f4 | pass |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for n,q in results.items():
        d={z["fold"]:z["auc"] for z in q["per_fold"]}
        ok=q["mean_fold_auc"]>=.65 and q["min_fold_auc"]>=.55
        lines.append(f"| {n} | {q['auc']:.3f} | {q['mean_fold_auc']:.3f} | {q['min_fold_auc']:.3f} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {d[4]:.3f} | {ok} |")
    lines+=["","## Best single inference-safe proxies","","| feature | direction in regressions | mean AUC | min fold AUC |","|---|---|---:|---:|"]
    for q in uni[:12]: lines.append(f"| {q['feature']} | {q['direction']} | {q['mean_auc']:.3f} | {q['min_auc']:.3f} |")
    lines+=["","No Exact-K output changed."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n"); print("\n".join(lines))
if __name__=="__main__": main()
