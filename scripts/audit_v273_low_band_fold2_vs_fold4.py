"""Describe low-register residual actions in probability band [0.60, 0.65).

Focus: why fold2 band is 2 K2 / 3 K3 while fold4 is 3 K2 / 0 K3.

No model fitting, no threshold search, no prediction changes.
All added audio features are inference-safe and reconstructed from the bundle +
raw PCM, using the same residual decomposition code as the main audits.
"""
from __future__ import annotations
import argparse,json,re,math
from collections import defaultdict
from pathlib import Path
import numpy as np

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_attack_novelty import raw_powers
from scripts.audit_v273_selected_f0_attack_exclusive import row_features as selected_f0_features
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    transition_spectrum,f0_pool,template,fit_best,harmonic_relation_distance
)
from scripts.v273_window_experiment import load_bundle
from scripts.v273_residual_audit import require

P0=.60
P1=.65
EPS=1e-12

def meta(member):
    stem=Path(member).stem
    m=re.match(r"^(\d{2})_([A-Za-z]+)(\d+)-(.+?)_(comp|solo)$",stem)
    if not m:return {"player":"?","style":"?","part":"?"}
    player,style,variant,rest,part=m.groups()
    return {"player":player,"style":style,"part":part,"variant":style+variant}

def load_band(root,fold):
    p=Path(root)/f"fold-{fold}"/"replay.npz"
    with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
    action=a["val_b_low"].astype(bool)&(a["val_base_k"].astype(int)==3)
    valid=a["val_valid"].astype(bool)
    ids=a["val_action_ids"][valid].astype(int)
    y=a["val_true_k"].astype(int)[action][valid]
    pr=a["val_probability"].astype(float)
    X=a["val_X"].astype(float)[valid]
    f0=a["val_f0"].astype(float)[valid]
    reg=np.asarray(a["val_register"]).astype(str)
    rec=a["val_recording"].astype(str)[action][valid]
    require(len(ids)==len(y)==len(pr)==len(X)==len(f0)==len(reg)==len(rec),"length drift")
    rows=[]
    for i in range(len(y)):
        if reg[i]!="low" or not (P0<=pr[i]<P1):continue
        rows.append({"row_id":int(ids[i]),"fold":fold,"true_k":int(y[i]),"probability":float(pr[i]),
                     "pair":float(X[i,0]),"triplet":float(X[i,1]),
                     "pair_minus_triplet":float(X[i,0]-X[i,1]),
                     "median_triplet_f0":float(f0[i]),"recording_id":str(rec[i]),**meta(str(rec[i]))})
    return rows

def cents(a,b):
    return abs(1200.0*math.log2((float(a)+EPS)/(float(b)+EPS)))

def inference_features(samples,start):
    tfreq,x=transition_spectrum(samples,start)
    pool=f0_pool(tfreq,x)
    require(len(pool)>=3,"small f0 pool")
    D=np.column_stack([template(tfreq,f) for f in pool])
    trip,_,_,_=fit_best(D,x,3)
    _,combo,amps=trip
    f0=np.asarray([pool[i] for i in combo],float)
    amps=np.asarray(amps,float)
    require(len(f0)==3 and len(amps)==3,"triplet drift")

    freq,powers,_=raw_powers(samples,start)
    sf=selected_f0_features({"decomposition":{"triplet_f0":f0.tolist()}},freq,powers)

    pair_c=[cents(f0[i],f0[j]) for i in range(3) for j in range(i+1,3)]
    rel=[harmonic_relation_distance(f0[i],f0[j]) for i in range(3) for j in range(i+1,3)]
    sa=np.sort(amps)[::-1]
    out={
      "amp_range":float(np.ptp(amps)),
      "amp_cv":float(amps.std()/(abs(amps.mean())+EPS)),
      "amp3_over_amp1":float(sa[2]/(sa[0]+EPS)),
      "amp3_over_amp2":float(sa[2]/(sa[1]+EPS)),
      "geom_min_pair_cents":float(min(pair_c)),
      "geom_span_cents":float(max(pair_c)),
      "geom_harmonic_relation_min_error":float(min(rel)),
      "geom_harmonic_relation_median_error":float(np.median(rel)),
    }
    for k in (
      "nov_onset_contrast_norm_median","nov_pre_norm_median","nov_post1_norm_median",
      "nov_positive_retained_fraction_median",
      "exc_unique_attack_fraction_median","exc_unique_attack_fraction_min",
      "exc_shared_attack_fraction_median","exc_shared_attack_fraction_range",
      "exc_attack_energy_total_median",
      "joint_unique_x_post_median","joint_unique_x_contrast_median",
      "weak_unique_fraction","weak_unique_contrast","weak_unique_post1_norm",
    ):
        out[k]=float(sf[k])
    return out

def summarize(rows,key,selector):
    z=np.asarray([r[key] for r in rows if selector(r)],float)
    return None if len(z)==0 else {"n":int(len(z)),"mean":float(np.mean(z)),"median":float(np.median(z)),
                                  "min":float(np.min(z)),"max":float(np.max(z))}

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","dataset","output"):ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    rows=load_band(a.internal_exports,2)+load_band(a.internal_exports,4)
    require(sum(r["fold"]==2 and r["true_k"]==2 for r in rows)==2,"fold2 K2 drift")
    require(sum(r["fold"]==2 and r["true_k"]==3 for r in rows)==3,"fold2 K3 drift")
    require(sum(r["fold"]==4 and r["true_k"]==2 for r in rows)==3,"fold4 K2 drift")
    require(sum(r["fold"]==4 and r["true_k"]==3 for r in rows)==0,"fold4 K3 drift")

    cache,_,_=load_bundle(a.bundle,a.config)
    require(all(0<=r["row_id"]<len(cache["exact"]) for r in rows),"bundle row id drift")
    for r in rows:
        require(str(cache["members"][r["row_id"]])==r["recording_id"],"recording mismatch")

    wanted={r["recording_id"] for r in rows}
    tracks={t.annotation_member:t for t in index_guitarset(a.dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for r in rows:by[r["recording_id"]].append(r)

    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for r in by[member]:
            start=int(cache["cluster_start_samples"][r["row_id"]])
            r["start_sample"]=start
            r.update(inference_features(samples,start))
        print(json.dumps({"recording":member,"rows":len(by[member])}),flush=True)

    groups={
      "fold2_K2":lambda r:r["fold"]==2 and r["true_k"]==2,
      "fold2_K3":lambda r:r["fold"]==2 and r["true_k"]==3,
      "fold4_K2":lambda r:r["fold"]==4 and r["true_k"]==2,
    }
    keys=[k for k,v in rows[0].items() if isinstance(v,(int,float)) and k not in ("row_id","fold","true_k","start_sample")]
    summary={g:{k:summarize(rows,k,sel) for k in keys} for g,sel in groups.items()}

    separators=[]
    for k in keys:
        a2=summary["fold2_K2"][k];a3=summary["fold2_K3"][k];a4=summary["fold4_K2"][k]
        if not a2 or not a3 or not a4:continue
        m2=a2["median"];m3=a3["median"];m4=a4["median"]
        lo=min(m2,m4);hi=max(m2,m4)
        outside=m3<lo or m3>hi
        margin=min(abs(m3-m2),abs(m3-m4))
        scale=max(abs(m2),abs(m3),abs(m4),1e-9)
        separators.append({"feature":k,"fold2_K2_median":m2,"fold2_K3_median":m3,
                           "fold4_K2_median":m4,"K3_outside_both_K2_medians":outside,
                           "normalized_margin":float(margin/scale)})
    separators.sort(key=lambda q:(q["K3_outside_both_K2_medians"],q["normalized_margin"]),reverse=True)

    rep={"status":"completed","experiment":"v273_low_band_fold2_vs_fold4",
         "population":{"fold2_K2":2,"fold2_K3":3,"fold4_K2":3},
         "rows":rows,"summary":summary,"descriptive_separators":separators,
         "warning":"Tiny descriptive cohort; do not fit or promote a rule from these 8 rows.",
         "prediction_changes":False,"threshold_search":False,"outer_fold_3_used":False}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Low-band fold2 vs fold4 audit","",
           "Cohort: fold2 = 2 K2 / 3 K3; fold4 = 3 K2 / 0 K3. Descriptive only.","",
           "## Exact rows","","| fold | K | p | pair | triplet | f0 | player | style | part |",
           "|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for r in sorted(rows,key=lambda z:(z["fold"],z["probability"])):
        lines.append(f"| {r['fold']} | {r['true_k']} | {r['probability']:.4f} | {r['pair']:.4f} | {r['triplet']:.4f} | {r['median_triplet_f0']:.2f} | {r['player']} | {r['style']} | {r['part']} |")
    lines+=["","## Largest descriptive differences","","| feature | f2 K2 med | f2 K3 med | f4 K2 med | K3 outside both K2? |",
            "|---|---:|---:|---:|---|"]
    for q in separators[:18]:
        lines.append(f"| {q['feature']} | {q['fold2_K2_median']:.6g} | {q['fold2_K3_median']:.6g} | {q['fold4_K2_median']:.6g} | {q['K3_outside_both_K2_medians']} |")
    lines+=["","No rule selected from this tiny cohort."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
