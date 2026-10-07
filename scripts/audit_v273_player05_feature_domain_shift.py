"""Diagnose feature-domain shift behind fresh player05 failure.

No retuning. Compares the frozen enriched residual action populations:
- internal OOF selected actions from folds 0/1/2/4 at p(K2)>=0.59;
- fresh player05 selected actions from run 37552019921.

For each of the 10 inference-safe features, report:
- internal K2 vs K3 selected-action medians and AUC orientation,
- player05 K2 vs K3 selected-action medians and same frozen orientation,
- within-class shift player05 vs internal,
- whether the K2-vs-K3 direction flips on player05.

The purpose is diagnostic only. Player05 is development data after this audit.
"""
from __future__ import annotations
import argparse,json,math
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
from scripts.audit_v273_internal_b_low_harmonic_strata import transition_spectrum,extract_one
from scripts.v273_window_experiment import load_bundle

FOLDS=(0,1,2,4)
TH=.59
BASE=("best_pair_residual_ratio","best_triplet_residual_ratio")
EXTRA=(
  "geom_harmonic_relation_min_error","geom_span_cents","amp3_over_amp2",
  "nov_onset_contrast_norm_median","exc_unique_attack_fraction_min",
  "joint_unique_x_post_median","weak_unique_fraction","weak_unique_post1_norm",
)
FEATURES=BASE+EXTRA

def require(c,m):
    if not c: raise RuntimeError(m)

def clf():
    return Pipeline([("scale",StandardScaler()),
      ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
      class_weight="balanced",random_state=28431))])

def load_internal_split(root,fold,split):
    p=Path(root)/f"fold-{fold}"/"replay.npz"
    with np.load(p,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
    action=a[split+"_b_low"].astype(bool)&(a[split+"_base_k"].astype(int)==3)
    valid=a[split+"_valid"].astype(bool)
    ids=a[split+"_action_ids"][valid].astype(int)
    y=a[split+"_true_k"].astype(int)[action][valid]
    X=a[split+"_X"].astype(float)[valid]
    rec=a[split+"_recording"].astype(str)[action][valid]
    require(len(ids)==len(y)==len(X)==len(rec),"internal split drift")
    return [{"row_id":int(ids[i]),"fold":int(fold),"true_k":int(y[i]),
             "recording_id":str(rec[i]),
             "best_pair_residual_ratio":float(X[i,0]),
             "best_triplet_residual_ratio":float(X[i,1])} for i in range(len(y))]

def internal_extras(rows,bundle,config,dataset):
    cache,_,_=load_bundle(bundle,config)
    ids=sorted({r["row_id"] for r in rows})
    wanted={str(cache["members"][i]) for i in ids}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"internal audio coverage")
    by=defaultdict(list)
    for rid in ids:by[str(cache["members"][rid])].append(rid)
    out={}
    for member in sorted(by):
        wav=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        s=np.asarray(wav.samples,float)/32768.
        for rid in by[member]:
            q=inference_features(s,int(cache["cluster_start_samples"][rid]))
            out[rid]={k:float(q[k]) for k in EXTRA}
    return out

def xmat(rows,extra):
    return np.asarray([[r[BASE[0]],r[BASE[1]],*[extra[r["row_id"]][k] for k in EXTRA]] for r in rows],float)

def build_internal_actions(root,bundle,config,dataset):
    folds={}
    all_rows=[]
    for f in FOLDS:
        fit=load_internal_split(root,f,"fit");val=load_internal_split(root,f,"val")
        folds[f]=(fit,val);all_rows+=fit+val
    ex=internal_extras(all_rows,bundle,config,dataset)
    selected=[]
    for f in FOLDS:
        fit,val=folds[f]
        tr=np.asarray([r["true_k"] in (2,3) for r in fit],bool)
        y=np.asarray([r["true_k"]==2 for r in fit],int)[tr]
        m=clf();m.fit(xmat(fit,ex)[tr],y)
        p=m.predict_proba(xmat(val,ex))[:,1]
        for r,pp in zip(val,p):
            if pp>=TH:
                q=dict(r);q["probability"]=float(pp);q.update(ex[r["row_id"]]);selected.append(q)
    require(sum(r["true_k"]==2 for r in selected)==53,"internal K2 selected drift")
    require(sum(r["true_k"]==3 for r in selected)==33,"internal K3 selected drift")
    return selected

def player05_tracks(dataset):
    import causal_note.guitarset as gs
    old=gs.ALLOWED_PLAYERS;gs.ALLOWED_PLAYERS=frozenset(("05",))
    try:t=gs.index_guitarset(dataset)
    finally:gs.ALLOWED_PLAYERS=old
    require(len(t)==60,"player05 track count drift")
    return t

def build_p05_actions(pred_path,score_path,dataset):
    with np.load(pred_path,allow_pickle=False) as z:p={k:np.asarray(z[k]) for k in z.files}
    with np.load(score_path,allow_pickle=False) as z:s={k:np.asarray(z[k]) for k in z.files}
    members=p["members"].astype(str);starts=p["cluster_start_samples"].astype(np.int64)
    ids=p["residual_candidate_ids"].astype(np.int64)
    probs=p["enriched_candidate_probability"].astype(float)
    mask=s["enriched_p059_action_mask"].astype(bool)
    y=s["true_k"].astype(int)
    require(len(mask)==len(y)==len(members)==len(starts),"p05 arrays drift")
    action_ids=np.flatnonzero(mask)
    require(set(map(int,action_ids))<=set(map(int,ids)),"p05 candidate/action mismatch")
    prob_by={int(i):float(pp) for i,pp in zip(ids,probs)}
    tracks={t.annotation_member:t for t in player05_tracks(dataset)}
    by=defaultdict(list)
    for rid in action_ids:by[members[rid]].append(int(rid))
    rows=[]
    for member in sorted(by):
        t=tracks[member]
        wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member);audio=np.asarray(wav.samples,float)/32768.
        for rid in by[member]:
            freq,x=transition_spectrum(audio,int(starts[rid]))
            d=extract_one(freq,x)
            require(d is not None,"p05 residual extraction missing")
            q=inference_features(audio,int(starts[rid]))
            row={"row_id":rid,"true_k":int(y[rid]),"recording_id":member,
                 "probability":prob_by[rid],
                 "best_pair_residual_ratio":float(d["best_pair_residual_ratio"]),
                 "best_triplet_residual_ratio":float(d["best_triplet_residual_ratio"])}
            row.update({k:float(q[k]) for k in EXTRA});rows.append(row)
    require(sum(r["true_k"]==2 for r in rows)==6,"p05 K2 action drift")
    require(sum(r["true_k"]==3 for r in rows)==21,"p05 K3 action drift")
    return rows

def med(rows,k,cls):
    z=np.asarray([r[k] for r in rows if r["true_k"]==cls],float)
    return float(np.median(z))

def auc(rows,k,sign):
    rr=[r for r in rows if r["true_k"] in (2,3)]
    y=np.asarray([r["true_k"]==2 for r in rr],int)
    x=np.asarray([r[k] for r in rr],float)*sign
    return float(roc_auc_score(y,x))

def main():
    ap=argparse.ArgumentParser()
    for n in ("internal-exports","bundle","config","dataset","player05-predictions","player05-scored","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    internal=build_internal_actions(a.internal_exports,a.bundle,a.config,a.dataset)
    p05=build_p05_actions(a.player05_predictions,a.player05_scored,a.dataset)

    stats=[]
    for k in FEATURES:
        i2,i3=med(internal,k,2),med(internal,k,3)
        sign=1.0 if i2>=i3 else -1.0
        p2,p3=med(p05,k,2),med(p05,k,3)
        ia=auc(internal,k,sign);pa=auc(p05,k,sign)
        direction_internal=np.sign(i2-i3)
        direction_p05=np.sign(p2-p3)
        def rel_shift(new,old):
            scale=max(abs(old),1e-9);return float((new-old)/scale)
        stats.append({
          "feature":k,"orientation_for_K2_from_internal":"high" if sign>0 else "low",
          "internal":{"K2_median":i2,"K3_median":i3,"oriented_auc_K2":ia},
          "player05":{"K2_median":p2,"K3_median":p3,"oriented_auc_K2":pa},
          "direction_flip":bool(direction_internal!=0 and direction_p05!=0 and direction_internal!=direction_p05),
          "K2_relative_shift_p05_vs_internal":rel_shift(p2,i2),
          "K3_relative_shift_p05_vs_internal":rel_shift(p3,i3),
          "auc_delta_p05_minus_internal":pa-ia,
        })
    stats.sort(key=lambda q:(q["direction_flip"],-q["auc_delta_p05_minus_internal"]),reverse=True)
    flips=[q["feature"] for q in stats if q["direction_flip"]]
    rep={"status":"completed","experiment":"v273_player05_feature_domain_shift",
         "populations":{"internal_selected":{"K2":53,"K3":33},
                        "player05_selected":{"K2":6,"K3":21}},
         "features":stats,"direction_flips":flips,
         "diagnostic_only":True,
         "player05_independence_consumed":True,
         "retuning_on_player05_permitted_for_future_development_but_not_independent_validation":True}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(rep,indent=2,sort_keys=True)+"\n")
    lines=["# Player05 enriched-residual feature domain shift","",
           "Diagnostic only; player05 is no longer an independent validation set after this audit.","",
           "| feature | internal K2/K3 med | p05 K2/K3 med | internal AUC | p05 AUC | direction flip |",
           "|---|---:|---:|---:|---:|---|"]
    for q in stats:
        i=q["internal"];p=q["player05"]
        lines.append(f"| {q['feature']} | {i['K2_median']:.4g}/{i['K3_median']:.4g} | {p['K2_median']:.4g}/{p['K3_median']:.4g} | {i['oriented_auc_K2']:.3f} | {p['oriented_auc_K2']:.3f} | {q['direction_flip']} |")
    lines+=["",f"Direction flips: **{flips}**."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
if __name__=="__main__":main()
