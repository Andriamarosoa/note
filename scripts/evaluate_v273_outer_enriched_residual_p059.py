"""Exploratory outer fold-3 evaluation of the enriched residual classifier.

Frozen from internal LOFO run 37547761255:
  variant: base_geom_attack
  features: pair/triplet residual + 3 geometry/harmonic + 5 attack/exclusivity
  classifier: StandardScaler + balanced LogisticRegression(C=1, random_state=28431)
  action threshold: p(K2) >= 0.59

Fold 3 is historically exposed. No outer tuning and no automatic promotion.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.guitarset import index_guitarset
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.audit_v273_low_band_fold2_vs_fold4 import inference_features
from scripts.evaluate_v273_outer_residual_two_stage_p060 import (
    load_npz,metrics_simple,effect,action_count,build_robust_model,reconstruct_blow
)
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.v273_window_experiment import load_bundle,require

TH_ENR=.59
TH_BASE60=.60
TH_BASE64=.64
GEOM=("geom_harmonic_relation_min_error","geom_span_cents","amp3_over_amp2")
ATTACK=("nov_onset_contrast_norm_median","exc_unique_attack_fraction_min",
        "joint_unique_x_post_median","weak_unique_fraction","weak_unique_post1_norm")
EXTRA=GEOM+ATTACK

def enriched_classifier():
    return Pipeline([("scale",StandardScaler()),
                     ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                                               class_weight="balanced",random_state=28431))])

def build_extra_map(cache,ids,dataset):
    ids=np.asarray(ids,int)
    wanted={str(cache["members"][i]) for i in ids}
    tracks={t.annotation_member:t for t in index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"audio coverage")
    by=defaultdict(list)
    for rid in ids:by[str(cache["members"][rid])].append(int(rid))
    out={}
    for member in sorted(by):
        au=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        samples=np.asarray(au.samples,float)/32768.
        for rid in sorted(set(by[member])):
            q=inference_features(samples,int(cache["cluster_start_samples"][rid]))
            v=np.asarray([q[k] for k in EXTRA],float)
            require(np.isfinite(v).all(),"nonfinite enriched feature")
            out[rid]=v
        print(json.dumps({"recording":member,"enriched_done":len(out)}),flush=True)
    require(len(out)==len(set(map(int,ids))),"extra map coverage")
    return out

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights","outer-group-predictions",
              "outer-b-low-predictions","dataset","feature-report","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")

    fr=json.loads(a.feature_report.read_text())
    cand=fr["selected_candidate_if_strictly_better_than_base"]
    require(cand is not None and cand["variant"]=="base_geom_attack","candidate drift")
    require(abs(float(cand["threshold"])-TH_ENR)<1e-12 and int(cand["net"])==20,"candidate threshold/net drift")

    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    require(len(outer)==15279 and not np.intersect1d(fit,outer).size,"partition drift")
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]

    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)

    frozen_group=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(frozen_group["global_index"],outer)
    np.testing.assert_array_equal(frozen_group["k"],yo)
    np.testing.assert_array_equal(frozen_group["predicted"],Go)
    np.testing.assert_allclose(frozen_group["probability"],Po,rtol=0,atol=1e-6)

    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,cstats,sstats,bid,lid=reconstruct_blow(Xf,Xo,yf,Gf)
    frozen_b=load_npz(a.outer_b_low_predictions)
    np.testing.assert_array_equal(frozen_b["global_index"],outer)
    np.testing.assert_array_equal(frozen_b["k"],yo)
    np.testing.assert_array_equal(frozen_b["base_predicted"],Go)
    frozen_mask=frozen_b["B_low_mask"].astype(bool)
    require(np.array_equal(frozen_mask,blow_o&np.isin(Go,(2,3,4))),"outer B_low mask mismatch")

    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    idsf=fit[cand_f];idso=outer[cand_o]
    rf=audio_rows(cache,idsf,a.dataset);ro=audio_rows(cache,idso,a.dataset)
    Xrf,_,vf,_=extracted_matrix(rf);Xro,_,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    train=vf&np.isin(yfc,(2,3))
    require(np.any(yfc[train]==2) and np.any(yfc[train]==3),"residual train collapse")

    # Baseline replay for direct comparison.
    base_lr=fit_classifier(Xrf[train],(yfc[train]==2).astype(int))
    pbase=np.full(len(idso),np.nan);pbase[vo]=base_lr.predict_proba(Xro[vo])[:,1]

    # Enriched inference-safe feature matrix, calculated once per valid row ID.
    needed=np.concatenate([idsf[vf],idso[vo]])
    extra=build_extra_map(cache,needed,a.dataset)
    Xef=np.column_stack([Xrf[vf],np.vstack([extra[int(i)] for i in idsf[vf]])])
    Xeo=np.column_stack([Xro[vo],np.vstack([extra[int(i)] for i in idso[vo]])])
    train_valid=np.isin(yfc[vf],(2,3))
    require(int(train_valid.sum())==int(train.sum()),"enriched train alignment")
    enr=enriched_classifier();enr.fit(Xef[train_valid],(yfc[vf][train_valid]==2).astype(int))
    penr=np.full(len(idso),np.nan);penr[vo]=enr.predict_proba(Xeo)[:,1]

    pos={int(g):i for i,g in enumerate(outer)}
    def global_mask(prob,th):
        local=vo&(prob>=th);gm=np.zeros(len(outer),bool)
        for gid,use in zip(idso,local):
            if use:gm[pos[int(gid)]]=True
        return gm

    masks={
      "base_p060":global_mask(pbase,TH_BASE60),
      "base_p064":global_mask(pbase,TH_BASE64),
      "enriched_p059":global_mask(penr,TH_ENR),
    }
    preds={};reports={}
    for name,m in masks.items():
        p=Go.copy();p[m]=2;preds[name]=p
        reports[name]={"actions":action_count(yo,m),"metrics":metrics_simple(yo,p),
                       "effect_vs_base":effect(yo,Go,p)}

    require(reports["base_p060"]["actions"]["corrections"]==7 and
            reports["base_p060"]["actions"]["regressions"]==5,"outer p060 replay drift")
    require(reports["base_p064"]["actions"]["corrections"]==1 and
            reports["base_p064"]["actions"]["regressions"]==1,"outer p064 replay drift")

    base_m=metrics_simple(yo,Go)
    report={"status":"completed","experiment":"v273_outer_enriched_residual_p059",
            "protocol":{"outer_fold":3,"fresh_independent_outer_validation":False,
                        "reason_not_fresh":"fold 3 historically exposed",
                        "outer_tuning":False,"internal_selection_run":37547761255,
                        "variant":"base_geom_attack","threshold":TH_ENR,
                        "classifier":"StandardScaler + balanced LogisticRegression(C=1, random_state=28431)",
                        "annotation_f0_used":False},
            "verification":{"outer_group_prediction_replay_equal":True,
                            "outer_B_low_mask_replay_equal":True,
                            "baseline_p060_replay_equal":True,
                            "baseline_p064_replay_equal":True},
            "internal_fit":{"coarse_clusters":{str(k):v for k,v in cstats.items()},
                            "B_like_cluster":int(bid),"B_subclusters":{str(k):v for k,v in sstats.items()},
                            "B_low_cluster":int(lid),"residual_K23_train_rows":int(train.sum()),
                            "enriched_features":["best_pair_residual_ratio","best_triplet_residual_ratio",*EXTRA]},
            "outer":{"base_metrics":base_m,**reports}}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",global_index=outer,k=yo,base_predicted=Go,
                        base_p060_predicted=preds["base_p060"],
                        base_p064_predicted=preds["base_p064"],
                        enriched_p059_predicted=preds["enriched_p059"],
                        base_p060_action_mask=masks["base_p060"].astype(np.uint8),
                        base_p064_action_mask=masks["base_p064"].astype(np.uint8),
                        enriched_p059_action_mask=masks["enriched_p059"].astype(np.uint8),
                        base_candidate_probability=pbase,enriched_candidate_probability=penr)
    def pct(x):return 100*x
    e=reports["enriched_p059"];q60=reports["base_p060"];q64=reports["base_p064"]
    lines=["# Exploratory outer fold 3 — enriched residual p0.59","",
           "**Fold 3 is historically exposed; not fresh independent validation. No outer tuning.**","",
           "| Measure | Robust base | base p>=.60 | base p>=.64 | enriched p>=.59 |",
           "|---|---:|---:|---:|---:|",
           f"| Exact-K global | {pct(base_m['exact']):.3f}% | {pct(q60['metrics']['exact']):.3f}% | {pct(q64['metrics']['exact']):.3f}% | {pct(e['metrics']['exact']):.3f}% |",
           f"| Exact-K poly | {pct(base_m['poly_exact']):.3f}% | {pct(q60['metrics']['poly_exact']):.3f}% | {pct(q64['metrics']['poly_exact']):.3f}% | {pct(e['metrics']['poly_exact']):.3f}% |","",
           f"base p>=.60: **{q60['actions']['corrections']}/{q60['actions']['regressions']}**, net **{q60['actions']['net']:+d}**.",
           f"base p>=.64: **{q64['actions']['corrections']}/{q64['actions']['regressions']}**, net **{q64['actions']['net']:+d}**.",
           f"enriched p>=.59: **{e['actions']['corrections']}/{e['actions']['regressions']}**, other-K {e['actions']['other_k']}, net **{e['actions']['net']:+d}**.",
           f"Enriched paired effect vs robust base: **{e['effect_vs_base']['net']:+d}**.","",
           "No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
