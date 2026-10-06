"""Exploratory outer fold-3 evaluation of the internally selected residual threshold 0.64.

Threshold 0.64 was selected in run 37546108223 using internal OOF constraints:
all folds, all GuitarSet players and all styles non-negative; maximize total net.

This script rebuilds the same robust base, pooled-internal B_low assignment and
pair/triplet residual classifier, then evaluates frozen thresholds 0.60 and 0.64
on historically exposed fold 3. No outer tuning.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

from scripts.evaluate_v273_outer_residual_two_stage_p060 import (
    load_npz,metrics_simple,effect,action_count,build_robust_model,reconstruct_blow
)
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import audio_rows
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.v273_window_experiment import load_bundle,require

TH60=.60
TH64=.64

def main():
    ap=argparse.ArgumentParser()
    for n in ("bundle","config","uniform-weights","freeze-weights","outer-group-predictions",
              "outer-b-low-predictions","dataset","threshold-report","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    sel=json.loads(a.threshold_report.read_text())
    require(sel["best"] is not None,"no robust threshold selected")
    require(abs(float(sel["best"]["threshold"])-TH64)<1e-12,"selected threshold drift")

    cache,parts,_=load_bundle(a.bundle,a.config)
    fit=np.asarray(parts["final_fit"],np.int64);outer=np.asarray(parts["outer"],np.int64)
    yall=np.minimum(cache["exact"].astype(np.int32),6);yf=yall[fit];yo=yall[outer]

    model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(model,cache,fit);Po,Go=predict(model,cache,outer)

    fg=load_npz(a.outer_group_predictions)
    np.testing.assert_array_equal(fg["global_index"],outer)
    np.testing.assert_array_equal(fg["predicted"],Go)

    Xf=structural_features(cache,fit,Pf);Xo=structural_features(cache,outer,Po)
    blow_f,blow_o,cstats,sstats,bid,lid=reconstruct_blow(Xf,Xo,yf,Gf)
    fb=load_npz(a.outer_b_low_predictions)
    frozen_mask=fb["B_low_mask"].astype(bool)
    require(np.array_equal(frozen_mask,blow_o&np.isin(Go,(2,3,4))),"B_low mismatch")

    cand_f=blow_f&(Gf==3);cand_o=blow_o&(Go==3)
    rf=audio_rows(cache,fit[cand_f],a.dataset);ro=audio_rows(cache,outer[cand_o],a.dataset)
    Xrf,_,vf,_=extracted_matrix(rf);Xro,_,vo,_=extracted_matrix(ro)
    yfc=yf[cand_f];yoc=yo[cand_o]
    train=vf&np.isin(yfc,(2,3))
    require(np.any(yfc[train]==2) and np.any(yfc[train]==3),"train collapse")
    residual=fit_classifier(Xrf[train],(yfc[train]==2).astype(int))
    p=np.full(len(Xro),np.nan);p[vo]=residual.predict_proba(Xro[vo])[:,1]

    pos={int(g):i for i,g in enumerate(outer)}
    masks={}
    preds={}
    reports={}
    for name,th in (("p060",TH60),("p064",TH64)):
        local=vo&(p>=th)
        gm=np.zeros(len(outer),bool)
        for gid,use in zip(outer[cand_o],local):
            if use:gm[pos[int(gid)]]=True
        pred=Go.copy();pred[gm]=2
        masks[name]=gm;preds[name]=pred
        reports[name]={"threshold":th,"actions":action_count(yo,gm),
                       "metrics":metrics_simple(yo,pred),"effect_vs_base":effect(yo,Go,pred)}

    base=metrics_simple(yo,Go)
    report={"status":"completed","experiment":"v273_outer_residual_p064",
            "protocol":{"outer_fold":3,"fresh_independent_outer_validation":False,
                        "reason_not_fresh":"fold 3 historically exposed",
                        "outer_tuning":False,"selected_threshold_run":37546108223,
                        "selected_threshold":TH64,
                        "selection_constraints":"all internal folds/players/styles net >=0; maximize total net"},
            "verification":{"outer_group_prediction_replay_equal":True,"outer_B_low_mask_replay_equal":True},
            "internal_fit":{"coarse_clusters":{str(k):v for k,v in cstats.items()},
                            "B_like_cluster":int(bid),"B_subclusters":{str(k):v for k,v in sstats.items()},
                            "B_low_cluster":int(lid),"residual_K23_train_rows":int(train.sum())},
            "outer":{"base_metrics":base,**reports}}
    a.output.mkdir(parents=True)
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"predictions.npz",global_index=outer,k=yo,base_predicted=Go,
                        p060_predicted=preds["p060"],p064_predicted=preds["p064"],
                        p060_action_mask=masks["p060"].astype(np.uint8),
                        p064_action_mask=masks["p064"].astype(np.uint8),
                        candidate_probability=p)
    def pct(x):return 100*x
    q60=reports["p060"];q64=reports["p064"]
    lines=["# Exploratory outer fold 3 — residual threshold p0.64","",
           "**Fold 3 is historically exposed; not fresh independent validation. No outer tuning.**","",
           "| Measure | Robust base | p>=0.60 | p>=0.64 |","|---|---:|---:|---:|",
           f"| Exact-K global | {pct(base['exact']):.3f}% | {pct(q60['metrics']['exact']):.3f}% | {pct(q64['metrics']['exact']):.3f}% |",
           f"| Exact-K poly | {pct(base['poly_exact']):.3f}% | {pct(q60['metrics']['poly_exact']):.3f}% | {pct(q64['metrics']['poly_exact']):.3f}% |","",
           f"p>=.60 actions: **{q60['actions']['corrections']}/{q60['actions']['regressions']}**, net **{q60['actions']['net']:+d}**.",
           f"p>=.64 actions: **{q64['actions']['corrections']}/{q64['actions']['regressions']}**, net **{q64['actions']['net']:+d}**.",
           f"p>=.64 paired effect vs robust base: **{q64['effect_vs_base']['net']:+d}**.","",
           "No automatic promotion."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n");print("\n".join(lines))
if __name__=="__main__":main()
