"""One-shot fresh unseen-performer validation on GuitarSet player 05.

The frozen proposal stack, robust count model, B_low construction, enriched
residual feature family and threshold were all selected before this run.

Important leakage guard:
- player 05 runtime clusters/features and all predictions are built first;
- predictions are saved and SHA-256 hashed;
- only then are player-05 annotations opened to assign true K and score.

This is performer-independent but not composition-independent: player 05 plays
the same GuitarSet composition inventory as players 00..04.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import causal_note.guitarset as gs
from scripts import candidate_timing as timing
from scripts.train_boundaries import decode_pcm16_mono_wav
from scripts.evaluate_v8_boundaries import _arrangement
from scripts.train_v86_state_transition_proposals import (
    _predict_score_tracks,_candidate_groups,_spectral_transition,_score_context,_records,HORIZONS
)
from scripts.train_v87_causal_candidate_memory import _encode_records,_sequence_arrays
from scripts.train_v88_regime_moe import _encode_v87,_feature_matrix,_predictions
from scripts.train_v91_ordinal_cardinality import _load_frozen_stack
from scripts.train_v90_structured_cluster_cardinality import (
    _extended_candidate_features,_cluster_arrays,MAX_CANDIDATES,CLUSTER_WINDOW_MS
)
from scripts.evaluate_v90_cardinality_realization import _clusters_window
from scripts.evaluate_v90_cluster_oracles import _references,_assign_refs
from scripts.train_v100_spectral_string_slots import _spectral_maps_for_runtime
from scripts.spectral_window import COVERED
from scripts.v273_window_experiment import load_bundle
from scripts.evaluate_v273_outer_residual_two_stage_p060 import (
    build_robust_model,reconstruct_blow,metrics_simple,effect,action_count
)
from scripts.audit_v273_internal_b_like_boundary_corrector import predict,structural_features
from scripts.audit_v273_internal_b_low_harmonic_strata import (
    audio_rows,transition_spectrum,extract_one
)
from scripts.v273_residual_audit import extracted_matrix,fit_classifier
from scripts.audit_v273_low_band_fold2_vs_fold4 import inference_features

PLAYER="05"
TH_ENR=.59
TH_BASE=.60
EXTRA=(
  "geom_harmonic_relation_min_error","geom_span_cents","amp3_over_amp2",
  "nov_onset_contrast_norm_median","exc_unique_attack_fraction_min",
  "joint_unique_x_post_median","weak_unique_fraction","weak_unique_post1_norm",
)

def require(c,m):
    if not c: raise RuntimeError(m)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def index_player05(dataset):
    old=gs.ALLOWED_PLAYERS
    gs.ALLOWED_PLAYERS=frozenset((PLAYER,))
    try:
        tracks=gs.index_guitarset(dataset)
    finally:
        gs.ALLOWED_PLAYERS=old
    require(len(tracks)==60,f"expected 60 player05 tracks, got {len(tracks)}")
    require({t.player_id for t in tracks}=={PLAYER},"player05 scope drift")
    require(len({t.annotation_member for t in tracks})==60,"duplicate player05 member")
    return tracks

def proposal_args(source_dir):
    a=SimpleNamespace(base_model=source_dir/"v84/control.epoch-01.keras")
    for version,stem in (("v86","v86-state-transition-refiner"),
                         ("v87","v87-causal-candidate-memory"),
                         ("v88","v88-regime-moe")):
        setattr(a,version+"_weights",source_dir/version/(stem+".weights.h5"))
        setattr(a,version+"_report",source_dir/version/"report.json")
    return a

def runtime_records(tracks,score_by_member,floor):
    """Build the exact V8.6 runtime inputs without reading reference annotations.

    The historical _records() function mixes runtime fields with supervision
    fields. Only class_id/birth/matched_reference_count depend on references;
    downstream inference tensors ignore those labels. They are set to neutral
    placeholders here.
    """
    records=[]
    for ordinal,track in enumerate(tracks,start=1):
        member=track.annotation_member
        audio=decode_pcm16_mono_wav(track.audio_zip,track.audio_member)
        normalized=np.asarray(audio.samples,dtype=np.float64)/32768.0
        groups=_candidate_groups(score_by_member[member],floor)
        presence=score_by_member[member]["presence"]
        multiplicity=score_by_member[member]["multiplicity"]
        for group in groups:
            sample=int(group["sample"])
            records.append({
              "member":member,
              "arrangement":_arrangement(member),
              "sample":sample,
              "count":int(group["count"]),
              "score":float(group["score"]),
              "sources":list(group["sources"]),
              "class_id":0,
              "birth":0,
              "matched_reference_count":0,
              "horizons":[_spectral_transition(normalized,sample,h) for h in HORIZONS],
              "score_context":_score_context(presence,multiplicity,sample),
            })
        print(f"runtime features {ordinal}/{len(tracks)}: {member} candidates={len(groups)}",flush=True)
    return records

def runtime_stack(records,enc86,enc87,model88):
    emb86,prob86=_encode_records(enc86,records)
    sequences,_=_sequence_arrays(records,emb86,prob86)
    hidden87,prob87=_encode_v87(enc87,sequences)
    x88=_feature_matrix(records,emb86,prob86,hidden87,prob87)
    out88=_predictions(model88,x88)
    return x88,out88

def runtime_equivalence_guard(dataset,base_model,floor,enc86,enc87,model88):
    """Prove on allowed internal tracks that removing labels changes no runtime tensor."""
    tracks=gs.index_guitarset(dataset)
    require(len(tracks)>0,"no internal tracks for runtime equivalence")
    probe=(tracks[0],tracks[-1])
    scores=_predict_score_tracks(base_model,probe)
    labelled=_records(probe,scores,floor)
    runtime=runtime_records(probe,scores,floor)
    require(len(labelled)==len(runtime)>0,"runtime record length mismatch")
    for a,b in zip(labelled,runtime):
        for key in ("member","arrangement","sample","count","score","sources"):
            require(a[key]==b[key],f"runtime record drift {key}")
        np.testing.assert_allclose(a["horizons"],b["horizons"],rtol=0,atol=0)
        np.testing.assert_allclose(a["score_context"],b["score_context"],rtol=0,atol=0)
    xl,ol=runtime_stack(labelled,enc86,enc87,model88)
    xr,orr=runtime_stack(runtime,enc86,enc87,model88)
    np.testing.assert_allclose(xl,xr,rtol=0,atol=0)
    for key in sorted(ol):
        np.testing.assert_allclose(ol[key],orr[key],rtol=0,atol=1e-7)
    return {"probe_members":[t.annotation_member for t in probe],
            "records":len(runtime),"x88_max_abs":float(np.max(np.abs(xl-xr))) if len(xl) else 0.0,
            "outputs_equal":True}

def label_free_holdout(tracks,source_dir,dataset):
    ma=proposal_args(source_dir)
    floor,_,enc86,enc87,model88=_load_frozen_stack(ma)
    equivalence=runtime_equivalence_guard(dataset,ma.base_model,floor,enc86,enc87,model88)
    print(json.dumps({"phase":"runtime_equivalence_passed",**equivalence}),flush=True)
    print(json.dumps({"phase":"player05_proposals","tracks":len(tracks)}),flush=True)
    score_streams=_predict_score_tracks(ma.base_model,tracks)
    records=runtime_records(tracks,score_streams,floor)
    x88,out88=runtime_stack(records,enc86,enc87,model88)
    candidate_features,fused=_extended_candidate_features(x88,out88)
    clusters=_clusters_window(records,CLUSTER_WINDOW_MS)
    sequence,mask,stats,_,_,truncated=_cluster_arrays(
        clusters,{},records,candidate_features,out88,fused
    )
    fields=timing.capture_timing(clusters,records,fused,MAX_CANDIDATES)
    spectra=_spectral_maps_for_runtime(tracks,clusters,records,window=COVERED)
    require(np.isfinite(sequence).all() and np.isfinite(stats).all() and np.isfinite(spectra).all(),
            "nonfinite player05 runtime input")
    members=np.asarray([cl["member"] for cl in clusters],dtype="U96")
    cache={
      "sequence":np.asarray(sequence,np.float32),
      "mask":np.asarray(mask,np.float32),
      "stats":np.asarray(stats,np.float32),
      "spectral":np.asarray(spectra,np.float32),
      "members":members,
      "cluster_start_samples":np.asarray(fields["cluster_start_samples"],np.int64),
    }
    require(len(cache["sequence"])==len(clusters)>0,"empty player05 clusters")
    return cache,clusters,records,x88,out88,fused,truncated,equivalence

def holdout_audio_rows(cache,ids,tracks):
    by={t.annotation_member:t for t in tracks}
    aud={};out=[]
    for row in np.asarray(ids,np.int64):
        member=str(cache["members"][row]);require(member in by,"holdout audio member missing")
        if member not in aud:
            wav=decode_pcm16_mono_wav(by[member].audio_zip,by[member].audio_member)
            aud[member]=np.asarray(wav.samples,np.float64)/32768.0
        freq,x=transition_spectrum(aud[member],int(cache["cluster_start_samples"][row]))
        out.append(extract_one(freq,x))
    return out

def extra_map_internal(cache,ids,dataset):
    ids=np.asarray(ids,np.int64);wanted={str(cache["members"][i]) for i in ids}
    tracks={t.annotation_member:t for t in gs.index_guitarset(dataset) if t.annotation_member in wanted}
    require(set(tracks)==wanted,"internal extra audio coverage")
    by=defaultdict(list)
    for rid in ids:by[str(cache["members"][rid])].append(int(rid))
    out={}
    for member in sorted(by):
        wav=decode_pcm16_mono_wav(tracks[member].audio_zip,tracks[member].audio_member)
        s=np.asarray(wav.samples,float)/32768.
        for rid in sorted(set(by[member])):
            q=inference_features(s,int(cache["cluster_start_samples"][rid]))
            v=np.asarray([q[k] for k in EXTRA],float);require(np.isfinite(v).all(),"internal extra nonfinite")
            out[rid]=v
    return out

def extra_map_holdout(cache,ids,tracks):
    ids=np.asarray(ids,np.int64);bytrack={t.annotation_member:t for t in tracks}
    by=defaultdict(list)
    for rid in ids:by[str(cache["members"][rid])].append(int(rid))
    out={}
    for member in sorted(by):
        t=bytrack[member]
        wav=decode_pcm16_mono_wav(t.audio_zip,t.audio_member)
        s=np.asarray(wav.samples,float)/32768.
        for rid in sorted(set(by[member])):
            q=inference_features(s,int(cache["cluster_start_samples"][rid]))
            v=np.asarray([q[k] for k in EXTRA],float);require(np.isfinite(v).all(),"holdout extra nonfinite")
            out[rid]=v
    return out

def enriched_classifier():
    return Pipeline([("scale",StandardScaler()),
                     ("lr",LogisticRegression(C=1.0,max_iter=3000,solver="lbfgs",
                                               class_weight="balanced",random_state=28431))])

def class_metrics(y,p):
    out={}
    for k in range(7):
        m=y==k
        out[str(k)]={"rows":int(m.sum()),"correct":int(np.sum((p==y)&m)),
                     "exact":None if not m.any() else float(np.mean(p[m]==y[m]))}
    return out

def freeze_predictions(path,cache,base_pred,base60_pred,enr_pred,base60_mask,enr_mask,
                       pbase,penr,candidate_ids):
    np.savez_compressed(path,
      members=np.asarray(cache["members"]),
      cluster_start_samples=np.asarray(cache["cluster_start_samples"],np.int64),
      robust_base_predicted=np.asarray(base_pred,np.int32),
      base_residual_p060_predicted=np.asarray(base60_pred,np.int32),
      enriched_p059_predicted=np.asarray(enr_pred,np.int32),
      base_residual_p060_action_mask=np.asarray(base60_mask,np.uint8),
      enriched_p059_action_mask=np.asarray(enr_mask,np.uint8),
      residual_candidate_ids=np.asarray(candidate_ids,np.int64),
      base_candidate_probability=np.asarray(pbase,float),
      enriched_candidate_probability=np.asarray(penr,float))
    return digest(path)

def read_labels_after_freeze(tracks,clusters,records,x88,out88,sequence,mask,stats,pred_sha):
    require(len(pred_sha)==64,"prediction hash missing before labels")
    old=gs.ALLOWED_PLAYERS;gs.ALLOWED_PLAYERS=frozenset((PLAYER,))
    try:
        refs=_references(tracks)
        assigned,diag=_assign_refs(clusters,records,refs)
        # Independent audit: labelled builder must reproduce all runtime arrays.
        _,_,_,seq2,mask2,stats2,_,exact2,_=_cluster_data_labelled(tracks,records,x88,out88)
    finally:
        gs.ALLOWED_PLAYERS=old
    exact=np.asarray([len(assigned.get(i,())) for i in range(len(clusters))],np.int32)
    require(np.array_equal(exact,np.asarray(exact2,np.int32)),"label assignment mismatch")
    np.testing.assert_allclose(sequence,seq2,rtol=0,atol=0)
    np.testing.assert_allclose(mask,mask2,rtol=0,atol=0)
    np.testing.assert_allclose(stats,stats2,rtol=0,atol=0)
    return np.minimum(exact,6),diag

def _cluster_data_labelled(tracks,records,x88,out88):
    # imported lazily only for the post-prediction leakage audit
    from scripts.train_v90_structured_cluster_cardinality import _cluster_data
    return _cluster_data(tracks,records,x88,out88)

def main():
    ap=argparse.ArgumentParser()
    for n in ("source-dir","dataset","bundle","config","uniform-weights","freeze-weights",
              "feature-report","protocol","output"):
        ap.add_argument("--"+n,type=Path,required=True)
    a=ap.parse_args();require(not a.output.exists(),"overwrite")
    a.output.mkdir(parents=True)

    feature=json.loads(a.feature_report.read_text())
    chosen=feature["selected_candidate_if_strictly_better_than_base"]
    require(chosen is not None and chosen["variant"]=="base_geom_attack","frozen feature candidate drift")
    require(abs(float(chosen["threshold"])-TH_ENR)<1e-12 and int(chosen["net"])==20,
            "frozen enriched threshold/net drift")
    protocol_sha=digest(a.protocol)

    # Fresh holdout inventory. No annotation content is read here.
    tracks=index_player05(a.dataset)
    internal_cache,parts,_=load_bundle(a.bundle,a.config)
    internal_members={str(x) for x in np.asarray(internal_cache["members"]).astype(str)}
    hold_members={t.annotation_member for t in tracks}
    from scripts.train_boundaries import group_stem
    dev_compositions=set(json.loads(a.config.read_text())["composition_folds"])
    hold_compositions={group_stem(m) for m in hold_members}
    overlap_compositions=hold_compositions & dev_compositions
    new_compositions=hold_compositions - dev_compositions
    require(not internal_members & hold_members,"player05 overlaps internal bundle")
    require({m[:2] for m in internal_members}<=set(("00","01","02","03","04")),"internal player scope drift")

    hold,clusters,records,x88,out88,_,truncated,runtime_equivalence=label_free_holdout(tracks,a.source_dir,a.dataset)
    hid=np.arange(len(hold["sequence"]),dtype=np.int64)

    # Frozen robust count base: internal FIT labels define B_low; holdout labels are unavailable here.
    fit=np.asarray(parts["final_fit"],np.int64)
    yf=np.minimum(internal_cache["exact"].astype(np.int32),6)[fit]
    count_model=build_robust_model(a.uniform_weights,a.freeze_weights)
    Pf,Gf=predict(count_model,internal_cache,fit)
    Ph,Gh=predict(count_model,hold,hid)
    Xf=structural_features(internal_cache,fit,Pf)
    Xh=structural_features(hold,hid,Ph)
    blow_f,blow_h,cstats,sstats,bid,lid=reconstruct_blow(Xf,Xh,yf,Gf)

    # Fit residual classifiers only from internal labelled data.
    cand_f=blow_f&(Gf==3);cand_h=blow_h&(Gh==3)
    idsf=fit[cand_f];idsh=hid[cand_h]
    rf=audio_rows(internal_cache,idsf,a.dataset)
    rh=holdout_audio_rows(hold,idsh,tracks)
    Xrf,_,vf,_=extracted_matrix(rf);Xrh,_,vh,_=extracted_matrix(rh)
    yfc=yf[cand_f]
    train=vf&np.isin(yfc,(2,3))
    require(np.any(yfc[train]==2) and np.any(yfc[train]==3),"residual train collapse")

    base_lr=fit_classifier(Xrf[train],(yfc[train]==2).astype(int))
    pbase=np.full(len(idsh),np.nan);pbase[vh]=base_lr.predict_proba(Xrh[vh])[:,1]

    ext_i=extra_map_internal(internal_cache,idsf[vf],a.dataset)
    ext_h=extra_map_holdout(hold,idsh[vh],tracks)
    Xef=np.column_stack([Xrf[vf],np.vstack([ext_i[int(i)] for i in idsf[vf]])])
    Xeh=np.column_stack([Xrh[vh],np.vstack([ext_h[int(i)] for i in idsh[vh]])])
    train_valid=np.isin(yfc[vf],(2,3));require(int(train_valid.sum())==int(train.sum()),"train align")
    enr=enriched_classifier();enr.fit(Xef[train_valid],(yfc[vf][train_valid]==2).astype(int))
    penr=np.full(len(idsh),np.nan);penr[vh]=enr.predict_proba(Xeh)[:,1]

    # Freeze predictions BEFORE opening player05 annotations.
    base60_local=vh&(pbase>=TH_BASE)
    enr_local=vh&(penr>=TH_ENR)
    pos={int(g):i for i,g in enumerate(hid)}
    base60_mask=np.zeros(len(hid),bool);enr_mask=np.zeros(len(hid),bool)
    for gid,b,e in zip(idsh,base60_local,enr_local):
        j=pos[int(gid)];base60_mask[j]=bool(b);enr_mask[j]=bool(e)
    pred60=Gh.copy();pred60[base60_mask]=2
    pred_enr=Gh.copy();pred_enr[enr_mask]=2
    frozen_path=a.output/"predictions-before-labels.npz"
    pred_sha=freeze_predictions(frozen_path,hold,Gh,pred60,pred_enr,base60_mask,enr_mask,pbase,penr,idsh)
    (a.output/"prediction-sha256.txt").write_text(pred_sha+"\n")
    print(json.dumps({"phase":"predictions_frozen_before_labels","sha256":pred_sha,
                      "clusters":len(hid),"tracks":len(tracks)}),flush=True)

    # Only now open annotations and score.
    y,assign_diag=read_labels_after_freeze(
        tracks,clusters,records,x88,out88,hold["sequence"],hold["mask"],hold["stats"],pred_sha
    )
    require(len(y)==len(Gh),"holdout label length drift")

    base_m=metrics_simple(y,Gh);b60_m=metrics_simple(y,pred60);enr_m=metrics_simple(y,pred_enr)
    base60_actions=action_count(y,base60_mask);enr_actions=action_count(y,enr_mask)
    enr_effect=effect(y,Gh,pred_enr);b60_effect=effect(y,Gh,pred60)
    pass_rule=bool(enr_effect["net"]>0 and enr_m["poly_exact"]>base_m["poly_exact"])

    report={
      "status":"completed",
      "experiment":"v273_fresh_player05_independent_validation",
      "protocol":{
        "protocol_sha256":protocol_sha,
        "player":"05","tracks":len(tracks),
        "independence":"unseen performer; same GuitarSet composition inventory",
        "proposal_source_player05_excluded":True,
        "runtime_label_free_equivalence_guard":runtime_equivalence,
        "count_training_player05_excluded":True,
        "development_folds_player05_excluded":True,
        "labels_read_after_prediction_sha_frozen":True,
        "prediction_sha256":pred_sha,
        "variant":"base_geom_attack","threshold":TH_ENR,
        "classifier":"StandardScaler + balanced LogisticRegression(C=1, random_state=28431)",
        "features":["best_pair_residual_ratio","best_triplet_residual_ratio",*EXTRA],
        "automatic_retuning_after_holdout":False,
      },
      "inventory":{"members":sorted(hold_members),
                   "compositions":sorted({m.split("_",1)[1].rsplit("_",1)[0] for m in hold_members}),
                   "track_count":len(hold_members),"clusters":len(y),
                   "truncated_candidates":int(np.sum(truncated))},
      "internal_fit":{"B_like_cluster":int(bid),"B_low_cluster":int(lid),
                      "coarse_clusters":{str(k):v for k,v in cstats.items()},
                      "B_subclusters":{str(k):v for k,v in sstats.items()},
                      "residual_K23_train_rows":int(train.sum())},
      "holdout":{
        "assignment":assign_diag,
        "robust_base_metrics":base_m,
        "base_residual_p060_metrics":b60_m,
        "enriched_p059_metrics":enr_m,
        "base_residual_p060_actions":base60_actions,
        "enriched_p059_actions":enr_actions,
        "base_residual_p060_effect_vs_base":b60_effect,
        "enriched_p059_effect_vs_base":enr_effect,
        "robust_base_by_k":class_metrics(y,Gh),
        "enriched_p059_by_k":class_metrics(y,pred_enr),
        "poly_rows":int(np.sum(y>=2)),
      },
      "predeclared_pass":{
        "corrections_gt_regressions":enr_effect["net"]>0,
        "poly_exact_strictly_higher":enr_m["poly_exact"]>base_m["poly_exact"],
        "passed":pass_rule,
      },
      "fresh_independent_validation":True,
      "composition_independent":len(overlap_compositions)==0,
      "prediction_changes":True,
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    np.savez_compressed(a.output/"scored-predictions.npz",
      true_k=y,robust_base_predicted=Gh,base_residual_p060_predicted=pred60,
      enriched_p059_predicted=pred_enr,base_residual_p060_action_mask=base60_mask.astype(np.uint8),
      enriched_p059_action_mask=enr_mask.astype(np.uint8))

    def pct(v):return 100*float(v)
    lines=["# Fresh independent validation — GuitarSet player 05","",
           "**No player-05 label was opened before predictions were frozen.**",
           f"Prediction SHA-256: `{pred_sha}`.","",
           "| Measure | robust base | base residual p>=.60 | enriched p>=.59 |",
           "|---|---:|---:|---:|",
           f"| Exact-K global | {pct(base_m['exact']):.3f}% | {pct(b60_m['exact']):.3f}% | {pct(enr_m['exact']):.3f}% |",
           f"| Exact-K poly | {pct(base_m['poly_exact']):.3f}% | {pct(b60_m['poly_exact']):.3f}% | {pct(enr_m['poly_exact']):.3f}% |","",
           f"Enriched actions: **{enr_actions['corrections']} corrections / {enr_actions['regressions']} regressions**, other-K {enr_actions['other_k']}, net **{enr_actions['net']:+d}**.",
           f"Paired Exact-K effect vs robust base: **{enr_effect['net']:+d}**.",
           f"Predeclared independent-validation pass: **{pass_rule}**.","",
           "This is performer-independent, not composition-independent. No retuning is permitted from this result."]
    (a.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__":main()
