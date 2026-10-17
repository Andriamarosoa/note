"""S651 — learned routing across independent NOTE harmonic heads.

Actual objective: identify *useful NoteOn corrections* versus regressions.
Four independent S650 heads give B/T/E/J probability opinions; an extra
router learns WHEN to trust their agreement or rescue using the current
native attack, native K distribution and physical/harmonic competition.

Proper NESTED group validation to avoid base-head/meta data leakage:
 - outer 5 composition folds hold out entire GuitarSet TRAIN songs
 - 3-way inner group folds train base heads and yield honest OOF
   scores for the *outer training* meta-router input
 - outer validation scores derive from base heads trained only on outer
   training groups; meta heads trained only on inner-OOF features
 - compare OOF router utility to the baseline S641 head, freeze
   model/floor on TRAIN before opening the 12 GuitarSet DEV sources
 - train final meta-router with full 5-fold OOF base scores, then fit
   base heads on all TRAIN
No TEST, no performance claims for microphone device latency.
"""
from __future__ import annotations

from collections import Counter,deque
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import CausalSelection,Selection,load_heads
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,GUITARSET_PINNED_CONTROL
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s640_learned_overlap_rescue import iter_acoustic,get_candidates,relay,pcm_truth
from scripts.s641_fundamental_consistency import original_fundamental_evidence
from scripts.s649_exclusive_harmonic_attack import (
    train_record,novelty_features,exclusive_harmonic_attack,historical,
    matrices,build_model
)
from scripts.s650_harmonic_head_selection_combos import (
    HEADS,HEAD_ALIAS,binary_utility,head_policies,COMBO_KINDS
)

ROUTERS=("opinions_only","conditional_context")
FLOORS=(.50,.60,.70,.75,.80,.85,.90,.92,.95,.97,.99,1.01)
BASELINE_MIN_IMPROVEMENT=2.0
INNER_FOLDS=3
OUTER_FOLDS=5
# This small controlled set was fixed before any DEV testing.
CONTEXT_FEATURE_INDICES=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,
                          14,17,20,23,26,29,32,35,38,40)
PHYSICAL_FEATURE_INDICES=(0,2,3,4,6,7,10,14,18,22)
HEAD_FLOORS=dict(B=.90,T=.95,E=.95,J=.92)  # S650 TRAIN-selected floors
VOTE_FLOORS=dict(B=.90,T=.95,E=.95,J=.92)
BLEND_FLOORS=dict(
    weighted_B60_T20_E20=.90,
    weighted_B40_T30_E30=.90,
    weighted_B40_T20_E20_J20=.90,
    geometric_BTE=.90,
)


def head_probabilities(trained,features):
    return np.column_stack([trained[tag].predict_proba(features[source])[:,1]
                            for tag,source in zip(HEAD_ALIAS,HEADS)])


def meta_features(scores,features,variant):
    """Only current/past source features and four independently trained heads.

    Physical K, register, onset, flux and f0 support become the *routing*
    context, not a separate global veto. Every scoring row keeps the
    same deterministic layout; labels and future samples never enter.
    """
    a=np.asarray(scores,np.float32)
    if a.ndim!=2 or a.shape[1]!=4 or not np.isfinite(a).all():
        raise ValueError("S651 requires four independent finite head outputs")
    b,t,e,j=a.T
    simple=np.column_stack((
        a, np.max(a,axis=1),np.min(a,axis=1),
        np.std(a,axis=1),b-t,b-e,b-j,t-e,
        b*(t+e)*.5,t*e,
        ((a[:,:3]>=.9).sum(axis=1)).astype(np.float32)/3.,
    ))
    if variant=="opinions_only":
        return np.asarray(simple,np.float32)
    if variant!="conditional_context":
        raise ValueError(f"Invalid router {variant}")
    control=np.asarray(features["control41"],np.float32)
    extended=np.asarray(features["exclusive65"],np.float32)
    physical=extended[:,control.shape[1]:control.shape[1]+24]
    if control.shape[1]!=41 or physical.shape[1]!=24:
        raise AssertionError("S651 baseline NOTE harmonic-context feature ABI changed")
    return np.asarray(np.column_stack((
        simple,control[:,CONTEXT_FEATURE_INDICES],
        physical[:,PHYSICAL_FEATURE_INDICES])),np.float32)


def make_router(name):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(
                             C=.35 if name=="opinions_only" else .08,
                             class_weight="balanced",max_iter=1600,
                             random_state=651))


def base_fit(matrix,y,idx):
    if np.unique(y[idx]).size!=2:
        raise ValueError("No corrected NoteOn in base-head train")
    return {tag:build_model(source).fit(matrix[source][idx],y[idx])
            for tag,source in zip(HEAD_ALIAS,HEADS)}


def nested_router_cv(records):
    y=np.concatenate([r["y"] for r in records])
    g=np.concatenate([np.repeat(r["composition"],len(r["y"])) for r in records])
    matrix={name:np.concatenate([r["matrices"][name] for r in records])
            for name in set(HEADS)|{"exclusive65","control41"}}
    if len(y)<400 or int(y.sum())<35 or len(set(g))!=9:
        raise ValueError("S651 unable to reproduce the original GuitarSet correction pool")
    outer_split=list(GroupKFold(n_splits=OUTER_FOLDS).split(matrix["control41"],y,g))
    outer_prob={tag:np.full(len(y),np.nan) for tag in HEAD_ALIAS}
    meta_predictions={name:np.full(len(y),np.nan) for name in ROUTERS}
    details=[]
    for nth,(itr,iva) in enumerate(outer_split,1):
        if set(g[itr])&set(g[iva]):
            raise AssertionError("S651 outer composition leakage")
        inner_preds=np.full((len(itr),4),np.nan)
        inner_split=list(GroupKFold(n_splits=INNER_FOLDS).split(
            matrix["control41"][itr],y[itr],g[itr]))
        for train_local,held_local in inner_split:
            train_idx=itr[train_local]
            val_idx=itr[held_local]
            if set(g[train_idx])&set(g[val_idx]):
                raise AssertionError("S651 inner composition leakage")
            fitted=base_fit(matrix,y,train_idx)
            scores=head_probabilities(fitted,{
                source:matrix[source][val_idx] for source in HEADS})
            inner_preds[held_local]=scores
        if not np.isfinite(inner_preds).all():
            raise AssertionError("S651 missing inner OOF predictions")
        final_inner=base_fit(matrix,y,itr)
        outer_scores=head_probabilities(final_inner,{
            source:matrix[source][iva] for source in HEADS})
        for i,tag in enumerate(HEAD_ALIAS):
            outer_prob[tag][iva]=outer_scores[:,i]
        for name in ROUTERS:
            fit_x=meta_features(inner_preds,{
                source:matrix[source][itr] for source in ("control41","exclusive65")},
                name)
            test_x=meta_features(outer_scores,{
                source:matrix[source][iva] for source in ("control41","exclusive65")},
                name)
            trained=make_router(name).fit(fit_x,y[itr])
            meta_predictions[name][iva]=trained.predict_proba(test_x)[:,1]
        details.append(dict(fold=nth,train_compositions=sorted(set(g[itr])),
                            held_compositions=sorted(set(g[iva])),
                            train_positive=int(y[itr].sum()),
                            held_positive=int(y[iva].sum())))
        print("S651 TRAIN NESTED",nth,"/5","train",len(itr),
              "held",len(iva),"positives",int(y[iva].sum()),flush=True)
    if not all(np.isfinite(p).all() for p in
               [*outer_prob.values(),*meta_predictions.values()]):
        raise AssertionError("Incomplete nested composition OOF")
    outcomes={}
    for name in ROUTERS:
        trials=[]
        for floor in FLOORS:
            choice=binary_utility(meta_predictions[name]>=floor,y)
            trials.append(dict(floor=floor,**choice))
        selected=max(trials,key=lambda v:(v["utility"],-v["FP"],v["floor"]))
        outcomes[name]=dict(chosen=selected,trials=trials)
    base_oof=binary_utility(outer_prob["B"]>=HEAD_FLOORS["B"],y)
    selected_name=max(ROUTERS,key=lambda n:(
        outcomes[n]["chosen"]["utility"],-outcomes[n]["chosen"]["FP"],n))
    selected=(selected_name if outcomes[selected_name]["chosen"]["utility"]>=
              base_oof["utility"]+BASELINE_MIN_IMPROVEMENT else "S641_only")
    print("S651 ROUTER TRAIN CV",json.dumps(dict(
        baseline=base_oof,selected=selected,router_choices={
            name:outcomes[name]["chosen"] for name in ROUTERS})),flush=True)
    # Freeze two router models on full-data OOF base head scores:
    # this OOF is NEVER trained on the label of its own composition.
    base_full_oof=np.full((len(y),4),np.nan)
    for tr,va in outer_split:
        fitted=base_fit(matrix,y,tr)
        base_full_oof[va]=head_probabilities(fitted,{
            source:matrix[source][va] for source in HEADS})
    router_models={}
    for name in ROUTERS:
        router_models[name]=make_router(name).fit(
            meta_features(base_full_oof,{
                source:matrix[source] for source in ("control41","exclusive65")},
                name),y)
    base_full=base_fit(matrix,y,np.arange(len(y)))
    return base_full,router_models,selected,outcomes,dict(
        nested_folds=details,baseline_S641_oof=base_oof,
        train_examples=len(y),train_positive=int(y.sum()),
        candidate_winner=selected_name,
        threshold_choice_from_nested_OOF_only=True)


def evaluate(spec,root,archive,heads,config,base,meta,choices):
    sound,truth,sha=pcm_truth(spec,root,archive)
    names=("S614","bass_raw","S641_only","majority_two_of_BTE",
           "router_opinions_only","router_conditional_context")
    states={name:CausalSelection(config) for name in names}
    events={name:[] for name in names}
    counts=Counter()
    history=deque(maxlen=16)
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        spectra_cache={}
        old4,old12=historical(history,x)
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        events["S614"].extend(states["S614"].step(native,k,int(end),int(end)))
        indices=list(candidates)
        matrices_batch={key:[] for key in HEADS}
        for i in indices:
            f=candidates[i]
            physical=original_fundamental_evidence(x,i)
            temporal=novelty_features(x,old4,old12,native,i)
            exclusive=exclusive_harmonic_attack(
                sound,int(end),i,native,states["S614"].active,spectra_cache)
            feats=matrices(f,physical,temporal,exclusive)
            for key in HEADS:
                matrices_batch[key].append(feats[key])
        if indices:
            batch={key:np.stack(matrices_batch[key]) for key in HEADS}
            head_scores=head_probabilities(base,batch)
            independently={tag:head_scores[:,j] for j,tag in enumerate(HEAD_ALIAS)}
            meta_prob={name:meta[name].predict_proba(
                meta_features(head_scores,batch,name))[:,1] for name in ROUTERS}
            B,T,E=[independently[tag]>=HEAD_FLOORS[tag] for tag in ("B","T","E")]
            decisions=dict(
                S641_only=B,
                majority_two_of_BTE=(B.astype(int)+T.astype(int)+E.astype(int))>=2,
                router_opinions_only=meta_prob["opinions_only"]>=choices["opinions_only"]["chosen"]["floor"],
                router_conditional_context=meta_prob["conditional_context"]>=choices["conditional_context"]["chosen"]["floor"],
            )
        else:
            decisions={name:[] for name in names if name not in ("S614","bass_raw")}
        raw,amount=relay(native,bass,states["bass_raw"],
                         {i:1. for i in indices},0.)
        events["bass_raw"].extend(states["bass_raw"].step(raw,k,int(end),int(end)))
        counts["bass_raw"]+=amount
        for name,yes in decisions.items():
            mask={i:1. for i,ok in zip(indices,yes) if ok}
            p,accepted=relay(native,bass,states[name],mask,.5)
            counts[name]+=accepted
            events[name].extend(states[name].step(p,k,int(end),int(end)))
        history.append(x.copy())
    groups=overlap_groups(truth)
    outputs={}
    baseline=None
    for name,state in states.items():
        events[name].extend(state.panic(len(sound)))
        sc,pairs=event_score(events[name],truth)
        gt={j for _,j in pairs}
        if name=="S614":baseline=gt
        outputs[name]=dict(score=sc,poly_TP=len(gt&groups["onset_overlap"]),
                           K2_TP=len(gt&groups["onset_K2"]),
                           K3plus_TP=len(gt&groups["onset_K3plus"]),
                           annotations=gt,
                           accepted_proposals=int(counts[name]))
    for name,row in outputs.items():
        matched=row.pop("annotations")
        row["new_gt_vs_S614"]=len(matched-baseline)
        row["lost_gt_vs_S614"]=len(baseline-matched)
    return dict(name=spec["name"],composition=spec["composition"],
                source_sha256=sha,policies=outputs)


def run(out):
    if out.exists():raise FileExistsError("S651 evidence file must be immutable")
    pinned=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==pinned["protocol_sha256"]
    assert digest(MODEL.read_bytes())==pinned["model_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in protocol["sources"]
           if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    held=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12 and len(set(s["composition"] for s in train))==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+held}
    heads=load_heads(MODEL,expected_sha256=pinned["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/filename,z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            records=[]
            for i,spec in enumerate(train,1):
                record=train_record(spec,root,archive,heads,cfg)
                records.append(record)
                print("S651 REAL TRAIN",i,"/36",record["name"],
                      "candidates",len(record["y"]),"useful",int(record["y"].sum()),
                      flush=True)
            base,meta,winner,choices,cv=nested_router_cv(records)
            print("S651 ROUTER CHOICE FROZEN BEFORE DEV",winner,flush=True)
            validated=[]
            for i,spec in enumerate(dev,1):
                row=evaluate(spec,root,archive,heads,cfg,base,meta,choices)
                validated.append(row)
                print("S651 REAL DEV",i,"/12",row["name"],
                      "B",row["policies"]["S641_only"]["score"]["TP"],
                      "router",row["policies"]["router_conditional_context"]["score"]["TP"],
                      flush=True)
    outcomes={}
    for name in validated[0]["policies"]:
        items=[r["policies"][name] for r in validated]
        s=summary([r["score"] for r in items])
        outcomes[name]=dict(
            score=s,
            overlapping_TP=sum(r["poly_TP"] for r in items),
            K2_TP=sum(r["K2_TP"] for r in items),
            K3plus_TP=sum(r["K3plus_TP"] for r in items),
            newly_correct=sum(r["new_gt_vs_S614"] for r in items),
            previously_correct_lost=sum(r["lost_gt_vs_S614"] for r in items),
            admitted=sum(r["accepted_proposals"] for r in items))
    assert all(outcomes["S614"]["score"][key]==value
               for key,value in GUITARSET_PINNED_CONTROL.items())
    assert outcomes["bass_raw"]["score"]["TP"]==1044
    assert outcomes["bass_raw"]["score"]["FP"]==834
    assert outcomes["S641_only"]["score"]["TP"]==1009
    assert outcomes["S641_only"]["score"]["FP"]==594
    s641=outcomes["S641_only"]["score"]
    for name,r in outcomes.items():
        s=r["score"]
        r["delta_F1_points_vs_S641"]=round(100*(s["F1"]-s641["F1"]),6)
        r["delta_TP_vs_S641"]=s["TP"]-s641["TP"]
        r["delta_FP_vs_S641"]=s["FP"]-s641["FP"]
        print("S651 SCORE",name,"TP",s["TP"],"FP",s["FP"],
              "F1",round(s["F1"],8),"delta",r["delta_F1_points_vs_S641"],
              "K3",r["K3plus_TP"],flush=True)
    report=dict(
        status="S651_TRAIN_ONLY_NESTED_CV_CONDITIONAL_HEAD_SELECTION_ROUTER",
        unchanged_S614_and_ExactK=True,
        unchanged_live_runtime=True,
        additional_future_lookahead_ms=0,
        original_guitarset_TRAIN=36,
        original_guitarset_DEV=12,
        guitarset_TEST_never_accessed=True,
        dev_already_exposed_exploratory=True,
        nested_inner_group_CV_used_to_prevent_router_training_leakage=True,
        training_only_preselected_winner=winner,
        baseline_head_thresholds=HEAD_FLOORS,
        router_cuts={n:v["chosen"]["floor"] for n,v in choices.items()},
        nested_cv=cv,
        train_router_selection=choices,
        frozen_real_DEV_aggregate=outcomes,
        real_DEV_per_recording=validated,
        elapsed_seconds=round(time.perf_counter()-start,2),
        no_promotion=True,
        note="S651 selects between NOTE independent harmonic heads using native onset, K and acoustic context, not heuristic thresholds chosen on DEV."),
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S651 FINAL",json.dumps(dict(
        winner=winner,
        train_baseline_utility=cv["baseline_S641_oof"]["utility"],
        train_routes={n:r["chosen"] for n,r in choices.items()},
        dev_results={name:dict(TP=v["score"]["TP"],FP=v["score"]["FP"],
                               F1=v["score"]["F1"],K3=v["K3plus_TP"])
                     for name,v in outcomes.items()},
        duration_seconds=report["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
