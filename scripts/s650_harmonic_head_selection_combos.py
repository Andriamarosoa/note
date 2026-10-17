"""S650 — NOTE heads and selection COMBINATIONS, not feature concatenation.

Independent heads:
 H0 baseline 41D (S641 proven rescue, always tracked)
 HT attack temporal evidence
 HE physically exclusive/share-partials evidence
 HJ joint temporal+exclusive evidence (ablation)

Selectors:
 independent, AND, OR, 2/3 consensus, base-veto, base-rescue,
 weighted confidence and geometric confidence with abstention.

Candidate events are exactly the real NoteOn proposals from S649's
ungated causal Bass ×2 rescue, labelled on GuitarSet TRAIN by whether
they save one missed S614 annotation. Each head trains using ONLY
composition-held TRAIN features; each decision threshold and combination
is chosen from TRAIN OOF only, before reading GuitarSet DEV. No TEST.
All policies use separate frozen CausalSelection, K unchanged, no
future audio or backdated NoteOn, no change to official microphone LIVE.
"""
from __future__ import annotations

import argparse
from collections import Counter,deque
import json
from pathlib import Path
import tempfile
import time
import zipfile

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

from causal_note.strict_live import CausalSelection,Selection,load_heads
from scripts.s613_prepare_strict_live import digest,download
from scripts.s613_train_strict_live import event_score,summary
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,GUITARSET_PINNED_CONTROL
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s640_learned_overlap_rescue import iter_acoustic,get_candidates,relay,pcm_truth
from scripts.s641_fundamental_consistency import original_fundamental_evidence
from scripts.s649_exclusive_harmonic_attack import (
    train_record,novelty_features,exclusive_harmonic_attack,
    historical,matrices,build_model,
)

HEADS=("control41","temporal65","exclusive65","exclusive_temporal65")
HEAD_ALIAS=("B","T","E","J")
FLOORS=(.65,.75,.80,.85,.90,.92,.95,.97,.99,1.01)
COST_FP=.50
MARGIN_MIN_GAIN=1.0
COMBO_KINDS=(
    "single_B","single_T","single_E","single_J",
    "B_AND_T","B_AND_E","B_AND_J",
    "B_AND_T_OR_E", # B AND (T or E)
    "B_AND_T_AND_E",
    "B_OR_T_AND_E",  # B OR (T AND E)
    "B_OR_J","T_AND_E",
    "vote_two_of_BTE",
    "vote_three_of_BTE",
    "weighted_B60_T20_E20",
    "weighted_B40_T30_E30",
    "weighted_B40_T20_E20_J20",
    "geometric_BTE",
)


def head_policies(scores,thresholds,blend_thresholds):
    """All selection decisions at ONE physical hop, independent predictions.

    No action changes another head's feature values. Only the final
    proposals pass through NOTE's original stateful CausalSelection.
    """
    B,T,E,J=[scores[k]>=thresholds[k] for k in HEAD_ALIAS]
    result=dict(
        single_B=B,single_T=T,single_E=E,single_J=J,
        B_AND_T=B&T,B_AND_E=B&E,B_AND_J=B&J,
        B_AND_T_OR_E=B&(T|E),B_AND_T_AND_E=B&T&E,
        B_OR_T_AND_E=B|(T&E),B_OR_J=B|J,
        T_AND_E=T&E,
        vote_two_of_BTE=((B.astype(int)+T.astype(int)+E.astype(int))>=2),
        vote_three_of_BTE=B&T&E,
    )
    result["weighted_B60_T20_E20"]=(.6*scores["B"]+.2*scores["T"]+.2*scores["E"])>=blend_thresholds["weighted_B60_T20_E20"]
    result["weighted_B40_T30_E30"]=(.4*scores["B"]+.3*scores["T"]+.3*scores["E"])>=blend_thresholds["weighted_B40_T30_E30"]
    result["weighted_B40_T20_E20_J20"]=(.4*scores["B"]+.2*scores["T"]+.2*scores["E"]+.2*scores["J"])>=blend_thresholds["weighted_B40_T20_E20_J20"]
    result["geometric_BTE"]=(np.clip(scores["B"]*scores["T"]*scores["E"],0,1)**(1/3))>=blend_thresholds["geometric_BTE"]
    assert set(result)==set(COMBO_KINDS)
    return result


def binary_utility(decisions,truth):
    take=np.asarray(decisions,bool)
    y=np.asarray(truth,bool)
    TP=int(y[take].sum())
    FP=int(take.sum())-TP
    return dict(TP=TP,FP=FP,FN=int(y.sum())-TP,accepted=int(take.sum()),
                utility=float(TP-COST_FP*FP))


def model_train_cv(records):
    labels=np.concatenate([r["y"] for r in records])
    groups=np.concatenate([np.repeat(r["composition"],len(r["y"])) for r in records])
    if len(labels)<450 or int(labels.sum())<35 or len(set(groups))!=9:
        raise ValueError("S650 insufficient original GuitarSet TRAIN correction examples")
    splits=list(GroupKFold(n_splits=5).split(np.zeros((len(labels),1)),labels,groups))
    fold_info=[]
    for itr,iva in splits:
        tr=set(groups[itr]);va=set(groups[iva])
        assert not tr&va
        fold_info.append(dict(train=sorted(tr),held=sorted(va),n_train=len(itr),
                              n_held=len(iva),positive_held=int(labels[iva].sum())))
        if len(np.unique(labels[itr]))!=2:
            raise ValueError("Fold lacks both repair and regression labels")
    models={}
    oof={}
    head_floors={}
    single_head_CV={}
    for tag,source in zip(HEAD_ALIAS,HEADS):
        x=np.concatenate([r["matrices"][source] for r in records])
        prediction=np.zeros(len(labels),np.float64)
        for tr,va in splits:
            model=build_model(source).fit(x[tr],labels[tr])
            prediction[va]=model.predict_proba(x[va])[:,1]
        oof[tag]=prediction
        best=None
        for floor in FLOORS:
            current=binary_utility(prediction>=floor,labels)
            current["floor"]=floor
            if best is None or (current["utility"],-current["FP"],floor)>(
                best["utility"],-best["FP"],best["floor"]):
                best=current
        head_floors[tag]=best["floor"]
        single_head_CV[tag]=best
        models[tag]=build_model(source).fit(x,labels)
        print("S650 TRAIN HEAD",tag,source,"n",len(labels),
              "repair",int(labels.sum()),best,flush=True)

    blend_metrics={
        "weighted_B60_T20_E20":.6*oof["B"]+.2*oof["T"]+.2*oof["E"],
        "weighted_B40_T30_E30":.4*oof["B"]+.3*oof["T"]+.3*oof["E"],
        "weighted_B40_T20_E20_J20":.4*oof["B"]+.2*oof["T"]+.2*oof["E"]+.2*oof["J"],
        "geometric_BTE":np.clip(oof["B"]*oof["T"]*oof["E"],0,1)**(1/3),
    }
    blend_floors={}
    for name,values in blend_metrics.items():
        best=None
        for floor in FLOORS:
            x=binary_utility(values>=floor,labels)
            x["floor"]=floor
            if best is None or (x["utility"],-x["FP"],floor)>(
                best["utility"],-best["FP"],best["floor"]):
                best=x
        blend_floors[name]=best["floor"]

    selections=head_policies(oof,head_floors,blend_floors)
    evaluations={name:binary_utility(dec,labels) for name,dec in selections.items()}
    # A winner must improve TRAIN OOF utility beyond S641 baseline by a
    # precommitted margin; otherwise keep base, even if a tie has fewer FP.
    base=evaluations["single_B"]
    other=max((name for name in COMBO_KINDS if name!="single_B"),
              key=lambda n:(evaluations[n]["utility"],-evaluations[n]["FP"],n))
    winner=(other if evaluations[other]["utility"]>=base["utility"]+MARGIN_MIN_GAIN
            else "single_B")
    print("S650 TRAIN COMBINATIONS",json.dumps(evaluations),flush=True)
    print("S650 FROZEN WINNER",winner,"vs baseline",base,flush=True)
    return (models,head_floors,blend_floors,winner,
            dict(sample_count=len(labels),positive=int(labels.sum()),
                 cv_folds=fold_info,single_heads=single_head_CV,
                 selection_utility=evaluations))


def validate(spec,root,archive,heads,config,models,floors,blend_floors):
    sound,truth,sha=pcm_truth(spec,root,archive)
    policies=("S614","bass_raw",*COMBO_KINDS)
    decoder={n:CausalSelection(config) for n in policies}
    emitted={n:[] for n in policies}
    admissions=Counter()
    history=deque(maxlen=16)
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        spectra_cache={} # one set of past-only FFTs across all candidates in a hop
        past4,past12=historical(history,x)
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        # Emit native baseline before reading active status for competitor features.
        emitted["S614"].extend(decoder["S614"].step(native,k,int(end),int(end)))
        index=list(candidates)
        passed={name:[] for name in HEADS}
        for i in index:
            base=candidates[i]
            physical=original_fundamental_evidence(x,i)
            temporal=novelty_features(x,past4,past12,native,i)
            exclusive=exclusive_harmonic_attack(
                sound,int(end),i,native,decoder["S614"].active,spectra_cache)
            features=matrices(base,physical,temporal,exclusive)
            for source in HEADS:passed[source].append(features[source])
        scoring={}
        for tag,source in zip(HEAD_ALIAS,HEADS):
            scoring[tag]=(models[tag].predict_proba(np.stack(passed[source]))[:,1]
                          if index else np.empty(0,np.float32))
        active=head_policies(scoring,floors,blend_floors)
        raw,count=relay(native,bass,decoder["bass_raw"],{i:1. for i in index},0.)
        admissions["bass_raw"]+=count
        emitted["bass_raw"].extend(decoder["bass_raw"].step(raw,k,int(end),int(end)))
        for name in COMBO_KINDS:
            selected={i:float(ok) for i,ok in zip(index,active[name]) if ok}
            p,count=relay(native,bass,decoder[name],selected,.5)
            admissions[name]+=count
            emitted[name].extend(decoder[name].step(p,k,int(end),int(end)))
        history.append(x.copy())
    group=overlap_groups(truth)
    score={}
    original_matches=None
    for name,state in decoder.items():
        emitted[name].extend(state.panic(len(sound)))
        measured,pairs=event_score(emitted[name],truth)
        annotations={j for _,j in pairs}
        if name=="S614":original_matches=annotations
        score[name]=dict(score=measured,
                         overlap_TP=len(annotations&group["onset_overlap"]),
                         K2_TP=len(annotations&group["onset_K2"]),
                         K3plus_TP=len(annotations&group["onset_K3plus"]),
                         matches=annotations,admissions=int(admissions[name]))
    for name,v in score.items():
        seen=v.pop("matches")
        v["new_onsets_vs_S614"]=len(seen-original_matches)
        v["lost_original_onsets"]=len(original_matches-seen)
    return dict(name=spec["name"],composition=spec["composition"],
                sha256=sha,policies=score)


def run(out):
    if out.exists():raise FileExistsError("S650 evidence folder must be new")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet"
           and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet"
         and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert len(set(s["composition"] for s in train))==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/filename,z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            rows=[]
            for i,s in enumerate(train,1):
                row=train_record(s,root,archive,heads,cfg)
                rows.append(row)
                print("S650 TRAIN",i,"/36",row["name"],
                      "candidates",len(row["y"]),"true",int(row["y"].sum()),flush=True)
            models,floors,blend_floors,winner,train_cv=model_train_cv(rows)
            tests=[]
            for i,s in enumerate(dev,1):
                row=validate(s,root,archive,heads,cfg,models,floors,blend_floors)
                tests.append(row)
                print("S650 DEV",i,"/12",row["name"],
                      "S614TP",row["policies"]["S614"]["score"]["TP"],
                      "winnerTP",row["policies"][winner]["score"]["TP"],flush=True)
    scores={}
    for name in tests[0]["policies"]:
        arr=[r["policies"][name] for r in tests]
        score=summary([r["score"] for r in arr])
        scores[name]=dict(score=score,
                          K2_TP=sum(r["K2_TP"] for r in arr),
                          K3plus_TP=sum(r["K3plus_TP"] for r in arr),
                          overlap_TP=sum(r["overlap_TP"] for r in arr),
                          new_vs_original=sum(r["new_onsets_vs_S614"] for r in arr),
                          lost_vs_original=sum(r["lost_original_onsets"] for r in arr),
                          accepted_proposals=sum(r["admissions"] for r in arr))
    assert all(scores["S614"]["score"][k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    assert scores["bass_raw"]["score"]["TP"]==1044
    assert scores["bass_raw"]["score"]["FP"]==834
    assert scores["single_B"]["score"]["TP"]==1009,scores["single_B"]
    assert scores["single_B"]["score"]["FP"]==594,scores["single_B"]
    orig=scores["S614"]["score"]
    control=scores["single_B"]["score"]
    for name,r in scores.items():
        score=r["score"]
        r["delta_TP_vs_S641"]=score["TP"]-control["TP"]
        r["delta_FP_vs_S641"]=score["FP"]-control["FP"]
        r["delta_F1points_vs_S641"]=round(100*(score["F1"]-control["F1"]),6)
        print("S650 SCORE",name,"TP",score["TP"],"FP",score["FP"],
              "F1",round(score["F1"],8),
              "deltaPtsVsS641",r["delta_F1points_vs_S641"],
              "K3",r["K3plus_TP"],flush=True)
    evidence=dict(
        status="S650_EXPLICIT_INDEPENDENT_HEADS_AND_SELECTION_COMBINATIONS",
        original_note_sha=frozen["model_sha256"],
        S614_unchanged=True,
        Exact_K_and_native_notes_unchanged=True,
        live_microphone_not_modified=True,
        past_only_features_no_extra_lookahead=True,
        dataset="GuitarSet real 36 TRAIN 12 DEV",
        DEV_previously_exposed=True,
        TEST_never_accessed=True,
        independent_heads=list(dict(zip(HEAD_ALIAS,HEADS)).items()),
        cv_selected_head_thresholds=floors,
        cv_selected_blend_thresholds=blend_floors,
        TRAIN_only_winner=winner,
        TRAIN_only_CV=train_cv,
        causal_note_on_results=scores,
        individual_source_results=tests,
        wall_time_seconds=round(time.perf_counter()-started,2),
        auto_promotion=False,
        note="Head-output combinations, unlike feature concatenation; independent training of 4 heads, grouped OOF threshold selection; comparison is exploratory DEV, not sealed TEST.")
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print("S650 FINAL",json.dumps(dict(
        chosen_from_TRAIN=winner,
        OOF_utility={name:d["utility"] for name,d in train_cv["selection_utility"].items()},
        outcomes={n:dict(TP=v["score"]["TP"],FP=v["score"]["FP"],
                         F1=v["score"]["F1"],K3=v["K3plus_TP"])
                  for n,v in scores.items()}
    )),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
