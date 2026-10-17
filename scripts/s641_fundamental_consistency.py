"""S641 — GuitarSet real-audio octave round-trip consistency experiment.

An up octave + remap down is an IDENTITY ON PITCH LABELS, not independent
acoustic proof. We test the physically meaningful criterion instead:
whether a proposed low fundamental exists, is spectrally localized and
onsets independently IN THE UNTRANSPOSED ORIGINAL PCM spectrum.

S640 is reproduced as TRAIN18 baseline. Experimental alternatives use
all 36 predefined GuitarSet TRAIN original recordings, not DEV/TEST.
Train-only group holdouts calibrate thresholds (9 composition groups).
Held GuitarSet DEV is read once after training and choices have been set.
S614 native heads + K and real-time output are never changed.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

from causal_note.strict_live import HOP,SR,Selection,CausalSelection,load_heads
from scripts.s640_learned_overlap_rescue import (
    iter_acoustic,get_candidates,relay,pcm_truth,
    feature_vector,FEATURE_NAMES,FROZEN_GATES
)
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,GUITARSET_PINNED_CONTROL
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s613_prepare_strict_live import download,digest
from scripts.s613_train_strict_live import event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups

# Spectral features are from the ORIGINAL right-aligned acoustic frames,
# independent of the remapped pitch *decision* (not independent of source
# audio). They require no additional FFT and exactly zero future samples.
ADDITIONAL_NAMES=tuple(
    f"{resolution}_{name}"
    for resolution in ("short","medium","long")
    for name in (
        "fundamental_minus_2nd",
        "fundamental_minus_4th",
        "fundamental_minus_local_semitone_neighbor",
        "fundamental_minus_octave_fundamental",
        "fundamental_attack_delta",
        "octave_attack_delta",
        "fundamental_attack_minus_octave_attack",
        "fundamental_relative_minus_octave_relative",
    )
)
ALL_NAMES=FEATURE_NAMES+ADDITIONAL_NAMES
GATES=(.70,.80,.85,.90,.92,.95,.97,.99)
BUDGET_FP_PER_NEW_TP=2.0


def original_fundamental_evidence(x,i):
    """Compare f0 to its harmonics, neighbors, octave and own onset novelty.

    If the upper octave itself is a true sounding note, its partials
    CANNOT create a subharmonic at f0. A strong independent spectral
    fundamental below it is evidence for two notes, but not a proof:
    microphones, strings, noise and true unisons can also cause energy.
    """
    if x.shape!=(49,126) or not 0<=i<24:
        raise ValueError("S641 expects 49x126 original features and bass MIDI 40..63")
    upper=i+12
    below=max(0,i-1);above=min(48,i+1)
    v=[]
    for j in range(3):
        off=8*j
        fund=float(x[i,off])
        second=float(x[i,off+1])
        fourth=float(x[i,off+3])
        near=max(float(x[below,off]),float(x[above,off]))
        octave=float(x[upper,off])
        # Physical = [score_{3}, f0_{3}]; lag=2 hops physical delta.
        delta=float(x[i,57+j])
        octave_delta=float(x[upper,57+j])
        rel=float(x[i,24+off])
        octave_rel=float(x[upper,24+off])
        v.extend((
            fund-second,
            fund-fourth,
            fund-near,
            fund-octave,
            delta,
            octave_delta,
            delta-octave_delta,
            rel-octave_rel,
        ))
    out=np.asarray(v,np.float32)
    if out.shape!=(len(ADDITIONAL_NAMES),) or not np.isfinite(out).all():
        raise ValueError("S641 spectrum consistency ABI")
    return out


def extended_features(x,native,bass,k,i):
    return np.concatenate((feature_vector(x,native,bass,k,i),
                           original_fundamental_evidence(x,i)))


def train_record(spec,root,archive,heads,selection):
    sound,truth,sha=pcm_truth(spec,root,archive)
    baseline=CausalSelection(selection)
    ungated=CausalSelection(selection)
    baseline_events=[]
    bass_events=[]
    labeled_candidates=[]
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        candidates=get_candidates(x,native,k,bass,valid,selection.onset)
        baseline_events.extend(baseline.step(native,k,int(end),int(end)))
        corrected,_=relay(native,bass,ungated,
                          {i:1. for i in candidates},0.)
        emitted=ungated.step(corrected,k,int(end),int(end))
        for e in emitted:
            i=int(e["pitch"])-40
            if e["type"]=="note_on" and i in candidates and native[i,0]<selection.onset:
                labeled_candidates.append((
                    int(end),int(e["pitch"]),
                    candidates[i],original_fundamental_evidence(x,i)))
        bass_events.extend(emitted)
    baseline_events.extend(baseline.panic(len(sound)))
    bass_events.extend(ungated.panic(len(sound)))
    _,matches_original=event_score(baseline_events,truth)
    _,matches_rescue=event_score(bass_events,truth)
    matched_original_truth={j for _,j in matches_original}
    matched_rescue_pred={i:j for i,j in matches_rescue}
    noteons=[e for e in bass_events if e["type"]=="note_on"]
    keymap={(int(e["decision_sample"]),int(e["pitch"])):i
            for i,e in enumerate(noteons)}
    base=[]
    full=[]
    target=[]
    for end,pitch,features,extra in labeled_candidates:
        j=matched_rescue_pred.get(keymap[(end,pitch)])
        valuable=(j is not None and j not in matched_original_truth)
        base.append(features);full.append(np.concatenate((features,extra)))
        target.append(int(valuable))
    return dict(name=spec["name"],composition=spec["composition"],sha256=sha,
                X=np.asarray(base,np.float32).reshape(-1,len(FEATURE_NAMES)),
                Xfull=np.asarray(full,np.float32).reshape(-1,len(ALL_NAMES)),
                y=np.asarray(target,np.int8),
                candidates=len(target),positives=sum(target))


def build_gate():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=.35,class_weight="balanced",
                                            max_iter=1200,random_state=641))


def train_only_group_threshold(x,y,groups):
    """Choose a threshold WITHOUT seeing any DEV labels.

    OOF scores for all 9 composition groups, no same composition in
    training and validation. Cost of FP is pre-fixed to 0.5 TP equivalent;
    this corresponds to requiring <2 extra FP / new TP.
    """
    if len(set(groups))<5 or int(y.sum())<8:
        raise ValueError("Train-only CV requires 5 compositions and positives")
    oof=np.zeros(len(y),np.float64)
    folds=[]
    for itrain,ival in GroupKFold(n_splits=5).split(x,y,groups):
        if len(set(groups[itrain])&set(groups[ival])):
            raise AssertionError("Composition leakage in train OOF")
        if np.unique(y[itrain]).size<2:
            raise ValueError("CV training fold missing positive class")
        head=build_gate()
        head.fit(x[itrain],y[itrain])
        oof[ival]=head.predict_proba(x[ival])[:,1]
        folds.append(dict(training_compositions=sorted(set(groups[itrain])),
                          validation_compositions=sorted(set(groups[ival])),
                          train_positives=int(y[itrain].sum()),
                          validation_positives=int(y[ival].sum())))
    trials=[]
    for t in (*GATES,1.01):
        accepted=oof>=t
        tp=int(np.sum(y[accepted]))
        fp=int(np.sum(accepted))-tp
        value=tp-fp/BUDGET_FP_PER_NEW_TP
        trials.append(dict(threshold=t,tp=tp,fp=fp,
                           accepted=tp+fp,
                           utility=value))
    # prefer fewer FPs on ties, and abstain completely if negative value
    best=max(trials,key=lambda z:(z["utility"],-z["fp"],z["threshold"]))
    return float(best["threshold"]),dict(folds=folds,candidates=trials)


def scores_for_validation(spec,root,archive,heads,config,gates):
    sound,truth,sha=pcm_truth(spec,root,archive)
    state={name:CausalSelection(config) for name in ("S614","bass_raw",*gates)}
    emitted={name:[] for name in state}
    admission=Counter()
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        predictions={}
        if candidates:
            raw=np.stack(list(candidates.values()))
            enhanced=np.stack([
                np.concatenate((candidates[i],original_fundamental_evidence(x,i)))
                for i in candidates
            ])
            for name,(model,threshold,extended) in gates.items():
                probs=model.predict_proba(enhanced if extended else raw)[:,1]
                predictions[name]={i:float(p) for i,p in zip(candidates,probs)}
        emitted["S614"].extend(state["S614"].step(native,k,int(end),int(end)))
        p,n=relay(native,bass,state["bass_raw"],{i:1. for i in candidates},0.)
        admission["bass_raw"]+=n
        emitted["bass_raw"].extend(state["bass_raw"].step(p,k,int(end),int(end)))
        for name,(model,threshold,extended) in gates.items():
            p,n=relay(native,bass,state[name],predictions.get(name,{}),threshold)
            admission[name]+=n
            emitted[name].extend(state[name].step(p,k,int(end),int(end)))
    groups=overlap_groups(truth)
    base_correct=None
    per={}
    for name,s in state.items():
        emitted[name].extend(s.panic(len(sound)))
        measure,pairs=event_score(emitted[name],truth)
        matched={j for _,j in pairs}
        if name=="S614":base_correct=matched
        per[name]=dict(
            score=measure,
            recovered_onset_overlap=len(matched&groups["onset_overlap"]),
            recovered_onset_K3plus=len(matched&groups["onset_K3plus"]),
            recovered_onset_K2=len(matched&groups["onset_K2"]),
            correct_truth=matched,
            accepted_proposals=int(admission[name]))
    for name in per:
        correct=per[name].pop("correct_truth")
        per[name]["new_vs_S614"]=len(correct-base_correct)
        per[name]["lost_vs_S614"]=len(base_correct-correct)
    return dict(name=spec["name"],composition=spec["composition"],sha256=sha,
                seconds=len(sound)/SR,truth=len(truth),policies=per)


def run(out):
    if out.exists():
        raise FileExistsError("S641 refuses to overwrite evidence")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    all_train=[s for s in protocol["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    train18=pick_original_guitar_training(protocol)
    dev=[s for s in protocol["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(all_train)==36 and len(train18)==18 and len(dev)==12
    assert len({s["composition"] for s in all_train})==9
    assert not {(s["corpus"],s["composition"]) for s in all_train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            z=protocol["guitar_archives"][key]
            download(z["url"],root/"data"/filename,z["size"],z["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            rows=[]
            for idx,s in enumerate(all_train,1):
                rec=train_record(s,root,archive,heads,cfg)
                rows.append(rec)
                print("S641 real TRAIN",idx,"/36",rec["name"],
                      "candidates",rec["candidates"],"positive",rec["positives"],flush=True)
            def extract(chosen,extended):
                values=np.concatenate([r["Xfull" if extended else "X"] for r in chosen])
                targets=np.concatenate([r["y"] for r in chosen])
                groups=np.concatenate([np.full(len(r["y"]),r["composition"],dtype=object) for r in chosen])
                return values,targets,groups
            names18={s["name"] for s in train18}
            xs18,ys18,groups18=extract([r for r in rows if r["name"] in names18],False)
            xs36,ys36,groups36=extract(rows,False)
            xf36,_,_=extract(rows,True)
            assert int(ys18.sum())>=10 and int(ys36.sum())>=int(ys18.sum())
            tbase,cvbase=train_only_group_threshold(xs18,ys18,groups18)
            t36,cv36=train_only_group_threshold(xs36,ys36,groups36)
            text,cvextra=train_only_group_threshold(xf36,ys36,groups36)
            print("S641 OOF",json.dumps(dict(
                old18=dict(positives=int(ys18.sum()),threshold=tbase),
                all36=dict(positives=int(ys36.sum()),threshold=t36),
                physical36=dict(positives=int(ys36.sum()),threshold=text))),flush=True)
            gbase=build_gate().fit(xs18,ys18)
            g36=build_gate().fit(xs36,ys36)
            gextra=build_gate().fit(xf36,ys36)
            # TRAIN-only OOF thresholds are frozen before touching DEV.
            gates=dict(
                original41_train18_oof=(gbase,tbase,False),
                original41_train36_oof=(g36,t36,False),
                physical65_train36_oof=(gextra,text,True),
                original41_train36_085=(g36,.85,False),
                physical65_train36_085=(gextra,.85,True),
                original41_train36_095=(g36,.95,False),
                physical65_train36_095=(gextra,.95,True),
            )
            output=[]
            for idx,s in enumerate(dev,1):
                r=scores_for_validation(s,root,archive,heads,cfg,gates)
                output.append(r)
                print("S641 real DEV",idx,"/12",r["name"],
                      "baseline",r["policies"]["S614"]["score"]["TP"],
                      "physical_oof",r["policies"]["physical65_train36_oof"]["score"]["TP"],
                      "physical_FP",r["policies"]["physical65_train36_oof"]["score"]["FP"],
                      flush=True)
    scores={}
    for name in output[0]["policies"]:
        rows_dev=[r["policies"][name] for r in output]
        score=summary([r["score"] for r in rows_dev])
        scores[name]=dict(
            score=score,
            onset_overlap_TP=sum(r["recovered_onset_overlap"] for r in rows_dev),
            onset_K2_TP=sum(r["recovered_onset_K2"] for r in rows_dev),
            onset_K3plus_TP=sum(r["recovered_onset_K3plus"] for r in rows_dev),
            new_events=sum(r["new_vs_S614"] for r in rows_dev),
            lost_events=sum(r["lost_vs_S614"] for r in rows_dev),
            accepted=sum(r["accepted_proposals"] for r in rows_dev))
    baseline=scores["S614"]["score"]
    assert all(baseline[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),baseline
    assert scores["bass_raw"]["score"]["TP"]==1044
    assert scores["bass_raw"]["score"]["FP"]==834
    for name,v in scores.items():
        s=v["score"]
        v["delta_TP"]=s["TP"]-baseline["TP"]
        v["delta_FP"]=s["FP"]-baseline["FP"]
        v["delta_F1_points"]=round(100*(s["F1"]-baseline["F1"]),6)
        print("S641 SCORES",name,
              "TP",s["TP"],"FP",s["FP"],"F1",round(s["F1"],7),
              "deltaPts",v["delta_F1_points"],
              "poly",v["onset_overlap_TP"],"K3+",v["onset_K3plus_TP"],flush=True)
    report=dict(
        status="S641_REAL_GUITARSET_CYCLE_AUDIT_PHYSICAL_FUNDAMENTAL_NOT_PROMOTED",
        pitch_round_trip_identity_is_not_independent_evidence=True,
        tested_independent_original_spectrum_fundamental_support=True,
        no_future_audio=True,original_K_unchanged=True,
        S614_unchanged=True,original_sha256=frozen["model_sha256"],
        training_only_GuitarSet_TRAIN=True,
        TRAIN36_sources=36,TRAIN18_control_sources=18,DEV12_sources=12,
        previously_exposed_DEV=True,TEST_never_opened=True,
        train_rows18=len(ys18),train_positives18=int(ys18.sum()),
        train_rows36=len(ys36),train_positives36=int(ys36.sum()),
        features_original41=list(FEATURE_NAMES),
        features_added24=list(ADDITIONAL_NAMES),
        threshold_selected_only_by_composition_group_cv=True,
        folds_and_thresholds=dict(
            original41_train18=dict(threshold=tbase,cv=cvbase),
            original41_train36=dict(threshold=t36,cv=cv36),
            physical65_train36=dict(threshold=text,cv=cvextra)),
        result=scores,
        per_source=output,
        seconds_elapsed=round(time.perf_counter()-start,2),
        automatic_promotion=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S641 FINAL",json.dumps({
        "train18_positives":report["train_positives18"],
        "train36_positives":report["train_positives36"],
        "elapsed_seconds":report["seconds_elapsed"],
        "policies":{k:dict(F1=v["score"]["F1"],TP=v["score"]["TP"],
                           FP=v["score"]["FP"],K3plus=v["onset_K3plus_TP"])
                    for k,v in scores.items()}
    }),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
