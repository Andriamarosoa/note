"""S640 — learned overlap-aware rescue selector trained on REAL GuitarSet TRAIN only.

The S639 threshold relay turns spectral candidate bass +12 evidence directly
into NoteOn without proof of an independent physical attack. This experiment
learns *when to abstain*, with complementary native harmonic energy, attack
novelty, cross-pitch competition and probability of spare K capacity.

Strict experimental discipline:
 - frozen S614 NOTE pitch/K heads, strict PCM up to current hop only;
 - GuitarSet metadata-separated TRAIN compositions (18 source tracks)
   for learning; GuitarSet DEV 12 recordings for one fixed comparison;
 - no TEST waveform, annotation, source or threshold tuning;
 - original NoteOn and K unchanged when rescue is off;
 - trained shadow candidate feature only, no modification to LIVE production.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import time
import zipfile
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_note.strict_live import SR,HOP,Selection,CausalSelection,load_heads
from causal_note.three_view_pitch import ThreeViewPitchProbe,remap_pitch_probabilities
from scripts.s637_guitarset_three_view_groundtruth import (
    FeatureCollector,ROOT,MODEL,raw_guitar_audio,
    GUITARSET_PINNED_CONTROL,
)
from scripts.s623_mixed_source_event_head import pick_original_guitar_training
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups

TRAIN_POLICY=("bass_080",.8,None,False,0.)
# fixed before examining DEV score, no posterior threshold optimization on DEV
FROZEN_GATES=(.35,.50,.70,.85)
PRIMARY_GATE=.70
BASS_PITCHES=24
FEATURE_NAMES=(
    "native_onset","native_sustain","native_release",
    "bass_onset","bass_sustain","bass_release",
    *(f"K_probability_{i}" for i in range(7)),
    "pitch_midi_normalized",
    "native_fundamental_short","native_fundamental_medium","native_fundamental_long",
    "native_fundamental_relative_short","native_fundamental_relative_medium","native_fundamental_relative_long",
    "native_spectral_score_short","native_spectral_score_medium","native_spectral_score_long",
    "native_first_lag_score_short","native_first_lag_score_medium","native_first_lag_score_long",
    "native_first_lag_fund_short","native_first_lag_fund_medium","native_first_lag_fund_long",
    "native_recent_avg_score_short","native_recent_avg_score_long",
    "native_flux_short","native_flux_medium","native_flux_long",
    "other_pitch_top_onset","other_pitch_second_onset",
    "octave_up_onset","octave_up_sustain",
    "native_octave_up_fundamental_short","native_octave_up_fundamental_long",
    "native_other_pitch_top_sustain",
    "native_pitch_onset_vs_other_top",
    "bass_pitch_onset_vs_native_octave_up",
)


def iter_acoustic(sound,heads,*,batch_frames=64):
    """Only past-captured real audio; pass native features to the selector."""
    capture=FeatureCollector()
    probe=ThreeViewPitchProbe(capture)
    for at in range(0,len(sound),batch_frames*HOP):
        record=probe.push(sound[at:at+batch_frames*HOP],start_sample=at)
        if not capture.features:
            continue
        feat=np.stack(capture.features).astype(np.float32)
        capture.features.clear()
        prediction,counts=heads.predict(feat.reshape(-1,49,126))
        prediction=prediction.reshape(len(feat),3,49,3)
        counts=counts.reshape(len(feat),3,7)
        for record_item,features,prob,ks in zip(record,feat,prediction,counts):
            valid=record_item["valid_rescue"]
            yield (record_item["decision_sample"],features[0],prob[0],ks[0],
                   remap_pitch_probabilities(prob[1],12) if valid
                   else np.zeros((49,3),np.float32),valid)


def feature_vector(x,native,bass,k,i):
    """Features at the present hop ONLY; no labels or future window.

    Harmonic rivalry is explicit: ratio/cross-pitch evidence is based on
    the ORIGINAL acoustic window at the same physical sample, so a
    +12 shifted spectrum cannot unilaterally prove a new fundamental.
    """
    onset=native[:,0];sustain=native[:,1]
    elsewhere=np.delete(onset,i)
    highest=np.partition(elsewhere,-2)[-2:]
    ix=i+12
    octave=native[ix] if ix<49 else np.zeros(3,np.float32)
    octave_fund=x[ix] if ix<49 else np.zeros(126,np.float32)
    vals=[
        *native[i],*bass[i],*k,float((i+40)/127.),
        *x[i,[0,8,16]],
        *x[i,[24,32,40]],
        *x[i,48:51],
        *x[i,54:57],
        *x[i,57:60],
        float(x[i,102]),float(x[i,104]),
        *x[i,120:123],
        float(highest[-1]),float(highest[-2]),
        float(octave[0]),float(octave[1]),
        float(octave_fund[0]),float(octave_fund[16]),
        float(np.max(np.delete(sustain,i))),
        float(onset[i]-highest[-1]),
        float(bass[i,0]-octave[0]),
    ]
    result=np.asarray(vals,np.float32)
    if result.shape!=(len(FEATURE_NAMES),):
        raise AssertionError(f"Wrong S640 ABI {result.shape} vs {len(FEATURE_NAMES)}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite present-only acoustic features")
    return result


def get_candidates(x,native,k,bass,valid,onset_threshold):
    if not valid:
        return {}
    allowed=np.flatnonzero(
        (np.arange(49)<BASS_PITCHES)
        & (native[:,0]<onset_threshold)
        & (bass[:,0]>=.8)
        & (bass[:,1]>=.5))
    return {int(i):feature_vector(x,native,bass,k,int(i)) for i in allowed}


def relay(native,bass,state,scores,threshold):
    """The ONLY difference from S614: gated bass proposals already at t."""
    result=native.copy()
    count=0
    for i,confidence in scores.items():
        if (confidence<threshold or state.active[i] or not state.armed[i]
                or native[i,0]>=state.config.onset):
            continue
        result[i,0]=state.config.onset
        result[i,1]=max(float(result[i,1]),float(bass[i,1]))
        count+=1
    return result,count


def pcm_truth(spec,root,archive):
    sound,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(sound)/SR)
    return sound,truth,sha


def training_source(spec,root,archive,heads,config):
    sound,truth,sha=pcm_truth(spec,root,archive)
    orig=CausalSelection(config)
    rescue=CausalSelection(config)
    orig_events=[];rescue_events=[]
    candidate_event_features=[]
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        orig_events.extend(orig.step(native,k,int(end),int(end)))
        proposal,_=relay(native,bass,rescue,
                         {i:1. for i in candidates},0.)
        emitted=rescue.step(proposal,k,int(end),int(end))
        for event in emitted:
            if event["type"]=="note_on":
                pitch=event["pitch"]-40
                if pitch in candidates and native[pitch,0]<config.onset:
                    prev_index=sum(e["type"]=="note_on" for e in rescue_events)
                    prev_index+=sum(e["type"]=="note_on" for e in emitted
                                    if e["type"]=="note_on"
                                    and e["decision_sample"]==event["decision_sample"]
                                    and e["pitch"]!=event["pitch"]
                                    and e["pitch"]<event["pitch"])
                    # Store event id later with actual ordered indices to
                    # avoid dependence on note_off ordering.
                    candidate_event_features.append((int(end),int(event["pitch"]),
                                                     candidates[pitch]))
        rescue_events.extend(emitted)
    orig_events.extend(orig.panic(len(sound)))
    rescue_events.extend(rescue.panic(len(sound)))
    _,pairs_orig=event_score(orig_events,truth)
    _,pairs_rescue=event_score(rescue_events,truth)
    orig_true={j for _,j in pairs_orig}
    matched_rescue={i:j for i,j in pairs_rescue}
    r_noteons=[e for e in rescue_events if e["type"]=="note_on"]
    # Each key (time,pitch) should be unique due to CausalSelection's
    # single emission per pitch per hop.
    key_to_ix={(e["decision_sample"],e["pitch"]):i
               for i,e in enumerate(r_noteons)}
    X=[];y=[];other_positive=0
    for end,pitch,vec in candidate_event_features:
        ix=key_to_ix[(end,pitch)]
        j=matched_rescue.get(ix)
        accepted=bool(j is not None and j not in orig_true)
        if j is not None and j in orig_true:
            other_positive+=1
        X.append(vec);y.append(int(accepted))
    return dict(name=spec["name"],composition=spec["composition"],
                source_sha256=sha,rows=np.asarray(X,np.float32).reshape(-1,len(FEATURE_NAMES)),
                targets=np.asarray(y,np.int8),positive_but_native_already_caught=other_positive,
                all_candidates=len(candidate_event_features),new_positives=int(sum(y)))


def validation_source(spec,root,archive,heads,config,gate):
    sound,truth,sha=pcm_truth(spec,root,archive)
    states={"S614_original":CausalSelection(config),
            "bass_080_ungated":CausalSelection(config)}
    for threshold in FROZEN_GATES:
        states[f"trained_gate_{threshold:g}"]=CausalSelection(config)
    events={name:[] for name in states}
    candidate_count=Counter()
    for end,x,native,k,bass,valid in iter_acoustic(sound,heads):
        candidates=get_candidates(x,native,k,bass,valid,config.onset)
        scores=(gate.predict_proba(np.stack(list(candidates.values())))[:,1]
                if candidates else np.empty(0,np.float32))
        index_scores=dict(zip(candidates,scores))
        events["S614_original"].extend(
            states["S614_original"].step(native,k,int(end),int(end)))
        for name,state in states.items():
            if name=="S614_original":continue
            floor=0. if name=="bass_080_ungated" else float(name.split("_")[-1])
            proposal,num=relay(native,bass,state,index_scores if floor
                                else {i:1. for i in candidates},floor)
            candidate_count[name]+=num
            events[name].extend(state.step(proposal,k,int(end),int(end)))
    results={}
    group=overlap_groups(truth)
    scored={}
    for name,state in states.items():
        events[name].extend(state.panic(len(sound)))
        scored[name]=event_score(events[name],truth)
    baseline_matched={j for _,j in scored["S614_original"][1]}
    for name,(metrics,pairs) in scored.items():
        matched={j for _,j in pairs}
        results[name]=dict(
            score=metrics,proposals_accepted=int(candidate_count[name]),
            true_onset_poly_recovered=len(matched&group["onset_overlap"]),
            true_K3plus_recovered=len(matched&group["onset_K3plus"]),
            newly_recovered_vs_original=len(matched-baseline_matched),
            lost_vs_original=len(baseline_matched-matched),
        )
    return dict(name=spec["name"],composition=spec["composition"],source_sha256=sha,
                truth_count=len(truth),real_audio_seconds=len(sound)/SR,results=results)


def run(out):
    if out.exists():raise FileExistsError("S640 evidence must be immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=pick_original_guitar_training(protocol)
    dev=[s for s in protocol["sources"]
         if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    held=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==18 and len(dev)==12
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+held}
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    since=time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);(root/"data").mkdir()
        archives=protocol["guitar_archives"]
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=archives[key]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zip_audio:
            records=[]
            for no,spec in enumerate(train,1):
                item=training_source(spec,root,zip_audio,heads,config)
                records.append(item)
                print("S640 train real guitar",no,"/",len(train),item["name"],
                      "proposals",item["all_candidates"],
                      "true_new",item["new_positives"],flush=True)
            x=np.concatenate([r["rows"] for r in records])
            y=np.concatenate([r["targets"] for r in records])
            if len(x)<50 or y.sum()<15 or len(y)-y.sum()<15:
                raise AssertionError("Too little actual GuitarSet train supervision")
            gate=make_pipeline(
                StandardScaler(),
                LogisticRegression(C=.35,class_weight="balanced",
                                   max_iter=1200,random_state=640))
            gate.fit(x,y)
            print("S640 FIT",len(y),"positive",int(y.sum()),
                  "negative",int(len(y)-y.sum()),
                  "crossing native",sum(r["positive_but_native_already_caught"] for r in records),
                  flush=True)
            validation=[]
            for no,spec in enumerate(dev,1):
                item=validation_source(spec,root,zip_audio,heads,config,gate)
                validation.append(item)
                print("S640 validate REAL GuitarSet",no,"/",len(dev),
                      item["name"],
                      "orig TP",item["results"]["S614_original"]["score"]["TP"],
                      "gate07 TP",item["results"]["trained_gate_0.7"]["score"]["TP"],
                      "gate07 FP",item["results"]["trained_gate_0.7"]["score"]["FP"],
                      flush=True)
    strategies={}
    for name in validation[0]["results"]:
        rows=[s["results"][name] for s in validation]
        metrics=summary([s["score"] for s in rows])
        strategies[name]=dict(
            score=metrics,
            poly_noteon_TP=sum(s["true_onset_poly_recovered"] for s in rows),
            K3plus_noteon_TP=sum(s["true_K3plus_recovered"] for s in rows),
            newly_recovered=sum(s["newly_recovered_vs_original"] for s in rows),
            lost=sum(s["lost_vs_original"] for s in rows),
            proposals_accepted=sum(s["proposals_accepted"] for s in rows),
        )
    baseline=strategies["S614_original"]["score"]
    assert all(baseline[k]==v for k,v in GUITARSET_PINNED_CONTROL.items()),baseline
    ungated=strategies["bass_080_ungated"]["score"]
    assert ungated["TP"]==1044 and ungated["FP"]==834,(ungated, "S637 non reproducible")
    for name,v in strategies.items():
        s=v["score"]
        v["delta_TP"]=s["TP"]-baseline["TP"]
        v["delta_FP"]=s["FP"]-baseline["FP"]
        v["delta_F1_points"]=round(100*(s["F1"]-baseline["F1"]),6)
        print("S640 RESULTS",name,"TP",s["TP"],"FP",s["FP"],
              "F1",round(s["F1"],5),
              "delta",v["delta_F1_points"],
              "poly_TP",v["poly_noteon_TP"],"K3plusTP",v["K3plus_noteon_TP"],flush=True)
    report=dict(
        status="S640_TRAIN_ONLY_LEARNED_HARMONIC_COMPETITION_GUITARSET_DEV",
        real_audio=True,test_wav_or_labels_never_read=True,
        annotation_as_training_target_only=True,
        physical_audio_lookahead_ms=0,native_K_unchanged=True,
        native_S614_frozen_sha256=frozen["model_sha256"],
        trained_only_on_GuitarSet_train_compositions=sorted(set(s["composition"] for s in train)),
        GuitarSet_dev_compositions=sorted(set(s["composition"] for s in dev)),
        train_recordings=len(train),dev_recordings=len(dev),
        train_rows=len(y),train_positives=int(y.sum()),
        train_negative_examples=int(len(y)-y.sum()),
        inference_features=list(FEATURE_NAMES),
        feature_width=len(FEATURE_NAMES),
        model="StandardScaler + LogisticRegression(class_weight=balanced,C=0.35)",
        primary_gate=PRIMARY_GATE,
        gates_fixed_before_DEV=True,
        dev_historically_exposed=True,
        not_promoted=True,
        scores=strategies,
        per_source=validation,
        train_source_counts=[dict(name=t["name"],new_positives=t["new_positives"],
                                  candidate_noteons=t["all_candidates"])
                             for t in records],
        elapsed_seconds=round(time.perf_counter()-since,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S640 FINAL",json.dumps(dict(
        train_rows=report["train_rows"],train_positives=report["train_positives"],
        elapsed_seconds=report["elapsed_seconds"],
        results={n:dict(TP=v["score"]["TP"],FP=v["score"]["FP"],F1=v["score"]["F1"],
                        polyTP=v["poly_noteon_TP"],K3plusTP=v["K3plus_noteon_TP"],
                        deltaF1=v["delta_F1_points"])
                 for n,v in strategies.items()})),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
