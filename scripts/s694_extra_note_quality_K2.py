"""S694: score *new* weak fundamental Onset proposals, not all 49x frames.

Original native NOTE Onsets/NoteOffs are bit-for-bit kept; no native
events are suppressed. Generate optional S686 acoustic-rescue events, and
add only disjoint new pitch intervals whose per-event past-only acoustic
features are judged genuine by a TRAIN-only random forest.

The candidate never alters original K posteriors, pitch, sustain or S675
weights. Exact-K *decoded intervals* and S675 full-note must still pass
their guards. TRAIN composition-held policy is frozen before DEV. Test
unopened; DEV historically visible; no production promotion.
"""
from __future__ import annotations
import argparse,json,tempfile,zipfile,time
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    ROOT,MODEL,raw_guitar_audio,native_posteriors,
    note_instances,native_rhythm_clock)
# The original S683 acoustic-feature helper is embedded below so that
# this branch does not rely on another unmerged experimental branch.
from scripts.s686_causal_fundamental_novelty import rescue as novelty_rescue
from scripts.s682_onset_four_lock_guard import (
    native_metrics,aggregate,score_release,full_aggregate,safe_full)

FEATURE_CHOICES=np.array([0,1,2,8,9,10,16,17,18,24,25,26,
    54,55,62,63,70,71,78,79,84,85,90,91,96,97,104,105,
    112,113,120,121],np.int64)
def onset_feature_rows(features,post,k,frame,pitch):
    """Pure causal features: current and historical frames only.

    frame and pitch must be 1D integer arrays of equal length;
    source NOTE feature dim is the frozen 126D native ABI.
    """
    frame=np.asarray(frame,dtype=np.int64)
    pitch=np.asarray(pitch,dtype=np.int64)
    if frame.shape!=pitch.shape or np.any(pitch<0) or np.any(pitch>48):
        raise ValueError("Pitch/frame dimensions outside NOTE ABI")
    if len(frame)==0:
        return np.empty((0,75),np.float32)
    now=features[frame,pitch]
    selected=now[:,FEATURE_CHOICES]
    old8=features[np.maximum(0,frame-8),pitch][:,FEATURE_CHOICES[:12]]
    now12=now[:,FEATURE_CHOICES[:12]]
    t=post[frame,pitch]
    lag2=post[np.maximum(0,frame-2),pitch]
    lag8=post[np.maximum(0,frame-8),pitch]
    lag24=post[np.maximum(0,frame-24),pitch]
    octave=post[frame,np.minimum(48,pitch+12),:2]
    low=post[frame,np.maximum(0,pitch-12),:2]
    krow=k[frame]
    globalpeak=np.max(post[frame,:,0],axis=1)[:,None]
    x=np.column_stack((selected,now12-old8,
                       t,lag2,lag8,lag24,t-lag2,t-lag8,
                       octave,low,krow,globalpeak,
                       pitch.astype(np.float32)/127.))
    if not np.isfinite(x).all():
        raise ValueError("Non-finite learned onset features")
    return x.astype(np.float32)



def expanded_train_specs(original, archive, annotation_zip):
    """Use additional authentic players 00..03 ONLY on preexisting TRAIN
    composition groups. Never use player 04 DEV or player 05 TEST.
    Reject accidental cross-composition exposure.
    """
    wav_names=set(archive.namelist())
    ann_names=set(annotation_zip.namelist())
    candidates={}
    for spec in original:
        stem=spec["name"][3:]
        for player in range(4):
            name=f"{player:02d}_"+stem
            audio=name+"_mic.wav"
            annotation=name+".jams"
            if audio not in wav_names or annotation not in ann_names:
                continue
            row=dict(spec,name=name,audio=audio,annotation=annotation)
            candidates[name]=row
    actual=list(candidates.values())
    if len(actual)<len(original) or len(actual)>len(original)*4:
        raise AssertionError("Player augmentation contract or archive paths failed")
    if {x["composition"] for x in actual}!={x["composition"] for x in original}:
        raise AssertionError("Player augmentation changed composition split")
    if any(x["name"].startswith(("04_","05_")) for x in actual):
        raise AssertionError("Unseen DEV/TEST player contaminated training")
    return sorted(actual,key=lambda x:(x["composition"],x["name"]))


S686_BASE="rescue_0.30_rise_0.12"
POLICIES=("native","gate_0.55_k2_0.10","gate_0.55_k2_0.25",
    "gate_0.65_k2_0.10","gate_0.65_k2_0.25","gate_0.65_k2_0.40",
    "gate_0.72_k2_0.10","gate_0.72_k2_0.25","gate_0.72_k2_0.40")
MATCH_NATIVE_MS=120
MAX_NATIVE_COLLISION_MS=35


def paired_intervals(events):
    active={}
    intervals=[]
    for e in events:
        pitch=int(e["pitch"])
        if e["type"]=="note_on":
            active[pitch]=e
        elif e["type"]=="note_off" and pitch in active:
            on=active.pop(pitch)
            if int(e["decision_sample"])>int(on["decision_sample"]):
                intervals.append((on,e))
    return intervals


def new_intervals(original,boosted,ends):
    """Potentially new source Onsets with complete causal NoteOff pair.
    Do not alter existing events or double trigger an already active pitch.
    """
    original_on=[e for e in original if e["type"]=="note_on"]
    original_pairs=paired_intervals(original)
    out=[]
    for on,off in paired_intervals(boosted):
        pitch=on["pitch"];start=int(on["decision_sample"])
        end=int(off["decision_sample"])
        if any(n["pitch"]==pitch and
               abs(int(n["decision_sample"])-start)<=MATCH_NATIVE_MS*SR/1000.
               for n in original_on):
            continue
        if any(start<int(y["decision_sample"])+MAX_NATIVE_COLLISION_MS*SR/1000.
               and end>int(x["decision_sample"])-MAX_NATIVE_COLLISION_MS*SR/1000.
               and x["pitch"]==pitch for x,y in original_pairs):
            continue
        out.append((on,off))
    return out


def proposal_features(item,proposals):
    if not proposals:return np.empty((0,75),np.float32)
    frames=np.asarray([min(len(item["ends"])-1,
       np.searchsorted(item["ends"],int(x["decision_sample"]),"left"))
       for x,_ in proposals],np.int64)
    pitch=np.asarray([int(x["pitch"])-40 for x,_ in proposals],np.int64)
    return onset_feature_rows(item["feat"],item["p"],item["k"],frames,pitch)


def accepted_merge(original,proposals,prob,gate):
    if len(proposals)!=len(prob):raise ValueError("Probability/proposal mismatch")
    chosen=[(on,off) for (on,off),p in zip(proposals,prob) if p>=gate]
    existing=list(paired_intervals(original))
    extra=[]
    for on,off in sorted(chosen,key=lambda z:int(z[0]["decision_sample"])):
        start=int(on["decision_sample"]);end=int(off["decision_sample"]);pitch=on["pitch"]
        if any(a["pitch"]==pitch and
               start<int(b["decision_sample"]) and
               end>int(a["decision_sample"]) for a,b in existing+extra):
            continue
        extra.append((on,off))
    extra_events=[event for pair in extra for event in pair]
    # Do not reorder simultaneous *native* events. Exact original event
    # sequence remains a subsequence, including original restrikes.
    result=original+extra_events
    source_index={id(e):i for i,e in enumerate(original)}
    extra_index={id(e):i for i,e in enumerate(extra_events)}
    result.sort(key=lambda e:(int(e["decision_sample"]),
                              0 if id(e) in source_index else 1,
                              source_index.get(id(e),extra_index.get(id(e),0))))
    native_ids=set(source_index)
    preserved=[e for e in result if id(e) in native_ids]
    if preserved!=original:
        raise AssertionError("S694 modified an original NOTE event sequence")
    return result


def prepare(spec,root,archive,heads,selection):
    wave,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,feat=feature_track(wave,block_size=4096)
    p,k=native_posteriors(feat,heads)
    original=decode(ends,p,k,selection,len(wave))
    p_plus=novelty_rescue(p,feat,S686_BASE,selection.onset)
    boosted=decode(ends,p_plus,k,selection,len(wave))
    if not np.array_equal(p_plus[:,:,1:],p[:,:,1:]):
        raise AssertionError("Native sustain/release mutated")
    proposals=new_intervals(original,boosted,ends)
    return dict(name=spec["name"],composition=spec["composition"],
                split=spec["split"],sha=sha,truth=truth,ends=ends,
                feat=feat,p=p,k=k,original=original,boosted=boosted,
                proposals=proposals,
                X=proposal_features(dict(ends=ends,feat=feat,p=p,k=k),proposals))


def training_labels(item):
    _,pairs=event_score(item["boosted"],item["truth"],tolerance=.080)
    boosted_on=[e for e in item["boosted"] if e["type"]=="note_on"]
    good={id(boosted_on[i]) for i,_ in pairs}
    return np.asarray([int(id(on) in good) for on,_ in item["proposals"]],
                      dtype=np.int8)


def train(records):
    samples=[];targets=[]
    for item in records:
        if len(item["X"]):
            samples.append(item["X"]);targets.append(training_labels(item))
    if not samples:raise AssertionError("No potential weak notes")
    X=np.vstack(samples);y=np.concatenate(targets)
    if len(y)<40 or int(y.sum())<5 or int((1-y).sum())<10:
        raise AssertionError("Insufficient counterfactual onset evidence")
    model=RandomForestClassifier(
        n_estimators=170,max_depth=6,min_samples_leaf=3,
        max_features="sqrt",class_weight="balanced_subsample",
        n_jobs=1,random_state=690)
    model.fit(X,y)
    return model,dict(rows=int(len(y)),true_new=int(y.sum()),
                      false_new=int((1-y).sum()),
                      features=int(X.shape[1]))


def accepted_by_quality_and_K(item,prob,policy):
    """Proof available on the source Onset frame only: native predicted
    K2..K6, never annotated K nor a future note or a DEV-selected gate."""
    if policy not in POLICIES:raise ValueError("Not preregistered")
    if policy=="native":return np.zeros(len(prob),bool)
    parts=policy.split("_")
    threshold=float(parts[1]);K2_threshold=float(parts[3])
    frames=np.asarray([
        min(len(item["ends"])-1,
            int(np.searchsorted(item["ends"],
                                int(on["decision_sample"]),"left")))
        for on,_ in item["proposals"]],np.int64)
    native_predicted_K2=item["k"][frames,2:].sum(axis=1) if len(frames) else np.empty(0)
    return (np.asarray(prob)>=threshold)&(native_predicted_K2>=K2_threshold)


def measure(item,model,policies,offset_model=None):
    predicted=(model.predict_proba(item["X"])[:,1]
               if len(item["X"]) else np.empty(0,np.float64))
    native=item["original"];metrics={};full={}
    for name in policies:
        # Gate probabilities by native K2+ posterior. Feed booleans to
        # accepted_merge with threshold .5 so all source MIDI stays intact.
        verdict=accepted_by_quality_and_K(item,predicted,name)
        stream=native if name=="native" else accepted_merge(
            native,item["proposals"],verdict.astype(np.float32),.5)
        metrics[name]=native_metrics(stream,item["truth"],item["ends"])
        if offset_model is not None:
            onset,pairs=event_score(stream,item["truth"],tolerance=.080)
            row=dict(truth=item["truth"],ends=item["ends"],feats=item["feat"],
                post=item["p"],k=item["k"],events=stream,onset=onset,
                pairs=pairs,notes=note_instances(stream,item["ends"]),
                clock=native_rhythm_clock(stream,item["ends"]))
            full[name]=score_release(row,offset_model)
    return metrics,full


def aggregate_guard(rows,policies):
    sums={name:aggregate([r["scores"] for r in rows],name) for name in policies}
    original=sums["native"];groups=sorted({x["composition"] for x in rows})
    qualified=[]
    for name in policies[1:]:
        s=sums[name]
        good=sum(
            aggregate([x["scores"] for x in rows if x["composition"]==comp],
                      name)["onset_F1"] >=
            aggregate([x["scores"] for x in rows if x["composition"]==comp],
                      "native")["onset_F1"]
            for comp in groups)
        deltaTP=s["tp"]-original["tp"];deltaFP=s["fp"]-original["fp"]
        if (deltaTP>=1 and deltaFP<=deltaTP and
            s["K_frames_correct"]>=original["K_frames_correct"] and
            s["onset_F1"]>original["onset_F1"] and good>=2):
            qualified.append(name)
    selected=max(qualified,key=lambda name:sums[name]["onset_F1"]) if qualified else "native"
    return selected,sums,qualified


def run(out,checkpoint):
    if out.exists():raise FileExistsError("S694 immutable report")
    started=time.perf_counter()
    freeze=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==freeze["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==freeze["protocol_sha256"]
    train_specs=[s for s in proto["sources"]
                 if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev_specs=[s for s in proto["sources"]
               if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({s["composition"] for s in train_specs})
    assert len(comps)==9 and len(train_specs)==36 and len(dev_specs)==12
    train_comps=set(comps[:5]);held_comps=set(comps[5:])
    heads=load_heads(MODEL,expected_sha256=freeze["model_sha256"])
    selection=Selection(**heads.metadata["selection"])
    from causal_note.s672_live_offset import S672TrialNoteOff
    frozen_release=S672TrialNoteOff.load(checkpoint,heads)
    report=dict(schema="S694_NATIVE_PRESERVED_ADDED_ONSET_QUALITY",
       native_model_sha256=freeze["model_sha256"],
       S675_checkpoint_sha256=digest(checkpoint.read_bytes()),
       predeclared_S686_proposal=S686_BASE,
       native_onsets_and_noteoffs_all_preserved=True,
       native_K_and_pitch_heads_unchanged=True,
       S675_release_unchanged=True,no_future_audio_in_inference=True,
       new_notes_gated_by_frozen_predicted_K2plus_only=True,
       TRAIN_groups=sorted(train_comps),TRAIN_held_groups=sorted(held_comps),
       TEST_unopened=True,DEV_previously_exposed=True,
       no_live_code_modified=True)
    policies=POLICIES
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            source=proto["guitar_archives"][key]
            download(source["url"],root/"data"/name,source["size"],source["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive, zipfile.ZipFile(
                 root/"data/GuitarSet_annotations.zip") as annotations:
            train_original=[s for s in train_specs if s["composition"] in train_comps]
            train_expanded=expanded_train_specs(train_original,archive,annotations)
            report["expanded_authentic_TRAIN_sources"]=len(train_expanded)
            report["extra_TRAIN_players"]=list(range(4))
            print("S694 REAL TRAIN SOURCES",len(train_expanded),flush=True)
            trainrows=[]
            for i,s in enumerate(train_expanded,1):
                trainrows.append(prepare(s,root,archive,heads,selection))
                print("S694 TRAIN",i,"/",len(train_expanded),s["name"],len(trainrows[-1]["proposals"]),flush=True)
            model,trained=train(trainrows)
            report["TRAIN_head"]=trained
            del trainrows
            heldrows=[]
            for i,s in enumerate([s for s in train_specs if s["composition"] in held_comps],1):
                item=prepare(s,root,archive,heads,selection)
                metrics,_=measure(item,model,policies)
                heldrows.append(dict(name=s["name"],composition=s["composition"],
                                     scores=metrics))
                print("S694 HELD",i,"/16",s["name"],len(item["proposals"]),flush=True)
            selected,held,qualified=aggregate_guard(heldrows,policies)
            report["TRAIN_held_selection"]=selected
            report["TRAIN_held_onset_and_K"]=held
            report["TRAIN_held_qualified"]=qualified
            print("S694 TRAIN FROZEN",selected,flush=True)
            chosen=("native",) if selected=="native" else ("native",selected)
            heldfull=[]
            for i,s in enumerate([s for s in train_specs if s["composition"] in held_comps],1):
                item=prepare(s,root,archive,heads,selection)
                _,offsets=measure(item,model,chosen,frozen_release.model)
                heldfull.append(offsets)
                print("S694 HELD S675",i,"/16",s["name"],flush=True)
            holdoff={name:full_aggregate(heldfull,name) for name in chosen}
            allowed=bool(selected!="native" and safe_full(holdoff["native"],holdoff[selected]))
            report["TRAIN_held_S675"]=holdoff;report["TRAIN_held_four_lock"]=allowed
            devrows=[];devfull=[]
            for i,s in enumerate(dev_specs,1):
                item=prepare(s,root,archive,heads,selection)
                metrics,offsets=measure(item,model,chosen,frozen_release.model)
                devrows.append(dict(name=s["name"],composition=s["composition"],
                                    scores=metrics,extras=len(item["proposals"])))
                devfull.append(offsets)
                print("S694 DEV",i,"/12",s["name"],flush=True)
    stats={name:aggregate([r["scores"] for r in devrows],name) for name in chosen}
    off={name:full_aggregate(devfull,name) for name in chosen}
    ok=bool(allowed and selected!="native" and
       stats[selected]["onset_F1"]>stats["native"]["onset_F1"] and
       stats[selected]["K_frames_correct"]>=stats["native"]["K_frames_correct"] and
       safe_full(off["native"],off[selected]))
    report["DEV_onset_K"]=stats;report["DEV_S675_full_note"]=off
    report["DEV_by_recording"]=devrows
    report["all_four_locks"]=ok
    report["promoted"]=False
    report["elapsed_s"]=round(time.perf_counter()-started,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S694 FINAL",json.dumps(dict(selection=selected,
       training=trained,original=stats["native"],candidate=stats[selected],
       S675_original=off["native"],S675_candidate=off[selected],
       four_locks=ok)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--checkpoint",required=True,type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.out,args.checkpoint)
