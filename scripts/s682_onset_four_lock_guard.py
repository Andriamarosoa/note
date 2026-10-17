"""S682: causal NOTE onset selection under four independent non-regression locks.

Base: S630 NOTE posteriors, GuitarSet true MIC, original NoteOn, frozen K
head and S675 TRAIN-frozen release model. A candidate may change *only*
the NOTE selector's onset and active_on thresholds; the native neural
weights, pitch classes, K posterior, release head and musical timebase
never change. TRAIN CAL 5 composition families chooses a single proposal;
TRAIN HELD 4 other compositions vetoes it; only then do we open DEV.

A score improvement is not a promotion. A candidate must pass:
(1) same original 7-way K posterior bytes, (2) no worsening in decoded
    framewise Exact-K, (3) improved NoteOn+correct MIDI pitch ±80ms,
(4) zero regression in S675 conditional NoteOff ±50ms and full note F1.
TEST is unopened, GuitarSet DEV historically exposed; research only.

Outputs replay evidence & provenance, never edits installed LIVE runner.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib,json,tempfile,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from causal_note.strict_live import HOP,SR,Selection,load_heads,feature_track
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import decode,event_score,real_notes
from scripts.s672_causal_offset_timing_grid import (
    MODEL,ROOT,raw_guitar_audio,native_posteriors,
    note_instances,native_rhythm_clock,hazard_predictions,
    choose_offset
)

# Pre-register all threshold variants. Original stays an independent baseline.
ONSET_THRESHOLDS=(.50,.54,.57,.60,.63,.66,.70)
SUSTAIN_THRESHOLDS=(.45,.50,.55)
MIN_CAL_GAIN_F1=.002
BATCH_HOPS=4
OFFSET_GATE=.60
OFFSET_PERSISTENCE=1
PITCH_LOW=40
PITCH_HIGH=88


def configurations(base):
    proposals={"native":base}
    for onset in ONSET_THRESHOLDS:
        for sustain in SUSTAIN_THRESHOLDS:
            if onset==base.onset and sustain==base.active_on:continue
            name=f"onset_{onset:.2f}_sustain_{sustain:.2f}"
            proposals[name]=replace(base,onset=onset,active_on=sustain)
    return proposals


def decoded_exact_k(events,truth,ends):
    """Count sustained native pitch intervals per received source frame.

    This evaluates the *selected* pitch count, not just K-head logits,
    which remain strictly invariant by architecture.
    """
    n=len(ends)
    d=np.zeros(n+1,dtype=np.int32)
    state=set()
    for e in events:
        pitch=int(e["pitch"])
        frame=int(np.searchsorted(ends,int(e["decision_sample"]),side="left"))
        if frame>=n:continue
        if e["type"]=="note_on":
            if pitch not in state:
                state.add(pitch);d[frame]+=1
        elif pitch in state:
            state.remove(pitch);d[frame]-=1
    for p in state:
        # Unclosed notes remain active until EOF.
        pass
    estimated=np.minimum(6,np.maximum(0,np.cumsum(d[:-1])))
    gt_diff=np.zeros(n+1,dtype=np.int32)
    end_times=np.asarray(ends,dtype=np.float64)/SR
    for note in truth:
        start=int(np.searchsorted(end_times,float(note["time"]),side="left"))
        finish=int(np.searchsorted(end_times,float(note["time"])+float(note["duration"]),side="left"))
        if finish>start and start<n:
            gt_diff[start]+=1
            gt_diff[min(n,finish)]-=1
    gt=np.minimum(6,np.cumsum(gt_diff[:-1]))
    return dict(correct=int(np.count_nonzero(estimated==gt)),frames=n,
                accuracy=float(np.mean(estimated==gt)))


def native_metrics(events,truth,ends):
    onset,_=event_score(events,truth,tolerance=.080)
    k=decoded_exact_k(events,truth,ends)
    return dict(tp=onset["TP"],fp=onset["FP"],fn=onset["FN"],
                f1=float(onset["F1"]),count_correct=k["correct"],
                count_frames=k["frames"],count_accuracy=k["accuracy"])


def aggregate(rows,name):
    tp=sum(r[name]["tp"] for r in rows)
    fp=sum(r[name]["fp"] for r in rows)
    fn=sum(r[name]["fn"] for r in rows)
    kc=sum(r[name]["count_correct"] for r in rows)
    frames=sum(r[name]["count_frames"] for r in rows)
    return dict(tp=tp,fp=fp,fn=fn,onset_F1=2*tp/max(1,2*tp+fp+fn),
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                decoded_exact_k_accuracy=kc/max(1,frames),
                K_frames_correct=kc,K_frames_total=frames)


def qualify_native(base,proposal,grouped,min_gain=MIN_CAL_GAIN_F1):
    """Upstream K/decode and NoteOn veto, applied before S675 evaluation."""
    if proposal["onset_F1"] < base["onset_F1"]+min_gain:
        return False
    if proposal["K_frames_correct"] < base["K_frames_correct"]:
        return False
    if proposal["tp"] < base["tp"]:
        return False
    if grouped is not None and sum(
            by_case["proposal"]["onset_F1"]>=by_case["native"]["onset_F1"]
            for by_case in grouped.values())<2:
        return False
    return True


def record_native(spec,root,archive,heads,variants):
    audio,audio_sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,features=feature_track(audio,block_size=4096)
    p,k=native_posteriors(features,heads)
    prior_sha=digest(k.tobytes())
    results={}
    for name,config in variants.items():
        events=decode(ends,p,k,config,len(audio))
        results[name]=native_metrics(events,truth,ends)
    assert digest(k.tobytes())==prior_sha
    return dict(id=spec["name"],composition=spec["composition"],
                pcm_sha256=audio_sha,results=results)


def prepare_release(spec,root,archive,heads,selector):
    audio,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(audio)/SR)
    ends,features=feature_track(audio,block_size=4096)
    post,k=native_posteriors(features,heads)
    events=decode(ends,post,k,selector,len(audio))
    onset,pairs=event_score(events,truth,tolerance=.080)
    return dict(spec=spec,truth=truth,ends=ends,feats=features,
                post=post,k=k,events=events,onset=onset,pairs=pairs,
                notes=note_instances(events,ends),
                clock=native_rhythm_clock(events,ends),pcm_sha256=sha)


def score_release(item,model):
    proposals=hazard_predictions(item,model)
    positions=[
        choose_offset(item,i,proposals[0][i],proposals[1][i],
                      OFFSET_GATE,"hazard",batch_hops=BATCH_HOPS,
                      persistence=OFFSET_PERSISTENCE)
        for i in range(len(item["notes"]))
    ]
    valid=correct=full=long_eligible=long_correct=0
    for index,j in item["pairs"]:
        gt=item["truth"][j]
        target=float(gt["time"])+float(gt["duration"])
        if target >=item["ends"][-1]/SR:continue
        valid+=1
        hit=int(positions[index]<item["ends"][-1]/SR-1e-7 and
                abs(positions[index]-target)<=.050)
        correct+=hit
        if gt["duration"]>=1.0:
            long_eligible+=1;long_correct+=hit
    predicted=len(item["notes"])
    gt_count=len(item["truth"])
    # Full note requires pitch + onset within 80 ms AND offset within 50 ms.
    full=2*correct/max(1,predicted+gt_count)
    return dict(correct=correct,eligible=valid,accuracy=correct/max(1,valid),
                full_note_f1=full,predicted=predicted,ground_truth=gt_count,
                long_correct=long_correct,long_eligible=long_eligible,
                NoteOn_TP=item["onset"]["TP"],NoteOn_FP=item["onset"]["FP"],
                NoteOn_FN=item["onset"]["FN"])


def full_aggregate(rows,name):
    tp=sum(r[name]["correct"] for r in rows)
    eligible=sum(r[name]["eligible"] for r in rows)
    denom=sum(r[name]["predicted"]+r[name]["ground_truth"] for r in rows)
    return dict(correct=tp,eligible=eligible,
                conditional_offset_accuracy=tp/max(1,eligible),
                full_note_F1=2*tp/max(1,denom),
                long_correct=sum(r[name]["long_correct"] for r in rows),
                long_eligible=sum(r[name]["long_eligible"] for r in rows))


def evaluate_release(specs,root,archive,heads,configs,model):
    rows=[]
    for i,spec in enumerate(specs,1):
        scores={}
        for name,selector in configs.items():
            item=prepare_release(spec,root,archive,heads,selector)
            scores[name]=score_release(item,model)
            del item
        rows.append(dict(name=spec["name"],composition=spec["composition"],
                         results=scores))
        print("S682 S675 SCORE",i,"/",len(specs),spec["name"],
              json.dumps({k:{n:v[n] for n in ("correct","eligible",
                                               "full_note_f1")}
                          for k,v in scores.items()}),flush=True)
    return rows


def safe_full(baseline,candidate):
    """Strict non-regression: protects true NOTE and offset together."""
    return (candidate["correct"]>=baseline["correct"] and
            candidate["conditional_offset_accuracy"]>=
            baseline["conditional_offset_accuracy"]-1e-12 and
            candidate["full_note_F1"]>=baseline["full_note_F1"]-1e-12 and
            candidate["long_correct"]>=baseline["long_correct"])


def run(out,checkpoint):
    if out.exists():raise FileExistsError("Refusing to overwrite audit report")
    start=time.perf_counter()
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    proto=json.loads((ROOT/"protocol.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    trains=[s for s in proto["sources"]
            if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"]
            if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    comps=sorted({x["composition"] for x in trains})
    assert len(trains)==36 and len(comps)==9 and len(dev)==12
    calcomps=set(comps[:5])
    calibration=[s for s in trains if s["composition"] in calcomps]
    held=[s for s in trains if s["composition"] not in calcomps]
    original_model=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    original_selection=Selection(**original_model.metadata["selection"])
    configs=configurations(original_selection)
    candidate_names=list(configs.keys())
    frozen_fingerprint=digest(MODEL.read_bytes())
    # Checkpoint is a previously signed trusted S675 workflow artifact, not a
    # user-provided joblib. Do not re-tune release threshold from onset DEV.
    from causal_note.s672_live_offset import S672TrialNoteOff
    trained=S672TrialNoteOff.load(checkpoint,original_model)
    assert trained.gate==OFFSET_GATE and trained.persistence==OFFSET_PERSISTENCE
    report=dict(schema="S682_TRAIN_HELD_FOUR_LOCK_ONSET",release_checkpoint=
                dict(path=str(checkpoint),sha256=digest(checkpoint.read_bytes())),
                native_head_sha256=frozen_fingerprint,
                native_7way_K_head_frozen=True,
                native_note_pitch_classes_frozen=True,
                original_LIVE_unmodified=True,
                original_S675_NoteOff_head_frozen=True,
                TEST_labels_unopened=True,DEV_previously_exposed=True,
                training_cal_compositions=sorted(calcomps),
                held_train_compositions=sorted(set(comps)-calcomps),
                proposed_variants=candidate_names)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            arc=proto["guitar_archives"][key]
            download(arc["url"],root/"data"/filename,arc["size"],arc["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zip_audio:
            calrows=[]
            for i,spec in enumerate(calibration,1):
                row=record_native(spec,root,zip_audio,original_model,configs)
                calrows.append(row)
                print("S682 CAL",i,len(calibration),spec["name"],flush=True)
            cal_agg={name:aggregate([r["results"] for r in calrows],name)
                     for name in configs}
            group={comp:{name:aggregate([r["results"] for r in calrows
                                        if r["composition"]==comp],name)
                         for name in configs}
                   for comp in sorted(calcomps)}
            candidates=[]
            for name in candidate_names[1:]:
                proposal=cal_agg[name]
                positives=sum(group[comp][name]["onset_F1"]>=
                              group[comp]["native"]["onset_F1"]
                              for comp in group)
                if (proposal["onset_F1"]>=cal_agg["native"]["onset_F1"]+
                    MIN_CAL_GAIN_F1 and
                    proposal["K_frames_correct"]>=cal_agg["native"]["K_frames_correct"] and
                    proposal["tp"]>=cal_agg["native"]["tp"] and positives>=3):
                    candidates.append(name)
            candidates.sort(key=lambda name:(cal_agg[name]["onset_F1"],
                                              cal_agg[name]["K_frames_correct"]),
                            reverse=True)
            chosen=candidates[0] if candidates else "native"
            report["TRAIN_calibration"]={
                "native":cal_agg["native"],
                "top_qualifying_variants":candidates[:7],
                "top_candidates":{name:cal_agg[name] for name in candidates[:7]},
                "frozen_selection":chosen,
            }
            print("S682 CAL FROZEN",json.dumps(dict(choice=chosen,
                onset=cal_agg[chosen],baseline=cal_agg["native"])),flush=True)
            # No other variant can be chosen from held or DEV scores.
            selected={"native":configs["native"]}
            if chosen!="native":selected[chosen]=configs[chosen]
            heldrows=[]
            for i,spec in enumerate(held,1):
                heldrows.append(record_native(spec,root,zip_audio,
                                              original_model,selected))
                print("S682 HELD NATIVE",i,len(held),spec["name"],flush=True)
            held_agg={name:aggregate([r["results"] for r in heldrows],name)
                      for name in selected}
            held_safe=bool(chosen!="native" and
                held_agg[chosen]["onset_F1"]>=held_agg["native"]["onset_F1"] and
                held_agg[chosen]["K_frames_correct"]>=held_agg["native"]["K_frames_correct"] and
                held_agg[chosen]["tp"]>=held_agg["native"]["tp"])
            report["TRAIN_held_native"]=held_agg
            report["TRAIN_held_native_safe"]=held_safe
            print("S682 HELD FREEZE GUARD",held_safe,flush=True)
            if held_safe:
                offset_rows=evaluate_release(held,root,zip_audio,
                                             original_model,selected,trained.model)
                offset_scores={name:full_aggregate([x["results"]
                          for x in offset_rows],name) for name in selected}
                report["TRAIN_held_S675_offset"]=offset_scores
                offset_safe=safe_full(offset_scores["native"],
                                      offset_scores[chosen])
            else:
                offset_safe=False
            report["TRAIN_held_offset_safe"]=offset_safe
            # DEV: full audit on native and frozen proposal whether safe or
            # not. A rejected candidate remains SHADOW (never integrated).
            devrows=[]
            for i,spec in enumerate(dev,1):
                devrows.append(record_native(spec,root,zip_audio,
                                             original_model,selected))
                print("S682 DEV NATIVE",i,len(dev),spec["name"],flush=True)
            dev_agg={name:aggregate([r["results"] for r in devrows],name)
                     for name in selected}
            report["DEV_native_onset_and_K"]=dev_agg
            dev_release=evaluate_release(dev,root,zip_audio,
                                         original_model,selected,trained.model)
            full={name:full_aggregate([r["results"] for r in dev_release],
                                      name) for name in selected}
            report["DEV_S675_offset_and_full_note"]=full
            report["DEV_by_recording"]=dev_release
    if digest(MODEL.read_bytes())!=frozen_fingerprint:
        raise AssertionError("Frozen NOTE original model mutated")
    dev_safe=bool(chosen!="native" and held_safe and offset_safe and
        dev_agg[chosen]["onset_F1"]>dev_agg["native"]["onset_F1"] and
        dev_agg[chosen]["K_frames_correct"]>=dev_agg["native"]["K_frames_correct"] and
        safe_full(full["native"],full[chosen]))
    report["DEV_four_locks_passed"]=dev_safe
    report["promotion"]="NEVER_PROMOTE_RETROSPECTIVE_DEV_ONLY"
    report["proposed_selection_remains_shadow"]=True
    report["elapsed_seconds"]=round(time.perf_counter()-start,2)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S682 FINAL",json.dumps(dict(
        frozen_cal_choice=chosen,held_native_passed=held_safe,
        held_s675_passed=offset_safe,
        DEV_f1_native=dev_agg["native"]["onset_F1"],
        DEV_f1_candidate=dev_agg[chosen]["onset_F1"],
        DEV_K_native=dev_agg["native"]["decoded_exact_k_accuracy"],
        DEV_K_candidate=dev_agg[chosen]["decoded_exact_k_accuracy"],
        DEV_S675_original=full["native"],
        DEV_S675_candidate=full[chosen],
        four_locks_passed=dev_safe,
        reference_modified=False)),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--checkpoint",type=Path,required=True)
    args=ap.parse_args()
    with threadpool_limits(limits=1):
        run(args.out,args.checkpoint)
