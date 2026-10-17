"""S667: strictly causal NOTE NoteOff release sweeps on actual GuitarSet PCM.

Use the frozen S614 streaming models and *same* native K, pitch/posterior
streams. Change ONLY independent release hysteresis settings through
CausalSelection/Selection; no code or checkpoints in production are touched.
Select policies using TRAIN calibration compositions alone, require held TRAIN
evidence before inspecting DEV. No TEST labels opened. No lookahead and
no backdating of emitted NoteOff.

Note: model was trained on GuitarSet TRAIN, so calibration TRAIN is not
prospective model evaluation. Strict-LIVE NoteOn/NoteOff scores on DEV are
also not interchangeable with offline S425-S666 GuitarSet cohorts.
"""
from __future__ import annotations
import argparse,collections,json,tempfile,time,zipfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from causal_note.strict_live import SR, Selection, load_heads
from scripts.s613_prepare_strict_live import download, digest, labels
from scripts.s613_train_strict_live import (
    real_notes,decode,event_score,offset_audit,summary,
)
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT, MODEL, raw_guitar_audio, GUITARSET_PINNED_CONTROL,
)
from scripts.s652_tone_filter_multiview import predict_causal_view

VARIANTS=("native",)+tuple(
    f"release_{steps}_off_{threshold:.2f}"
    for steps in (1,2,3,5,8,12)
    for threshold in (.15,.25,.35))
BASELINE="native"


def config(name,original):
    if name==BASELINE:
        return original
    _,frames,_,cutoff=name.split("_")
    return replace(original,release_frames=int(frames),
                   active_off=float(cutoff))


def exact_offset_summary(events,truth,pairs,seconds):
    """Offset strict ±50ms and flexible duration tolerance; NoteOff must
    be emitted acoustically, never EOF panic.
    """
    active={}
    intervals=[]
    for event in events:
        pitch=event["pitch"]
        if event["type"]=="note_on":
            active[pitch]=len(intervals)
            intervals.append({"on":event,"off":None})
        elif pitch in active:
            intervals[active.pop(pitch)]["off"]=event
    out=collections.Counter()
    errors=[]
    for pred,gt in pairs:
        note=truth[gt]
        finish=note["time"]+note["duration"]
        if finish>=seconds:
            continue
        out["eligible"]+=1
        off=intervals[pred]["off"]
        if off is None or off["reason"] in ("stream_end","shutdown","stream_error"):
            out["forced"]+=1
            continue
        err=(off["time"]-finish)*1000.
        errors.append(err)
        success50=abs(err)<=50.
        successflex=abs(err)<=max(50.,200*note["duration"])
        out["strict50"]+=int(success50)
        out["flex"]+=int(successflex)
        if note["duration"]>=1.0:
            out["long_eligible"]+=1
            out["long50"]+=int(success50)
        elif note["duration"]>=.5:
            out["medium_eligible"]+=1
            out["medium50"]+=int(success50)
    return dict(
       eligible=int(out["eligible"]),strict50=int(out["strict50"]),
       flexible_correct=int(out["flex"]),forced=int(out["forced"]),
       medium_eligible=int(out["medium_eligible"]),
       medium50=int(out["medium50"]),
       long_eligible=int(out["long_eligible"]),
       long50=int(out["long50"]),
       offset_error_median_ms=float(np.median(errors)) if errors else None,
    )


def record(spec,root,archive,heads,original):
    wave,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(wave)/SR)
    ends,post,k=predict_causal_view(wave,heads)
    records={}
    for name in VARIANTS:
        events=decode(ends,post,k,config(name,original),len(wave))
        noteon,pairs=event_score(events,truth)
        offset=exact_offset_summary(events,truth,pairs,len(wave)/SR)
        records[name]=dict(onset=noteon,offset=offset)
    return dict(name=spec["name"],composition=spec["composition"],
                pcm_sha256=sha,results=records)


def aggregate(rows):
    acc={}
    for name in VARIANTS:
        onset=summary([r["results"][name]["onset"] for r in rows])
        keys=("eligible","strict50","flexible_correct","forced",
              "long_eligible","long50","medium_eligible","medium50")
        parts=[r["results"][name]["offset"] for r in rows]
        offsets={key:sum(p[key] for p in parts) for key in keys}
        offsets.update(
            accuracy50=offsets["strict50"]/max(1,offsets["eligible"]),
            accuracy_flexible=offsets["flexible_correct"]/max(1,offsets["eligible"]),
            accuracy_long=offsets["long50"]/max(1,offsets["long_eligible"]),
        )
        acc[name]=dict(onset=onset,offset=offsets)
    return acc


def choose(cal,held):
    a=aggregate(cal);h=aggregate(held)
    base=a[BASELINE]
    safe=[]
    for key,info in a.items():
        if key==BASELINE:continue
        if (info["onset"]["F1"] < base["onset"]["F1"]-.002 or
            info["offset"]["accuracy_flexible"] <
            base["offset"]["accuracy_flexible"]-.002):
            continue
        safe.append(key)
    best=max(safe,key=lambda name:a[name]["offset"]["strict50"],
             default=BASELINE)
    ref=h[BASELINE]
    improvements=[]
    for comp in sorted({r["composition"] for r in held}):
        s=aggregate([r for r in held if r["composition"]==comp])
        improvements.append(dict(
            composition=comp,
            offset_delta=s[best]["offset"]["strict50"]-
                         s[BASELINE]["offset"]["strict50"],
            baseline_eligible=s[BASELINE]["offset"]["eligible"]))
    enough=sum(q["offset_delta"]>0 for q in improvements)>=2
    retain=(best!=BASELINE and enough and
        h[best]["offset"]["strict50"]>=ref["offset"]["strict50"]+5 and
        h[best]["onset"]["F1"]>=ref["onset"]["F1"]-.005)
    selected=best if retain else BASELINE
    return selected,dict(
      train_calibration_chosen=best,
      chosen_after_held_safety=selected,
      held_by_composition=improvements,
      train_calibration_scores=a,
      train_held_scores=h,
      no_DEV_selection=True,
      no_TEST_selection=True,
    )


def run(out):
    if out.exists():
        raise FileExistsError("S667 immutable evidence output already exists")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(train)==36 and len(dev)==12
    assert not {s["composition"] for s in train}&{s["composition"] for s in dev}
    comps=sorted({s["composition"] for s in train})
    assert len(comps)==9
    calibration=set(comps[:5])
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    original=Selection(**heads.metadata["selection"])
    assert original.release_frames==5
    assert abs(original.active_off-.25)<1e-10
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            src=protocol["guitar_archives"][key]
            download(src["url"],root/"data"/path,src["size"],src["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            training=[]
            for i,s in enumerate(train,1):
                training.append(record(s,root,arc,heads,original))
                print("S667 TRAIN",i,"/36",s["name"],flush=True)
            cal=[q for q in training if q["composition"] in calibration]
            held=[q for q in training if q["composition"] not in calibration]
            selected,proof=choose(cal,held)
            print("S667 FROZEN TRAIN SELECTION",
                  json.dumps(dict(selected=selected,cal=proof["train_calibration_chosen"])),
                  flush=True)
            testing=[]
            for i,s in enumerate(dev,1):
                testing.append(record(s,root,arc,heads,original))
                print("S667 DEV",i,"/12",s["name"],flush=True)
    scores=aggregate(testing)
    base=scores[BASELINE]["onset"]
    assert all(base[key]==value for key,value in GUITARSET_PINNED_CONTROL.items())
    for name,info in scores.items():
        print("S667 SCORE",name,
              "F1_NoteOn",round(info["onset"]["F1"],5),
              "offset_strict50",round(info["offset"]["accuracy50"],6),
              "offset_flexible",round(info["offset"]["accuracy_flexible"],6),
              "long",info["offset"]["long50"],
              "eligible",info["offset"]["eligible"],flush=True)
    report=dict(status="S667_STRICT_CAUSAL_NOTE_OFF_HYSTERESIS_ABLATION",
       native_K_posteriors_unchanged=True,
       original_heads_and_NoteOn_threshold_unchanged=True,
       no_lookahead=True,no_backdating=True,
       test_annotations_unopened=True,
       original_LIVE_unmodified=True,
       calibrated_only_on_TRAIN=True,
       native_selection_frozen_on_TRAIN=selected,
       no_official_promotion=True,
       noteon_can_change_due_to_active_state=True,
       does_not_claim_frozen_exact_K_predictions_changed=True,
       S425_historical_not_comparable_to_strict_LIVE=True,
       train_selection_proof=proof,
       DEV_results=scores,DEV_by_recording=testing,
       elapsed_seconds=round(time.perf_counter()-started,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S667 FINAL",json.dumps(dict(
      selected=selected,
      DEV_offset50={n:round(s["offset"]["accuracy50"],5) for n,s in scores.items()},
      DEV_flexible={n:round(s["offset"]["accuracy_flexible"],5) for n,s in scores.items()})),
      flush=True)

def self_test():
    native=Selection(onset=.6,active_off=.25,release_frames=5)
    candidate=config("release_2_off_0.35",native)
    assert candidate.release_frames==2 and candidate.active_off==.35
    assert native.active_off==.25 and native.release_frames==5
    print("S667 SELFTEST PASS: native state unmodified")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:self_test()
    else:
        assert a.out
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            run(a.out)
