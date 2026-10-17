"""S655 causal per-hop tone routing by native NOTE polyphony confidence.

The S652 lowpass helps some K2/K3 frames but harms K1. We retain
the true original NOTE K posterior as routing evidence (not label),
and ask whether tone filtering should be ON or OFF separately at
each 256-sample decision step.

Eleven pre-registered deterministic routing variants, one selected
on 5 TRAIN compositions and confirmed on 4 different compositions,
then frozen DEV evaluation. All audio causal and same NoteOn clock.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import tempfile
import time
import zipfile
from collections import Counter

import numpy as np
from causal_note.strict_live import SR,Selection,CausalSelection,load_heads
from scripts.s652_tone_filter_multiview import (
    FILTERS,causal_lowpass,predict_causal_view,aggregate as _old_aggregate
)
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio,GUITARSET_PINNED_CONTROL
from scripts.s638_guitarset_true_overlap import overlap_groups

POLICIES=(
 "S614_original","S652_avg_three",
 "tone_poly_ge025","tone_poly_ge050","tone_poly_ge075",
 "tone_K3_ge025","tone_K3_ge050",
 "tone_poly_high_register","tone_poly_low_register",
 "tone_uncertain_count",
 "tone_poly_mild_half",
 "tone_poly_original_or_filtered",
)
TRAIN_MIN_DELTA=.004
HELD_MIN_DELTA=.002
IMPROVED_GROUPS=3


def local_probs(name,orig,mild,dark,k,config):
    """Select only from past/current K, not true-K annotation labels."""
    if name=="S614_original":return orig
    average=(orig+mild+dark)/3.
    if name=="S652_avg_three":return average
    ppoly=float(np.sum(k[2:]))
    p3=float(np.sum(k[3:]))
    if name=="tone_poly_ge025":return average if ppoly>=.25 else orig
    if name=="tone_poly_ge050":return average if ppoly>=.5 else orig
    if name=="tone_poly_ge075":return average if ppoly>=.75 else orig
    if name=="tone_K3_ge025":return average if p3>=.25 else orig
    if name=="tone_K3_ge050":return average if p3>=.5 else orig
    if name=="tone_uncertain_count":
        return average if float(np.max(k))<.55 else orig
    if name=="tone_poly_mild_half":
        return .5*(orig+mild) if ppoly>=.35 else orig
    if name=="tone_poly_low_register":
        out=orig.copy()
        if ppoly>=.35:out[:17]=average[:17]
        return out
    if name=="tone_poly_high_register":
        out=orig.copy()
        if ppoly>=.35:out[17:]=average[17:]
        return out
    if name=="tone_poly_original_or_filtered":
        out=orig.copy()
        if ppoly>=.35:
            mask=(orig[:,0]<config.onset)&(mild[:,0]>=.80)&(
                dark[:,0]>=.80)&(mild[:,1]>=config.active_on)&(
                dark[:,1]>=config.active_on)
            out[mask,0]=np.maximum(mild[mask,0],dark[mask,0])
            out[mask,1]=np.maximum(mild[mask,1],dark[mask,1])
        return out
    raise ValueError("Unknown S655 policy")


def measure(spec,root,archive,heads,selection):
    wav,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(wav)/SR)
    ends,native,k=predict_causal_view(wav,heads)
    views=[]
    for fc in FILTERS.values():
        e,pi,ki=predict_causal_view(causal_lowpass(wav,fc),heads)
        if not np.array_equal(e,ends):raise ValueError("K/Tone grid changed")
        views.append(pi)
    mild,dark=views
    selectors={name:CausalSelection(selection) for name in POLICIES}
    events={name:[] for name in POLICIES}
    for j,end in enumerate(ends):
        for name in POLICIES:
            p=local_probs(name,native[j],mild[j],dark[j],k[j],selection)
            events[name].extend(selectors[name].step(p,k[j],int(end),int(end)))
    gt_context=overlap_groups(truth)
    results={}
    seen_original=None
    for name,selector in selectors.items():
        events[name].extend(selector.panic(len(wav)))
        sc,pairs=event_score(events[name],truth)
        matched={idx for _,idx in pairs}
        if name=="S614_original":seen_original=matched
        results[name]=dict(score=sc,
            overlap_TP=len(matched & gt_context["onset_overlap"]),
            K2_TP=len(matched & gt_context["onset_K2"]),
            K3plus_TP=len(matched & gt_context["onset_K3plus"]),
            E2_TP=sum(truth[i]["pitch"]==40 for i in matched),
            E3_TP=sum(truth[i]["pitch"]==52 for i in matched),
            matched=matched)
    for row in results.values():
        m=row.pop("matched")
        row["new"]=len(m-seen_original)
        row["lost"]=len(seen_original-m)
    return dict(name=spec["name"],composition=spec["composition"],
                SHA256=sha,results=results)


def aggregate(records):
    scores={}
    for name in POLICIES:
        values=[r["results"][name] for r in records]
        scores[name]=dict(
            score=summary([v["score"] for v in values]),
            K2_TP=sum(v["K2_TP"] for v in values),
            K3plus_TP=sum(v["K3plus_TP"] for v in values),
            overlap_TP=sum(v["overlap_TP"] for v in values),
            E2_TP=sum(v["E2_TP"] for v in values),
            E3_TP=sum(v["E3_TP"] for v in values),
            new_vs_original=sum(v["new"] for v in values),
            lost_vs_original=sum(v["lost"] for v in values))
    return scores


def train_choice(calib,held):
    a=aggregate(calib);b=aggregate(held)
    first=a["S614_original"]["score"]
    second=b["S614_original"]["score"]
    best=max(POLICIES,key=lambda n:(a[n]["score"]["F1"],
                                   -a[n]["score"]["FP"],
                                   n=="S614_original"))
    if a[best]["score"]["F1"]<first["F1"]+TRAIN_MIN_DELTA:
        best="S614_original"
    positive={}
    for comp in sorted({r["composition"] for r in held}):
        sub=[r for r in held if r["composition"]==comp]
        results=aggregate(sub)
        positive[comp]=round(results[best]["score"]["F1"]-
                             results["S614_original"]["score"]["F1"],8)
    select=(best if best!="S614_original" and
            b[best]["score"]["F1"]>=second["F1"]+HELD_MIN_DELTA and
            sum(v>0 for v in positive.values())>=IMPROVED_GROUPS
            else "S614_original")
    return select,dict(calibration_best=best,selected=select,
            calibration_F1={n:r["score"]["F1"] for n,r in a.items()},
            held_F1={n:r["score"]["F1"] for n,r in b.items()},
            held_delta_by_composition=positive)


def run(out):
    if out.exists():raise FileExistsError("S655 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    keys=sorted(set(s["composition"] for s in train))
    a=set(keys[:5]);b=set(keys[5:])
    assert len(keys)==9 and not a&b
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    selection=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            d=proto["guitar_archives"][key]
            download(d["url"],root/"data"/filename,d["size"],d["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            calib=[];held=[]
            for ix,s in enumerate(train,1):
                r=measure(s,root,arc,heads,selection)
                (calib if r["composition"] in a else held).append(r)
                print("S655 REAL TRAIN",ix,"/36",r["name"],flush=True)
            selected,decision=train_choice(calib,held)
            print("S655 TRAIN CHOICE",json.dumps(decision),flush=True)
            devrows=[]
            for ix,s in enumerate(dev,1):
                r=measure(s,root,arc,heads,selection)
                devrows.append(r)
                print("S655 REAL DEV",ix,"/12",r["name"],flush=True)
    scores=aggregate(devrows)
    ref=scores["S614_original"]["score"]
    assert all(ref[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    for name,r in scores.items():
        sc=r["score"]
        r["delta_F1_points_vs_S614"]=round((sc["F1"]-ref["F1"])*100,6)
        print("S655 SCORE",name,"TP",sc["TP"],"FP",sc["FP"],
              "F1",sc["F1"],"K3",r["K3plus_TP"],flush=True)
    report=dict(status="S655_NOTE_TONE_CONDITIONED_BY_ORIGINAL_K",
       untouched_NOTE_pitch_K_and_live=True,
       native_original_K_router_only=True,
       extra_future_audio_ms=0,
       TRAIN_guitarset_compositions=9,
       TRAIN_calibration=sorted(a),TRAIN_held=sorted(b),
       DEV_previously_exposed=True,TEST_not_opened=True,
       chosen_only_by_TRAIN=selected,
       TRAIN_choice=decision,
       DEV=scores,per_recording=devrows,
       S641_historical_F1=0.6339930882814955,
       run_seconds=round(time.perf_counter()-start,2),promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S655 FINAL",json.dumps(dict(selected=selected,
      scores={n:r["score"]["F1"] for n,r in scores.items()})),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
