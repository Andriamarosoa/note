"""S658 real natural GuitarSet audit of the S657 causal concurrent octave gate.

The candidate selection was developed on 8 constructed TRAIN E3/E4
pairs. Now evaluate on *natural* full GuitarSet recording tracks.
All variants use original native NOTE S614 pitch and K outputs and
live CausalSelection; the ONLY intervention is a lower-octave
proposal gated by native high onset history and filtered low evidence.

36 TRAIN across 9 compositions, five chosen calibration compositions
and four independent composition holdouts approve/reject the candidate;
12 DEV recorded in GuitarSet benchmark examined only after TRAIN
decision frozen. Never open TEST; avoid converting constructed 8/8
to general F1 claims. No mic production modification.
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

from causal_note.strict_live import SR,Selection,CausalSelection,load_heads
from scripts.s657_coincident_tone_lower_repair import condition,POLICIES
from scripts.s652_tone_filter_multiview import FILTERS,causal_lowpass,predict_causal_view
from scripts.s637_guitarset_three_view_groundtruth import ROOT,MODEL,raw_guitar_audio,GUITARSET_PINNED_CONTROL
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s638_guitarset_true_overlap import overlap_groups

MIN_CALIBRATION_GAIN=.003
MIN_HELD_GAIN=.002
MIN_POSITIVE_HELD=3


def measure_one(spec,root,archive,heads,config):
    wav,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(wav)/SR)
    ends,native,k=predict_causal_view(wav,heads)
    views=[]
    for fc in FILTERS.values():
        e,p,ki=predict_causal_view(causal_lowpass(wav,fc),heads)
        if not np.array_equal(ends,e):raise AssertionError("S658 Tone frame changed")
        views.append(p)
    mild,dark=views
    states={n:CausalSelection(config) for n in POLICIES}
    events={n:[] for n in POLICIES}
    ages=np.full(49,10**7,np.int32)
    for z,end in enumerate(ends):
        ages=np.minimum(ages+1,10**7)
        high=(native[z,:,0]>=config.onset)&(native[z,:,1]>=config.active_on)
        ages[high]=0
        for n in POLICIES:
            p=condition(native[z],mild[z],dark[z],config,ages,n,
                        states[n].armed,states[n].active)
            events[n].extend(states[n].step(p,k[z],int(end),int(end)))
    groups=overlap_groups(truth)
    scored={}
    base_truth=None
    for name,state in states.items():
        events[name].extend(state.panic(len(wav)))
        sc,pairs=event_score(events[name],truth)
        match={j for _,j in pairs}
        if name=="S614_original":base_truth=match
        scored[name]=dict(score=sc,
                          overlap_TP=len(match & groups["onset_overlap"]),
                          K2_TP=len(match & groups["onset_K2"]),
                          K3plus_TP=len(match & groups["onset_K3plus"]),
                          E2_TP=sum(truth[j]["pitch"]==40 for j in match),
                          E3_TP=sum(truth[j]["pitch"]==52 for j in match),
                          true_match=match)
    for n in POLICIES:
        match=scored[n].pop("true_match")
        scored[n]["new_true"]=len(match-base_truth)
        scored[n]["lost_true"]=len(base_truth-match)
    return dict(name=spec["name"],composition=spec["composition"],
                SHA256=sha,results=scored)


def aggregate(rows):
    out={}
    for name in POLICIES:
        sub=[row["results"][name] for row in rows]
        out[name]=dict(score=summary([v["score"] for v in sub]),
                       overlap_TP=sum(v["overlap_TP"] for v in sub),
                       K2_TP=sum(v["K2_TP"] for v in sub),
                       K3plus_TP=sum(v["K3plus_TP"] for v in sub),
                       E2_TP=sum(v["E2_TP"] for v in sub),
                       E3_TP=sum(v["E3_TP"] for v in sub),
                       new_gt=sum(v["new_true"] for v in sub),
                       lost_gt=sum(v["lost_true"] for v in sub))
    return out


def select_train_only(calib,held):
    a=aggregate(calib);b=aggregate(held)
    base=a["S614_original"]["score"]
    best=max(POLICIES,key=lambda name:(a[name]["score"]["F1"],
                                      -a[name]["score"]["FP"],
                                      name=="S614_original"))
    if a[best]["score"]["F1"]<base["F1"]+MIN_CALIBRATION_GAIN:
        best="S614_original"
    positive={}
    for comp in sorted({v["composition"] for v in held}):
        c=aggregate([v for v in held if v["composition"]==comp])
        positive[comp]=c[best]["score"]["F1"]-c["S614_original"]["score"]["F1"]
    accepted=(best!="S614_original" and
              b[best]["score"]["F1"]>=
                  b["S614_original"]["score"]["F1"]+MIN_HELD_GAIN
              and sum(x>0 for x in positive.values())>=MIN_POSITIVE_HELD)
    winner=best if accepted else "S614_original"
    return winner,dict(calibration_best=best,selected=winner,
       calibration_scores={n:r["score"]["F1"] for n,r in a.items()},
       held_scores={n:r["score"]["F1"] for n,r in b.items()},
       held_delta_by_composition=positive,
       fixed_gain_thresholds=[MIN_CALIBRATION_GAIN,MIN_HELD_GAIN],
       fixed_min_improved_compositions=MIN_POSITIVE_HELD)


def run(out):
    if out.exists():raise FileExistsError("S658 evidence immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    proto=json.loads((ROOT/"protocol.json").read_text())
    train=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in proto["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in proto["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12 and len({s["composition"] for s in train})==9
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    allcomps=sorted({s["composition"] for s in train})
    tr=set(allcomps[:5]);hv=set(allcomps[5:])
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    cfg=Selection(**heads.metadata["selection"])
    started=time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for key,name in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            d=proto["guitar_archives"][key]
            download(d["url"],root/"data"/name,d["size"],d["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as arc:
            a=[];b=[]
            for ix,s in enumerate(train,1):
                row=measure_one(s,root,arc,heads,cfg)
                (a if row["composition"] in tr else b).append(row)
                print("S658 TRAIN",ix,"/36",row["name"],
                      "origTP",row["results"]["S614_original"]["score"]["TP"],flush=True)
            selected,train_choice=select_train_only(a,b)
            print("S658 SELECT TRAIN",json.dumps(train_choice),flush=True)
            validated=[]
            for ix,s in enumerate(dev,1):
                row=measure_one(s,root,arc,heads,cfg)
                validated.append(row)
                print("S658 DEV",ix,"/12",row["name"],flush=True)
    scores=aggregate(validated)
    native=scores["S614_original"]["score"]
    assert all(native[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    for name,r in scores.items():
        sc=r["score"]
        r["delta_TP"]=sc["TP"]-native["TP"]
        r["delta_FP"]=sc["FP"]-native["FP"]
        r["delta_F1_points"]=round(100*(sc["F1"]-native["F1"]),6)
        print("S658 SCORE",name,"TP",sc["TP"],"FP",sc["FP"],
              "F1",sc["F1"],"poly",r["overlap_TP"],
              "K3",r["K3plus_TP"],flush=True)
    report=dict(status="S658_NATURAL_GUITARSET_LOWER_OCTAVE_TONE_GATE",
       original_checkpoint_sha=frozen["model_sha256"],
       no_tuned_DEV_policy=True,
       natural_sources=True,not_a_real_Stratocaster_Tone_measurement=True,
       original_K_not_modified=True,
       native_S614_model_and_live_unchanged=True,
       source_only_TRAIN=36,source_only_DEV=12,TEST_closed=True,
       DEV_previously_exposed=True,
       S657_controlled_development_pairs=8,
       selected_from_disjoint_TRAIN=selected,
       train_choice=train_choice,
       DEV_actual_onset_results=scores,
       per_recording=validated,
       elapsed_seconds=round(time.perf_counter()-started,2),
       promoted=False)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S658 FINAL",json.dumps(dict(selected=selected,
         F1={n:v["score"]["F1"] for n,v in scores.items()})),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(a.out)
