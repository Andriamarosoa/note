"""S658 — natural GuitarSet independent validation of S657 coincident-Tone lower-note gate.

The S657 candidate was devised using only 8 *constructed* E3+E4
mixtures. S658 tests that very SAME precommitted 5 gate rules
on 36 naturally played GuitarSet TRAIN recordings and then on
12 naturally played DEV recordings, with no changes to model weights.

Instruments are original GuitarSet, not documented Stratocaster Tone
knob settings. Causal digital 2800/1100Hz first-order lowpass only.
All prediction probabilities K come from the original signal. Each
selector has its own CausalSelection; no future audio or backdating.
Gate selection is based on 5 TRAIN composition calibration and must
confirm on the other 4 TRAIN compositions; never select on DEV.
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
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT,MODEL,raw_guitar_audio,GUITARSET_PINNED_CONTROL)
from scripts.s638_guitarset_true_overlap import overlap_groups
from scripts.s652_tone_filter_multiview import (
    causal_lowpass,predict_causal_view,FILTERS)
from scripts.s657_coincident_tone_lower_repair import (
    condition,POLICIES)

CALIB_GAIN_MIN=.002
HELD_GAIN_MIN=.002
MIN_SUCCESS_COMPOSITIONS=3


def decode_natural(spec,root,archive,heads,config):
    wav,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(wav)/SR)
    ends,native,original_k=predict_causal_view(wav,heads)
    views=[]
    for cutoff in FILTERS.values():
        tone=causal_lowpass(wav,cutoff)
        e,p,_=predict_causal_view(tone,heads)
        if not np.array_equal(ends,e):
            raise ValueError("S658 Tone view changed native sample clock")
        views.append(p)
    mild,dark=views
    states={name:CausalSelection(config) for name in POLICIES}
    events={name:[] for name in POLICIES}
    recent=np.full(49,10**7,np.int32)
    # Only original prediction can change age of *native* higher note.
    for j,end in enumerate(ends):
        recent=np.minimum(recent+1,10**7)
        fired=(native[j,:,0]>=config.onset)&(
               native[j,:,1]>=config.active_on)
        recent[fired]=0
        for name,selector in states.items():
            probs=condition(native[j],mild[j],dark[j],config,recent,
                            name,selector.armed,selector.active)
            events[name].extend(selector.step(
                probs,original_k[j],int(end),int(end)))
    grouped_truth=overlap_groups(truth)
    output={}
    original_matched=None
    for name,selector in states.items():
        events[name].extend(selector.panic(len(wav)))
        scores,pairs=event_score(events[name],truth)
        matched={j for _,j in pairs}
        if name=="S614_original":
            original_matched=matched
        correct=Counter(int(truth[t]["pitch"]) for t in matched)
        output[name]=dict(
            score=scores,
            overlap_TP=len(matched&grouped_truth["onset_overlap"]),
            K2_TP=len(matched&grouped_truth["onset_K2"]),
            K3plus_TP=len(matched&grouped_truth["onset_K3plus"]),
            E2_TP=int(correct[40]),
            E3_TP=int(correct[52]),
            matched=matched)
    for item in output.values():
        m=item.pop("matched")
        item["new_onsets_vs_S614"]=len(m-original_matched)
        item["lost_onsets_vs_S614"]=len(original_matched-m)
    return dict(name=spec["name"],composition=spec["composition"],
                audio_sha256=sha,annotations=len(truth),stats=output)


def aggregate(rows):
    result={}
    for name in POLICIES:
        vals=[r["stats"][name] for r in rows]
        result[name]=dict(
            score=summary([v["score"] for v in vals]),
            overlap_TP=sum(v["overlap_TP"] for v in vals),
            K2_TP=sum(v["K2_TP"] for v in vals),
            K3plus_TP=sum(v["K3plus_TP"] for v in vals),
            E2_TP=sum(v["E2_TP"] for v in vals),
            E3_TP=sum(v["E3_TP"] for v in vals),
            repaired=sum(v["new_onsets_vs_S614"] for v in vals),
            regressed=sum(v["lost_onsets_vs_S614"] for v in vals))
    return result


def choose_by_TRAIN(calibration,held):
    calib=aggregate(calibration);check=aggregate(held)
    base=calib["S614_original"]["score"]
    choice=max(POLICIES,key=lambda n:(calib[n]["score"]["F1"],
                                    -calib[n]["score"]["FP"],
                                    n=="S614_original"))
    if calib[choice]["score"]["F1"]<base["F1"]+CALIB_GAIN_MIN:
        choice="S614_original"
    held_compositions={}
    for comp in sorted({r["composition"] for r in held}):
        group=[r for r in held if r["composition"]==comp]
        scores=aggregate(group)
        held_compositions[comp]=dict(
            original_f1=scores["S614_original"]["score"]["F1"],
            selected_f1=scores[choice]["score"]["F1"],
            delta_TP=scores[choice]["score"]["TP"]-
                     scores["S614_original"]["score"]["TP"],
            delta_FP=scores[choice]["score"]["FP"]-
                     scores["S614_original"]["score"]["FP"])
    positive=sum(x["selected_f1"]>x["original_f1"] for x in
                 held_compositions.values())
    confirmed=(choice!="S614_original" and
               check[choice]["score"]["F1"]>=
               check["S614_original"]["score"]["F1"]+HELD_GAIN_MIN
               and positive>=MIN_SUCCESS_COMPOSITIONS)
    selected=choice if confirmed else "S614_original"
    return selected,dict(calibration_candidate=choice,
            held_success_compositions=positive,
            selected_without_DEV=selected,
            calibration_F1={n:v["score"]["F1"] for n,v in calib.items()},
            held_F1={n:v["score"]["F1"] for n,v in check.items()},
            held_compositions=held_compositions)


def run(out):
    if out.exists():
        raise FileExistsError("S658 report directory must be immutable")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train=[x for x in protocol["sources"]
           if x["corpus"]=="GuitarSet" and x["split"]=="train"]
    dev=[x for x in protocol["sources"]
         if x["corpus"]=="GuitarSet" and x["split"]=="dev"]
    test=[x for x in protocol["sources"] if x["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert not {(x["corpus"],x["composition"]) for x in train}&{
        (x["corpus"],x["composition"]) for x in dev+test}
    comps=sorted({x["composition"] for x in train})
    assert len(comps)==9
    calibration=set(comps[:5]);held=set(comps[5:])
    assert calibration.isdisjoint(held) and len(held)==4
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    t0=time.perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);(root/"data").mkdir()
        for key,path in (("GuitarSet_audio","GuitarSet_audio.zip"),
                         ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            x=protocol["guitar_archives"][key]
            download(x["url"],root/"data"/path,x["size"],x["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as zipfile_audio:
            train_calibration=[];train_held=[]
            for i,spec in enumerate(train,1):
                row=decode_natural(spec,root,zipfile_audio,heads,config)
                (train_calibration if row["composition"] in calibration
                 else train_held).append(row)
                print("S658 TRAIN",i,"/36",row["name"],
                      "originalTP",row["stats"]["S614_original"]["score"]["TP"],
                      flush=True)
            selected,train_result=choose_by_TRAIN(train_calibration,train_held)
            print("S658 TRAIN CHOICE",json.dumps(train_result),flush=True)
            validation=[]
            for i,spec in enumerate(dev,1):
                row=decode_natural(spec,root,zipfile_audio,heads,config)
                validation.append(row)
                print("S658 DEV",i,"/12",row["name"],
                      "original",row["stats"]["S614_original"]["score"]["TP"],
                      "conditional",row["stats"]["gate_last4_050"]["score"]["TP"],
                      flush=True)
    scores=aggregate(validation)
    original=scores["S614_original"]["score"]
    assert all(original[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    for name,r in scores.items():
        score=r["score"]
        r["delta_TP_vs_S614"]=score["TP"]-original["TP"]
        r["delta_FP_vs_S614"]=score["FP"]-original["FP"]
        r["delta_F1_points_vs_S614"]=round(
            100*(score["F1"]-original["F1"]),6)
        print("S658 SCORE",name,"TP",score["TP"],"FP",score["FP"],
              "F1",round(score["F1"],8),"E2",r["E2_TP"],
              "K3plus",r["K3plus_TP"],
              "change_F1_points",r["delta_F1_points_vs_S614"],flush=True)
    record=dict(
        status="S658_NATURAL_GUITARSET_S657_TONE_OCTAVE_GATE_TRANSFER",
        paired_S657_not_a_natural_song_validation=True,
        nature="real GuitarSet microphone chord performances",
        native_NOTE_S614_and_Exact_K_unchanged=True,
        original_K_used_by_every_selector=True,
        filter_is_causal_digital_Tone_proxy_not_real_Stratocaster=True,
        no_future_audio_and_no_backdated_NoteOn=True,
        no_model_promotion=True,
        selected_exclusively_by_TRAIN_compositions=selected,
        TRAIN_compositions_calibration=sorted(calibration),
        TRAIN_compositions_hold=sorted(held),
        TRAIN_decision=train_result,
        TEST_never_opened=True,
        DEV_previously_exposed_exploratory=True,
        DEV_GuitarSet=scores,per_dev_recording=validation,
        S641_historical_held_DEV_f1=.6339930882814955,
        sample_elapsed_seconds=round(time.perf_counter()-t0,2))
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(record,indent=2)+"\n")
    print("S658 FINAL",json.dumps(dict(
        preselected=selected,dev={k:{z:v["score"][z]
                        for z in ("TP","FP","FN","F1")}
                        for k,v in scores.items()})),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
