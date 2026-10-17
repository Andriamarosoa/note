"""S644: Transfer the S643 octave-duet detector to NATURAL GuitarSet DEV.

Fit exactly the S643 frozen composition-disjoint recipe from 36 original
GuitarSet TRAIN recordings. Then evaluate WITHOUT tuning on the 12 real
DEV tracks on actual low-note onset contexts. A positive label means
an additional note one octave higher starts within ±30ms, overlaps the
low note, and lives for >=80ms. A negative means the upper octave is
absent through the 190ms feature window. Cases with upper octave already
sounding, delayed >30ms, or ambiguous double-onsets are explicitly
excluded rather than silently relabeled.

This tests transfer from a **digitally mixed controlled** corpus to natural
guitar performance. No official NOTE production change or held-out TEST
source is read; the 190ms observation period is causal but is NOT a
low-latency strict-live NoteOn.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,accuracy_score

from causal_note.strict_live import SR
from scripts.s613_prepare_strict_live import labels,download,digest
from scripts.s613_train_strict_live import real_notes
from scripts.s637_guitarset_three_view_groundtruth import ROOT,raw_guitar_audio
from scripts.s642_real_octave_pair_comparison import BASE_T,DURATION,source_clip,normalize_loudness
from scripts.s643_octave_duet_head import (
    extract_sources,select_composition_halves,make_dataset,
    classifier,extracted_features,
)

ONSET_TOL=.030
MIN_DURATION=.080
OBSERVE_AFTER=.19
PITCH_LOW_RANGE=range(40,65)


def truth_case(truth,i,*,tolerance=ONSET_TOL):
    """Post-hoc truth; never available to the spectral classifier.

    Annotated note i is the lower target pitch. Distinguish independent
    high onset from a mere octave harmonic and exclude ambiguous cases.
    """
    current=truth[i]
    t=float(current["time"])
    low=int(current["pitch"])
    if low not in PITCH_LOW_RANGE or float(current["duration"])<.25:
        return None
    high=low+12
    if t<BASE_T+.10 or t>24.-(DURATION-BASE_T)-.10:
        return None
    upper=[n for n in truth if int(n["pitch"])==high
           and float(n["time"])+float(n["duration"])>=t-.05
           and float(n["time"])<=t+OBSERVE_AFTER+.01]
    simultaneous=[n for n in upper if abs(float(n["time"])-t)<=tolerance
                  and float(n["duration"])>=MIN_DURATION]
    if simultaneous:
        y=1
    elif upper:
        return None
    else:
        y=0
    other={int(n["pitch"]) for j,n in enumerate(truth)
           if j!=i and int(n["pitch"])!=low
           and float(n["time"])<=t+.03
           and float(n["time"])+float(n["duration"])>=t+.03}
    k=len(other)+1
    return dict(target=y,low_pitch=low,upper_pitch=high,
                true_chord_K=int(k),at_s=round(t,5),
                simultaneous_upper_true_notes=len(simultaneous))


def standard_metrics(y,probs):
    if len(y)==0:
        return dict(n=0,positives=0,negatives=0,TP=0,FP=0,
                    TN=0,FN=0,F1=None,precision=None,recall=None,accuracy=None)
    guessed=np.asarray(probs)>=.5
    a=np.asarray(y,dtype=int)
    cm=confusion_matrix(a,guessed.astype(int),labels=[0,1])
    return dict(n=len(a),positives=int(a.sum()),negatives=int(len(a)-a.sum()),
                TP=int(cm[1,1]),FP=int(cm[0,1]),TN=int(cm[0,0]),FN=int(cm[1,0]),
                F1=float(f1_score(a,guessed,zero_division=0)) if a.sum() else None,
                precision=float(precision_score(a,guessed,zero_division=0)) if a.sum() else None,
                recall=float(recall_score(a,guessed,zero_division=0)) if a.sum() else None,
                accuracy=float(accuracy_score(a,guessed)))


def dev_source(spec,root,archive):
    pcm,sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(pcm)/SR)
    rows=[];excluded=Counter()
    for i,n in enumerate(truth):
        case=truth_case(truth,i)
        if case is None:
            if 40<=int(n["pitch"])<=64:
                excluded["unscored_low_onsets"]+=1
            continue
        try:
            clip=normalize_loudness(source_clip(pcm,n["time"]))
        except ValueError:
            excluded["silence_or_incomplete_clip"]+=1
            continue
        row=extracted_features(clip,case["low_pitch"])
        rows.append(dict(**case,feature=row))
    return dict(name=spec["name"],composition=spec["composition"],
                source_sha256=sha,total_annotated_notes=len(truth),
                excluded=dict(excluded),rows=rows)


def run(out):
    if out.exists():raise FileExistsError("S644 immutable results only")
    protocol=json.loads((ROOT/"protocol.json").read_text())
    freeze=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest((ROOT/"protocol.json").read_bytes())==freeze["protocol_sha256"]
    train=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="train"]
    dev=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    test=[s for s in protocol["sources"] if s["split"]=="test"]
    assert len(train)==36 and len(dev)==12
    assert not {(s["corpus"],s["composition"]) for s in train}&{
        (s["corpus"],s["composition"]) for s in dev+test}
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);(root/"data").mkdir()
        for name,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                              ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=protocol["guitar_archives"][name]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            clips,_=extract_sources(train,root,archive)
            split=select_composition_halves(clips)
            source_pairs=split["ab_pairs"]+split["ba_pairs"]
            x,y,_,_=make_dataset(source_pairs)
            assert len(x)==2*len(source_pairs) and int(y.sum())==len(source_pairs)
            classifier_frozen=classifier().fit(x,y)
            rows=[]
            for no,s in enumerate(dev,1):
                record=dev_source(s,root,archive)
                if record["rows"]:
                    probabilities=classifier_frozen.predict_proba(
                        np.stack([r["feature"] for r in record["rows"]]))[:,1]
                    for item,p in zip(record["rows"],probabilities):
                        item["probability_duet"]=float(p)
                        item.pop("feature")
                rows.append(record)
                labels_now=[v["target"] for v in record["rows"]]
                predictions_now=[v["probability_duet"] for v in record["rows"]]
                m=standard_metrics(labels_now,predictions_now)
                print("S644 NATURAL",no,"/",len(dev),s["name"],
                      "cases",m["n"],"true_octave",m["positives"],
                      "correct_octave",m["TP"],"false_octave",m["FP"],flush=True)
    all_rows=[r for f in rows for r in f["rows"]]
    y=np.asarray([r["target"] for r in all_rows],np.int32)
    score=np.asarray([r["probability_duet"] for r in all_rows],np.float64)
    measures=standard_metrics(y,score)
    by_pitch={}
    for pitch in PITCH_LOW_RANGE:
        indices=[i for i,r in enumerate(all_rows) if r["low_pitch"]==pitch]
        if indices:
            by_pitch[str(pitch)]=standard_metrics(y[indices],score[indices])
    by_K={}
    for name,fn in (("single_at_low_onset",lambda r:r["true_chord_K"]==1),
                    ("K2",lambda r:r["true_chord_K"]==2),
                    ("K3plus",lambda r:r["true_chord_K"]>=3)):
        indices=[i for i,r in enumerate(all_rows) if fn(r)]
        by_K[name]=standard_metrics(y[indices],score[indices])
    e2=by_pitch.get("40",{})
    report=dict(
        status="S644_S643_SINGLE_VS_DUET_TRANSFER_TO_REAL_NATURAL_GUITARSET",
        S643_head_fitted_only_on_GuitarSet_TRAIN_mixed_recordings=True,
        target_natural_DEV_mic_recordings=True,
        TEST_not_read=True,DEV_historically_exposed=True,
        training_guitarset_compositions=sorted({s["composition"] for s in train}),
        testing_guitarset_compositions=sorted({s["composition"] for s in dev}),
        trained_pairs=len(source_pairs),
        classifier_fixed_untuned_threshold=.5,
        onset_simultaneity_tolerance_ms=30,
        octave_absent_through_analysis_window_to_count_negative=True,
        ambiguous_existing_or_delayed_octaves_excluded=True,
        observation_time_after_low_note_onset_ms=190,
        no_S614_pitch_or_K_changes=True,
        previous_s643_constructed_duet_F1=0.8291571753986332,
        real_natural_annotations_used_for_scoring_only=True,
        result=measures,
        E2_case=e2,
        by_low_pitch=by_pitch,
        by_true_chord_K=by_K,
        by_recording=rows,
        no_automatic_promotion=True,
        result_interpretation="Natural NOTE onset-context classifier diagnostic, not whole transcription MIDI F1 or real-time latency proof.",
    )
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S644 FINAL",json.dumps(dict(natural_context_result=measures,
           E2=e2,contextK=by_K,by_low_pitch=by_pitch)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
