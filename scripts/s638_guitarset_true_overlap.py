"""S638 — TRUE overlapping notes on real annotated GuitarSet audio.

The previous S637 'comp' versus 'solo' split is NOT a split by concurrent
notes. Here individual annotated NOTE ONSETS are classified by actual
temporal MIDI-note overlap and their matched predictions are counted.
GuitarSet dev (12 recordings), S614 frozen and native K remain unchanged.
Held-out test and all future audio are never used to make predictions.

Report counts of newly recovered AND newly lost annotations separately:
aggregate delta TP alone obscures the number of fragile corrections.
False positives are also attributed to ground-truth polyphonic passages
for diagnostic purposes ONLY, never as a runtime oracle.
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

from causal_note.strict_live import Selection,CausalSelection,load_heads
from scripts.s613_prepare_strict_live import download,digest,labels
from scripts.s613_train_strict_live import real_notes,event_score,summary
from scripts.s637_guitarset_three_view_groundtruth import (
    ROOT,MODEL,acoustic_batches,corrected,raw_guitar_audio,
    GUITARSET_PINNED_CONTROL,
)

# Pre-committed on previous S637 observation; this is a DIAGNOSTIC breakdown,
# not a tuned or independently blind policy-selection benchmark.
POLICIES=(
    ("S614_original",None,None,False,0.),
    ("bass_080",.80,None,False,0.),
    ("bass_090",.90,None,False,0.),
    ("high_080",None,.80,False,0.),
    ("both_080",.80,.80,False,0.),
    ("both_080_octave_veto_K35",.80,.80,True,.35),
)
GROUPS=("onset_overlap","onset_single","any_overlap","never_overlap",
        "onset_K2","onset_K3plus")


def overlap_groups(truth,*,min_shared_s=.020,same_chord_tolerance_s=.030):
    """Ground truth grouping, NEVER read by prediction.

    A distinct pitch j contributes to the onset concurrency of note i
    if it began before t_i+30ms (allows imperfect strum alignment),
    remains sounding at t_i and shares >=20ms with i.
    Also report any note with >=20ms overlap at any point in its duration.
    Note-level K is unique MIDI pitches; same-pitch unisons don't count twice.
    """
    if not 0<min_shared_s<.2 or not 0<=same_chord_tolerance_s<=.1:
        raise ValueError("Invalid overlap definition")
    by={g:[] for g in GROUPS}
    for i,n in enumerate(truth):
        start=float(n["time"])
        stop=start+float(n["duration"])
        active=set([int(n["pitch"])])
        overlaps=False
        for j,m in enumerate(truth):
            if i==j or int(m["pitch"])==int(n["pitch"]):
                continue
            s=float(m["time"]);e=s+float(m["duration"])
            shared=min(stop,e)-max(start,s)
            if shared < min_shared_s:
                continue
            overlaps=True
            if s<=start+same_chord_tolerance_s and e>=start+min_shared_s:
                active.add(int(m["pitch"]))
        k=len(active)
        by["onset_overlap" if k>=2 else "onset_single"].append(i)
        by["any_overlap" if overlaps else "never_overlap"].append(i)
        if k==2:by["onset_K2"].append(i)
        if k>=3:by["onset_K3plus"].append(i)
    return {k:set(v) for k,v in by.items()}


def truth_active_pitch_k(truth, t, *,wiggle=.030):
    """Diagnostic, after matching, attribution of unmatched FP to contexts.

    We consider nearby note windows (±30ms) so a predicted onset marginally
    before the chord gets the same context as other attacks.
    No use of this function in runtime selection.
    """
    return len({int(n["pitch"]) for n in truth
                if float(n["time"])-wiggle <= t
                <float(n["time"])+float(n["duration"])+wiggle})


def coverage(notes, events, pairs, groups, baseline_pairs):
    matched_truth={j for _,j in pairs}
    base_truth={j for _,j in baseline_pairs}
    predicted=[e for e in events if e["type"]=="note_on"]
    matched_pred={i for i,_ in pairs}
    fp=list(e for i,e in enumerate(predicted) if i not in matched_pred)
    fp_poly=sum(truth_active_pitch_k(notes,float(e["time"]))>=2 for e in fp)
    result={}
    for name,indices in groups.items():
        rec=len(indices & matched_truth)
        result[name]=dict(
            true_notes=len(indices),
            recovered=rec,
            missed=len(indices)-rec,
            recall=rec/len(indices) if indices else None,
            additional_correct_vs_S614=len((matched_truth-base_truth)&indices),
            lost_correct_vs_S614=len((base_truth-matched_truth)&indices),
            net_correct_vs_S614=len(matched_truth&indices)-len(base_truth&indices))
    return result,dict(false_positives=len(fp),
                       FP_in_true_polyphonic_passage=fp_poly,
                       FP_outside_true_polyphonic_passage=len(fp)-fp_poly)


def source_score(spec,root,archive,heads,config):
    pcm,source_sha=raw_guitar_audio(spec,archive)
    truth=real_notes(labels(spec,root)["notes"],len(pcm)/44100.)
    groups=overlap_groups(truth)
    states={policy[0]:CausalSelection(config) for policy in POLICIES}
    events={name:[] for name in states}
    nframes=0
    for end,native,k,bass,high,valid in acoustic_batches(pcm,heads):
        nframes+=1
        for policy in POLICIES:
            name=policy[0]
            p,_=corrected(native,bass,high,k,states[name],policy,valid)
            events[name].extend(states[name].step(p,k,int(end),int(end)))
    for name in states:
        events[name].extend(states[name].panic(len(pcm)))
    metric_pairs={}
    for name in states:
        metric_pairs[name]=event_score(events[name],truth)
    baseline_pairs=metric_pairs["S614_original"][1]
    rows={}
    for name in states:
        score,pairs=metric_pairs[name]
        categories,fp=coverage(truth,events[name],pairs,groups,baseline_pairs)
        rows[name]=dict(score=score,groups=categories,false_positive_context=fp)
    return dict(
        name=spec["name"],composition=spec["composition"],
        recording_type="solo" if "_solo" in spec["name"] else "comp",
        real_sha256=source_sha,truth_notes=len(truth),hop_frames=nframes,
        truth_onset_overlap=len(groups["onset_overlap"]),
        truth_any_overlap=len(groups["any_overlap"]),
        results=rows)


def aggregate(sources):
    by={}
    for policy in POLICIES:
        name=policy[0]
        rows=[item["results"][name] for item in sources]
        score=summary([r["score"] for r in rows])
        grouped={}
        for group in GROUPS:
            sr=[r["groups"][group] for r in rows]
            truth=sum(r["true_notes"] for r in sr)
            recovered=sum(r["recovered"] for r in sr)
            grouped[group]=dict(
                true_notes=truth,recovered=recovered,missed=truth-recovered,
                recall=recovered/truth if truth else None,
                additional_correct_vs_S614=sum(r["additional_correct_vs_S614"] for r in sr),
                lost_correct_vs_S614=sum(r["lost_correct_vs_S614"] for r in sr),
                net_correct_vs_S614=sum(r["net_correct_vs_S614"] for r in sr),
            )
        fp=dict(
            false_positives=sum(r["false_positive_context"]["false_positives"] for r in rows),
            FP_in_true_polyphonic_passage=sum(r["false_positive_context"]["FP_in_true_polyphonic_passage"] for r in rows),
            FP_outside_true_polyphonic_passage=sum(r["false_positive_context"]["FP_outside_true_polyphonic_passage"] for r in rows))
        by[name]=dict(score=score,groups=grouped,false_positive_context=fp)
    return by


def run(out):
    if out.exists():raise FileExistsError("Do not overwrite evidence")
    frozen=json.loads((ROOT/"selection_frozen.json").read_text())
    assert digest(MODEL.read_bytes())==frozen["model_sha256"]
    assert digest((ROOT/"protocol.json").read_bytes())==frozen["protocol_sha256"]
    protocol=json.loads((ROOT/"protocol.json").read_text())
    train={(s["corpus"],s["composition"]) for s in protocol["sources"] if s["split"]=="train"}
    test={(s["corpus"],s["composition"]) for s in protocol["sources"] if s["split"]=="test"}
    sources=[s for s in protocol["sources"] if s["corpus"]=="GuitarSet" and s["split"]=="dev"]
    assert len(sources)==12 and len({s["composition"] for s in sources})==3
    assert not {("GuitarSet",s["composition"]) for s in sources}&(train|test)
    heads=load_heads(MODEL,expected_sha256=frozen["model_sha256"])
    config=Selection(**heads.metadata["selection"])
    start=time.perf_counter()
    outputs=[]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        (root/"data").mkdir()
        z=protocol["guitar_archives"]
        for key,filename in (("GuitarSet_audio","GuitarSet_audio.zip"),
                             ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            item=z[key]
            download(item["url"],root/"data"/filename,item["size"],item["sha256"])
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as archive:
            for i,s in enumerate(sources,1):
                record=source_score(s,root,archive,heads,config)
                outputs.append(record)
                print("S638 overlapping REAL GuitarSet",i,"/",len(sources),
                      record["name"],
                      "onset_overlap",record["truth_onset_overlap"],
                      "baseline",record["results"]["S614_original"]["score"]["TP"],
                      flush=True)
    result=aggregate(outputs)
    baseline=result["S614_original"]["score"]
    assert all(baseline[k]==v for k,v in GUITARSET_PINNED_CONTROL.items())
    for policy,row in result.items():
        g=row["groups"]
        print("S638 STRATA",policy,
              "onset_poly_true",g["onset_overlap"]["true_notes"],
              "onset_poly_TP",g["onset_overlap"]["recovered"],
              "onset_poly_new",g["onset_overlap"]["additional_correct_vs_S614"],
              "onset_poly_lost",g["onset_overlap"]["lost_correct_vs_S614"],
              "any_overlap_TP",g["any_overlap"]["recovered"],
              "F1",round(row["score"]["F1"],5),
              "FP_poly",row["false_positive_context"]["FP_in_true_polyphonic_passage"],
              flush=True)
    report=dict(
        status="S638_REAL_GUITARSET_TRUE_NOTE_OVERLAP_DEV_AUDIT",
        model_sha256=frozen["model_sha256"],
        original_K_frozen=True,lookahead_ms_added=0,
        guitarset_dev_recordings=12,
        composition_groups=sorted({s["composition"] for s in sources}),
        no_TEST_groundtruth_touched=True,
        already_exposed_DEV=True,
        overlap_definition=dict(min_true_shared_ms=20,
                                same_chord_onset_jitter_ms=30,
                                different_MIDI_pitches_only=True,
                                onset_overlap="at least 2 distinct active annotated pitches at a note's start (30ms chord onset jitter)",
                                any_overlap="at least 20ms simultaneous active annotated notes during a note's life",
                                FP_poly_context="at least 2 distinct real MIDI pitches active within +/-30ms of a predicted unmatched NoteOn",
                                truth_used_for_evaluation_only=True),
        scores=result,
        by_recording=outputs,
        seconds_elapsed=round(time.perf_counter()-start,2),
        not_promoted=True)
    out.mkdir(parents=True)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    print("S638 RESULT",json.dumps(dict(
        status=report["status"],
        baseline_F1=result["S614_original"]["score"]["F1"],
        onset_overlap={name:dict(recall=row["groups"]["onset_overlap"]["recall"],
                                 tp=row["groups"]["onset_overlap"]["recovered"],
                                 new=row["groups"]["onset_overlap"]["additional_correct_vs_S614"],
                                 lost=row["groups"]["onset_overlap"]["lost_correct_vs_S614"],
                                 fp_in_overlap=row["false_positive_context"]["FP_in_true_polyphonic_passage"])
                       for name,row in result.items()}
    )),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.out)
