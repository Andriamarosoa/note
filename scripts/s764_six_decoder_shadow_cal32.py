"""S764: honest GuitarSet CAL32 SHADOW audit of six S763 independent decoders.

Each head gets only the causal S708 input tensors; predictions are fully
finalized before opening CAL32 ground truth. Native S708 MIDI is immutable,
and no decoder's hypothesis is used to re-decide another decoder's output.

Per-axis SHADOW metrics are intentionally NOT the official six S704 metrics
from the common frozen MIDI stream. No fitting or promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import torch

from causal_note.strict_live import SR
from causal_note.s763_independent_decoders import (
    AXES, Fragment, IsolatedDecoderBank, ShadowArbiter, DECISION_SAMPLES)
from causal_note.s708_cardinality_selection import GuidedPolicy
from scripts.s613_prepare_strict_live import download, labels, digest
from scripts.s613_train_strict_live import event_score, real_notes
from scripts.s699_player_held_full_train_K import expanded_train_specs
from scripts.s704_joint_metrics import SCORES, aggregate, metric
from scripts.s705_train_temporal_loops import setup
from scripts.s706_train_sharp_temporal_loops import CAL_COMPS, FIT_COMPS, load_model
from scripts.s733_train_onset_only import baseline_input

def target_rows(ends, truth):
    """Ground truth ONLY for offline scoring, never passed to any decoder."""
    clock=np.asarray(ends,np.int64)
    k=np.zeros(len(clock),np.int32)
    active=np.zeros((len(clock),49),bool)
    for note in truth:
        sample_start=note["time"]
        sample_end=sample_start+note["duration"]
        mask=(clock/SR>=sample_start)&(clock/SR<sample_end)
        k+=mask.astype(np.int32)
        active[mask,int(note["pitch"])-40]=True
    return np.minimum(k,6), active

def intent_event(intent):
    return dict(type=intent.kind,pitch=int(intent.pitch),
                decision_sample=int(intent.sample),
                available_sample=int(intent.sample),
                time=float(intent.sample/SR),
                reason=intent.reason,
                velocity=96 if intent.kind=="note_on" else 0)

def replay_six(ends,post,k,source,samples):
    """Emit all independent hypotheses WITHOUT reading annotations."""
    bank=IsolatedDecoderBank()
    arbiter=ShadowArbiter()
    onset,offset,full=[],[],[]
    counts={axis:[] for axis in ("exact_k","active_exact_k")}
    frame_pitch=[]
    disagreement=Counter()
    last={}
    for end,frame,cardinality in zip(ends,post,k):
        dec=bank.step(Fragment(int(end),frame,cardinality))
        diagnosis=arbiter.observe(dec)
        onset.extend(intent_event(x) for x in dec["onset"].intents)
        offset.extend(intent_event(x) for x in dec["offset"].intents)
        full.extend(intent_event(x) for x in dec["full_note"].intents)
        counts["exact_k"].append(dec["exact_k"].count)
        counts["active_exact_k"].append(dec["active_exact_k"].count)
        frame_pitch.append(dec["frame_pitch"].pitches)
        disagreement["khead_activek_disagree"]+=int(
            diagnosis.exact_k!=diagnosis.active_k)
        disagreement["activek_pitchcount_disagree"]+=int(
            diagnosis.active_k!=diagnosis.active_pitch_count)
        disagreement["onset_proposal_fragments"]+=int(diagnosis.onset_proposals>0)
        disagreement["offset_proposal_fragments"]+=int(diagnosis.offset_proposals>0)
        disagreement["full_note_proposal_fragments"]+=int(diagnosis.full_note_proposals>0)
        last=dec
    full.extend(intent_event(x) for x in bank.decoders["full_note"].finish(int(samples)))
    original=arbiter.preserve_signed_stream(source)
    if original!=source:
        raise AssertionError("S764 shadow arbiter mutated signed reference")
    return dict(onset=onset,offset=offset,full=full,
                counts=counts,pitches=frame_pitch,disagreement=dict(disagreement),
                signed=original)

def compare_track(predictions,truth,decision_ends,all_ends):
    """Per-axis diagnostics are separate from the frozen MIDI score."""
    ktrue,pitchtrue=target_rows(decision_ends,truth)
    answers={}
    for axis in ("exact_k","active_exact_k"):
        predicted=np.asarray(predictions["counts"][axis],np.int32)
        correct=predicted==ktrue
        per={}
        for k in range(7):
            subset=(ktrue==k)
            n=int(subset.sum())
            per[str(k)]=dict(frames=n,correct=int(correct[subset].sum()),
                             under=int(np.count_nonzero(predicted[subset]<k)),
                             over=int(np.count_nonzero(predicted[subset]>k)))
        answers[axis]=dict(total=len(ktrue),correct=int(correct.sum()),
                           under=int(np.count_nonzero(predicted<ktrue)),
                           over=int(np.count_nonzero(predicted>ktrue)),
                           per_true_k=per)
    guessed=np.zeros_like(pitchtrue)
    for i,pitches in enumerate(predictions["pitches"]):
        if pitches:
            guessed[i,np.asarray(pitches,dtype=int)-40]=True
    answers["frame_pitch"]=dict(
        tp=int(np.count_nonzero(guessed & pitchtrue)),
        fp=int(np.count_nonzero(guessed & ~pitchtrue)),
        fn=int(np.count_nonzero(~guessed & pitchtrue)))
    onset,_=event_score(predictions["onset"],truth,tolerance=.080)
    answers["onset"]=dict(tp=onset["TP"],fp=onset["FP"],fn=onset["FN"])
    # OffsetDecoder emits UNPAIRED END HYPOTHESES. This is precision/recall
    # of boundaries, not the S704 matched-note conditional Offset accuracy.
    duration=all_ends[-1]/SR
    endings=[dict(time=n["time"]+n["duration"],pitch=n["pitch"])
             for n in truth if n["time"]+n["duration"]<duration]
    virtual=[dict(type="note_on",pitch=e["pitch"],time=e["time"])
             for e in predictions["offset"]]
    off,_=event_score(virtual,endings,tolerance=.050)
    answers["offset"]=dict(tp=off["TP"],fp=off["FP"],fn=off["FN"],
                            actual_endings=len(endings))
    # Independently paired events are a separate FULL NOTE hypothesis
    # stream. Strict S704 metrics applied to *that* stream only.
    full=metric(predictions["full"],truth,all_ends)
    answers["full_note"]=dict(
        tp=full["tp"],fp=full["fp"],fn=full["fn"],
        offset_eligible=full["offset_eligible"],
        offset_correct=full["offset_correct"],
        k_correct=full["k_correct"],frames=full["frames"],
        active_k_correct=full["active_k_correct"],active_frames=full["active_frames"],
        six_score_diagnostic={m:float(full[m]) for m in SCORES})
    return answers

def merge_details(records):
    keys=("exact_k","active_exact_k")
    result={}
    for axis in keys:
        total=sum(x[axis]["total"] for x in records)
        correct=sum(x[axis]["correct"] for x in records)
        per={}
        for n in range(7):
            raw=[x[axis]["per_true_k"][str(n)] for x in records]
            p={name:sum(z[name] for z in raw)
               for name in ("frames","correct","under","over")}
            p["rate"]=p["correct"]/max(1,p["frames"])
            per[str(n)]=p
        result[axis]=dict(accuracy=correct/max(1,total),correct=correct,
                          total=total,under=sum(x[axis]["under"] for x in records),
                          over=sum(x[axis]["over"] for x in records),per_true_k=per)
    for axis in ("onset","offset","frame_pitch"):
        sums={name:sum(x[axis][name] for x in records) for name in ("tp","fp","fn")}
        tp,fp,fn=(sums[k] for k in ("tp","fp","fn"))
        sums.update(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                    f1=2*tp/max(1,2*tp+fp+fn))
        if axis=="offset":
            sums["definition"]="unpaired release-boundary F1 (+/-50ms), NOT official S704 offset_accuracy"
            sums["actual_endings"]=sum(x[axis]["actual_endings"] for x in records)
        result[axis]=sums
    fs=[x["full_note"] for x in records]
    tp=sum(x["tp"] for x in fs)
    fp=sum(x["fp"] for x in fs)
    fn=sum(x["fn"] for x in fs)
    good=sum(x["offset_correct"] for x in fs)
    eligible=sum(x["offset_eligible"] for x in fs)
    result["full_note"]=dict(
        tp=tp,fp=fp,fn=fn,offset_correct=good,offset_eligible=eligible,
        onset_f1=2*tp/max(1,2*tp+fp+fn),
        offset_accuracy=good/max(1,eligible),
        full_note_f1=2*good/max(1,2*tp+fp+fn),
        definition="independent FULL NOTE decoder's own complete paired MIDI hypothesis; not frozen S708")
    return result

def write(path,data):
    path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")

def run(args):
    if args.out.exists():
        raise FileExistsError("S764 immutable result directory must be new")
    args.out.mkdir(parents=True)
    signed=json.loads((args.s708_dir/"report.json").read_text())
    checkpoint=digest((args.s708_dir/"s708_selected.pt").read_bytes())
    if checkpoint!=signed["checkpoint_sha256"]:
        raise AssertionError("S764 signed S708 hash mismatch")
    choice=signed["frozen_candidate"]["choice"]
    release=choice["release"]
    if release!=dict(gate=.8,persistence=1,grace=24):
        raise AssertionError("S764 S707 release is not frozen")
    policy=GuidedPolicy(**choice["policy"])
    net=load_model(args.s708_dir/"s708_selected.pt")[0]
    torch.set_num_threads(2)
    report=dict(schema="S764_SHADOW_SIX_INDEPENDENT_DECODERS_CAL32",
        model_sha256=checkpoint,complete=False,promoted=False,
        LIVE_changed=False,HELD_opened=False,TEST_opened=False,
        training=False,selection="S708_frozen",calibration_only=True,
        source_reference_parity=False,recordings_done=0,
        recordings=[],shadows={},source={})
    write(args.out/"report.json",report)
    rows=[]
    details=[]
    discrepancies=Counter()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        (root/"data").mkdir()
        protocol,heads,selection,offset=setup(root,args.offset_checkpoint)
        for key,archive in (("GuitarSet_audio","GuitarSet_audio.zip"),
                            ("GuitarSet_annotations","GuitarSet_annotations.zip")):
            remote=protocol["guitar_archives"][key]
            download(remote["url"],root/"data"/archive,
                     remote["size"],remote["sha256"])
        baseline_spec=[s for s in protocol["sources"]
                       if s["corpus"]=="GuitarSet" and s["split"]=="train"]
        with zipfile.ZipFile(root/"data/GuitarSet_audio.zip") as wav, \
             zipfile.ZipFile(root/"data/GuitarSet_annotations.zip") as ann:
            expanded=expanded_train_specs(baseline_spec,wav,ann)
            cal=[s for s in expanded if s["composition"] in CAL_COMPS]
            fit=[s for s in expanded if s["composition"] in FIT_COMPS]
            if len(cal)!=32 or len(fit)!=80 or set(FIT_COMPS)&set(CAL_COMPS):
                raise AssertionError("S764 composition split not trustworthy")
            for i,spec in enumerate(cal):
                item,post,k,source,_,_,_=baseline_input(
                    spec,wav,root,heads,selection,offset,net,policy,release)
                source_before=hashlib.sha256(json.dumps(
                    source,sort_keys=True).encode()).hexdigest()
                replay=replay_six(item["decision_ends"],post,k,source,item["samples"])
                source_after=hashlib.sha256(json.dumps(
                    source,sort_keys=True).encode()).hexdigest()
                if source_before!=source_after or replay["signed"]!=source:
                    raise AssertionError("S764 modified source MIDI")
                # All six proposals exist BEFORE looking at annotations.
                truth=real_notes(labels(spec,root)["notes"],item["samples"]/SR)
                detail=compare_track(replay,truth,item["decision_ends"],item["ends"])
                frozen_metric=metric(source,truth,item["ends"])
                rows.append(frozen_metric)
                details.append(detail)
                discrepancies.update(replay["disagreement"])
                report["recordings"].append(dict(
                    name=spec["name"],composition=spec["composition"],
                    signed_source_sha256=source_before,
                    signed_events_untouched=True,diagnostics=detail,
                    conflicts=replay["disagreement"],
                    source={m:float(frozen_metric[m]) for m in SCORES},
                    proposal_counts={x:len(replay[x]) for x in ("onset","offset","full")}))
                report["recordings_done"]=i+1
                write(args.out/"report.json",report)
                print("S764 SHADOW",i+1,32,spec["name"],
                      "onset_proposals",len(replay["onset"]),
                      "full_note_pairs",sum(e["type"]=="note_on" for e in replay["full"]),
                      flush=True)
    scored=aggregate(rows)
    signed_scores=signed["frozen_candidate"]["CAL"]
    delta={m:float(scored[m]-signed_scores[m]) for m in SCORES}
    parity=all(abs(d)<1e-8 for d in delta.values())
    report.update(source_reference_parity=parity,
                  source_reference_delta=delta,
                  source={m:float(scored[m]) for m in SCORES},
                  shadows=merge_details(details),
                  conflicts=dict(discrepancies),
                  complete=parity)
    write(args.out/"report.json",report)
    if not parity:
        raise AssertionError("S764 S708 official six-score reference mismatch")
    print("S764_COMPLETE",json.dumps({
        "reference":report["source"],"shadows":report["shadows"],
        "conflicts":report["conflicts"],"promoted":False}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--s708-dir",type=Path,required=True)
    ap.add_argument("--offset-checkpoint",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    run(ap.parse_args())
