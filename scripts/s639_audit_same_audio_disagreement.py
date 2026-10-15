"""S639: explain SAME-WAV differences with fixed S635 causal matches.

Loads only S638's immutable event/annotation manifest and metric output.
Never trains a model, retunes thresholds, reruns acoustics, or accesses TEST.
Summarises pitch registers, local simultaneous onset multiplicity, and
which reference events were matched by NOTE, MIDI, both or neither.

The local onset-group size is not historical S80 Exact-K and must not be
presented as its score.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
from causal_note.s635_shared_midi_metric import (
    compare_manifest,extract_events,extract_reference
)

BANDS={"bass":(40,52),"middle":(53,64),"high":(65,76)}
COMMON_RANGE=(40,76)


def band(pitch):
    for name,(lo,hi) in BANDS.items():
        if lo<=pitch<=hi:return name
    raise ValueError("Outside common pitch range")


def local_k(reference,index,seconds=.020):
    """Number of distinct pitches starting within ±20 ms of an onset."""
    t,p,_=reference[index]
    return len({pitch for onset,pitch,_ in reference
                if abs(onset-t)<=seconds})


def category(k):
    return "K1" if k==1 else "K2" if k==2 else "K3plus"


def audit(manifest,expected_metric):
    actual=compare_manifest(manifest)
    if actual["overall"]!=expected_metric["overall"]:
        raise AssertionError("S638 F1 report no longer matches event/annotation inputs")
    if actual["recordings"]!=expected_metric["recordings"]:
        raise AssertionError("S638 source count changed")
    output=[]
    aggregates={method:defaultdict(Counter) for method in ("note","midi")}
    differences=defaultdict(Counter)
    detail={}
    for clip in manifest["clips"]:
        clipid=clip["clip_id"]
        duration=clip["duration_s"]
        true_notes=[r for r in extract_reference(clip["reference"],duration)
                    if COMMON_RANGE[0]<=r[1]<=COMMON_RANGE[1]]
        active_truth={i for _,_,i in true_notes}
        onset_k={i:local_k(true_notes,j) for j,(_,_,i) in enumerate(true_notes)}
        matches={}
        for engine in ("note","midi"):
            on=[e for e in extract_events(clip[engine]["events"],duration)
                if COMMON_RANGE[0]<=e[1]<=COMMON_RANGE[1]]
            matched=actual["by_clip"][clipid][engine]["80"]["matches"]
            matched_truth={int(pair[0]) for pair in matched}
            matched_pred={int(pair[1]) for pair in matched}
            if len(matched_truth)!=len(matched_pred):
                raise AssertionError("Non-unique correct note match")
            if not matched_truth<=active_truth:
                raise AssertionError("Reference index outside common pitch")
            by_register=defaultdict(Counter)
            by_k=defaultdict(Counter)
            for _,pitch,ridx in true_notes:
                success=ridx in matched_truth
                aggregate=by_register[band(pitch)]
                aggregate["TP" if success else "FN"]+=1
                by_k[category(onset_k[ridx])]["TP" if success else "FN"]+=1
            for t,pitch,pidx in on:
                if pidx not in matched_pred:
                    by_register[band(pitch)]["FP"]+=1
            for group,s in by_register.items():
                aggregates[engine][group].update(s)
            for group,s in by_k.items():
                aggregates[engine][group].update({}) # keep global register metrics separate
            matches[engine]=matched_truth
        common=matches["note"] & matches["midi"]
        note_only=matches["note"]-matches["midi"]
        midi_only=matches["midi"]-matches["note"]
        neither=active_truth-(matches["note"]|matches["midi"])
        assert sum(map(len,(common,note_only,midi_only,neither)))==len(active_truth)
        differences["all"].update({"both":len(common),"NOTE_only":len(note_only),
            "MIDI_only":len(midi_only),"neither":len(neither)})
        source={}
        for _,pitch,ridx in true_notes:
            scope=band(pitch)
            status=("both" if ridx in common else
                    "NOTE_only" if ridx in note_only else
                    "MIDI_only" if ridx in midi_only else "neither")
            differences[scope][status]+=1
            differences[category(onset_k[ridx])][status]+=1
        source=dict(clip=clipid,truth=len(active_truth),
            matched_by_both=len(common),NOTE_only=len(note_only),
            MIDI_only=len(midi_only),missed_by_both=len(neither),
            NOTE_F1=actual["by_clip"][clipid]["note"]["80"]["F1"],
            MIDI_F1=actual["by_clip"][clipid]["midi"]["80"]["F1"])
        output.append(source)
        detail[clipid]=dict(
            NOTE_only_reference_indices=sorted(note_only),
            MIDI_only_reference_indices=sorted(midi_only),
            missed_by_both_reference_indices=sorted(neither))
    by_register={}
    for engine in ("note","midi"):
        by_register[engine]={}
        for region in BANDS:
            c=aggregates[engine][region]
            tp,fp,fn=c["TP"],c["FP"],c["FN"]
            by_register[engine][region]=dict(TP=tp,FP=fp,FN=fn,
                   F1=2*tp/max(1,2*tp+fp+fn))
        totals={key:sum(by_register[engine][band][key] for band in BANDS)
                for key in ("TP","FP","FN")}
        if totals!={k:actual["overall"][engine]["80"][k] for k in totals}:
            raise AssertionError("Pitch register totals fail causal matcher invariant")
    return dict(status="S639_NO_TUNING_S638_FROZEN_CAUSAL_ERRORS",
        recordings=actual["recordings"],shared_pitch_range=[40,76],
        deadline_ms=80,
        original_models_unchanged=True,
        no_independent_test_claim=True,
        no_promotion=True,
        matched_reference_outcomes={k:dict(v) for k,v in differences.items()},
        F1_by_register=by_register,by_source=output,
        individual_truth_reference_indices=detail)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--metrics",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("No overwriting S639 result")
    manifest=json.loads(args.input.read_text())
    prior=json.loads(args.metrics.read_text())
    result=audit(manifest,prior)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print("S639 TRUE CAUSAL DIFFERENCES",json.dumps({
        "recordings":result["recordings"],
        "events":result["matched_reference_outcomes"]["all"],
        "by_source":result["by_source"],
        "register":result["F1_by_register"]},indent=2),flush=True)


if __name__=="__main__":main()
