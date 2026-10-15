"""S635 cross-repository equivalence against the ORIGINAL midi causal matcher.

This does not score music. It certifies that the shared evaluation contract
applies the identical one-to-one, past-only rule to both engines.
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import random
import sys

from causal_note.s635_shared_midi_metric import (
    DEADLINES_MS,causal_score,extract_reference,extract_events)


def source_module(path:Path):
    name="midi_reference_causal_event_metrics"
    spec=importlib.util.spec_from_file_location(name,str(path))
    if spec is None or spec.loader is None:raise FileNotFoundError(path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def verify(midi,seed,kind):
    r=random.Random(seed)
    if kind=="random":
        refs=[]
        predictions=[]
        for i in range(280):
            pitch=r.choice([40,45,52,52,52,64,69,76])
            start=round(r.random()*20,4)
            refs.append(dict(pitch=pitch,start_s=start,end_s=start+r.uniform(.04,.4)))
        for i in range(350):
            pitch=r.choice([40,45,52,52,64,69,76])
            start=round(r.random()*20,4)
            predictions.append(dict(pitch=pitch,time_s=start))
    elif kind=="repeated":
        refs=[dict(pitch=52,start_s=i*.087,end_s=i*.087+.06)
              for i in range(100)]
        predictions=[dict(pitch=52,time_s=i*.087+d)
                     for i in range(100) for d in (.025,.037)]
    elif kind=="chord":
        refs=[dict(pitch=p,start_s=t,end_s=t+.18)
              for t in (.0,.22,.45) for p in (40,52,64,76)]
        predictions=[dict(pitch=p,time_s=t+d)
                     for t in (.0,.22,.45) for p in (40,52,64,76)
                     for d in (.0,.015,.07)]
    else:raise ValueError(kind)
    duration=25.
    a=extract_reference(refs,duration)
    b=extract_events(predictions,duration)
    theirs_ref=[midi.ReferenceNote(row["pitch"],row["start_s"],row["end_s"])
                for row in refs]
    theirs_pred=[midi.NoteOnPrediction(row["pitch"],row["time_s"])
                 for row in predictions]
    for deadline in DEADLINES_MS:
        original=midi.match_causal_note_ons(theirs_ref,theirs_pred,
                                            max_latency_ms=deadline)
        current=causal_score(a,b,deadline)
        their_set={(v.reference_index,v.prediction_index) for v in original.matches}
        our_set={(i,j) for i,j,_ in current["matches"]}
        if their_set!=our_set:
            raise AssertionError(f"{kind} deadline={deadline} mismatched pairs "+
                f"midi-only={list(their_set-our_set)[:4]} NOTE-only={list(our_set-their_set)[:4]}")
        if current["FP"]!=len(original.false_prediction_indices):
            raise AssertionError("False prediction tally mismatch")
        if current["FN"]!=len(original.missed_reference_indices):
            raise AssertionError("Missed reference tally mismatch")
    return dict(test=kind,reference=len(refs),predictions=len(predictions),
                time_deadlines=len(DEADLINES_MS),comparison="exact_pair_parity")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--reference-midi",type=Path,required=True)
    args=p.parse_args()
    source=source_module(args.reference_midi)
    print("S635 CROSS-REPO PARITY",[
        verify(source,635+i,kind)
        for i,kind in enumerate(("random","repeated","chord"))])
