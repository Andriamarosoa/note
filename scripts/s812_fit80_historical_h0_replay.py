"""S812 replay: authentic signed 80 FIT OOF tracks, strict S704 MIDI256 parity.

Six independent protected H0 outputs MUST reproduce the historical stream,
including exact K/Active-K/FramePitch frame counts and Onset/Offset events.
No candidate H1 model changes reference output; no new musical gains claimed.
"""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from causal_note.s812_historical_axis_h0 import AXES,SixHistoricalH0,signature,checked_h1_view
from scripts.s800_oof_trace import SCHEMA as TRACE_SCHEMA,verify_snapshot
from scripts.s704_joint_metrics import SCORES,metric,aggregate
from scripts.s795_train_activek_midi256 import COUNT_KEYS

SCHEMA="S812_GUITARSET_FIT80_SIX_PROTECTED_HISTORICAL_H0_V1"

def assert_six_axis_parity(out,truth,ends,original):
    n=len(ends)
    exact=np.array([r.count for r in out["exact_k"][:n]],np.int32)
    active=np.array([r.count for r in out["active_exact_k"][:n]],np.int32)
    guessed=np.zeros((n,49),bool)
    for i,r in enumerate(out["frame_pitch"][:n]):
        if r.pitches:
            guessed[i,np.array(r.pitches,dtype=int)-40]=True
    true_pitch=np.zeros((n,49),bool)
    kreal=np.zeros(n,np.int32)
    times=ends/44100.
    for note in truth:
        mask=(times>=note["time"])&(times<note["time"]+note["duration"])
        true_pitch[mask,int(note["pitch"])-40]=True
        kreal+=mask.astype(np.int32)
    true_k=np.minimum(kreal,6)
    measured=dict(
      frames=n,
      k_correct=int((exact==true_k).sum()),
      active_frames=int((kreal>0).sum()),
      active_k_correct=int(((active==true_k)&(kreal>0)).sum()),
      frame_pitch_tp=int((guessed&true_pitch).sum()),
      frame_pitch_fp=int((guessed&~true_pitch).sum()),
      frame_pitch_fn=int((~guessed&true_pitch).sum()))
    for key,value in measured.items():
        if value!=int(original[key]):
            raise AssertionError("historical H0 "+key+" not equal to official S704 MIDI scorer")
    for axis,typ in (("onset","note_on"),("offset","note_off"),("full_note",None)):
        emitted=[dict(e) for snapshot in out[axis] for e in snapshot.emitted]
        expected=[e for e in original["source_events"] if typ is None or e["type"]==typ]
        if emitted!=expected:
            raise AssertionError("historical event projection differs: "+axis)
    return measured

def evaluate(manifest,traces):
    m=json.loads(Path(manifest).read_text("utf-8"))
    if m.get("schema")!=TRACE_SCHEMA or not m.get("complete") or not m.get("FIT_source_parity_S795"):
        raise AssertionError("signed S800 FIT OOF provenance missing")
    if len(m.get("fit_files",[]))!=80 or len({x["index"] for x in m["fit_files"]})!=80:
        raise AssertionError("80 distinct signed FIT tracks required")
    if any(m.get(k) for k in ("CAL_opened","HELD_opened","TEST_opened")):
        raise AssertionError("reserved validation data must not open")
    scores=[];by_composition=defaultdict(list)
    tracks=0;events_copied=0;sample_decisions=0;foreign_comparisons=0
    for row in sorted(m["fit_files"],key=lambda x:x["index"]):
        z=verify_snapshot(row,traces)
        with np.load(Path(traces)/f"fit_{row['index']:03d}.npz",allow_pickle=False) as npz:
            ends=npz["metric_ends"]
        source=z["source_events"];source_sig=signature(source)
        ref=metric(source,z["fit_truth"],ends)
        if any(int(ref[k])!=int(z["source_counts"][k]) for k in COUNT_KEYS):
            raise AssertionError("signed original source track count mismatch")
        decisions,bank=SixHistoricalH0.replay(source,ends,z["samples"])
        if signature(source)!=source_sig:
            raise AssertionError("input reference mutated")
        if len({id(x) for x in bank.lanes.values()})!=6:
            raise AssertionError("six private states are not isolated")
        assert_six_axis_parity(decisions,z["fit_truth"],ends,dict(ref,source_events=source))
        native_h0=bank.lanes["full_note"].events
        outcome=metric(native_h0,z["fit_truth"],ends)
        if any(int(ref[k])!=int(outcome[k]) for k in COUNT_KEYS) or any(
               abs(float(ref[k])-float(outcome[k]))>1.e-12 for k in SCORES):
            raise AssertionError("S708 native full-note H0 MIDI six-score parity failed")
        # Each hypothetical H1 axis is overlaid in its OWN output only.
        original_axis={axis:decisions[axis][-1] for axis in AXES}
        for axis in AXES:
            shadow=checked_h1_view(original_axis,("unpromoted_H1",row["index"],axis),axis)
            if shadow[axis]==original_axis[axis] or any(
                    shadow[other]!=original_axis[other] for other in AXES if other!=axis):
                raise AssertionError("cross-axis H1 contamination")
            foreign_comparisons+=5
        tracks+=1;sample_decisions+=len(ends)
        events_copied+=len(source)*6
        scores.append(outcome);by_composition[z["composition"]].append(outcome)
        print("S812_REAL_S708_H0_FIT",row["index"],z["name"],flush=True)
    combined=aggregate(scores)
    for k in COUNT_KEYS:
        if int(combined[k])!=int(m["S795_archived_fit_counts"][k]):
            raise AssertionError("80-track S708 original source counters drift: "+k)
    for k in SCORES:
        if abs(float(combined[k])-float(m["FIT_source_scores"][k]))>1.e-12:
            raise AssertionError("80-track S708 six score parity failed: "+k)
    if len(by_composition)!=5:raise AssertionError("five real compositions required")
    return dict(schema=SCHEMA,complete=True,source_run=39352880035,
      reference="S708",tracks=tracks,compositions=sorted(by_composition),
      native_256_frame_decisions=sample_decisions,signed_native_event_copies=events_copied,
      six_independent_original_MIDI_projections=True,
      protected_H0_bit_exact=True,original_source_MIDI_unchanged=True,
      thirty_cross_axis_shadow_invariance_checks=foreign_comparisons,
      original_S707_offset_events_unmodified=True,
      H0_is_a_projection_not_six_recovered_trained_neural_weights=True,
      S810_new_models_are_never_promoted_automatically=True,
      original_source_counts={k:int(combined[k]) for k in COUNT_KEYS},
      original_six_MIDI256_scores={k:float(combined[k]) for k in SCORES},
      by_composition={c:{k:float(aggregate(v)[k]) for k in SCORES} for c,v in by_composition.items()},
      no_future_event_consumed=True,added_lookahead_samples=0,added_hold_samples=0,
      actual_CPU_iRig_latency_measured=False,production_LIVE_modified=False,
      CAL_opened=False,HELD_opened=False,TEST_opened=False,promotion=False)

def main():
    p=argparse.ArgumentParser()
    for name in ("manifest","traces","out"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("S812 immutable output exists")
    result=evaluate(args.manifest,args.traces)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("S812_NATIVE_H0_80_TRACK_SIX_AXIS_PASS",
          result["tracks"],result["original_six_MIDI256_scores"],flush=True)

if __name__=="__main__":main()
