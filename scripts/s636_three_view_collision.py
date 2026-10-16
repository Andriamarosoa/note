"""S636: evaluate native/rescue pitch coincidences, NOT transcription accuracy.

This audit uses three-view S634/S635 attack proposal JSONs and compares
each rescue to native attack candidates. Offline ±80ms coincidences may
include FUTURE events. Separate CAUSAL classification uses exclusively
native attack candidates occurring at or before rescue, never future.
Octave coincidences are risk flags, NOT definitive false-positive labels:
real music often contains simultaneous octave notes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def audit_three_view_report(report, *, tolerance_ms=80.):
    if not 0 < tolerance_ms <= 250:
        raise ValueError("tolerance_ms must be within (0,250]")
    if report.get("kind") != "S634_three_view_shadow_candidates_only":
        raise ValueError("S636 requires the S634/S635 frozen three-view result")
    candidates=report.get("candidates",[])
    if not isinstance(candidates,list):
        raise ValueError("Candidate list required")
    if len(candidates)>250000:
        raise ValueError("Unbounded candidate payload")
    native=[]
    rescue=[]
    for e in candidates:
        if not isinstance(e,dict) or e.get("view") not in ("original","bass_x2","high_half"):
            raise ValueError("Unknown candidate view")
        if not isinstance(e.get("detected_pitch"),int) or not 40<=e["detected_pitch"]<=88:
            raise ValueError("Invalid MIDI pitch")
        try:
            t=float(e["time_s"])
        except (KeyError,TypeError,ValueError) as exc:
            raise ValueError("Invalid candidate event time") from exc
        if not 0<=t<=float(report.get("model_audio_seconds",1e9))+1e-3:
            raise ValueError("Out-of-range candidate time")
        if e["view"]=="original":
            native.append(e)
        else:
            rescue.append(e)
    detailed=[]
    for rescue_event in rescue:
        pitches=int(rescue_event["detected_pitch"])
        t=float(rescue_event["time_s"])
        matches=[]
        for reference in native:
            delta_ms=1000*(float(reference["time_s"])-t)
            if abs(delta_ms)>tolerance_ms+1e-7:
                continue
            q=int(reference["detected_pitch"])
            if q==pitches:
                kind="same_pitch"
            elif abs(q-pitches)==12:
                kind="octave_relation"
            else:
                continue
            matches.append(dict(kind=kind,
                                native_pitch=q,
                                native_time_s=reference["time_s"],
                                native_minus_rescue_ms=round(delta_ms,2)))
        matches.sort(key=lambda a:(a["kind"]!="same_pitch",abs(a["native_minus_rescue_ms"])))
        retrospect_same=next((a for a in matches if a["kind"]=="same_pitch"),None)
        retrospect_oct=next((a for a in matches if a["kind"]=="octave_relation"),None)
        causal_same=next((a for a in matches if a["kind"]=="same_pitch" and a["native_minus_rescue_ms"]<=0),None)
        causal_oct=next((a for a in matches if a["kind"]=="octave_relation" and a["native_minus_rescue_ms"]<=0),None)
        retrospective=("same_pitch" if retrospect_same else
                       "octave_relation" if retrospect_oct else "unmatched")
        causal=("same_pitch_already_known" if causal_same else
                "octave_already_known" if causal_oct else "requires_more_evidence")
        detailed.append(dict(
            view=rescue_event["view"],pitch=pitches,time_s=t,
            onset_confidence=rescue_event.get("attack_confidence"),
            native_K_at_rescue=rescue_event.get("native_active_K"),
            retrospective=retrospective,
            live_causal=causal,
            coincidences=matches,
            # A candidate detected earlier by rescue isn't a missed note
            # unless later independent annotations prove it.
            earlier_than_native_same_pitch_ms=(
                round(retrospect_same["native_minus_rescue_ms"],2)
                if retrospect_same and retrospect_same["native_minus_rescue_ms"]>0 else None),
        ))
    return dict(
        kind="S636_three_view_collision_audit",
        interpretation="coincidence audit, not true positive/false positive or F1",
        reference_frame="original NOTE candidate threshold crossings",
        max_nearby_time_delta_ms=tolerance_ms,
        original_candidate_count=len(native),rescue_candidate_count=len(rescue),
        retrospective_counts=dict(Counter(e["retrospective"] for e in detailed)),
        strictly_causal_counts=dict(Counter(e["live_causal"] for e in detailed)),
        octave_warning="Octave coincidences may be real chords, not proven aliases",
        never_use_future_native_for_live_decisions=True,
        details=detailed,
    )


def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True,
                    help="S635 three_views_mic.json (no WAV file needed)")
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--tolerance-ms",type=float,default=80.)
    args=ap.parse_args(argv)
    if args.out.exists():
        ap.error("Refusing to overwrite previous collision audit")
    report=json.loads(args.input.read_text(encoding="utf-8"))
    result=audit_three_view_report(report,tolerance_ms=args.tolerance_ms)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in (
        "kind","original_candidate_count","rescue_candidate_count",
        "retrospective_counts","strictly_causal_counts","octave_warning"
    )},indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
