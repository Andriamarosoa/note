"""S635 common strict causal NoteOn evaluator for MIDI vs NOTE.

Shares the matching *contract*, not models, between Andriamarosoa/midi
(causal_event_metrics.py) and Andriamarosoa/note. Evaluates two systems on
EXACTLY the same clips, hashes, labels and latency deadlines. Incompatible
historical F1 numbers must never be compared as if this were already run.

A predicted MIDI NoteOn can only match an earlier or simultaneous true
onset of the SAME pitch, each reference and prediction at most once.
Retriggers prefer the latest eligible reference. No backdated events.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import math
import re

DEADLINES_MS=(12.,18.,24.,35.,46.,80.,250.)
SHA256_HEX=re.compile(r"^[0-9a-f]{64}$")


def _finite_number(value,label):
    if isinstance(value,bool) or not isinstance(value,(float,int)):
        raise ValueError(f"{label} must be a number")
    v=float(value)
    if not math.isfinite(v):
        raise ValueError(f"{label} must be finite")
    return v


def _pitch(x):
    if type(x) is not int or not 0<=x<=127:
        raise ValueError("MIDI pitch must be an integer from 0..127")
    return x


def event_time(event):
    """NOTE events use audio availability; no decoder-retrospective timestamps."""
    if "available_sample" in event:
        if type(event["available_sample"]) is not int or event["available_sample"]<0:
            raise ValueError("available_sample must be nonnegative integer")
        if "decision_sample" in event:
            if type(event["decision_sample"]) is not int or event["decision_sample"]>event["available_sample"]:
                raise ValueError("backdated or noncausal emission")
        t=event["available_sample"]/44100.
        if "time" in event and abs(_finite_number(event["time"],"time")-t)>1e-6:
            raise ValueError("NOTE event time must match actual sample availability")
        return t
    for name in ("time_s","time"):
        if name in event:return _finite_number(event[name],name)
    raise ValueError("NoteOn must have causal output time or sample availability")


def extract_events(events,duration_s):
    """Accept unified explicit lists of emitted events, never inferred frames."""
    output=[]
    for index,item in enumerate(events):
        if item.get("type","note_on") not in ("note_on","note_off"):
            raise ValueError("Unexpected MIDI event type")
        if item.get("type","note_on")!="note_on":continue
        pitch=_pitch(item["pitch"])
        ts=event_time(item)
        if not 0<=ts<=duration_s+1e-6:
            raise ValueError("Output beyond evaluated audio")
        output.append((ts,pitch,index))
    output.sort(key=lambda v:(v[0],v[1],v[2]))
    return output


def extract_reference(notes,duration_s):
    output=[]
    for index,n in enumerate(notes):
        pitch=_pitch(n["pitch"])
        start=_finite_number(n["start_s"],"start_s")
        end=_finite_number(n["end_s"],"end_s")
        if not 0<=start<end<=duration_s+1e-6:
            raise ValueError("Invalid reference note interval")
        output.append((start,pitch,index))
    output.sort(key=lambda v:(v[0],v[1],v[2]))
    return output


def causal_score(reference,predictions,deadline_ms):
    """Greedy causal one-to-one matching following midi's latest-onset rule."""
    deadline=_finite_number(deadline_ms,"deadline_ms")
    if not 0<=deadline<=250:
        raise ValueError("Unsupported deadline")
    pending=defaultdict(list)
    ri=0
    paired=[]
    late_or_false=0
    for t,pitch,index in predictions:
        while ri<len(reference) and reference[ri][0]<=t+1e-12:
            rt,rpitch,rindex=reference[ri]
            pending[rpitch].append((rt,rindex))
            ri+=1
        possibilities=pending[pitch]
        possibilities[:]=[(rt,refid) for rt,refid in possibilities
                         if t-rt<=deadline/1000.+1e-12]
        if possibilities:
            rt,rindex=possibilities[-1]
            possibilities.clear()
            paired.append((rindex,index,(t-rt)*1000.))
        else:
            late_or_false+=1
    tp=len(paired);fp=len(predictions)-tp;fn=len(reference)-tp
    f1=2*tp/max(1,2*tp+fp+fn)
    precision=tp/max(1,tp+fp)
    recall=tp/max(1,tp+fn)
    values=sorted(x[2] for x in paired)
    def percentile(q):
        if not values:return None
        position=(len(values)-1)*q
        first=int(math.floor(position));second=int(math.ceil(position))
        return values[first]*(second-position)+values[second]*(position-first) if first!=second else values[first]
    return dict(TP=tp,FP=fp,FN=fn,precision=precision,recall=recall,F1=f1,
                observed_noteon_delay_ms=dict(p50=percentile(.50),
                   p95=percentile(.95)),matches=paired)


def compare_manifest(manifest):
    if manifest.get("protocol")!="NOTE_MIDI_SHARED_ONSET_V1":
        raise ValueError("Unsupported shared evaluation protocol")
    clips=manifest.get("clips",[])
    if not isinstance(clips,list) or not clips:raise ValueError("Nonempty clips required")
    counted={}
    for clip in clips:
        clipid=clip["clip_id"]
        if clipid in counted or not isinstance(clipid,str) or not clipid.strip():
            raise ValueError("Duplicate or invalid clip id")
        duration=_finite_number(clip["duration_s"],"duration_s")
        if duration<=0:raise ValueError("Invalid clip duration")
        source=clip["audio_sha256"]
        if not isinstance(source,str) or not SHA256_HEX.fullmatch(source):
            raise ValueError("Each evaluated audio must be hashed")
        refs=extract_reference(clip["reference"],duration)
        engines={}
        for system in ("note","midi"):
            info=clip[system]
            if info.get("audio_sha256")!=source:
                raise ValueError("Both systems must process identical audio bytes")
            if info.get("audio_sample_rate")!=44100:
                raise ValueError("Both systems must process PCM at 44.1kHz")
            events=extract_events(info["events"],duration)
            engines[system]={str(int(deadline) if deadline.is_integer() else deadline):
                causal_score(refs,events,deadline)
                for deadline in DEADLINES_MS}
        counted[clipid]=dict(corpus=clip.get("corpus","unspecified"),
                             audio_sha256=source,reference_notes=len(refs),
                             note=engines["note"],midi=engines["midi"])
    aggregated={}
    for system in ("note","midi"):
        aggregated[system]={}
        for deadline in DEADLINES_MS:
            key=str(int(deadline))
            totals={k:sum(c[system][key][k] for c in counted.values())
                   for k in ("TP","FP","FN")}
            tp,fp,fn=totals["TP"],totals["FP"],totals["FN"]
            totals["precision"]=tp/max(1,tp+fp)
            totals["recall"]=tp/max(1,tp+fn)
            totals["F1"]=2*tp/max(1,2*tp+fp+fn)
            aggregated[system][key]=totals
    return dict(protocol="NOTE_MIDI_SHARED_ONSET_V1",recordings=len(counted),
       matching="strictly causal, same pitch, one-to-one, latest reference onset",
       common_deadlines_ms=DEADLINES_MS,
       emitted_time_not_retroactively_shifted=True,
       overall=aggregated,by_clip=counted,
       no_physical_microphone_latency_claim=True,
       no_historical_S614_test_comparison_implied=True)


def load_and_compare(path):
    return compare_manifest(json.loads(Path(path).read_text(encoding="utf-8")))
