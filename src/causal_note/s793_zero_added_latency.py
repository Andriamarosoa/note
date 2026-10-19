"""S793 strict rejection gate: zero incremental latency, not zero physical latency.

Portable, standard-library-only. This is a *veto* for proposed upgrades,
not evidence that a design is genuinely real-time on an actual device.
The final decision requires paired runtime traces from the same device.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

SOURCE_ID = "s793_source_id"
SCORES = ("onset_f1", "exact_k", "active_exact_k", "offset_accuracy",
          "full_note_f1", "frame_pitch_f1")

class LatencyViolation(AssertionError):
    """The proposal MUST NOT be promoted, even if its accuracy improves."""

@dataclass(frozen=True)
class SchedulingContract:
    extra_lookahead_samples: int = 0
    extra_buffer_samples: int = 0
    extra_confirmation_samples: int = 0
    extra_hold_samples: int = 0
    extra_scheduler_hops: int = 0

    def validate(self):
        for name, value in vars(self).items():
            if type(value) is not int or value != 0:
                raise LatencyViolation(f"S793 no extra latency: {name}={value!r}")
        return self


def mark_reference(events):
    """Preserve explicit native event lineage before modifying MIDI proposals."""
    tagged=[]
    for index, event in enumerate(events):
        e=dict(event)
        if SOURCE_ID in e:
            raise LatencyViolation("preexisting source lineage tag")
        e[SOURCE_ID] = index
        tagged.append(e)
    return tagged


def _check_event(e, who):
    if e.get("type") not in ("note_on", "note_off") or type(e.get("pitch")) is not int \
            or not 40 <= e["pitch"] <= 88:
        raise LatencyViolation(f"{who}: malformed note event")
    for k in ("decision_sample", "available_sample"):
        if type(e.get(k)) is not int or e[k] < 0:
            raise LatencyViolation(f"{who}: invalid {k}")
    if e["available_sample"] < e["decision_sample"]:
        raise LatencyViolation(f"{who}: backdated MIDI event")


def veto_added_event_delay(reference, candidate):
    """No later emission for EVERY existing native NoteOn/NoteOff.

    Missing or duplicated lineage rejects fail-closed. Novel events need a
    logged first eligible decision; its provenance requires independent audit.
    """
    ref={}
    for e in reference:
        _check_event(e,"reference")
        key=e.get(SOURCE_ID)
        if type(key) is not int or key in ref:
            raise LatencyViolation("reference event IDs absent or repeated")
        ref[key]=e
    seen=set()
    last_time=-1
    for e in candidate:
        _check_event(e,"candidate")
        when=e["available_sample"]
        if when<last_time:
            raise LatencyViolation("candidate MIDI event stream unsorted")
        last_time=when
        key=e.get(SOURCE_ID)
        if key is None:
            first=e.get("first_eligible_sample")
            if type(first) is not int or first<0 or first!=when:
                raise LatencyViolation("new NoteOn/Off was held or lacks independent first-eligible proof")
            if e["decision_sample"] > first:
                raise LatencyViolation("new decision made after first eligible sample")
            continue
        if type(key) is not int or key not in ref or key in seen:
            raise LatencyViolation("source event lineage altered, lost or duplicated")
        seen.add(key)
        native=ref[key]
        if e["type"]!=native["type"] or e["pitch"]!=native["pitch"]:
            raise LatencyViolation("baseline MIDI note identity replaced")
        if when>native["available_sample"]:
            raise LatencyViolation(f"added {e['type']} latency for pitch {e['pitch']}: "
                f"{when-native['available_sample']} samples")
        if e["decision_sample"]>native["decision_sample"]:
            raise LatencyViolation(f"later audio decision for source event {key}")
    if seen!=set(ref):
        raise LatencyViolation("original MIDI events missing from candidate")
    return dict(native_events_checked=len(ref),candidate_events_checked=len(candidate),
                max_added_event_samples=0)


def _number(value, name):
    if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
        raise LatencyViolation(f"missing/invalid paired physical timing {name}")
    return float(value)


def veto_runtime_regression(baseline, candidate):
    """Require same-device measured per-path timing; never infer from FLOPs.

    Mandatory capture->NoteOn, capture->NoteOff, end-to-end and CPU
    processing p50, p95, maximum, all in ms. Missing data vetoes promotion.
    """
    if not isinstance(baseline,dict) or not isinstance(candidate,dict):
        raise LatencyViolation("measured baseline and candidate required")
    for key in ("hardware_id","audio_device_id","driver_mode","sample_rate_hz",
                "block_size_samples","hop_size_samples","workload_sha256"):
        if not baseline.get(key) or baseline.get(key)!=candidate.get(key):
            raise LatencyViolation(f"not a paired same-device experiment: {key}")
    if not baseline.get("paired_repeated_trials") or not candidate.get("paired_repeated_trials"):
        raise LatencyViolation("missing repeated paired hardware latency evidence")
    measurements={}
    for path in ("capture_to_noteon","capture_to_noteoff","total_audio_to_midi","cpu_inference"):
        for statistic in ("p50_ms","p95_ms","max_ms"):
            key=f"{path}_{statistic}"
            b=_number(baseline.get(key),key+" baseline")
            c=_number(candidate.get(key),key+" candidate")
            if c>b:
                raise LatencyViolation(f"latency regression {key}: {c-b:.6f} ms")
            measurements[key]=dict(baseline=b,candidate=c,delta=c-b)
    return dict(measured_paths=measurements,max_allowed_added_ms=0.)


def veto_metric_regression(baseline,candidate):
    """Existing six-metric non-regression remains a separate hard gate."""
    for key in SCORES:
        b=_number(baseline.get(key),"baseline "+key)
        c=_number(candidate.get(key),"candidate "+key)
        if b>1 or c>1 or c<b:
            raise LatencyViolation(f"S793 six-metric regression or invalid score: {key}")
    return dict(axes_checked=list(SCORES),metric_regressions=0)


def veto_promotion(*,contract,reference,candidate,baseline_runtime,candidate_runtime,
                   baseline_metrics,candidate_metrics):
    """All four independent proofs mandatory; fail closed if any are missing."""
    return dict(scheduling=contract.validate(),
                events=veto_added_event_delay(reference,candidate),
                runtime=veto_runtime_regression(baseline_runtime,candidate_runtime),
                metrics=veto_metric_regression(baseline_metrics,candidate_metrics),
                permitted=True)
