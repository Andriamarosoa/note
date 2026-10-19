"""S746: preserve the signed source MIDI stream while routing independent births.

All supplemental notes are proposals from NOTE's native dynamic multi-head
selector. The scorer sees only features available at the proposal NoteOn;
the merger reads baseline and supplemental events in availability order.
Never edit or re-run S707 over a previously accepted S708 note instance.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class InstanceGate:
    risk: float = 0.80
    k_tail: float = 0.40

    def validate(self):
        if not (0.0 <= self.risk <= 1.0 and 0.0 <= self.k_tail <= 1.0):
            raise ValueError("S746 gate outside [0,1]")
        return self


def extra_pairs(events):
    """Extract paired S743 proposed notes from an *ungated* S707 stream.

    Offsets are never inspected to accept or reject a NoteOn; they are
    carried only to close already accepted proposals at their actual
    causal availability time. S707 emits one off for every accepted on.
    """
    opened = {}
    result = []
    for e in events:
        pitch = int(e["pitch"])
        if e["type"] == "note_on":
            if pitch in opened:
                raise AssertionError("S746 candidate duplicate open pitch")
            opened[pitch] = e
        elif e["type"] == "note_off":
            on = opened.pop(pitch, None)
            if on is None:
                continue
            if on.get("reason") == "s743_multifundamental":
                result.append((on, e))
    if opened:
        raise AssertionError("Unclosed native candidate")
    return result


def onset_features(pairs, ends, birth_input, acoustic_heads, k):
    """Causal per-event features, with no duration or future NoteOff fields."""
    ends = np.asarray(ends, np.int64)
    x = np.asarray(birth_input, np.float32)
    h = np.asarray(acoustic_heads, np.float32)
    counts = np.asarray(k, np.float32)
    if not len(ends) or x.ndim != 3 or x.shape[:2] != (len(ends), 49) or \
       h.shape != (len(ends), 49, 6) or counts.shape != (len(ends), 7):
        raise ValueError("S746 candidate acoustic ABI mismatch")
    rows, indices = [], []
    for on, _ in pairs:
        frame = int(np.searchsorted(ends, on["decision_sample"]))
        pitch = int(on["pitch"]) - 40
        if frame >= len(ends) or int(ends[frame]) != on["decision_sample"] or \
           not (0 <= pitch < 49):
            raise AssertionError("S746 proposal is not at a causal fragment")
        indices.append((frame, pitch))
        rows.append(np.concatenate((x[frame, pitch], h[frame, pitch],
                                   counts[frame]), axis=0))
    features = np.asarray(rows, dtype=np.float32) if rows else \
        np.empty((0, x.shape[2] + 13), dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError("S746 nonfinite causal feature")
    return features, indices


def fit_event_labels(pairs, truth, baseline_matched):
    """FIT-only supervision: a truly NEW note with correct onset and release.

    More conservative than an onset-only classifier. Truth must NEVER be
    passed to onset_features(), merge_events() or inference.
    """
    result = []
    missing = set(range(len(truth))) - set(baseline_matched)
    for on, off in pairs:
        yes = any(
            j in missing and int(n["pitch"]) == int(on["pitch"])
            and abs(float(n["time"]) - float(on["time"])) <= .080
            and off["reason"] not in ("stream_end", "shutdown", "stream_error")
            and abs(float(off["time"]) - float(n["time"] + n["duration"])) <= .050
            for j, n in enumerate(truth)
        )
        result.append(int(yes))
    return np.asarray(result, dtype=np.int32)


def merge_events(reference, pairs, probabilities, indices, k, gate):
    """Online-equivalent priority mux: S708 originals always win.

    Neither an extra note's future off, nor future baseline events,
    influence the current extra NoteOn decision. A later S708 NoteOn can
    preempt an extra note (emitting a causal Off), never the reverse.
    """
    gate.validate()
    if len(pairs) != len(probabilities) or len(pairs) != len(indices):
        raise ValueError("S746 probabilities and events differ")
    counts = np.asarray(k, np.float32)
    if counts.ndim != 2 or counts.shape[1] != 7:
        raise ValueError("S746 invalid native Exact-K posterior")
    timeline = []
    for seq, event in enumerate(reference):
        # Preserve the original order of ALL frozen events, including ties.
        timeline.append((int(event["available_sample"]), 0, seq, "base", event, -1))
    for i, (on, off) in enumerate(pairs):
        timeline.append((int(on["available_sample"]), 2, i, "extra_on", on, i))
        off_priority = 3 if off["available_sample"] == on["available_sample"] else 1
        timeline.append((int(off["available_sample"]), off_priority, i, "extra_off", off, i))
    timeline.sort(key=lambda z: (z[0], z[1], z[2]))
    out, base, extra = [], set(), {}
    detail = dict(accepted=0, blocked_risk=0, blocked_k=0,
                  blocked_occupancy=0, preempted=0)
    for available, _, _, origin, e, i in timeline:
        pitch = int(e["pitch"])
        if origin == "base":
            if e["type"] == "note_off":
                if pitch not in base:
                    raise AssertionError("S746 frozen baseline off without on")
                base.remove(pitch)
            else:
                if pitch in extra:
                    out.append(dict(type="note_off", pitch=pitch,
                                    available_sample=available,
                                    decision_sample=available,
                                    time=float(e["time"]), velocity=0,
                                    reason="s746_source_priority"))
                    del extra[pitch]
                    detail["preempted"] += 1
                if pitch in base:
                    raise AssertionError("S746 duplicate signed source")
                while len(base) + len(extra) >= 6:
                    if not extra:
                        raise AssertionError("Frozen S708 exceeded six voices")
                    worst = min(extra, key=lambda p: (extra[p][1], extra[p][0]))
                    out.append(dict(type="note_off", pitch=int(worst),
                                    available_sample=available,
                                    decision_sample=available,
                                    time=float(e["time"]), velocity=0,
                                    reason="s746_source_budget_priority"))
                    del extra[worst]
                    detail["preempted"] += 1
                base.add(pitch)
            out.append(e.copy())  # Exact original event metadata and times.
            continue
        if origin == "extra_off":
            if pitch in extra and extra[pitch][0] == i:
                emitted = e.copy()
                emitted["_s746_extra"] = True
                out.append(emitted)
                del extra[pitch]
            continue
        if not np.isfinite(probabilities[i]) or probabilities[i] < gate.risk:
            detail["blocked_risk"] += 1
            continue
        frame, ix = indices[i]
        if frame < 0 or frame >= len(counts) or ix != pitch - 40:
            raise AssertionError("S746 fragment is not source-aligned")
        active = len(base) + len(extra)
        if active >= 6 or pitch in base or pitch in extra:
            detail["blocked_occupancy"] += 1
            continue
        if float(counts[frame, active+1:].sum()) < gate.k_tail:
            detail["blocked_k"] += 1
            continue
        emitted = e.copy()
        emitted["_s746_extra"] = True
        out.append(emitted)
        extra[pitch] = (i, float(probabilities[i]))
        detail["accepted"] += 1
    if extra or base:
        raise AssertionError("S746 merger left notes active")
    # Every signed baseline event remains byte-for-byte equivalent, with
    # no added event bearing its original reason.
    original = [(e["type"], e["pitch"], e["available_sample"], e["reason"])
                for e in reference]
    retained = [(e["type"], e["pitch"], e["available_sample"], e["reason"])
                for e in out if not e.get("_s746_extra")
                and not str(e.get("reason", "")).startswith("s746_")]
    if original != retained:
        raise AssertionError("S746 replaced a frozen source MIDI event")
    return out, detail
