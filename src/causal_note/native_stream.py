"""Bounded streaming of NOTE's original acoustic features and pitch selectors.

This is NOT an audio-only S80 frontend. Canonical event boundaries and predicted
K still come from the original S80/S397 producer. No event detector, count
heuristic, foreign recognition model, learned offset or fallback is substituted.

The original -80/+160 ms acoustic window is evaluated only after its real audio
has arrived. PR16 geometry additionally needs the next canonical event, or a
producer watermark certifying no unseen event up to +240 ms. A watermark means
all events with start < watermark have been delivered, including K0 events.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s82_external_pr16_neural_inference import ownership

SR = 44100
PRE = 3528
POST = 7056
GEOMETRY_HORIZON = 10584


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class NativeEvent:
    row_id: str
    start_sample: int
    end_sample: int

    def __post_init__(self):
        if not isinstance(self.row_id, str) or not self.row_id.strip():
            raise ValueError("Original native row_id is mandatory")
        _integer(self.start_sample, "start_sample")
        _integer(self.end_sample, "end_sample", self.start_sample)
        if self.end_sample - self.start_sample > 1764:
            raise ValueError("Original native groups span at most 1764 samples")


@dataclass(frozen=True)
class NativeFeatures:
    event: NativeEvent
    available_sample: int
    summary58: np.ndarray
    sequence42x49: np.ndarray
    flow245: np.ndarray
    geometry3: np.ndarray
    morphology490: np.ndarray
    physics294: np.ndarray
    geometry46: np.ndarray | None


class NativeFeatureStream:
    """Single-consumer, contiguous 44.1 kHz mono audio; fixed-size audio storage.

    Events must arrive before their -80 ms context is overwritten. Missing
    samples, late events, queue overflow and invalid watermarks raise explicitly;
    they never become silence, synthetic candidates or a different K. On an
    audio discontinuity the caller must end this stream and start a new one.
    """

    def __init__(self, *, include_geometry46=False, history_samples=2 * SR,
                 max_pending=128):
        _integer(history_samples, "history_samples", PRE + GEOMETRY_HORIZON)
        _integer(max_pending, "max_pending", 1)
        self.include_geometry46 = bool(include_geometry46)
        self.capacity = history_samples
        self.max_pending = max_pending
        self._audio = np.empty(history_samples, np.float64)
        self._received = 0
        self._watermark = 0
        self._pending = deque()
        self._previous = None
        self._last_event = None

    @property
    def received_samples(self):
        return self._received

    @property
    def pending_events(self):
        return tuple(self._pending)

    def _slice(self, start, end):
        if start < max(0, self._received - self.capacity) or end > self._received:
            raise BufferError("Required real acoustic context is unavailable")
        index = np.arange(start, end) % self.capacity
        return self._audio[index]

    def push(self, samples, *, start_sample, events=(), event_watermark=None):
        """Append one block and return newly mature original feature packets.

        An empty block may deliver delayed event decisions/watermarks. Advancing
        a watermark never fabricates audio; both dependencies must be satisfied.
        """
        _integer(start_sample, "start_sample")
        if start_sample != self._received:
            raise ValueError("Audio gap/overlap: a new stream is required")
        x = np.asarray(samples)
        if x.ndim != 1 or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise ValueError("Audio must be finite normalized mono floating point")
        if len(x) > self.capacity:
            raise BufferError("Audio block exceeds bounded history")
        end = start_sample + len(x)
        new = tuple(events)
        if len(self._pending) + len(new) > self.max_pending:
            raise BufferError("Native event queue full; no event may be dropped")
        previous = self._last_event
        active_ids = {e.row_id for e in self._pending}
        if previous is not None:
            active_ids.add(previous.row_id)
        for event in new:
            if not isinstance(event, NativeEvent):
                raise TypeError("Expected canonical NativeEvent")
            if event.end_sample >= end:
                raise ValueError("Event end has not yet been observed in the audio")
            if event.start_sample < self._watermark:
                raise ValueError("Event contradicts the producer's committed watermark")
            if previous is not None and event.start_sample <= previous.end_sample:
                raise ValueError("Native events must be ordered and non-overlapping")
            if event.row_id in active_ids:
                raise ValueError("Duplicate pending native event identity")
            active_ids.add(event.row_id)
            previous = event
        watermark = self._watermark if event_watermark is None else event_watermark
        _integer(watermark, "event_watermark")
        if not self._watermark <= watermark <= end:
            raise ValueError("Watermark must advance monotonically within received audio")
        pending = tuple(self._pending) + new
        if pending and max(0, pending[0].start_sample - PRE) < max(0, end - self.capacity):
            raise BufferError("Native producer delayed beyond retained acoustic history")
        # Validation above is atomic with respect to bad audio/event input.
        self._audio[np.arange(start_sample, end) % self.capacity] = x
        self._received = end
        self._watermark = watermark
        self._pending.extend(new)
        self._last_event = previous
        output = []
        while self._pending:
            event = self._pending[0]
            if end < event.start_sample + POST:
                break
            if self.include_geometry46 and watermark < event.start_sample + GEOMETRY_HORIZON:
                break
            following = self._pending[1] if len(self._pending) > 1 else None
            output.append(self._extract(event, self._previous, following))
            self._pending.popleft()
            self._previous = event
        return tuple(output)

    def _extract(self, event, previous, following):
        begin = max(0, event.start_sample - PRE)
        window = self._slice(begin, event.start_sample + POST)
        spec, times = _frames(window, event.start_sample - begin)
        summary, activation = features_from_spectrogram(spec, times, return_states=True)
        seq = np.log1p(activation).astype(np.float32)
        if len(summary) != 58 or seq.shape != (42, 49):
            raise ValueError("Original NOTE acoustic feature ABI changed")
        trajectory = seq[None]
        flow = flow_profile(trajectory)[0]
        morph, physics = audio_temporal_views(trajectory)
        # S97 stores the inter-start gap as float32 BEFORE the nonlinear views.
        gap = np.array([2. if previous is None else
                        np.clip((event.start_sample - previous.start_sample) / SR, 0, 2)],
                       np.float32)
        geometry3 = np.column_stack([
            [(event.end_sample - event.start_sample) / 1764.],
            np.log1p(gap), np.minimum(gap, .2) / .2])[0].astype(np.float32)
        geometry46 = None
        if self.include_geometry46:
            neighbors = ([previous] if previous else []) + [event] + ([following] if following else [])
            starts = np.array([e.start_sample for e in neighbors], np.int64)
            ends = np.array([e.end_sample for e in neighbors], np.int64)
            query = event.start_sample + np.rint(times * SR / 1000.).astype(np.int64)
            mask = ownership(starts, ends, int(previous is not None), query)
            before = (event.start_sample - previous.end_sample) / 10584. if previous else 1.
            after = (following.start_sample - event.start_sample) / 10584. if following else 1.
            geometry46 = np.r_[mask, (event.end_sample - event.start_sample) / 1764.,
                               np.clip(before, 0, 1), np.clip(after, 0, 1), mask.mean()].astype(np.float32)
        return NativeFeatures(event, self._received,
                              np.asarray([summary[k] for k in sorted(summary)], np.float32),
                              seq, flow, geometry3, morph[0], physics[0], geometry46)


class NativePitchHeads:
    """Load S430's twelve NOTE checkpoints once, retaining both native gates.

    The four-fold new-input ensemble is the existing S430 policy. It is not the
    composition-held evaluation protocol of S402/S416/S428 and has no certified
    live accuracy. K is a model input AND the exact final note budget.
    """

    def __init__(self, s402, s416, s428, report419, report428):
        from scripts.s430_infer_note_names_from_new_S80_sound_events import load_models
        from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import sha
        self.head402, self.head416, self.head428, self.gates, self.hashes = load_models(
            *map(Path, (s402, s416, s428, report419, report428)))
        self.hashes.update(S419_gate=sha(Path(report419)), S428_gate=sha(Path(report428)))

    def predict(self, sequence, counts):
        import torch
        from scripts.s402_k_input_first_supervised_midi_pitch_network import feature_bank
        from scripts.s428_train_sound_to_note_names_harmonic_ranker import build_features, FEATURES
        from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import choose_pitches
        s = np.asarray(sequence, np.float32)
        k = np.asarray(counts)
        if (s.ndim != 3 or s.shape[1:] != (42, 49) or not np.isfinite(s).all()
                or np.any(s < 0) or k.shape != (len(s),) or k.dtype.kind not in "iu"
                or not np.isin(k, range(7)).all()):
            raise ValueError("Native 42x49 features and original integer S80 K0..K6 required")
        if len(k) == 0:
            return np.empty((0, 6), np.int16), ()
        ft = feature_bank(s, k)
        prob402 = np.zeros((len(k), 49), np.float32)
        for net in self.head402:
            prob402 += net["network"].predict_proba(
                net["transform"].transform(ft)).astype(np.float32) / 4
        prob416 = np.zeros_like(prob402)
        with torch.inference_mode():
            for net in self.head416:
                prob416 += net(torch.from_numpy(s), torch.from_numpy(k.astype(np.int64))).sigmoid().numpy() / 4
        old, _ = choose_pitches(prob402, k)
        cnn, _ = choose_pitches(prob416, k)
        reference = old.copy()
        for i, count in enumerate(k):
            if count and self.gates[str(int(count))]["S419_use_CNN"]:
                reference[i] = cnn[i]
        physical = build_features(s, k, prob416, reference, old)
        confidence = np.zeros_like(prob402)
        for net in self.head428:
            confidence += net.predict_proba(physical.reshape(-1, FEATURES))[:, 1].reshape(-1, 49).astype(np.float32) / 4
        acoustic, _ = choose_pitches(confidence, k)
        methods = []
        for i, count in enumerate(k):
            if not count:
                methods.append("K0")
                continue
            gate = self.gates[str(int(count))]
            method = "S428" if gate["S428_use_harmonic_ranker"] else ("S416" if gate["S419_use_CNN"] else "S402")
            methods.append(method)
            if method == "S428":
                reference[i] = acoustic[i]
        if not np.array_equal((reference >= 0).sum(1), k):
            raise AssertionError("NOTE pitch selection changed original S80 cardinality")
        return reference, tuple(methods)


class NativeNoteStream:
    """Streaming note names with externally supplied canonical S80 events/K.

    ``decisions`` pairs each NativeEvent with its original predicted K. Output
    carries both acoustic onset and actual availability in the received stream.
    It does not invent offsets; K0 means no NEW notes, not a silence decision.
    """

    def __init__(self, heads, **feature_options):
        self.heads = heads
        self.features = NativeFeatureStream(**feature_options)
        self._counts = {}

    def push(self, samples, *, start_sample, decisions=(), event_watermark=None):
        from scripts.s399_s80_exactK_constrained_midi_pitch_decoder import note_name
        decisions = tuple(decisions)
        for event, k in decisions:
            _integer(k, "original S80 K")
            if k > 6:
                raise ValueError("S80 K outside K0..K6")
        packets = self.features.push(samples, start_sample=start_sample,
                                     events=[event for event, k in decisions],
                                     event_watermark=event_watermark)
        self._counts.update((e.row_id, k) for e, k in decisions)
        if not packets:
            return ()
        k = np.asarray([self._counts[p.event.row_id] for p in packets], np.int8)
        pitches, methods = self.heads.predict(np.stack([p.sequence42x49 for p in packets]), k)
        result = []
        for packet, slots, method in zip(packets, pitches, methods):
            event = packet.event
            count = self._counts.pop(event.row_id)
            notes = [int(p) for p in slots if p >= 0]
            result.append(dict(source_row_id=event.row_id, start_sample=event.start_sample,
                               end_sample=event.end_sample, K=count, MIDI_pitches=notes,
                               notes=[note_name(p) for p in notes], selected_head=method,
                               available_sample=packet.available_sample,
                               acoustic_wait_ms=(packet.available_sample - event.start_sample) * 1000. / SR))
        return tuple(result)
