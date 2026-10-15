"""Bounded S511/S515 acoustic frontend for NOTE's recent learned heads.

The original whole-recording median/p95 and flux median/MAD are noncausal.
`rolling3s` is an explicitly different, unpromoted normalization using only
frames available at the event's fixed right edge. A supplied `reference`
profile reproduces the offline ABI for audit, never certifies causal Live.
Original V88 events and finalized BP observations are required from upstream.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np
from scipy.signal import stft

SR, FFT, HOP = 44100, 8192, 441
MIDIS = np.arange(40, 89)


def spectra(sound):
    return np.abs(stft(np.asarray(sound, np.float32), fs=SR, window="hann",
                       nperseg=FFT, noverlap=FFT-HOP, nfft=FFT,
                       padded=False, boundary=None)[2]).astype(np.float32)


def pitch_scores(amp):
    """Original S511 harmonic weighting and floating-point reduction order."""
    outputs = []
    for pitch in MIDIS:
        fundamental = 440. * 2. ** ((pitch - 69.) / 12.)
        parts, weights = [], []
        for h in range(1, 9):
            freq = fundamental * h
            if freq > SR / 2 - 30:
                continue
            centre, radius = int(np.rint(freq * FFT / SR)), 2 if h <= 3 else 3
            parts.append(np.max(amp[max(1, centre-radius):min(len(amp), centre+radius+1)], axis=0))
            weights.append(1. / h ** .66)
        parts, weights = np.stack(parts), np.asarray(weights, np.float32)
        weighted_parts = np.log1p(parts * 600.) * weights[:, None]
        # S511 reduces a noncontiguous time column in harmonic order. NumPy
        # otherwise switches to pairwise summation for a single-frame block.
        weighted = weighted_parts[0].copy()
        for partial in weighted_parts[1:]:
            weighted += partial
        weighted /= weights.sum()
        outputs.append(.75 * weighted + .25 * np.log1p(parts[0] * 600.))
    return np.asarray(outputs, np.float32)


@dataclass(frozen=True)
class Normalization:
    median: np.ndarray
    scale: np.ndarray
    flux_median: float
    flux_scale: float

    @classmethod
    def fit(cls, pitch, flux):
        median = np.median(pitch, axis=1, keepdims=True)
        scale = np.maximum(.2, np.quantile(pitch, .95, axis=1, keepdims=True) - median)
        robust = float(np.median(flux))
        mad = float(np.median(np.abs(flux - robust)))
        return cls(median, scale, robust, max(.005, 1.4826 * mad))


def reference_profile(sound):
    """Audit only: this explicitly reads the complete recording, including future."""
    amp = spectra(sound)
    log = np.log1p(amp * 800.)
    flux = np.maximum(0, np.diff(log, axis=1, prepend=log[:, :1])).mean(axis=0)
    return Normalization.fit(pitch_scores(amp), flux)


@dataclass(frozen=True)
class RecentEvent:
    source: str
    ident: str
    time: float
    k: int
    births: tuple
    active: tuple
    support: tuple
    producer_ready_sample: int

    def __post_init__(self):
        if (not self.source or not self.ident or not np.isfinite(self.time) or self.time < 0
                or type(self.k) is not int or self.k not in range(7)
                or type(self.producer_ready_sample) is not int
                or self.producer_ready_sample < int(np.ceil((self.time + .030) * SR))
                or len(self.support) != 49 or not np.isfinite(self.support).all()
                or any(type(p) is not int or p not in range(40, 89) for p in (*self.births, *self.active))):
            raise ValueError("Original event and finalized BP birth/+30ms-active/support observations required")

    @property
    def near(self):
        # S511 frame centers begin at FFT/2; nearest tie goes to lower frame.
        position = (self.time * SR - FFT / 2) / HOP
        low = max(0, int(np.floor(position)))
        candidates = (low, low + 1)
        return min(candidates, key=lambda i: abs((FFT/2 + i*HOP)/SR - self.time))


class RecentAudioStream:
    """Continuous mono 44.1 kHz, bounded audio/feature memory, no future padding.

    Instances are single-consumer and one-source. Producers must send ordered
    canonical events with final BP observations before the 512-frame history
    expires. Output availability includes producer delay, never a fabricated
    NoteOff. `finish` is an explicit observed end-of-source, not live silence.
    """

    def __init__(self, *, normalization="rolling3s", profile=None):
        if normalization not in ("rolling3s", "reference"):
            raise ValueError("Choose rolling3s or explicit noncausal reference mode")
        if (normalization == "reference") != (profile is not None):
            raise ValueError("Only noncausal reference mode accepts a whole-recording profile")
        self.normalization, self.profile = normalization, profile
        self.received, self.frame_count, self.tail = 0, 0, np.empty(0, np.float32)
        self.frames, self.pending = deque(maxlen=512), deque()
        self.previous_log, self.last_key, self.source = None, None, None
        self.closed = False

    def push(self, sound, *, start_sample, events=()):
        x = np.asarray(sound)
        if self.closed or type(start_sample) is not int or start_sample != self.received:
            raise ValueError("Closed stream or audio gap/overlap: start a new stream")
        if x.ndim != 1 or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise ValueError("Finite mono floating-point PCM required")
        if len(x) > SR or len(self.pending) + len(events) > 128:
            raise BufferError("Audio block or pending event queue exceeds bounded capacity")
        end = self.received + len(x)
        key, source = self.last_key, self.source
        for e in events:
            if not isinstance(e, RecentEvent):
                raise TypeError("RecentEvent required")
            if e.producer_ready_sample > end or (source is not None and e.source != source):
                raise ValueError("Producer observations unavailable or another source")
            current = (e.time, e.ident)
            if key is not None and (current <= key or e.ident == key[1]):
                raise ValueError("Events must be unique and ordered by time then ID")
            source, key = e.source, current
        # Late observations cannot silently lose their acoustic/normalization history.
        first = self.frames[0][0] if self.frames else 0
        for e in (*self.pending, *events):
            need = max(0, e.near + 2 - (299 if self.profile is None else 27))
            if need < first:
                raise BufferError("Producer arrived after the retained acoustic history")
        self.pending.extend(events)
        self.source, self.last_key, self.received = source, key, end
        sound = np.concatenate((self.tail, x.astype(np.float32)))
        out = self._drain(False)
        if len(sound) >= FFT:
            amp = spectra(sound)
            raw = pitch_scores(amp)
            log = np.log1p(amp * 800.)
            previous = log[:, :1] if self.previous_log is None else self.previous_log[:, None]
            flux = np.maximum(0, np.diff(log, axis=1, prepend=previous)).mean(axis=0)
            for i in range(amp.shape[1]):
                self.frames.append((self.frame_count, raw[:, i].copy(), float(flux[i])))
                self.frame_count += 1
                out.extend(self._drain(False))
            self.previous_log = log[:, -1].copy()
            sound = sound[amp.shape[1] * HOP:]
        self.tail = sound
        return out

    def _drain(self, eof):
        output = []
        while self.pending and self.frames:
            e = self.pending[0]
            if not eof and self.frame_count <= e.near + 2:
                break
            near = min(e.near, self.frame_count - 1)
            right = min(near + 2, self.frame_count - 1)
            first = self.frames[0][0]
            earliest = max(0, right - (299 if self.profile is None else 27))
            if earliest < first:
                raise BufferError("Required original event context was overwritten")
            records = list(self.frames)[earliest-first:right-first+1]
            raw = np.stack([z[1] for z in records], axis=1)
            flux = np.asarray([z[2] for z in records], np.float32)
            profile = self.profile or Normalization.fit(raw, flux)
            rel = np.clip((raw - profile.median) / profile.scale, 0, 2.5)
            def indices(lo, hi):
                return np.arange(lo, hi) - earliest
            local = indices(max(0, near-2), right+1)
            past = indices(max(0, near-13), max(1, near-4))
            before = np.maximum(0, np.arange(max(0, near-2), right+1)-12) - earliest
            energy = raw[:, local[0]:local[-1]+1].mean(axis=1)
            spread = np.quantile(energy, .9) - np.quantile(energy, .1)
            absolute = np.clip((energy - np.quantile(energy, .1)) / max(.05, spread), 0, 2.5)
            strength = rel[:, local].max(axis=1)
            burst = np.maximum(0, rel[:, local] - rel[:, before]).max(axis=1)
            sustain = rel[:, past[0]:past[-1]+1].mean(axis=1)
            novelty = np.maximum(strength - sustain, 0)
            physical = max(0, (float(flux[near-earliest]) - profile.flux_median) / profile.flux_scale)
            x = np.asarray([[p/127., absolute[j], strength[j], burst[j], sustain[j], novelty[j],
                             float(p in e.births), float(p in e.active), e.support[j],
                             float(any(p-base in (12, 19, 24) for base in e.active if base < p)),
                             float(np.clip(physical / 5., 0, 2.))]
                            for j, p in enumerate(MIDIS)], np.float32)
            row = dict(source_recording_sha=e.source, V88_original_event_id=e.ident,
                       original_V88_candidate_start_s=e.time, original_V88_K_soft_prior_ONLY=e.k,
                       old_basicpitch_birth_pitches=sorted(set(e.births)),
                       old_basicpitch_active_pitches=sorted(set(e.active)),
                       per_midi_features_49x11=x.round(6).tolist())
            output.append(dict(row=row, available_sample=self.received,
                               audio_context_end_sample=FFT + right*HOP,
                               producer_ready_sample=e.producer_ready_sample,
                               normalization=self.normalization, end_of_source=eof))
            self.pending.popleft()
        return output

    def finish(self):
        if self.closed:
            raise ValueError("Stream already closed")
        if self.pending and not self.frames:
            raise ValueError("Source too short for a genuine 8192-sample FFT")
        result = self._drain(True)
        self.closed = True
        return result
