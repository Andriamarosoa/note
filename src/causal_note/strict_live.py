"""S613: native NOTE heads and selections driven only by received PCM.

This is a new, separately trained branch, not a causal claim about S80 or
S578. It retains NOTE's harmonic, competing-fundamental, temporal-head and
cardinality-selection design. No BP, V88 sidecar, future frame or score is
an inference input. The longest FFT is HISTORY, not a post-onset timer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SR, HOP = 44100, 256
PITCHES = np.arange(40, 89)
WINDOWS = (2048, 4096, 8192)
SCHEMA = "S613_native_causal_harmonic_heads_v1"


class CausalPitchFeatures:
    """Right-aligned multi-resolution native harmonic features, bounded state.

    Past-only temporal differences extend the S511/S562/S569 acoustic idea;
    unlike those frozen ABIs this schema has no BP flags or event dependency.
    Inputs must be contiguous mono float PCM at 44.1 kHz. Reset on any gap.
    """

    def __init__(self):
        self.received = 0
        self.next_end = HOP
        self.tail = np.empty(0, np.float32)
        self.history = deque(maxlen=32)
        self.last_spectra = [None] * len(WINDOWS)
        self.tapers = [np.hanning(n).astype(np.float32) for n in WINDOWS]
        self.indices = []
        hz = 440 * 2. ** ((PITCHES - 69) / 12.)
        for n in WINDOWS:
            center = np.rint(hz[:, None] * np.arange(1, 9) * n / SR).astype(int)
            self.indices.append(np.clip(center[..., None] + np.arange(-1, 2), 1, n // 2))
        self.weights = (1. / np.arange(1, 9) ** .66).astype(np.float32)

    def _frame(self, signal):
        partials, flux = [], []
        for j, (n, taper, ix) in enumerate(zip(WINDOWS, self.tapers, self.indices)):
            spectrum = (np.abs(np.fft.rfft(signal[-n:] * taper)) * (2 / taper.sum())).astype(np.float32)
            log = np.log1p(600 * spectrum).astype(np.float32)
            last = self.last_spectra[j]
            flux.append(float(np.maximum(log - last, 0).mean()) if last is not None else 0.)
            self.last_spectra[j] = log
            partials.append(log[ix].max(axis=-1))
        h = np.stack(partials, axis=1)  # pitch, resolution, harmonic
        score = .75 * (h * self.weights).sum(-1) / self.weights.sum() + .25 * h[..., 0]
        physical = np.concatenate((score, h[..., 0]), axis=1)
        zero = np.zeros_like(physical)
        older = [self.history[-lag] if len(self.history) >= lag else zero for lag in (2, 4, 8, 16)]
        relative = h / (.1 + h.max(axis=0, keepdims=True))
        features = [h.reshape(49, -1), relative.reshape(49, -1), physical]
        features.extend(physical - old for old in older)
        for offset in (-24, -19, -12, -1, 1, 12, 19, 24):
            other = np.zeros_like(score)
            if offset < 0:
                other[-offset:] = score[:offset]
            else:
                other[:-offset] = score[offset:]
            features.append(score - other)
        history = np.stack((*self.history, physical))
        features.extend((history[-8:].mean(0), history.mean(0), physical - history[-8:].min(0)))
        scalar = [*flux, np.log1p(100 * np.sqrt(np.mean(signal[-HOP:] ** 2))),
                  np.log1p(100 * np.sqrt(np.mean(signal ** 2)))]
        features.extend((np.broadcast_to(scalar, (49, len(scalar))), PITCHES[:, None] / 127.))
        self.history.append(physical)
        return np.concatenate(features, axis=1).astype(np.float32)

    def push(self, samples, *, start_sample):
        x = np.asarray(samples)
        if type(start_sample) is not int or start_sample != self.received:
            raise ValueError("Audio discontinuity: reset the stream, never bridge a gap")
        if x.ndim != 1 or x.dtype.kind != "f" or not np.isfinite(x).all():
            raise ValueError("Finite mono floating-point PCM required")
        if len(x) > 4096:
            raise BufferError("At most 4096 samples per input block")
        end = self.received + len(x)
        joined = np.concatenate((self.tail, x.astype(np.float32, copy=False)))
        origin = self.received - len(self.tail)
        result = []
        while self.next_end <= end:
            right = self.next_end - origin
            frame = joined[max(0, right - WINDOWS[-1]):right]
            if len(frame) < WINDOWS[-1]:
                frame = np.pad(frame, (WINDOWS[-1] - len(frame), 0))
            result.append((self.next_end, self._frame(frame)))
            self.next_end += HOP
        self.tail = joined[-WINDOWS[-1]:].copy()
        self.received = end
        return result


def feature_track(audio, block_size=256):
    """Training/audit cache made by the SAME stateful PCM frontend as Live."""
    stream = CausalPitchFeatures()
    frames, ends = [], []
    for start in range(0, len(audio), block_size):
        for end, row in stream.push(audio[start:start + block_size], start_sample=start):
            ends.append(end)
            frames.append(row)
    if not frames:
        raise ValueError("At least one hop of real audio is required")
    return np.asarray(ends, np.int64), np.stack(frames)


def count_features(x):
    """Native count head sees the whole 49-pitch acoustic competition."""
    # Absolute + relative first three harmonics at each of three resolutions.
    ix = [j * 8 + h for j in range(3) for h in range(3)]
    return np.asarray(x, np.float32)[..., ix].reshape(*x.shape[:-2], -1)


class NativeLiveHeads:
    """Small trained MLP heads exported as NumPy; no training runtime in Live."""

    def __init__(self, path, *, expected_sha256=None):
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Live checkpoint digest mismatch")
        with np.load(path, allow_pickle=False) as z:
            self.arrays = {k: z[k].copy() for k in z.files}
        if str(self.arrays.pop("schema")) != SCHEMA:
            raise ValueError("Wrong causal feature/checkpoint schema")
        self.metadata = json.loads(str(self.arrays.pop("metadata")))
        if not all(np.isfinite(v).all() for v in self.arrays.values()):
            raise ValueError("Non-finite model")

    def _net(self, name, x):
        a = self.arrays
        value = (x - a[name + ".mean"]) / a[name + ".scale"]
        for layer in range(3):
            value = value @ a[f"{name}.w{layer}"] + a[f"{name}.b{layer}"]
            if layer < 2:
                value = np.maximum(value, 0)
        return value

    def predict(self, features):
        x = np.asarray(features, np.float32)
        batched = x.ndim == 3
        if not batched:
            x = x[None]
        k = self._net("count", count_features(x))
        k = np.exp(k - k.max(-1, keepdims=True))
        k /= k.sum(-1, keepdims=True)
        # Predicted K distribution conditions the pitch heads, as in S402.
        joint = np.concatenate((x, np.broadcast_to(k[:, None], (*x.shape[:2], 7))), -1)
        logits = self._net("pitch", joint)
        p = 1. / (1. + np.exp(-np.clip(logits, -30, 30)))
        return (p, k) if batched else (p[0], k[0])


class NativeTreeLiveHeads(NativeLiveHeads):
    """Alternative small native tree heads on the identical causal features."""

    def __init__(self, path, *, expected_sha256=None):
        import joblib
        path = Path(path)
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and self.sha256 != expected_sha256:
            raise ValueError("Live checkpoint digest mismatch")
        model = joblib.load(path)
        if model["schema"] != SCHEMA or model["architecture"] != "native_causal_tree_heads":
            raise ValueError("Wrong causal tree schema")
        self.arrays, self.forests, self.metadata = model["count_arrays"], model["forests"], model["metadata"]

    def predict(self, features):
        x = np.asarray(features, np.float32)
        batched = x.ndim == 3
        if not batched:
            x = x[None]
        k = self._net("count", count_features(x))
        k = np.exp(k - k.max(-1, keepdims=True)); k /= k.sum(-1, keepdims=True)
        joint = np.concatenate((x, np.broadcast_to(k[:, None], (*x.shape[:2], 7))), -1)
        flat = joint.reshape(-1, joint.shape[-1])
        p = np.stack([model.predict_proba(flat)[:, 1] for model in self.forests], -1).reshape(*x.shape[:2], 3)
        return (p, k) if batched else (p[0], k[0])


class NativeONNXLiveHeads(NativeLiveHeads):
    """Exact native trained trees in a compiled CPU inference runtime."""

    def __init__(self, path, *, expected_sha256=None):
        super().__init__(path, expected_sha256=expected_sha256)
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.sessions = [ort.InferenceSession(self.arrays.pop(f"tree{i}").tobytes(),
                         sess_options=options, providers=["CPUExecutionProvider"]) for i in range(3)]

    def predict(self, features):
        x = np.asarray(features, np.float32)
        batched = x.ndim == 3
        if not batched:
            x = x[None]
        k = self._net("count", count_features(x))
        k = np.exp(k - k.max(-1, keepdims=True)); k /= k.sum(-1, keepdims=True)
        joint = np.concatenate((x, np.broadcast_to(k[:, None], (*x.shape[:2], 7))), -1)
        flat = np.ascontiguousarray(joint.reshape(-1, joint.shape[-1]), dtype=np.float32)
        probabilities = [session.run([session.get_outputs()[1].name], {session.get_inputs()[0].name: flat})[0][:, 1]
                         for session in self.sessions]
        p = np.stack(probabilities, -1).reshape(*x.shape[:2], 3)
        return (p, k) if batched else (p[0], k[0])


class NativeHybridLiveHeads(NativeONNXLiveHeads):
    """Native neural/tree consensus, with a development-frozen mixing weight."""

    def predict(self, features):
        tree, k = NativeONNXLiveHeads.predict(self, features)
        neural, _ = NativeLiveHeads.predict(self, features)
        alpha = self.metadata["tree_weight"]
        combined = np.power(np.clip(neural, 1e-8, 1), 1 - alpha) * np.power(np.clip(tree, 1e-8, 1), alpha)
        return combined, k


def load_heads(path, *, expected_sha256=None):
    if Path(path).suffix == ".joblib":
        kind = NativeTreeLiveHeads
    else:
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata"]))
        kind = {"native_causal_tree_heads_onnx": NativeONNXLiveHeads,
                "native_causal_hybrid": NativeHybridLiveHeads}.get(metadata.get("architecture"), NativeLiveHeads)
    return kind(path, expected_sha256=expected_sha256)


@dataclass(frozen=True)
class Selection:
    onset: float = .5
    active_on: float = .5
    active_off: float = .25
    onset_reset: float = .2
    release_frames: int = 5
    refractory_frames: int = 10
    max_notes: int = 6
    count_confidence: float = 1.1  # >1 disables learned count veto for ablation
    confirmation_frames: int = 1
    restrike_onset: float = .0

    def __post_init__(self):
        for key in ("onset", "active_on", "active_off", "onset_reset"):
            if not 0 < getattr(self, key) < 1:
                raise ValueError("Probability threshold must be in (0, 1)")
        if not 0 <= self.count_confidence <= 1.1:
            raise ValueError("Count confidence outside [0, 1.1]")
        if not 0 <= self.restrike_onset <= 1:
            raise ValueError("Restrike threshold outside [0, 1]")
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 1
               for key in ("release_frames", "refractory_frames", "max_notes", "confirmation_frames")):
            raise ValueError("Positive integer selection bounds required")
        if self.max_notes > 6:
            raise ValueError("Native Live budget is at most six notes")


class CausalSelection:
    """Per-pitch attack/restrike/release state; output time is never backdated."""

    def __init__(self, config=Selection()):
        self.config = config
        self.active = np.zeros(49, bool)
        self.armed = np.ones(49, bool)
        self.low = np.zeros(49, np.int32)
        self.high = np.zeros(49, np.int32)
        self.last_on = np.full(49, -10**12, np.int64)
        self.last_end = 0

    def step(self, probability, count, end_sample, available_sample):
        p, k, c = np.asarray(probability), np.asarray(count), self.config
        if (p.shape != (49, 3) or k.shape != (7,) or not np.isfinite(p).all()
                or not np.isfinite(k).all() or np.any((p < 0) | (p > 1))
                or end_sample != self.last_end + HOP or available_sample < end_sample):
            raise ValueError("Invalid probability stream or unavailable audio")
        self.last_end = end_sample
        onset, sustain, release = p.T
        self.armed |= onset < c.onset_reset
        self.low = np.where((sustain < c.active_off) | (release > .75), np.minimum(self.low + 1, c.release_frames), 0)
        events = []

        def emit(kind, i, reason):
            events.append(dict(type=kind, pitch=int(PITCHES[i]), decision_sample=int(end_sample),
                               available_sample=int(available_sample), time=available_sample / SR,
                               reason=reason, velocity=96 if kind == "note_on" else 0))

        for i in np.flatnonzero(self.active & (self.low >= c.release_frames)):
            emit("note_off", i, "causal_release")
            self.active[i] = False
        confident = (onset >= np.where(self.active, max(c.onset, c.restrike_onset), c.onset)) & (sustain >= c.active_on)
        self.high = np.where(confident, np.minimum(self.high + 1, c.confirmation_frames), 0)
        ready = (self.high >= c.confirmation_frames) & self.armed
        ready &= end_sample - self.last_on >= c.refractory_frames * HOP
        budget = c.max_notes
        if k.max() >= c.count_confidence:
            budget = min(budget, max(int(k.argmax()), int(self.active.sum())))
        for i in sorted(np.flatnonzero(ready), key=lambda j: (-float(onset[j] * sustain[j]), int(j))):
            if self.active[i]:
                emit("note_off", i, "restrike")
            elif self.active.sum() >= budget:
                continue
            emit("note_on", i, "causal_attack")
            self.active[i] = True
            self.armed[i] = False
            self.low[i] = 0
            self.last_on[i] = end_sample
        return events

    def panic(self, sample, reason="stream_end"):
        events = [dict(type="note_off", pitch=int(PITCHES[i]), decision_sample=int(sample),
                       available_sample=int(sample), time=sample / SR, velocity=0,
                       reason=reason) for i in np.flatnonzero(self.active)]
        self.active[:] = False
        return events


class StrictLiveStream:
    """Complete PCM -> native count/pitch/onset/off heads -> note events."""

    def __init__(self, heads, selection=None, *, harmonic_mode="off",
                 harmonic_checkpoint=None, harmonic_threshold=0.10):
        self.heads = heads
        self.frontend = CausalPitchFeatures()
        self.selector = CausalSelection(selection or Selection(**heads.metadata["selection"]))
        self.harmonic_selector = None
        if harmonic_mode not in ("off", "shadow", "guard"):
            raise ValueError("Unknown harmonic experiment mode")
        if harmonic_mode == "off" and harmonic_checkpoint is not None:
            raise ValueError("Harmonic checkpoint provided but experiment is disabled")
        if harmonic_mode != "off":
            # Imported lazily: the default S613 path is untouched.
            from causal_note.harmonic_expert import HarmonicSpecialistSelector
            self.harmonic_selector = HarmonicSpecialistSelector(
                self.selector, mode=harmonic_mode, checkpoint=harmonic_checkpoint,
                genuine_threshold=harmonic_threshold,
            )
        self.closed = False
        self.timings_ms = deque(maxlen=4096)

    def push(self, samples, *, start_sample):
        if self.closed:
            raise ValueError("Closed stream")
        before = time.perf_counter()
        # The PortAudio callback may deliver arbitrarily sized positive
        # blocks. Keep the native frontend's 4096-sample invariant, splitting
        # only in this transport adapter (no extra lookahead or FFT window).
        x = np.asarray(samples)
        events = []
        # Every sample in this one transport callback was already received.
        available_sample = start_sample + len(x)
        for offset in range(0, len(x), 4096):
            frames = self.frontend.push(
                x[offset:offset + 4096], start_sample=start_sample + offset,
            )
            for end, features in frames:
                p, k = self.heads.predict(features)
                if self.harmonic_selector is None:
                    events.extend(self.selector.step(
                        p, k, end, available_sample,
                    ))
                else:
                    events.extend(self.harmonic_selector.step(
                        features, p, k, end, available_sample,
                    ))
        self.timings_ms.append((time.perf_counter() - before) * 1000)
        return events

    def finish(self, reason="stream_end"):
        if self.closed:
            return []
        self.closed = True
        # EOF/shutdown closes notes, but cannot manufacture an acoustic offset.
        return self.selector.panic(self.frontend.received, reason)
