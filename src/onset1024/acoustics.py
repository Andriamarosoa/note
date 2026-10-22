"""Exact-K ONLY: frozen S708 input pipeline; no MIDI output or other axis decoder.
Copies audited PCM preprocessing from S613/S706 to maintain ABI.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import hashlib, json
import numpy as np

SR, HOP = 44100, 1024
PITCHES = np.arange(40, 89)
WINDOWS = (2048, 4096, 8192)
SCHEMA = "S613_native_causal_harmonic_heads_v1"


class CausalPitchFeatures:
    """Right-aligned multi-resolution native harmonic features, bounded state.

    Past-only temporal differences extend the S511/S562/S569 acoustic idea;
    unlike those frozen ABIs this schema has no BP flags or event dependency.
    Inputs must be contiguous mono float PCM at 44.1 kHz. Reset on any gap.
    """

    def __init__(self, *, hop=HOP):
        if hop != 1024:
            raise ValueError("Experimental onset branch must use exactly 1024-sample inference hop")
        self.hop = int(hop)
        # Preserve the TRAINING-time physical interval of native S613 lags.
        # This is an approximation, not an independently trained checkpoint.
        self.lags = tuple(max(1, round(lag * HOP / self.hop)) for lag in (2, 4, 8, 16))
        self.recent_frames = max(1, round(8 * HOP / self.hop))
        self.history_frames = max(1, round(32 * HOP / self.hop))
        self.received = 0
        self.next_end = self.hop
        self.tail = np.empty(0, np.float32)
        self.history = deque(maxlen=self.history_frames)
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
        older = [self.history[-lag] if len(self.history) >= lag else zero for lag in self.lags]
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
        features.extend((
            history[-self.recent_frames:].mean(0), history.mean(0),
            physical - history[-self.recent_frames:].min(0),
        ))
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
            self.next_end += self.hop
        self.tail = joined[-WINDOWS[-1]:].copy()
        self.received = end
        return result



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


def load_native_heads(path, *, expected_sha256=None):
    if Path(path).suffix != ".npz":
        raise ValueError("Pinned S834/S828 acoustic source must be the original .npz")
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(str(z["metadata"]))
    kind = {"native_causal_tree_heads_onnx": NativeONNXLiveHeads,
            "native_causal_hybrid": NativeHybridLiveHeads}.get(metadata.get("architecture"), NativeLiveHeads)
    return kind(path, expected_sha256=expected_sha256)


def sharp_track(audio, ends):
    ends=np.asarray(ends,np.int64)
    if audio.ndim!=1 or not np.isfinite(audio).all():
        raise ValueError('Finite mono PCM required')
    if np.any(ends>len(audio)) or np.any(ends<1) or np.any(np.diff(ends)<=0):
        raise ValueError('Unavailable or non-monotonic decision sample')
    hz=(440*2.**((PITCHES-69)/12.))[:,None]*np.arange(1,9)[None,:]
    centers=[];windows=[];phase=[];grid=[]
    for n in WINDOWS:
        centers.append(hz*n/SR)
        windows.append(np.hanning(n))
        grid.append(np.arange(n//2+1))
        # Align FFT-bin phases to the common center of the PAST window
        # before interpolation; interpolation of raw opposing phases cancels.
        phase.append(np.exp(1j*2*np.pi*np.arange(n//2+1)*(n-1)/(2*n)))
    result=[];previous=None;previous_end=None
    for end in ends:
        logs=[];contrasts=[];complex_values=[]
        for n,window,center,rot,bins in zip(WINDOWS,windows,centers,phase,grid):
            pcm=np.asarray(audio[max(0,int(end)-n):int(end)],np.float64)
            if len(pcm)<n:pcm=np.pad(pcm,(n-len(pcm),0))
            spectrum=np.fft.rfft(pcm*window)*(2/window.sum())
            amplitude=np.abs(spectrum)
            def interp(values,freq):
                return np.interp(freq.ravel(),bins,values).reshape(49,8)
            own=interp(amplitude,center)
            sides=(interp(amplitude,center*2**(-.5/12))+
                   interp(amplitude,center*2**(.5/12)))*.5
            logs.append(np.log1p(600*own))
            contrasts.append(np.log1p(600*own)-np.log1p(600*sides))
            aligned=spectrum*rot
            value=interp(aligned.real,center)+1j*interp(aligned.imag,center)
            complex_values.append(value)
        value=np.stack(complex_values,1)
        cos=np.zeros((49,3,8));sin=np.zeros_like(cos)
        if previous is not None:
            advance=np.exp(-1j*2*np.pi*hz[:,None,:]*(int(end)-previous_end)/SR)
            prod=value*np.conj(previous)*advance
            norm=np.abs(value)*np.abs(previous)
            valid=norm>1e-12
            coherence=np.divide(prod,np.maximum(norm,1e-12),out=np.zeros_like(prod),where=valid)
            cos=coherence.real;sin=coherence.imag
        previous=value;previous_end=int(end)
        result.append(np.concatenate((np.stack(logs,1).reshape(49,24),
            np.stack(contrasts,1).reshape(49,24),cos.reshape(49,24),sin.reshape(49,24)),1))
    out=np.asarray(result,np.float32)
    if out.shape!=(len(ends),49,96) or not np.isfinite(out).all():
        raise AssertionError('Sharp harmonic feature ABI failure')
    return out
