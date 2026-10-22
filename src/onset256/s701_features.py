"""S701 39D past-only features, original TRAIN ABI; one fresh row per PCM256."""
from collections import deque
import numpy as np

def lag(x, steps):
    """Look backward only; stable for the first frame."""
    t = np.maximum(0, np.arange(len(x)) - steps)
    return x[t]


def onset_features(feat, post, k):
    """Per-note 3D feature tensor from NOTE's source 126D ABI, current/past only."""
    feat = np.asarray(feat, np.float32)
    post = np.asarray(post, np.float32)
    k = np.asarray(k, np.float32)
    n = len(feat)
    if feat.shape != (n, 49, 126) or post.shape != (n, 49, 3) or k.shape != (n, 7):
        raise ValueError("S701 frozen NOTE ABI mismatch")
    if n == 0:
        return np.empty((0, 49, 39), np.float32)
    onset, sus, release = post.transpose(2, 0, 1)
    fundamental = np.maximum(feat[:, :, [0, 8, 16]], 0).mean(axis=2)
    overtone2 = np.maximum(feat[:, :, [1, 9, 17]], 0).mean(axis=2)
    overtone3 = np.maximum(feat[:, :, [2, 10, 18]], 0).mean(axis=2)
    near = np.maximum(feat[:, :, 54:57], 0).mean(axis=2)
    distant = np.maximum(feat[:, :, 66:69], 0).mean(axis=2)
    f_rise = (.65 * near + .35 * distant) / (.2 + fundamental)
    # Own fundamental vs evidence copied from lower-octave / fifth harmonics.
    cols = np.arange(49)
    lower12 = np.maximum(cols - 12, 0)
    lower19 = np.maximum(cols - 19, 0)
    low12_ok = np.broadcast_to((cols >= 12)[None, :], (n, 49))
    low19_ok = np.broadcast_to((cols >= 19)[None, :], (n, 49))
    sus12 = np.where(low12_ok, sus[:, lower12], 0)
    ons12 = np.where(low12_ok, onset[:, lower12], 0)
    sus19 = np.where(low19_ok, sus[:, lower19], 0)
    ons19 = np.where(low19_ok, onset[:, lower19], 0)
    pitch = np.broadcast_to((cols[None, :] + 40.) / 127., (n, 49))
    # This list is deliberately fixed; every column is inferable from past audio.
    values = [
        onset, sus, release, lag(onset, 2), lag(onset, 8),
        lag(sus, 2), lag(sus, 8), onset - lag(onset, 2),
        onset - lag(onset, 8), sus - lag(sus, 8),
        fundamental, overtone2, overtone3, near, distant, f_rise,
        fundamental - lag(fundamental, 2),
        fundamental - lag(fundamental, 8),
        overtone2 - lag(overtone2, 8),
        overtone3 - lag(overtone3, 8),
        overtone2 / (.2 + fundamental),
        overtone3 / (.2 + fundamental),
        sus12, ons12, sus19, ons19,
        f_rise - f_rise[:, lower12],
        f_rise - f_rise[:, lower19],
        np.maximum(0, onset - ons12), np.maximum(0, onset - ons19),
        pitch,
        *[np.broadcast_to(k[:, i, None], (n, 49)) for i in range(7)],
        np.broadcast_to(np.max(k[:, 2:], axis=1)[:, None], (n, 49)),
    ]
    out = np.stack(values, axis=2).astype(np.float32, copy=False)
    if out.shape[-1] != 39 or not np.isfinite(out).all():
        raise AssertionError("S701 feature layout error")
    return out



class LiveS701Features:
    def __init__(self):
        self.f=deque(maxlen=9);self.p=deque(maxlen=9);self.k=deque(maxlen=9)
    def step(self,features,post,k):
        f,p,q=np.asarray(features,np.float32),np.asarray(post,np.float32),np.asarray(k,np.float32)
        if f.shape!=(49,126) or p.shape!=(49,3) or q.shape!=(7,):
            raise ValueError("S701 original 256-trained ABI mismatch")
        self.f.append(f.copy());self.p.append(p.copy());self.k.append(q.copy())
        return onset_features(np.stack(self.f),np.stack(self.p),np.stack(self.k))[-1]
