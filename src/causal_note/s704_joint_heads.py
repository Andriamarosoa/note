"""Experimental S704 acoustic heads and one causal emitted-note state.

No annotation is accepted by inference. Four native hops form a decision
block; every event is stamped at the end of the block actually received.
The production S613/S630 model and selector are never changed.
"""
from dataclasses import dataclass
import numpy as np
from causal_note.strict_live import HOP, SR
from scripts.s701_train_causal_onset_shadow import onset_features

BATCH = 4


def acoustic_features(feat, post, k):
    """Full native acoustics plus 371 ms of causal state/energy history."""
    n = len(feat)
    if feat.shape != (n, 49, 126):
        raise ValueError('Native harmonic ABI changed')
    state = np.concatenate((feat[:, :, 48:54], post), axis=-1)
    history = [state[np.maximum(np.arange(n)-lag, 0)] for lag in (8, 32, 64)]
    return np.concatenate((feat, onset_features(feat, post, k), *history), -1).astype(np.float32)


def count_input(feat, post, k):
    ix = [r*8+h for r in range(3) for h in range(3)]
    spectrum = feat[:, :, ix].reshape(len(feat), -1)
    tops = np.sort(post, axis=1)[:, -6:, :].reshape(len(feat), -1)
    return np.concatenate((spectrum, tops, post.sum(1), k), -1).astype(np.float32)


@dataclass(frozen=True)
class JointPolicy:
    onset: float = .45
    sustain_on: float = .35
    sustain_off: float = .20
    release: float = .70
    release_steps: int = 1
    count_confidence: float = 1.1
    blend: float = 1.


class JointSelector:
    """Single state determines onsets, offsets and decoded Exact-K."""
    def __init__(self, policy, *, decision_hop_samples=BATCH*HOP):
        if decision_hop_samples not in (HOP, BATCH*HOP):
            raise ValueError("only original 1024 or strict256 decision grid allowed")
        self.decision_hop_samples = int(decision_hop_samples)
        self.policy = policy
        self.active = np.zeros(49, bool)
        self.armed = np.ones(49, bool)
        self.low = np.zeros(49, np.int32)
        self.last_on = np.full(49, -10**12, np.int64)
        self.last_end = 0

    def step(self, post, count, end):
        if end != self.last_end + self.decision_hop_samples:
            raise ValueError('Non-contiguous decision blocks')
        self.last_end = int(end)
        c = self.policy
        on, active, release = post.T
        self.armed |= on < c.onset*.4
        self.low = np.where((active < c.sustain_off) | (release >= c.release),
                            self.low + 1, 0)
        events = []

        def emit(kind, pitch, reason):
            events.append(dict(type=kind, pitch=int(pitch+40),
                decision_sample=int(end), available_sample=int(end),
                time=float(end/SR), reason=reason,
                velocity=96 if kind=='note_on' else 0))

        old_enough = end-self.last_on >= .035*SR
        for p in np.flatnonzero(self.active & old_enough & (self.low >= c.release_steps)):
            emit('note_off', p, 'joint_causal_release')
            self.active[p] = False
        ready = (on >= c.onset) & (active >= c.sustain_on) & self.armed
        ready &= end-self.last_on >= .065*SR
        budget = 6
        if count.max() >= c.count_confidence:
            budget = max(int(self.active.sum()), min(6, int(count.argmax())))
        for p in sorted(np.flatnonzero(ready), key=lambda j: (-float(on[j]*active[j]), int(j))):
            if self.active[p]:
                emit('note_off', p, 'joint_restrike')
            elif self.active.sum() >= budget:
                continue
            emit('note_on', p, 'joint_causal_attack')
            self.active[p] = True
            self.armed[p] = False
            self.low[p] = 0
            self.last_on[p] = end
        return events

    def finish(self, sample):
        result = [dict(type='note_off', pitch=int(p+40), decision_sample=int(sample),
                       available_sample=int(sample), time=float(sample/SR),
                       reason='stream_end', velocity=0) for p in np.flatnonzero(self.active)]
        self.active[:] = False
        return result


def decode_joint(ends, post, count, policy, samples):
    selector = JointSelector(policy)
    result = []
    for end, p, k in zip(ends, post, count):
        result.extend(selector.step(p, k, int(end)))
    result.extend(selector.finish(samples))
    return result


def probabilities(models, x, batch=16384):
    flat = x.reshape(-1, x.shape[-1])
    p = np.empty((len(flat), 3), np.float32)
    for start in range(0, len(flat), batch):
        z = flat[start:start+batch]
        for i, model in enumerate(models):
            p[start:start+len(z), i] = model.predict_proba(z)[:, 1]
    return p.reshape(*x.shape[:-1], 3)


def count_probabilities(model, x):
    p = np.zeros((len(x), 7), np.float32)
    p[:, model.classes_] = model.predict_proba(x)
    return p
