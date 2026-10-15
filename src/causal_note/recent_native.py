"""Persistent S569 -> S578/S585 NOTE inference and original note selections.

This is downstream of the original V88/event and Basic Pitch producers. It is
not a replacement S80 counter. K0 sites are retained, as in S589/S595; the
recent note policies never used K as a hard output quota. No offset is inferred.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
from pathlib import Path
import numpy as np

POLICIES = ("S578_global28", "S585_state_strict100", "S585_state_restrike")
MODEL_SPECS = {
    "global28": ("S578/S578_FULL28_general28_52D_TRUE_AUDIO_NOTE_MODEL.joblib",
                 "21b234796441fa5594d3a748d5b9ca42377e1181ec6cc2fe5bf68816c4c687af"),
    "state4": ("S585/S585_FULL28_four_state_52D.joblib",
               "555cb092accb6ea0b42394280f646c90ccac0b48746040811ac57da382837d36"),
}


def row_values(row):
    source, ident = row["source_recording_sha"], row["V88_original_event_id"]
    time = float(row["original_V88_candidate_start_s"])
    x = np.asarray(row["per_midi_features_49x11"], np.float32)
    births = row["old_basicpitch_birth_pitches"]
    k = row["original_V88_K_soft_prior_ONLY"]
    if (not isinstance(source, str) or not source or not isinstance(ident, str) or not ident
            or not np.isfinite(time) or time < 0 or x.shape != (49, 11)
            or not np.isfinite(x).all() or isinstance(k, bool) or not isinstance(k, int)
            or k not in range(7) or any(type(p) is not int or p not in range(40, 89) for p in births)):
        raise ValueError("Original event/source identity, K0..6 and finite 49x11 features required")
    return source, ident, time, x, sorted(set(births))


def physical34(x):
    """S562 competing_fundamental34, evaluated for one event, no learned change."""
    x = x[None]
    parts = [x[:, :, [0, 1, 2, 3, 4, 5, 10]]]
    for step in (12, 19, 24):
        lower = np.zeros_like(x[:, :, [2, 3, 5]])
        lower[:, step:] = x[:, :-step, :][:, :, [2, 3, 5]]
        parts.extend((lower, x[:, :, [2, 3, 5]] - lower))
    for j in (2, 3, 5):
        z = x[:, :, j:j+1]
        parts.append(z / np.maximum(.001, z.max(axis=1, keepdims=True)))
    for j in (2, 3, 5):
        z = x[:, :, j:j+1]
        left = np.pad(z[:, :-1], ((0, 0), (1, 0), (0, 0)))
        right = np.pad(z[:, 1:], ((0, 0), (0, 1), (0, 0)))
        parts.append(z - np.maximum(left, right))
    parts.extend((x[:, :, 4:5], x[:, :, 3:4] - x[:, :, 4:5],
                  x[:, :, 5:6] - x[:, :, 4:5]))
    return np.concatenate(parts, axis=2)[0].astype(np.float32)


class RecentFeatureState:
    """Exact S569 temporal52 with just two prior events, one source per stream."""

    def __init__(self):
        self.history = deque(maxlen=2)
        self.source = None
        self.last_key = None

    def push(self, row):
        source, ident, time, x, _ = row_values(row)
        key = (time, ident)
        if self.source is not None and source != self.source:
            raise ValueError("Start a new stream for another recording")
        if self.last_key is not None and (key <= self.last_key or ident == self.last_key[1]):
            raise ValueError("Canonical events must be unique and ordered by time then ID")
        prev = np.zeros((49, 4), np.float32)
        dif = np.zeros_like(prev)
        two = np.zeros((49, 2), np.float32)
        lag, ratio = np.zeros_like(two), np.zeros_like(two)
        if self.history and time - self.history[-1][0] <= .350:
            t, old = self.history[-1]
            dt = time - t
            prev[:] = old[:, [2, 3, 4, 5]]
            dif[:] = x[:, [2, 3, 4, 5]] - prev
            lag[:, 0], lag[:, 1] = np.exp(-dt / .080), np.exp(-dt / .250)
            ratio[:, 0] = np.clip((x[:, 2] + .01) / (old[:, 2] + .01), 0, 8)
            ratio[:, 1] = np.clip((x[:, 5] + .01) / (old[:, 5] + .01), 0, 8)
            if len(self.history) == 2 and time - self.history[0][0] <= .350:
                two[:] = self.history[0][1][:, [2, 5]]
        out = np.concatenate((physical34(x), prev, dif, two, lag, ratio,
                              x[:, [6, 7, 8, 9]]), axis=1).astype(np.float32)
        self.history.append((time, x.copy()))
        self.source, self.last_key = source, key
        return out


class RecentHeads:
    """Load unchanged full-28 weights once; deterministic serial tree reduction."""

    def __init__(self, root):
        import joblib
        self.models, self.hashes = {}, {}
        for name, (relative, expected) in MODEL_SPECS.items():
            path = Path(root) / relative
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"Original {name} checkpoint SHA mismatch")
            model = joblib.load(path)
            expected_classes = [0, 1] if name == "global28" else [0, 1, 2, 3]
            if list(model.classes_) != expected_classes or model.n_features_in_ != 52:
                raise ValueError(f"Original {name} class/feature ABI changed")
            # Thread scheduling overhead matters for 49 rows, and summation order
            # should not depend on a worker race. No weights/thresholds are changed.
            model.n_jobs = 1
            self.models[name], self.hashes[name] = model, actual

    def predict(self, x):
        x = np.asarray(x, np.float32)
        if x.shape != (49, 52) or not np.isfinite(x).all():
            raise ValueError("S569 49x52 finite features required")
        return (self.models["global28"].predict_proba(x)[:, 1],
                self.models["state4"].predict_proba(x))


class RecentSelector:
    """Incremental original S536/S585 policy; at most 49 last-note entries.

    BP births bypass the learned classifier, dedup and harmonic veto exactly as
    in the frozen policies. S585 state 'active' is not a validated NoteOff head.
    """

    def __init__(self, policy):
        if policy not in POLICIES:
            raise ValueError("Unknown original NOTE policy")
        self.policy, self.last, self.counts = policy, {}, Counter()

    def select(self, row, global_prob, state_prob):
        _, _, time, _, births = row_values(row)
        # Keep the JSON float64 comparisons of the original selectors, not the
        # float32 model input conversion; threshold ties must retain their ABI.
        acoustic = np.asarray(row["per_midi_features_49x11"], float)
        gp, sp = np.asarray(global_prob), np.asarray(state_prob)
        if (gp.shape != (49,) or sp.shape != (49, 4)
                or not np.isfinite(gp).all() or not np.isfinite(sp).all()
                or np.any((gp < 0) | (gp > 1)) or np.any((sp < 0) | (sp > 1))):
            raise ValueError("Original head probability ABI required")
        if births:
            for p in births:
                self.last[p] = time
            return births
        global_head = self.policy == POLICIES[0]
        attack = gp if global_head else sp[:, 2] + sp[:, 3]
        keep = []
        for j in np.flatnonzero(attack >= .7):
            pitch = int(j + 40)
            dt = time - self.last[pitch] if pitch in self.last else None
            rapid = dt is not None and (0 <= 1000 * dt <= 100 if global_head else 0 <= dt <= .100)
            if rapid:
                if not (self.policy == POLICIES[2] and sp[j, 3] >= .5):
                    self.counts["duplicate_candidates_dropped" if global_head else "rapid_candidate_suppressed"] += 1
                    continue
                self.counts["rapid_candidate_passed_by_restrike_state"] += 1
            harmonic = any(j - d in keep
                           and acoustic[j, 3] < .8 * max(.001, acoustic[j-d, 3])
                           and acoustic[j, 5] < .8 * max(.001, acoustic[j-d, 5])
                           for d in (12, 19, 24))
            if harmonic:
                self.counts["weak_harmonics_dropped" if global_head else "weak_harmonic_suppressed"] += 1
            else:
                keep.append(int(j))
        notes = [j + 40 for j in keep]
        for pitch in notes:
            self.last[pitch] = time
        return notes


class RecentNoteStream:
    """One source, shared loaded heads, separate state for each original policy."""

    def __init__(self, heads):
        self.heads, self.features = heads, RecentFeatureState()
        self.selectors = {p: RecentSelector(p) for p in POLICIES}
        self.available_sample = 0

    def push(self, row, *, available_sample):
        _, _, time, _, _ = row_values(row)
        if (type(available_sample) is not int
                or available_sample < max(self.available_sample, int(np.ceil(time * 44100)))):
            raise ValueError("Actual availability must be monotone and after the acoustic onset")
        x = self.features.push(row)
        gp, sp = self.heads.predict(x)
        result = dict(source_recording_sha=row["source_recording_sha"],
                      V88_original_event_id=row["V88_original_event_id"], time=time,
                      original_V88_K_soft_prior_ONLY=row["original_V88_K_soft_prior_ONLY"],
                      available_sample=available_sample,
                      notes={p: s.select(row, gp, sp) for p, s in self.selectors.items()})
        self.available_sample = available_sample
        return result, x, gp, sp
