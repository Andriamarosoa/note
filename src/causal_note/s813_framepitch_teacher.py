"""S813 independent Frame Pitch teacher from fresh, causal MIDI256 DSP.

Input ABI: float16-cached native 126 + sharp 96 features, [T,49,222].
There are no S708 posteriors, emitted events, labels, K decisions or peer
teacher outputs in the prediction API. Native temporal differences are
already computed at hop 256 by the PCM extractor; this module adds no
lookahead, smoothing, hold or delay. Its CPU time still requires measurement.

The two fitted heads own separate trees. The second specializes in bass
and lower-fundamental competition. Three declared selectors and three cuts
are calibrated exclusively on composition-disjoint inner-calibration data.
No inference uses the calibration labels. This is a teacher, not a MIDI
unification or a claim of measured >85% musical performance.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from types import MappingProxyType

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

SCHEMA = "S813_INDEPENDENT_FRAMEPITCH_FRESH_DSP256_V1"
FEATURE_DIM = 222
PITCHES = 49
HEAD_NAMES = ("general", "bass_harmonic")
SELECTORS = ("general", "specialist_region", "mean_region")
# The tie-breaker keeps the conventional cut and simplest selector.
CUTS = (.5, .3, .7)
FIXED_MODEL_PARAMS = dict(
    loss="log_loss", learning_rate=.1, max_iter=64, max_leaf_nodes=15,
    max_depth=4, min_samples_leaf=80, l2_regularization=1., max_bins=63,
    early_stopping=False, categorical_features=None, class_weight=None,
)


def _features(X):
    x = np.array(X, dtype=np.float32, copy=True)
    if x.ndim != 3 or x.shape[1:] != (PITCHES, FEATURE_DIM):
        raise ValueError("fresh MIDI256 DSP [T,49,222] required")
    if not np.isfinite(x).all():
        raise ValueError("non-finite acoustic features")
    return x


def specialist_region(X):
    """Current acoustic evidence only, recomputed separately for each pitch.

    Sharp log-amplitudes start at 126, eight harmonics per resolution.
    A competing lower pitch is considered only when its fundamental has
    appreciable energy and exceeds the current candidate fundamental.
    This determines routing, never labels or an unconditional pitch veto.
    """
    x = np.asarray(X)
    if x.ndim != 3 or x.shape[1:] != (PITCHES, FEATURE_DIM):
        raise ValueError("invalid DSP feature ABI")
    fundamental = x[:, :, (126, 134, 142)].mean(axis=-1)
    parent = np.zeros_like(fundamental)
    for interval in (12, 19, 24):
        parent[:, interval:] = np.maximum(
            parent[:, interval:], fundamental[:, :-interval])
    bass = np.broadcast_to(np.arange(PITCHES)[None, :] < 8,
                           fundamental.shape)
    competition = (parent > .5) & (parent > fundamental + .25)
    return bass | competition


def _validate_rows(rows, role):
    rows = list(rows)
    if not rows:
        raise ValueError(role + " rows are empty")
    names = []
    for row in rows:
        if not isinstance(row.get("name"), str) or not row["name"]:
            raise ValueError("each row needs a nonempty recording name")
        if not isinstance(row.get("composition"), str) or not row["composition"]:
            raise ValueError("each row needs a composition")
        names.append(row["name"])
        X = np.asarray(row["X"])
        y = np.asarray(row["y"])
        if X.ndim != 3 or X.shape[1:] != (PITCHES, FEATURE_DIM) or not len(X):
            raise ValueError("fresh nonempty [T,49,222] DSP required")
        if y.shape != X.shape[:2] or not np.isin(y, (0, 1)).all():
            raise ValueError("binary MIDI256 occupancy labels [T,49] required")
    if len(set(names)) != len(names):
        raise ValueError("duplicate recording name in " + role)
    return sorted(rows, key=lambda row: row["name"])


def sample_training_rows(rows, head, *, max_points=4000, seed=813):
    """Stratified samples with inverse-inclusion weights, no class balancing.

    Each selected positive represents N_positive/n_selected_positive real
    candidates, and analogously for negatives. Global normalization keeps
    the original stream prior and relative recording durations. Specialist
    weighting preserves the natural prior conditional on its acoustic region.
    """
    if head not in HEAD_NAMES or max_points < 4:
        raise ValueError("invalid head or sampling budget")
    xs, ys, ws, proofs = [], [], [], []
    for row in _validate_rows(rows, "training"):
        X = _features(row["X"])
        y = np.asarray(row["y"], dtype=np.uint8).ravel()
        eligible = (np.ones(X.shape[:2], bool) if head == "general"
                    else specialist_region(X)).ravel()
        pos = np.flatnonzero(eligible & (y == 1))
        neg = np.flatnonzero(eligible & (y == 0))
        pos_take = min(len(pos), max_points // 2)
        neg_take = min(len(neg), max_points - pos_take)
        pos_take = min(len(pos), max_points - neg_take)
        name_seed = int.from_bytes(hashlib.sha256(
            (head + "/" + row["name"]).encode()).digest()[:4], "big")
        rng = np.random.default_rng(seed + name_seed)
        p = rng.choice(pos, pos_take, replace=False)
        n = rng.choice(neg, neg_take, replace=False)
        selected = np.concatenate((p, n))
        if not len(selected):
            continue
        w = np.concatenate((
            np.full(pos_take, len(pos) / max(pos_take, 1), np.float64),
            np.full(neg_take, len(neg) / max(neg_take, 1), np.float64)))
        order = np.argsort(selected, kind="stable")
        selected, w = selected[order], w[order]
        xs.append(X.reshape(-1, FEATURE_DIM)[selected].copy())
        ys.append(y[selected].copy())
        ws.append(w)
        proofs.append(dict(name=row["name"], composition=row["composition"],
                           natural_positive=len(pos), natural_negative=len(neg),
                           sampled_positive=pos_take, sampled_negative=neg_take))
    if not xs:
        raise ValueError("head has no eligible training samples")
    X, y, weights = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("each independently trained head requires both classes")
    natural_positive = sum(p["natural_positive"] for p in proofs)
    natural_negative = sum(p["natural_negative"] for p in proofs)
    weighted_prior = float(np.sum(weights * y) / np.sum(weights))
    expected_prior = natural_positive / (natural_positive + natural_negative)
    if not np.isclose(weighted_prior, expected_prior, atol=1e-12, rtol=0):
        raise AssertionError("sampling weights changed natural class prior")
    weights /= weights.mean()
    return X, y, weights, dict(
        rows=proofs, samples=len(y), natural_positive=natural_positive,
        natural_negative=natural_negative, weighted_positive_prior=weighted_prior,
        inverse_inclusion_weighted=True, class_weight=None)


@dataclass
class OwnedHead:
    name: str
    model: HistGradientBoostingClassifier

    def probabilities(self, flat):
        if not len(flat):
            return np.empty(0, np.float32)
        positive = list(self.model.classes_).index(1)
        parts = []
        # CPU teacher is deliberately evaluated outside an audio callback.
        with threadpool_limits(limits=1):
            for start in range(0, len(flat), 32768):
                parts.append(self.model.predict_proba(
                    flat[start:start + 32768])[:, positive])
        return np.concatenate(parts).astype(np.float32)


def _combine(general, specialist, region, selector):
    if selector == "general":
        return general.copy()
    if selector == "specialist_region":
        return np.where(region, specialist, general)
    if selector == "mean_region":
        return np.where(region, .5 * (general + specialist), general)
    raise ValueError("undeclared selector")


class FramePitchTeacher:
    """Owned, independently calibrated teacher; inference accepts only DSP."""

    def __init__(self, heads, selector, threshold, metadata):
        if set(heads) != set(HEAD_NAMES) or selector not in SELECTORS:
            raise ValueError("two distinct declared heads and selector required")
        if heads["general"].model is heads["bass_harmonic"].model:
            raise ValueError("heads cannot share fitted model ownership")
        if threshold not in CUTS:
            raise ValueError("undeclared calibration cut")
        self._heads = dict(heads)
        self.selector = selector
        self.threshold = float(threshold)
        self._metadata = deepcopy(metadata)

    @property
    def heads(self):
        return MappingProxyType(self._heads)

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def predict_proba(self, X):
        x = _features(X)
        flat = x.reshape(-1, FEATURE_DIM)
        general = self._heads["general"].probabilities(flat).reshape(x.shape[:2])
        if self.selector == "general":
            result = general.copy()
        else:
            region = specialist_region(x)
            specialist = np.zeros_like(general)
            specialist[region] = self._heads["bass_harmonic"].probabilities(
                flat[region.ravel()])
            result = _combine(general, specialist, region, self.selector)
        result = np.array(result, dtype=np.float32, copy=True)
        result.setflags(write=False)
        return result

    def predict(self, X):
        result = self.predict_proba(X) >= self.threshold
        result.setflags(write=False)
        return result


def fit_teacher(train_rows, cal_rows, *, seed=813, max_points=4000):
    """Fit only train rows; choose selector/cut only from inner CAL rows.

    The caller owns outer folds and must never supply their labels here.
    No refit uses the calibration compositions. Fixed model hyperparameters
    were declared before outer results; metadata records all nine candidates.
    """
    train = _validate_rows(train_rows, "training")
    cal = _validate_rows(cal_rows, "inner calibration")
    train_comps = {r["composition"] for r in train}
    cal_comps = {r["composition"] for r in cal}
    if train_comps & cal_comps or ({r["name"] for r in train} &
                                  {r["name"] for r in cal}):
        raise ValueError("inner calibration must be recording/composition-disjoint")
    heads, training = {}, {}
    for index, name in enumerate(HEAD_NAMES):
        X, y, weights, proof = sample_training_rows(
            train, name, max_points=max_points, seed=seed)
        model = HistGradientBoostingClassifier(
            **FIXED_MODEL_PARAMS, random_state=seed + index)
        with threadpool_limits(limits=1):
            model.fit(X, y, sample_weight=weights)
        heads[name] = OwnedHead(name, model)
        training[name] = proof

    counts = {(selector, cut): [0, 0, 0]
              for selector in SELECTORS for cut in CUTS}
    for row in cal:
        X = _features(row["X"])
        y = np.asarray(row["y"], dtype=bool)
        flat = X.reshape(-1, FEATURE_DIM)
        general = heads["general"].probabilities(flat).reshape(y.shape)
        region = specialist_region(X)
        specialist = np.zeros_like(general)
        specialist[region] = heads["bass_harmonic"].probabilities(
            flat[region.ravel()])
        for selector in SELECTORS:
            prob = _combine(general, specialist, region, selector)
            for cut in CUTS:
                pred = prob >= cut
                c = counts[selector, cut]
                c[0] += int((pred & y).sum())
                c[1] += int((pred & ~y).sum())
                c[2] += int((~pred & y).sum())
    candidates = []
    for (selector, cut), (tp, fp, fn) in counts.items():
        candidates.append(dict(selector=selector, threshold=cut,
                               tp=tp, fp=fp, fn=fn,
                               f1=2 * tp / max(1, 2 * tp + fp + fn)))
    # Stable first-wins ties retain simpler general selector and cut .5.
    chosen = max(candidates, key=lambda row: row["f1"])
    metadata = dict(
        schema=SCHEMA, target="frame_pitch_f1", target_strictly_above=.85,
        input_features=FEATURE_DIM, native_features=126, sharp_features=96,
        sample_rate=44100, decision_hop_samples=256,
        supplied_feature_cache="float16 to float32", supervised_inputs=False,
        additional_lookahead_samples=0, additional_hold_samples=0,
        temporal_lags="upstream native hop256; no additional teacher state",
        heads=list(HEAD_NAMES), model_params=deepcopy(FIXED_MODEL_PARAMS),
        train_compositions=sorted(train_comps),
        calibration_compositions=sorted(cal_comps),
        calibration_names=[r["name"] for r in cal], training=training,
        selector_candidates=list(SELECTORS), threshold_candidates=list(CUTS),
        calibration_results=candidates, selected=chosen,
        physical_latency_measured=False, promotion=False)
    return FramePitchTeacher(heads, chosen["selector"], chosen["threshold"], metadata)
