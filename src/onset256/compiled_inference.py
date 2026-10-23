"""Optional inference acceleration of already SHA-verified project models.

No training, threshold changes, history extension, or additional audio hop.
Validate emitted-event parity before using an exported model as a reference.
"""
import numpy as np
from scipy.special import expit


class NumpyBinaryMLP:
    """Remove sklearn validation overhead while preserving scaler arithmetic."""
    def __init__(self, pipeline):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        scaler, model = [x[1] for x in pipeline.steps]
        if (not isinstance(scaler, StandardScaler) or not isinstance(model, MLPClassifier)
                or model.activation != "relu" or model.out_activation_ != "logistic"
                or list(model.classes_) != [0, 1] or not scaler.with_mean or not scaler.with_std):
            raise ValueError("unsupported immutable binary MLP")
        self.mean = scaler.mean_.copy()
        self.scale = scaler.scale_.copy()
        self.weights = [w.copy() for w in model.coefs_]
        self.bias = [b.copy() for b in model.intercepts_]
        self.n_features_in_ = pipeline.n_features_in_

    def predict_proba(self, features):
        x = np.asarray(features, np.float32).copy()
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("wrong MLP evidence dimension")
        # sklearn's StandardScaler performs these in-place on float32.
        x -= self.mean
        x /= self.scale
        for i, (w, b) in enumerate(zip(self.weights, self.bias)):
            x = x @ w + b
            if i < len(self.weights) - 1:
                np.maximum(x, 0, out=x)
        expit(x, out=x)
        return np.hstack((1 - x, x))


def _tree_logits(x, feature, threshold, left, right, leaf, missing_left, value, baseline):
    result = np.empty(x.shape[0], np.float64)
    for row in range(x.shape[0]):
        total = baseline
        for tree in range(feature.shape[0]):
            node = 0
            while not leaf[tree, node]:
                v = x[row, feature[tree, node]]
                go_left = missing_left[tree, node] if np.isnan(v) else v <= threshold[tree, node]
                node = left[tree, node] if go_left else right[tree, node]
            total += value[tree, node]
        result[row] = total
    return result


class CompiledBinaryTrees:
    """Compile frozen HGB traversal, preserving float64 sums and tree order."""
    def __init__(self, model):
        from sklearn.ensemble import HistGradientBoostingClassifier
        from numba import njit
        if not isinstance(model, HistGradientBoostingClassifier) or list(model.classes_) != [0, 1]:
            raise ValueError("only verified project binary HGB models supported")
        self.n_features_in_ = model.n_features_in_
        nodes = [stage[0].nodes for stage in model._predictors]
        if any(len(stage) != 1 for stage in model._predictors) or any(
                np.any(n["is_categorical"]) for n in nodes):
            raise ValueError("only numerical binary trees supported")
        shape = (len(nodes), max(map(len, nodes)))
        fields = (("feature_idx", np.int64), ("num_threshold", np.float64),
                  ("left", np.int64), ("right", np.int64), ("is_leaf", np.bool_),
                  ("missing_go_to_left", np.bool_), ("value", np.float64))
        arrays = []
        for field, dtype in fields:
            arr = np.zeros(shape, dtype)
            for i, n in enumerate(nodes):
                arr[i, :len(n)] = n[field]
            arrays.append(arr)
        self.arrays = arrays
        self.baseline = float(model._baseline_prediction[0, 0])
        self.kernel = njit(cache=True, fastmath=False)(_tree_logits)
        # JIT finishes at construction, before any audio is accepted.
        self.predict_proba(np.zeros((1, self.n_features_in_), np.float32))

    def predict_proba(self, features):
        x = np.ascontiguousarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("wrong tree evidence dimension")
        logits = self.kernel(x, *self.arrays, self.baseline)
        q = expit(logits)
        return np.column_stack((1 - q, q))


def accelerated(model):
    from sklearn.pipeline import Pipeline
    if isinstance(model, (NumpyBinaryMLP, CompiledBinaryTrees)):
        return model
    if isinstance(model, Pipeline):
        return NumpyBinaryMLP(model)
    return CompiledBinaryTrees(model)


class SharedEvidenceRisk:
    """Compute the common 69D snapshot once, before both current-hop heads."""
    requires_selector_state = True
    trained_hop_samples = 256

    def __init__(self, risk):
        self.model = accelerated(risk.model)
        self.cut = risk.cut
        self.calls = 0
        self.evidence = None

    def predict_correct_probability(self, feat, p, k, rescue, history, state):
        from .error_evidence69 import hard_matrix
        self.evidence = hard_matrix(feat, p, k, rescue, history, state)
        result = np.asarray(self.model.predict_proba(self.evidence), np.float32)
        self.calls += 1
        return result[:, 1].copy()


class SharedEvidenceRecovery:
    trained_hop_samples = 256

    def __init__(self, recovery, risk):
        self.model = accelerated(recovery.model)
        self.candidate_policy = recovery.candidate_policy
        self.calibrated_cut = recovery.calibrated_cut
        self.enabled = recovery.enabled
        self.risk = risk
        self.calls = 0

    def predict_recovery_probability(self, feat, p, k, rescue, history, state):
        if self.risk.calls != self.calls + 1 or self.risk.evidence is None:
            raise ValueError("recovery may only reuse this hop's preselection evidence")
        result = np.asarray(self.model.predict_proba(self.risk.evidence), np.float32)
        self.calls += 1
        return result[:, 1].copy()
