"""Separate causal quality models for native and residual onset candidates."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class NativeRescueHeads:
    def __init__(self):
        self.models = {}
        self.training_counts = {}

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or x.shape[1] != 49:
            raise ValueError('S703 requires the frozen 49-column event ABI')
        native = x[:, 40] > .5
        for name, mask in (('native', native), ('rescue', ~native)):
            target = y[mask]
            if len(target) < 100 or len(np.unique(target)) != 2:
                raise ValueError('Insufficient positive/negative '+name+' events')
            model = HistGradientBoostingClassifier(
                max_iter=145, learning_rate=.055, max_leaf_nodes=15,
                min_samples_leaf=25 if name == 'native' else 15,
                l2_regularization=12., early_stopping=False, random_state=703)
            model.fit(x[mask], target)
            self.models[name] = model
            self.training_counts[name] = dict(rows=int(mask.sum()),
                positive=int(target.sum()), negative=int((target == 0).sum()))
        return self

    def predict_proba(self, x):
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != 49:
            raise ValueError('S703 requires the frozen 49-column event ABI')
        result = np.empty((len(x), 2), np.float64)
        native = x[:, 40] > .5
        for name, mask in (('native', native), ('rescue', ~native)):
            if mask.any():
                result[mask] = self.models[name].predict_proba(x[mask])
        return result
