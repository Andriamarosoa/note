import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from onset256.compiled_inference import accelerated


class TestCompiledWeights(unittest.TestCase):
    def test_mlp_preserves_float32_scaling_and_probabilities(self):
        rng = np.random.default_rng(913)
        x = rng.normal(size=(120, 69)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(8, 4), max_iter=8, random_state=2))
        model.fit(x, y)
        fast = accelerated(model)
        np.testing.assert_array_equal(model.predict_proba(x), fast.predict_proba(x))

    def test_tree_thresholds_missing_values_and_float64_accumulation(self):
        rng = np.random.default_rng(818)
        x = rng.normal(size=(200, 5)).astype(np.float32)
        x[::7, 0] = np.nan
        y = (np.nan_to_num(x[:, 0]) + x[:, 1] > 0).astype(int)
        model = HistGradientBoostingClassifier(
            max_iter=12, max_leaf_nodes=7, min_samples_leaf=8,
            early_stopping=False, random_state=4).fit(x, y)
        fast = accelerated(model)
        np.testing.assert_array_equal(model.predict_proba(x), fast.predict_proba(x))

    def test_stream_rejects_multiple_or_partial_hops_before_reading_audio(self):
        from onset256.live_event_stream import LowLatencyOnsetStream
        # Shape guards run before any access to stream state.
        s = object.__new__(LowLatencyOnsetStream)
        for n, start in ((512, 0), (255, 0), (256, 1)):
            with self.assertRaises(ValueError):
                s.push(np.zeros(n, np.float32), start_sample=start)
