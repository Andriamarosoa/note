"""S829 fully isolated Exact-K: synthetic contracts, not musical FIT80 scores."""
import unittest
import numpy as np
from causal_note.s829_exactk_s809_native256 import (
    HOP, S829AcousticFrontEnd, S829Stream, S829FittedStream, score_exact,
    fit_exact, proposed, SCHEMA
)


def fake(name, comp, seed, n=147):
    rng = np.random.default_rng(seed)
    k = (np.arange(n) % 7).astype(np.uint8)
    rng.shuffle(k)
    x = rng.normal(0, 1, (n, 369)).astype(np.float32)
    x[:, 0] = k + rng.normal(0, .4, n)
    baseline = np.clip(k.astype(int) + rng.integers(-1, 2, n),
                       0, 6).astype(np.uint8)
    return dict(name=name, composition=comp, X=x, k=k, baseline=baseline)


class TestS829(unittest.TestCase):
    def test_exactk_includes_k0(self):
        s = score_exact([0, 1, 2, 0, 3], [0, 0, 2, 1, 3])
        self.assertEqual((s["correct"], s["frames"], s["undercount"],
                          s["overcount"]), (3, 5, 1, 1))
        with self.assertRaises(ValueError):
            score_exact([0, 7], [0, 1])

    def test_native256_no_future_and_chunk_invariance(self):
        rng = np.random.default_rng(829)
        signal = rng.normal(0, .07, 8*HOP).astype(np.float32)
        changed = signal.copy()
        changed[4*HOP:] = -signal[4*HOP:]
        results = []
        for samples, chunks in ((signal, [128]*16),
                                (signal, [256]*8),
                                (changed, [256]*8)):
            stream = S829Stream()
            out = []
            at = 0
            for size in chunks:
                out.extend(stream.push(samples[at:at+size], start_sample=at))
                at += size
            self.assertEqual(at, len(samples))
            x, b = stream.collected()
            self.assertEqual(x.shape, (8, 369))
            self.assertEqual(b.shape, (8,))
            self.assertEqual([d.decision_sample for d in out],
                             list(range(HOP, 9*HOP, HOP)))
            self.assertEqual(stream.diagnostics()["schema"], SCHEMA)
            self.assertFalse(stream.diagnostics()["source_neural_encoder_used"])
            self.assertFalse(stream.diagnostics()["shared_model_weights"])
            results.append((x, b))
        np.testing.assert_allclose(results[0][0], results[1][0], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(results[0][1], results[1][1])
        np.testing.assert_array_equal(results[1][0][:4], results[2][0][:4])
        np.testing.assert_array_equal(results[1][1][:4], results[2][1][:4])

    def test_two_trained_models_disjoint_data_and_private_selection(self):
        tr = [fake("tr1", "C1", 1), fake("tr2", "C2", 2),
              fake("tr3", "C3", 3)]
        cal = [fake("cal", "C4", 4)]
        fit = fit_exact(tr, cal)
        self.assertIsNot(fit.harmonic, fit.temporal)
        self.assertTrue(fit.metadata["independent_pcm_frontend"])
        self.assertEqual(fit.metadata["independent_trained_heads"], 2)
        self.assertFalse(fit.metadata["shared_neural_checkpoint"])
        self.assertFalse(fit.metadata["any_other_axis_as_input"])
        self.assertEqual(fit.metadata["decision_hop_samples"], 256)
        val = fake("outer", "C5", 5)
        out = fit.predict(val["X"], val["baseline"])
        self.assertEqual(len(out), 147)
        self.assertFalse(out.flags.writeable)
        with self.assertRaisesRegex(ValueError, "composition-disjoint"):
            fit_exact(tr+cal, cal)

    def test_trained_fitted_online_is_causal_at_256(self):
        tr = [fake("tr1", "C1", 11), fake("tr2", "C2", 12),
              fake("tr3", "C3", 13)]
        fit = fit_exact(tr, [fake("cal", "C4", 14)])
        rng = np.random.default_rng(28)
        audio = rng.normal(0, .05, HOP*6).astype(np.float32)
        future = audio.copy()
        future[3*HOP:] = 0
        all_outputs = []
        for wave in (audio, future):
            stream = S829FittedStream(fit)
            decisions = []
            for i in range(0, len(wave), HOP):
                produced = stream.push(wave[i:i+HOP], start_sample=i)
                self.assertEqual(len(produced), 1)
                decisions.extend(produced)
            self.assertEqual(stream.model_calls, 6)
            self.assertTrue(stream.diagnostics()["fitted_inference_online"])
            self.assertTrue(stream.diagnostics()["no_shared_encoder"])
            self.assertEqual([z.decision_sample for z in decisions],
                             list(range(HOP, 7*HOP, HOP)))
            all_outputs.append([z.count for z in decisions])
        self.assertEqual(all_outputs[0][:3], all_outputs[1][:3])

    def test_selector_uses_only_owned_heads_and_pcm_baseline(self):
        h = np.zeros((2, 7), dtype=np.float32)
        t = h.copy()
        h[:, 2] = 1
        t[:, 3] = 1
        raw = np.array([1, 0])
        for mode in ("harmonic", "temporal", "fusion",
                     "confidence_router", "acoustic"):
            y = proposed(h, t, raw, mode, 0)
            self.assertTrue(np.isin(y, range(7)).all())
        np.testing.assert_array_equal(proposed(h, t, raw, "acoustic", 0), raw)

    def test_each_stream_owns_its_fft_buffers(self):
        a, b = S829Stream(), S829Stream()
        self.assertIsNot(a.frontend, b.frontend)
        self.assertIsNot(a.frontend.previous, b.frontend.previous)
        self.assertIsNot(a.rows, b.rows)
        self.assertIsNot(a.baselines, b.baselines)
        self.assertIsInstance(a.frontend, S829AcousticFrontEnd)


if __name__ == "__main__":
    unittest.main()
