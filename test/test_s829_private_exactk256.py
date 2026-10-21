"""S829 Exact-K native256 fail-closed contracts (synthetic smoke, not FIT80)."""
import unittest
import numpy as np
from causal_note.s829_exactk_s809_native256 import (
    S829Stream, score_exact, fit_exact, SCHEMA
)
from causal_note.s828_activek_s708_private_readout import DIM_SELECT
from test.test_s827_s708_activek_only import StubNative, model, policy, wave


def fake(name, comp, seed, n=147):
    rng = np.random.default_rng(seed)
    k = (np.arange(n) % 7).astype(np.uint8)
    rng.shuffle(k)
    x = rng.normal(0, 1, (n, 3*len(DIM_SELECT)+6)).astype(np.float32)
    x[:, 0] = k + rng.normal(0, .4, n)
    baseline = np.clip(k.astype(int) + rng.integers(-1, 2, n), 0, 6).astype(np.uint8)
    return dict(name=name, composition=comp, X=x, k=k, baseline=baseline)


class TestS829(unittest.TestCase):
    def test_exactk_must_count_k0(self):
        s = score_exact([0, 1, 2, 0, 3], [0, 0, 2, 1, 3])
        self.assertEqual((s["correct"], s["frames"], s["under"], s["over"])
                         if "under" in s else (s["correct"], s["frames"], s["undercount"], s["overcount"]),
                         (3, 5, 1, 1))
        with self.assertRaises(ValueError):
            score_exact([0, 7], [0, 1])

    def test_true256_and_no_future_or_activek_state(self):
        signal = wave(9)
        changed = signal.copy()
        changed[4*256:] = 0
        before = []
        for samples in (signal, changed):
            stream = S829Stream(StubNative(), model(), policy())
            emitted = []
            for pos in range(0, len(samples), 128):
                emitted += list(stream.push(samples[pos:pos+128], start_sample=pos))
            x, counts = stream.collected()
            self.assertEqual(x.shape, (9, 3*len(DIM_SELECT)+6))
            self.assertEqual([z.decision_sample for z in emitted],
                             list(range(256, 10*256, 256)))
            self.assertEqual(stream.decoder.step_calls, 0)
            self.assertEqual(stream.model_calls, 9)
            self.assertEqual(len(counts), 9)
            self.assertEqual(stream.diagnostics()["schema"], SCHEMA)
            before.append(x[:4].copy())
        np.testing.assert_array_equal(*before)

    def test_independent_fit_and_selection(self):
        tr = [fake("tr1", "C1", 1), fake("tr2", "C2", 2),
              fake("tr3", "C3", 3)]
        cal = [fake("cal", "C4", 4)]
        fit = fit_exact(tr, cal)
        self.assertTrue(fit.metadata["independent_exactk_readout"])
        self.assertTrue(fit.metadata["shared_original_s708_weights_frozen"])
        self.assertFalse(fit.metadata["other_five_axes_modified"])
        self.assertEqual(fit.metadata["decision_hop_samples"], 256)
        out = fit.predict(fake("outer", "C5", 5)["X"],
                          fake("outer", "C5", 5)["baseline"])
        self.assertEqual(len(out), 147)
        self.assertFalse(out.flags.writeable)
        with self.assertRaisesRegex(ValueError, "composition-disjoint"):
            fit_exact(tr + cal, cal)


if __name__ == "__main__":
    unittest.main()
