"""Safety contracts for opt-in harmonic expert. No extra lookahead or native change."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_note.strict_live import (
    CausalSelection, HOP, Selection, StrictLiveStream,
)
from causal_note.harmonic_expert import (
    FEATURE_NAMES, SCHEMA, N_FEATURES,
    HarmonicGenuineOnsetHead, HarmonicSpecialistSelector,
    harmonic_pair_features,
)


class HarmonicExpertTests(unittest.TestCase):
    def inputs(self):
        x = np.zeros((49, 126), np.float32)
        p = np.zeros((49, 3), np.float32)
        k = np.eye(7, dtype=np.float32)[2]
        # A genuine E2 + E3 attack is spectrally compatible with H2(E2)
        # == F0(E3); this coincidence must NOT alone veto E3.
        x[0, 16] = 1.2          # parent E2, 8192 H1
        x[0, 17] = 1.7          # E2 H2
        x[12, 16] = 1.7         # E3 H1, same spectral energy
        x[0, 50] = 1.1         # parent harmonic score
        x[12, 50] = 1.6
        x[12, 54] = .4         # child acoustic novelty
        p[0, :2] = .95
        p[12, :2] = .92
        return x, p, k

    def checkpoint(self, folder, bias):
        path = Path(folder) / "expert.npz"
        np.savez(path, schema=np.asarray(SCHEMA),
                 feature_names=np.asarray(FEATURE_NAMES),
                 mean=np.zeros(N_FEATURES, np.float32),
                 scale=np.ones(N_FEATURES, np.float32),
                 weights=np.zeros(N_FEATURES, np.float32),
                 bias=np.asarray(bias, np.float32))
        return path

    def test_shadow_is_bitwise_equivalent_to_original_decisions(self):
        x, p, k = self.inputs()
        reference = CausalSelection(Selection())
        specialist = HarmonicSpecialistSelector(CausalSelection(Selection()), mode="shadow")
        self.assertEqual(reference.step(p, k, HOP, HOP),
                         specialist.step(x, p, k, HOP, HOP))
        self.assertEqual(specialist.stats.vetoed_upper_onsets, 0)
        self.assertGreater(specialist.stats.pairs_considered, 0)
        self.assertEqual(specialist.head.predict(np.zeros((1, N_FEATURES), np.float32)), None)

    def test_pair_features_contain_both_simultaneous_notes(self):
        x, p, k = self.inputs()
        pairs, X = harmonic_pair_features(x, p, k, np.zeros(49, bool))
        ix = np.flatnonzero((pairs[:, 0] == 40) & (pairs[:, 1] == 52))
        self.assertEqual(len(ix), 1)
        self.assertEqual(int(pairs[ix[0], 2]), 2)
        self.assertEqual(X.shape[1], N_FEATURES)
        self.assertAlmostEqual(float(X[ix[0], 1]), float(X[ix[0], 2]))

    def test_guard_refuses_fabricated_untrained_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "requires a separately trained"):
            HarmonicSpecialistSelector(CausalSelection(), mode="guard")

    def test_guard_can_inhibit_but_not_force_extra_notes(self):
        x, p, k = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            path = self.checkpoint(temp, -10.)
            guard = HarmonicSpecialistSelector(CausalSelection(Selection()), mode="guard",
                                               checkpoint=path)
            first = guard.step(x, p, k, HOP, HOP)
            self.assertEqual([e["pitch"] for e in first if e["type"] == "note_on"], [40])
            self.assertGreaterEqual(guard.stats.vetoed_upper_onsets, 1)

    def test_guard_can_preserve_true_octave_if_head_accepts_it(self):
        x, p, k = self.inputs()
        with tempfile.TemporaryDirectory() as temp:
            path = self.checkpoint(temp, 10.)
            guard = HarmonicSpecialistSelector(CausalSelection(Selection()), mode="guard",
                                               checkpoint=path)
            first = guard.step(x, p, k, HOP, HOP)
            self.assertEqual([e["pitch"] for e in first if e["type"] == "note_on"], [40, 52])

    def test_default_stream_does_not_allocate_or_import_specialist(self):
        class FixedHeads:
            metadata = {"selection": {"onset": .6}}
            def predict(self, x):
                p = np.zeros((49, 3), np.float32)
                p[12, :2] = .8
                return p, np.eye(7, dtype=np.float32)[1]
        heads = FixedHeads()
        baseline = StrictLiveStream(heads)
        shadow = StrictLiveStream(heads, harmonic_mode="shadow")
        self.assertIsNone(baseline.harmonic_selector)
        self.assertIsNotNone(shadow.harmonic_selector)
        rng = np.random.default_rng(911)
        audio = rng.normal(0, .05, 32 * HOP).astype(np.float32)
        for start in range(0, len(audio), HOP):
            b = baseline.push(audio[start:start + HOP], start_sample=start)
            s = shadow.push(audio[start:start + HOP], start_sample=start)
            self.assertEqual(b, s)
        self.assertEqual(baseline.finish(), shadow.finish())

    def test_nonfinite_features_fail_closed(self):
        x, p, k = self.inputs()
        x[0, 10] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            harmonic_pair_features(x, p, k, np.zeros(49, bool))


if __name__ == "__main__":
    unittest.main()
