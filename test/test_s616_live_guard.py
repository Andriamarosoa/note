"""S616 LIVE: deterministic causal wrapper and original S614 preservation."""
from __future__ import annotations

import unittest
import numpy as np

from causal_note.strict_live import Selection, HOP
from causal_note.s616_live_guard import GuardParameters, NativeCausalGuard, replay, candidate_variants


class NativeS616GuardTests(unittest.TestCase):
    def test_pass_through_equals_original_selection(self):
        from causal_note.strict_live import CausalSelection
        rng = np.random.default_rng(616)
        config = Selection(onset=.6, active_on=.5)
        original = CausalSelection(config)
        guard = NativeCausalGuard(config, GuardParameters())
        for i in range(150):
            p = rng.uniform(0, 1, (49, 3)).astype(np.float32)
            k = np.eye(7, dtype=np.float32)[i % 7]
            x = rng.uniform(0, 3, (49, 126)).astype(np.float32)
            a = original.step(p, k, (i + 1) * HOP, (i + 1) * HOP)
            b = guard.step(p, k, (i + 1) * HOP, (i + 1) * HOP, x)
            self.assertEqual(a, b)
        self.assertEqual(original.panic(151 * HOP), guard.finish(151 * HOP))

    def test_causal_unchanged_prefix_if_future_mutates(self):
        rng = np.random.default_rng(617)
        ends = np.arange(1, 151) * HOP
        p = rng.random((150, 49, 3)).astype(np.float32)
        k = np.broadcast_to(np.eye(7, dtype=np.float32)[3], (150,7)).copy()
        x = rng.random((150, 49, 126)).astype(np.float32)
        params = GuardParameters(.4, .4)
        a = replay(ends, p, k, x, Selection(), params, int(ends[-1]))
        pm = p.copy(); xm = x.copy(); km = k.copy()
        pm[90:] = rng.random(pm[90:].shape); xm[90:] = rng.random(xm[90:].shape)
        km[90:] = np.eye(7, dtype=np.float32)[5]
        b = replay(ends, pm, km, xm, Selection(), params, int(ends[-1]))
        self.assertEqual([e for e in a if e["decision_sample"] <= ends[89]],
                         [e for e in b if e["decision_sample"] <= ends[89]])

    def test_guard_is_not_hard_chord_veto(self):
        config = Selection(onset=.6, active_on=.5, release_frames=5)
        guard = NativeCausalGuard(config, GuardParameters(1., 0.))
        p = np.zeros((49, 3), np.float32)
        k = np.eye(7, dtype=np.float32)[2]
        x = np.ones((49,126), np.float32)
        p[2, 0] = .8; p[2, 1] = .9
        self.assertEqual(len(guard.step(p, k, HOP, HOP, x)), 1)
        p[2, 0] = 0
        p[14, 0] = 1.; p[14, 1] = .9
        x[14, [0,8,16]] = 2.0
        # New note's independent fundamental is stronger: guard cannot veto.
        got = guard.step(p, k, 2*HOP, 2*HOP, x)
        self.assertIn(54, [e["pitch"] for e in got if e["type"] == "note_on"])

    def test_invalid_inputs(self):
        guard = NativeCausalGuard(Selection(), GuardParameters())
        with self.assertRaises(ValueError):
            guard.step(np.zeros((49,3)), np.ones(7), HOP, HOP, np.zeros((49,125)))
        with self.assertRaises(ValueError):
            GuardParameters(-.01,0)
        self.assertEqual(len(candidate_variants()), 15)


if __name__ == "__main__":
    unittest.main()
