"""Demonstrate the count-veto regression without a GPU, WAV, or native model."""
import unittest

import numpy as np

from causal_note.strict_live import CausalSelection, HOP, Selection


class NativeCountVetoTests(unittest.TestCase):
    def probabilities(self, k=1):
        p = np.zeros((49, 3), dtype=np.float32)
        for pitch in (52, 59, 61, 64, 68):
            p[pitch - 40, 0] = .85
            p[pitch - 40, 1] = .9
        count = np.zeros(7, dtype=np.float32)
        count[k] = .9
        count[0] = .1
        return p, count

    def note_ons(self, conf, predicted_k=1):
        p, k = self.probabilities(predicted_k)
        s = CausalSelection(Selection(onset=.6, count_confidence=conf))
        return [e["pitch"] for e in s.step(p, k, HOP, HOP)
                if e["type"] == "note_on"]

    def test_original_1p1_never_uses_count_budget(self):
        self.assertEqual(len(self.note_ons(1.1)), 5)

    def test_count_gate_can_restrict_false_chord_if_count_is_correct(self):
        self.assertEqual(len(self.note_ons(.5)), 1)

    def test_count_gate_cannot_fix_a_wrong_count_head(self):
        self.assertEqual(len(self.note_ons(.5, predicted_k=5)), 5)


if __name__ == "__main__":
    unittest.main()
