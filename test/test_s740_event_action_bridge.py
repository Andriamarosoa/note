"""S740 shadow MIDI actions: causality, reattack, real note identities."""
import unittest
import numpy as np
from causal_note.s740_event_action_bridge import decode_pitch_actions


class ActionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((5, 49, 5), np.float32)
        self.a[:, :, 0] = 1.
        self.ends = np.array([1024, 2048, 3072, 4096, 5120])

    def action(self, frame, midi, kind):
        row = self.a[frame, midi-40]
        row[:] = 0.
        row[kind] = 1.

    def test_add_reattack_change_pitch_remove_as_midi(self):
        self.action(0, 40, 1)
        self.action(1, 40, 4)
        self.action(2, 40, 3)
        self.action(3, 42, 2)
        destination = np.full((5, 49), -1)
        destination[2, 0] = 42
        result = decode_pitch_actions(
            self.a, self.ends, replacement_pitches=destination)
        observed = [(e["type"], e["pitch"]) for e in result["events"]]
        self.assertEqual(observed, [
            ("note_on", 40),
            ("note_off", 40), ("note_on", 40),
            ("note_off", 40), ("note_on", 42),
            ("note_off", 42)])
        self.assertEqual(result["final_active"], ())
        self.assertEqual(result["skipped"]["CHANGE_PITCH"], 0)
        self.assertEqual(
            [e["available_sample"] for e in result["events"]],
            sorted(e["available_sample"] for e in result["events"]))

    def test_change_pitch_never_invents_destination(self):
        self.action(0, 40, 1)
        self.action(1, 40, 3)
        result = decode_pitch_actions(self.a, self.ends)
        self.assertEqual(result["final_active"], (40,))
        self.assertEqual(result["skipped"]["CHANGE_PITCH"], 1)

    def test_six_note_limit(self):
        for pitch in range(40, 47):
            self.action(0, pitch, 1)
        result = decode_pitch_actions(self.a, self.ends)
        self.assertEqual(len(result["final_active"]), 6)
        self.assertEqual(result["skipped"]["ADD"], 1)

    def test_strict_causal_prefix(self):
        self.action(0, 40, 1)
        self.action(1, 40, 4)
        first = decode_pitch_actions(self.a, self.ends)
        changed = self.a.copy()
        changed[-1, 40-40] = [0, 0, 1, 0, 0]
        second = decode_pitch_actions(changed, self.ends)
        self.assertEqual(
            [e for e in first["events"] if e["available_sample"] < self.ends[-1]],
            [e for e in second["events"] if e["available_sample"] < self.ends[-1]])
        with self.assertRaises(ValueError):
            decode_pitch_actions(self.a, self.ends[::-1])


if __name__ == "__main__":
    unittest.main()
