from __future__ import annotations
import io
import json
from pathlib import Path
import unittest
import numpy as np

from causal_note.strict_live import (
    HOP, SR, CausalPitchFeatures, feature_track, CausalSelection, Selection,
    StrictLiveStream, load_heads,
)
from scripts.live_strict_notes import blocks_stdin


class StrictLiveTests(unittest.TestCase):
    def test_future_mutation_and_input_chunking_cannot_change_features(self):
        rng = np.random.default_rng(613)
        audio = rng.normal(0, .05, SR).astype(np.float32)
        cutoff = HOP * 85
        changed = audio.copy()
        changed[cutoff:] = rng.normal(0, .9, len(audio) - cutoff)
        ends, expected = feature_track(audio)
        _, mutant = feature_track(changed, 997)
        _, prefix = feature_track(audio[:cutoff], 127)
        np.testing.assert_array_equal(expected[ends <= cutoff], mutant[ends <= cutoff])
        np.testing.assert_array_equal(prefix, expected[ends <= cutoff])
        _, rechunked = feature_track(audio, 997)
        np.testing.assert_array_equal(rechunked, expected)

    def test_bounded_memory_and_discontinuity_fail_closed(self):
        frontend = CausalPitchFeatures()
        for i in range(500):
            frontend.push(np.zeros(HOP, np.float32), start_sample=i * HOP)
        self.assertLessEqual(len(frontend.tail), 8192)
        self.assertEqual(len(frontend.history), 32)
        with self.assertRaisesRegex(ValueError, "discontinuity"):
            frontend.push(np.zeros(HOP, np.float32), start_sample=0)
        with self.assertRaises(BufferError):
            frontend.push(np.zeros(4097, np.float32), start_sample=500 * HOP)
        with self.assertRaises(ValueError):
            frontend.push(np.array([np.nan], np.float32), start_sample=500 * HOP)
        self.assertEqual(frontend.received, 500 * HOP)

    def test_note_on_restrike_note_off_and_count_selection(self):
        selection = CausalSelection(Selection(refractory_frames=2, release_frames=2, count_confidence=.8))
        p = np.zeros((49, 3), np.float32)
        p[12, :2] = .9
        p[16, :2] = .8
        k = np.eye(7)[1]
        events = selection.step(p, k, HOP, HOP)
        self.assertEqual([(e["type"], e["pitch"]) for e in events], [("note_on", 52)])
        p[:, 0] = 0
        self.assertEqual(selection.step(p, k, 2 * HOP, 2 * HOP), [])
        p[12, 0] = .9
        events = selection.step(p, k, 3 * HOP, 3 * HOP)
        self.assertEqual([e["type"] for e in events], ["note_off", "note_on"])
        p[:] = 0
        self.assertEqual(selection.step(p, k, 4 * HOP, 4 * HOP), [])
        events = selection.step(p, k, 5 * HOP, 5 * HOP + 7)
        self.assertEqual(events[0]["type"], "note_off")
        self.assertEqual(events[0]["time"], (5 * HOP + 7) / SR)
        self.assertEqual(selection.panic(6 * HOP), [])

    def test_fragmented_pcm16_and_truncated_sample(self):
        class Fragmented(io.BytesIO):
            def read(self, n=-1):
                return super().read(min(n, 7))
        pcm = np.arange(700, dtype="<i2")
        actual = np.concatenate(list(blocks_stdin(Fragmented(pcm.tobytes()), HOP)))
        np.testing.assert_array_equal(actual, pcm.astype(np.float32) / 32768)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            list(blocks_stdin(Fragmented(pcm.tobytes() + b"x"), HOP))

    def test_learned_pipeline_future_invariance_and_balanced_shutdown(self):
        path = Path(__file__).resolve().parents[1] / "model/live/strict_note/NOTE_strict_live.npz"
        self.assertTrue(path.is_file(), "The deployed native checkpoint must be included")
        heads = load_heads(path)
        rng = np.random.default_rng(615)
        audio = rng.normal(0, .05, SR).astype(np.float32)
        cut = 80 * HOP
        changed = audio.copy(); changed[cut:] *= 10
        def run(signal, block):
            stream = StrictLiveStream(heads)
            events = []
            for start in range(0, len(signal), block):
                events.extend(stream.push(signal[start:start + block], start_sample=start))
            events.extend(stream.finish())
            self.assertEqual(stream.finish(), [])
            active = set()
            for e in events:
                if e["type"] == "note_on":
                    self.assertNotIn(e["pitch"], active); active.add(e["pitch"])
                else:
                    self.assertIn(e["pitch"], active); active.remove(e["pitch"])
                self.assertLessEqual(len(active), 6)
                self.assertLessEqual(e["decision_sample"], e["available_sample"])
            self.assertEqual(active, set())
            return events
        before, after = run(audio, HOP), run(changed, HOP)
        self.assertEqual([e for e in before if e["decision_sample"] <= cut], [e for e in after if e["decision_sample"] <= cut])
        other = run(audio, 997)
        project = lambda rows: [(r["type"], r["pitch"], r["decision_sample"]) for r in rows]
        self.assertEqual(project(before), project(other))

    def test_trained_stream_does_not_emit_notes_on_silence(self):
        from scripts.live_strict_notes import DEFAULT_MODEL
        stream = StrictLiveStream(load_heads(DEFAULT_MODEL))
        events = []
        for start in range(0, 2 * SR, HOP):
            events.extend(stream.push(np.zeros(min(HOP, 2 * SR - start), np.float32), start_sample=start))
        events.extend(stream.finish())
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
