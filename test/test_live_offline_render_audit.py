"""Pure-Python regression tests for LIVE-vs-offline audits and synthesis timing."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch
import tempfile
import unittest

import numpy as np

from scripts.audit_live_vs_offline import summarize_notes, event_key, parse_expected
from scripts.render_live_midi_offline import render_midi_sequence, load_events


class FakeSynth:
    def __init__(self):
        self.active = {}
        self.calls = []
        self.audio_events = []

    def noteon(self, channel, pitch, velocity):
        self.active[pitch] = True
        self.calls.append(("on", pitch, velocity))

    def noteoff(self, channel, pitch):
        self.active.pop(pitch, None)
        self.calls.append(("off", pitch))

    def get_samples(self, frames):
        self.audio_events.extend([sorted(self.active) for _ in range(frames)])
        return np.full(frames * 2, 3000 if self.active else 0, dtype=np.int16)


class LiveOfflineAuditTests(unittest.TestCase):
    def test_short_note_is_rendered_between_events_not_cancelled_in_batch(self):
        events = [
            {"type": "note_on", "pitch": 52, "velocity": 96, "decision_sample": 12},
            {"type": "note_off", "pitch": 52, "velocity": 0, "decision_sample": 20},
        ]
        synth = FakeSynth()
        audio = np.concatenate(list(render_midi_sequence(synth, events, 35, chunk=8)))
        self.assertEqual(audio.size, 70)
        self.assertEqual(sum(52 in active for active in synth.audio_events), 8)
        self.assertEqual(synth.calls, [("on", 52, 96), ("off", 52)])

    def test_same_sample_restrike_preserves_event_order(self):
        events = [
            {"type": "note_off", "pitch": 52, "decision_sample": 100},
            {"type": "note_on", "pitch": 52, "velocity": 90, "decision_sample": 100},
        ]
        synth = FakeSynth()
        list(render_midi_sequence(synth, events, 120))
        self.assertEqual(synth.calls, [("off", 52), ("on", 52, 90)])
        self.assertTrue(synth.active[52])

    def test_counts_real_note_durations(self):
        events = [
            {"type": "note_on", "pitch": 40, "decision_sample": 256},
            {"type": "note_off", "pitch": 40, "decision_sample": 6 * 256, "reason": "causal_release"},
        ]
        counts = summarize_notes(events)
        self.assertEqual(counts["note_on"], 1)
        self.assertEqual(counts["note_off"], 1)
        self.assertAlmostEqual(counts["notes"][0]["duration_ms"], 5 * 256 / 44.1, places=2)
        self.assertEqual(counts["pairing_problems"], [])

    def test_piano_callback_exposes_collapsed_midi_note_and_output_status(self):
        from scripts.live_piano_output import PianoOutput

        class Synth:
            def __init__(self, *args, **kwargs):
                self.playing = False
            def sfload(self, *args):
                return 1
            def program_select(self, *args):
                pass
            def noteon(self, *args):
                self.playing = True
            def noteoff(self, *args):
                self.playing = False
            def get_samples(self, frames):
                return np.full(frames * 2, 5000 if self.playing else 0, np.int16)
            def delete(self):
                pass

        class Stream:
            active = True
            def stop(self):
                self.active = False
            def close(self):
                self.active = False

        callbacks = []
        def output_stream(**kwargs):
            callbacks.append(kwargs["callback"])
            return Stream(), SimpleNamespace(
                host_api="TEST", device_name="TEST_OUTPUT",
                device_index=0, latency_ms=4.0)

        with patch.dict(sys.modules, {"fluidsynth": SimpleNamespace(Synth=Synth)}):
            with patch("scripts.live_piano_output.open_output_stream",
                       side_effect=output_stream):
                piano = PianoOutput("irrelevant-in-mock.sf2")
                try:
                    piano.send({"type": "note_on", "pitch": 52, "velocity": 96})
                    piano.send({"type": "note_off", "pitch": 52})
                    arr = np.empty((128, 2), np.float32)
                    callbacks[0](arr, 128, None, "output_underflow")
                    self.assertTrue(np.all(arr == 0))
                    status = piano.diagnostics()
                    self.assertEqual(status["output_status_count"], 1)
                    self.assertEqual(status["same_callback_on_off_pairs"], 1)
                    self.assertEqual(status["max_control_events_in_callback"], 2)
                finally:
                    piano.close()

    def test_event_validation_and_marker_parser(self):
        with self.assertRaises(Exception):
            parse_expected("E2 1.3")
        self.assertEqual(parse_expected("40@1.23"), (40, 1.23))
        with tempfile.TemporaryDirectory() as td:
            fn = Path(td) / "events.jsonl"
            fn.write_text('{"type":"note_on","pitch":40,"decision_sample":500}\n'
                          '{"type":"note_off","pitch":40,"decision_sample":300}\n')
            with self.assertRaises(ValueError):
                load_events(fn)


if __name__ == "__main__":
    unittest.main()
