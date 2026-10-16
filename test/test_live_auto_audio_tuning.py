"""Transport-only tests: arbitrary block size and bounded automatic queue.

These do not change NOTE's 256-sample causal decision hop or native weights.
"""
from __future__ import annotations

import argparse
import unittest

import numpy as np

from causal_note.strict_live import HOP, SR, StrictLiveStream
from scripts.live_strict_notes import (
    apply_model_input,
    parse_block_size,
    parse_queue_blocks,
    resolve_queue_blocks,
)
from scripts.live_input_level import NoteAutoLevel


class FixedHeads:
    metadata = {"selection": {"onset": 0.6}}

    def predict(self, features):
        p = np.zeros((49, 3), np.float32)
        p[12, :2] = .99
        k = np.zeros(7, np.float32)
        k[1] = 1.
        return p, k


def replay(samples, transport_size):
    stream = StrictLiveStream(FixedHeads())
    events = []
    for start in range(0, len(samples), transport_size):
        events += stream.push(samples[start:start + transport_size], start_sample=start)
    events += stream.finish()
    return events, stream


class LiveAutoTuningTests(unittest.TestCase):
    def test_arbitrary_user_block_sizes_including_portaudio_auto(self):
        self.assertEqual(parse_block_size("0"), 0)
        for value in ("1", "127", "333", "4097", "8192", "65536"):
            self.assertEqual(parse_block_size(value), int(value))
        for value in ("-1", "invalid"):
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_block_size(value)

    def test_queue_is_auto_and_measured_in_audio_duration(self):
        self.assertEqual(resolve_queue_blocks("auto", 256), 13)
        self.assertEqual(resolve_queue_blocks("auto", 512), 7)
        self.assertEqual(resolve_queue_blocks("auto", 1024), 4)
        self.assertEqual(resolve_queue_blocks("auto", 4096), 2)
        self.assertEqual(resolve_queue_blocks("auto", 0), 13)
        self.assertEqual(resolve_queue_blocks(32, 1024), 32)
        self.assertEqual(parse_queue_blocks("auto"), "auto")
        self.assertEqual(parse_queue_blocks("16"), 16)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_queue_blocks("0")

    def test_large_input_packets_preserve_native_note_decisions(self):
        audio = np.random.default_rng(773).normal(
            0, .01, HOP * 40).astype(np.float32)
        original, base = replay(audio, 256)
        for physical_block in (333, 2048, 4097, 8192, 20000):
            events, stream = replay(audio, physical_block)
            self.assertEqual(
                [(e["type"], e["pitch"], e["decision_sample"])
                 for e in original],
                [(e["type"], e["pitch"], e["decision_sample"])
                 for e in events],
            )
            self.assertLessEqual(len(stream.frontend.tail), 8192)
            self.assertEqual(stream.frontend.received, len(audio))
        # The microphone delivered the first large packet at its end:
        # decisions unchanged, real audio availability is correctly delayed.
        big, _ = replay(audio, 8192)
        first = next(e for e in big if e["type"] == "note_on")
        self.assertEqual(first["decision_sample"], HOP)
        self.assertEqual(first["available_sample"], 8192)

    def test_auto_gain_handles_variable_portaudio_callback_lengths(self):
        leveler = NoteAutoLevel(sample_rate=SR, block_size=HOP)
        signal = np.random.default_rng(711).normal(
            0, .02, 5009).astype(np.float32)
        processed = apply_model_input(leveler, signal)
        self.assertEqual(processed.shape, signal.shape)
        self.assertTrue(np.isfinite(processed).all())
        self.assertEqual(
            leveler.total_blocks, int(np.ceil(len(signal) / HOP)))


if __name__ == "__main__":
    unittest.main()
