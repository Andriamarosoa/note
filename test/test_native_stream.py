"""Real NOTE feature/head parity, bounded future, failures and K conservation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from causal_note.native_stream import (
    NativeEvent, NativeFeatureStream, NativeNoteStream, POST, PRE, GEOMETRY_HORIZON,
)
from scripts.live_native_notes import load_heads, DEFAULT_MODELS, read_decisions
from scripts.extract_v273_harmonic_trajectory import _frames, features_from_spectrogram
from scripts.s97_external_s45_original_features import geometry
from scripts.s351_S71_S72_dataset1_477 import geometry46
from scripts.loop_v273_birth_flow import flow_profile
from scripts.v273_extended_head_bank import audio_temporal_views
from scripts.s430_infer_note_names_from_new_S80_sound_events import run as original_s430


def signal(length=100000):
    t = np.arange(length) / 44100.
    x = .05 * np.sin(2 * np.pi * 110 * t) + .03 * np.sin(2 * np.pi * 220 * t)
    return x + np.random.default_rng(611).normal(0, .002, length)


def feed(stream, audio, events, blocks):
    rows = []
    at = index = step = 0
    while at < len(audio):
        end = min(at + blocks[step % len(blocks)], len(audio))
        new = []
        while index < len(events) and events[index].end_sample < end:
            new.append(events[index]); index += 1
        # Original groups may span 40 ms; unclosed future events are not committed.
        rows.extend(stream.push(audio[at:end], start_sample=at, events=new,
                                event_watermark=max(0, end - 1764)))
        at = end; step += 1
    return rows


class NativeFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=1)
        cls.audio = signal()
        cls.events = [NativeEvent(str(i), start, start + 70)
                      for i, start in enumerate([0, 500, 2350, 9000, 15000, 20000, 50000, 58000, 70000])]

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_original_features_and_geometry_match_across_ring_wraps_and_block_sizes(self):
        events = self.events
        data = dict(member=np.asarray(["song"] * len(events)),
                    start_samples=np.asarray([e.start_sample for e in events]),
                    end_samples=np.asarray([e.end_sample for e in events]))
        expected_geom46 = geometry46(data)
        expected_geom3 = geometry(data["member"], data["start_samples"], data["end_samples"])
        sequences, summaries = [], []
        for event in events:
            spectrum, times = _frames(self.audio, event.start_sample)
            summary, activation = features_from_spectrogram(spectrum, times, return_states=True)
            sequences.append(np.log1p(activation).astype(np.float32))
            summaries.append(np.asarray([summary[k] for k in sorted(summary)], np.float32))
        sequences = np.asarray(sequences)
        flow = flow_profile(sequences)
        morph, physics = audio_temporal_views(sequences)
        for sizes in ((256,), (1, 127, 997, 4096)):
            stream = NativeFeatureStream(include_geometry46=True, history_samples=32768)
            packets = feed(stream, self.audio, events, sizes)
            self.assertEqual([p.event for p in packets], events)
            self.assertEqual(stream.pending_events, ())
            self.assertEqual(stream._audio.nbytes, 32768 * 8)
            for i, packet in enumerate(packets):
                for actual, expected in ((packet.sequence42x49, sequences[i]),
                                         (packet.summary58, summaries[i]),
                                         (packet.flow245, flow[i]),
                                         (packet.morphology490, morph[i]),
                                         (packet.physics294, physics[i]),
                                         (packet.geometry3, expected_geom3[i]),
                                         (packet.geometry46, expected_geom46[i])):
                    np.testing.assert_array_equal(actual, expected)
                self.assertGreaterEqual(packet.available_sample, packet.event.start_sample + GEOMETRY_HORIZON)

    def test_no_future_padding_or_early_emission(self):
        event = NativeEvent("source:0", 1000, 1100)
        stream = NativeFeatureStream()
        end = event.start_sample + POST
        self.assertEqual(stream.push(self.audio[:end-1], start_sample=0, events=[event]), ())
        actual = stream.push(self.audio[end-1:end], start_sample=end-1)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].available_sample, end)
        self.assertIsNone(actual[0].geometry46)
        # Changing the entire unobserved future cannot influence this packet.
        changed = self.audio.copy(); changed[end:] = 50
        spectrum, times = _frames(changed, event.start_sample)
        _, states = features_from_spectrogram(spectrum, times, return_states=True)
        np.testing.assert_array_equal(actual[0].sequence42x49, np.log1p(states).astype(np.float32))

    def test_geometry_waits_for_certified_next_event_horizon(self):
        event = NativeEvent("one", 1000, 1100)
        stream = NativeFeatureStream(include_geometry46=True)
        end = 1000 + GEOMETRY_HORIZON
        self.assertEqual(stream.push(self.audio[:end], start_sample=0, events=[event]), ())
        result = stream.push(np.empty(0), start_sample=end, event_watermark=end)
        self.assertEqual(len(result), 1)
        with self.assertRaisesRegex(ValueError, "watermark"):
            stream.push(np.empty(0), start_sample=end, events=[NativeEvent("late", 2000, 2100)])

    def test_invalid_input_is_atomic(self):
        stream = NativeFeatureStream()
        for audio, start, events in ((np.array([np.nan]), 0, []),
                                     (np.zeros(3), 1, []),
                                     (np.zeros(3), 0, [NativeEvent("future", 3, 3)]),
                                     (np.zeros(1000), 0, [NativeEvent("a", 5, 100), NativeEvent("b", 90, 110)])):
            with self.assertRaises(ValueError):
                stream.push(audio, start_sample=start, events=events)
            self.assertEqual(stream.received_samples, 0)
            self.assertEqual(stream.pending_events, ())

    def test_overflow_and_late_events_fail_without_dropping(self):
        stream = NativeFeatureStream(history_samples=PRE + GEOMETRY_HORIZON, max_pending=1)
        event = NativeEvent("a", 100, 200)
        stream.push(np.zeros(300), start_sample=0, events=[event])
        with self.assertRaisesRegex(BufferError, "queue"):
            stream.push(np.zeros(1000), start_sample=300, events=[NativeEvent("b", 400, 500)])
        self.assertEqual(stream.pending_events, (event,))
        self.assertEqual(stream.received_samples, 300)
        delayed = NativeFeatureStream(history_samples=PRE + GEOMETRY_HORIZON)
        for at in range(0, 20000, 1000):
            delayed.push(np.zeros(1000), start_sample=at)
        with self.assertRaises(BufferError):
            delayed.push(np.empty(0), start_sample=20000, events=[event])

    def test_incomplete_end_keeps_event_pending(self):
        stream = NativeFeatureStream()
        event = NativeEvent("unfinished", 5000, 5100)
        self.assertEqual(stream.push(self.audio[:7000], start_sample=0, events=[event]), ())
        self.assertEqual(stream.push(np.empty(0), start_sample=7000), ())
        self.assertEqual(stream.pending_events, (event,))


class NativeHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.limits = threadpool_limits(limits=1)
        cls.heads = load_heads(DEFAULT_MODELS)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_all_original_trained_heads_and_gates_match_unchanged_s430(self):
        rng = np.random.default_rng(611)
        seq = rng.random((14, 42, 49)).astype(np.float32) * .05
        for i in range(len(seq)):
            seq[i, 13:25, (i * 3) % 49] += .7
        k = np.tile(np.arange(7, dtype=np.int8), 2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez_compressed(root / "audio.npz", row_id=np.arange(14).astype(str), sequence42x49=seq)
            np.savez_compressed(root / "k.npz", row_id=np.arange(14).astype(str), s80_K=k)
            original_s430(root / "audio.npz", root / "k.npz", DEFAULT_MODELS / "S402",
                          DEFAULT_MODELS / "S416", DEFAULT_MODELS / "S428",
                          DEFAULT_MODELS / "S419/report.json", DEFAULT_MODELS / "S428/report.json", root / "oracle")
            with np.load(root / "oracle/notes_from_new_sound.npz") as z:
                expected = z["midi_pitch_slots"].copy()
        # Subsequent events never reload checkpoints or recalculate gate votes.
        with patch("scripts.s430_infer_note_names_from_new_S80_sound_events.load_models", side_effect=AssertionError("reload")):
            batch, methods = self.heads.predict(seq, k)
            streamed = np.concatenate([self.heads.predict(seq[i:i+1], k[i:i+1])[0] for i in range(len(k))])
        np.testing.assert_array_equal(batch, expected)
        np.testing.assert_array_equal(streamed, expected)
        np.testing.assert_array_equal((streamed >= 0).sum(1), k)
        self.assertEqual(methods[:7], ("K0", "S428", "S428", "S428", "S428", "S416", "S402"))

    def test_native_stream_keeps_K0_and_original_event_identity(self):
        events = [(NativeEvent(f"original:{i}", 1000 + i * 3000, 1100 + i * 3000), i) for i in range(7)]
        audio = signal(40000)
        runtime = NativeNoteStream(self.heads)
        output = []; index = 0
        for at in range(0, len(audio), 256):
            end = min(at + 256, len(audio)); new = []
            while index < len(events) and events[index][0].end_sample < end:
                new.append(events[index]); index += 1
            output.extend(runtime.push(audio[at:end], start_sample=at, decisions=new))
        self.assertEqual([r["source_row_id"] for r in output], [e.row_id for e, k in events])
        self.assertEqual([len(r["notes"]) for r in output], list(range(7)))
        self.assertTrue(all(160 <= r["acoustic_wait_ms"] < 160 + 256000 / 44100 for r in output))
        self.assertEqual(runtime._counts, {})

    def test_no_K_fallback_and_no_truth_input(self):
        seq = np.zeros((1, 42, 49), np.float32)
        for k in ([1.2], [True], [7], [-1]):
            with self.assertRaises(ValueError):
                self.heads.predict(seq, k)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            path.write_text(json.dumps(dict(row_id="one", start_sample=1, end_sample=2, s80_K=1, true_K=1)))
            with self.assertRaises(ValueError):
                read_decisions(path)


if __name__ == "__main__":
    unittest.main()
