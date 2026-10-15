"""Behavioral contracts for microphone scheduling, causal notes and cleanup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

import numpy as np

from causal_note.live_engine import DEFAULT_BUNDLE, HOP, WINDOW, LatestAudioQueue, LiveEngine, TimingStats
from causal_note.live_reference.decoder import PolyphonicDecoder, PolyphonicDecoderConfig
from causal_note.live_reference.tflite_runtime import PolyphonicPrediction
from scripts.live_notes import EventSink, run, wav_blocks


META = json.loads((DEFAULT_BUNDLE / "metadata.json").read_text())


class FakeModel:
    def __init__(self):
        self.windows = []
        self.pitch = 12  # E3
        self.onset = True
        self.present = True

    def infer(self, waveform, visible_window=WINDOW):
        self.windows.append((waveform.copy(), visible_window))
        frame = np.zeros(37, np.float32)
        onset = np.zeros_like(frame)
        if self.present:
            frame[self.pitch] = .99
            onset[self.pitch] = .99 if self.onset else 0
        return PolyphonicPrediction(frame, onset, np.zeros((37, 20), np.float32),
                                    np.zeros((37, 20), np.float32), .1)


class LiveEngineTests(unittest.TestCase):
    def engine(self):
        return LiveEngine(FakeModel(), META, prime_silence=True)

    def play(self, engine, hops=3, value=.1):
        result = []
        for _ in range(hops):
            result.extend(engine.process(np.full(HOP, value, np.float32), start_sample=engine.last_end))
        return result

    def test_note_names_sample_end_and_terminal_noteoff(self):
        engine = self.engine()
        events = self.play(engine)
        self.assertEqual([(x.type, x.pitch, x.sample) for x in events], [("note_on", 52, 512)])
        self.assertEqual(events[0].json()["note"], "E3")
        self.assertEqual(events[0].json()["clock"], "decision_sample_end")
        final = engine.finalize(800, reason="eof")
        self.assertEqual((final[0].type, final[0].sample, final[0].note_id),
                         ("note_off", 800, events[0].note_id))
        self.assertEqual(engine.finalize(), [])
        with self.assertRaises(RuntimeError):
            self.play(engine)

    def test_waveform_prefix_is_unchanged_by_different_future(self):
        a, b = self.engine(), self.engine()
        left, right = self.play(a, 12), self.play(b, 12)
        self.assertEqual(left, right)
        for (wa, va), (wb, vb) in zip(a.runtime.windows, b.runtime.windows):
            np.testing.assert_array_equal(wa, wb)
            self.assertEqual(va, vb)
        prefix = [w.copy() for w, _ in a.runtime.windows]
        self.play(a, 3, .8)
        self.play(b, 3, -.8)
        for before, (after, _) in zip(prefix, a.runtime.windows):
            np.testing.assert_array_equal(before, after)

    def test_audio_gap_closes_notes_and_clears_past_audio(self):
        engine = self.engine()
        old = self.play(engine)[0]
        start = engine.last_end + 4 * HOP
        events = engine.process(np.full(HOP, .2, np.float32), start_sample=start)
        self.assertEqual([(x.type, x.reason, x.sample, x.note_id) for x in events],
                         [("note_off", "audio_gap", start, old.note_id)])
        self.assertEqual(engine.ring.available, HOP)
        self.assertEqual(engine.recoveries, 1)
        self.play(engine, 1, .2)
        window, visible = engine.runtime.windows[-1]
        np.testing.assert_array_equal(window[:-2 * HOP], 0)
        self.assertEqual(visible, 512)

    def test_callback_status_resets_even_with_contiguous_positions(self):
        engine = self.engine()
        self.play(engine)
        events = engine.process(np.zeros(HOP, np.float32), start_sample=engine.last_end,
                                discontinuity=True)
        self.assertEqual([x.type for x in events], ["note_off"])
        self.assertFalse(engine.active_ids)

    def test_silence_and_release_close_notes_without_reinventing_attacks(self):
        engine = self.engine()
        self.play(engine)
        engine.runtime.onset = False
        self.assertEqual(self.play(engine, 5), [])
        engine.runtime.present = False
        released = self.play(engine, 3)
        self.assertEqual([(x.type, x.reason) for x in released], [("note_off", "release")])
        self.assertEqual(self.play(engine, 20, 0), [])

    def test_silence_vetoes_even_high_model_probabilities(self):
        engine = self.engine()
        self.assertEqual(self.play(engine, 20, 0), [])

    def test_real_calibration_never_emits_notes(self):
        engine = LiveEngine(FakeModel(), META)
        self.assertEqual(self.play(engine, 20), [])
        self.assertFalse(engine.evidence.calibrated)

    def test_invalid_input_fails_without_clock_mutation(self):
        engine = self.engine()
        for samples in (np.zeros(20), np.full(HOP, np.nan), np.full(HOP, np.inf)):
            with self.assertRaises(ValueError):
                engine.process(samples, start_sample=0)
            self.assertEqual(engine.last_end, 0)
        self.play(engine, 2)
        with self.assertRaises(ValueError):
            engine.process(np.zeros(HOP), start_sample=0)

    def test_skipped_inference_keeps_audio_and_original_attack_clock(self):
        engine = LiveEngine(FakeModel(), META, prime_silence=True)
        self.play(engine, 3)
        previous_calls = len(engine.runtime.windows)
        physical_attack_hop = engine.last_end // HOP
        onset = SimpleNamespace(is_onset=True)
        evidence = SimpleNamespace(calibrated=True, onset=onset,
                                   activity=SimpleNamespace(active=True),
                                   audio_hop_index=physical_attack_hop)
        with patch.object(engine.evidence, "process", return_value=evidence):
            engine.process(np.full(HOP, .25, np.float32), start_sample=engine.last_end,
                           skip_inference=True)
        self.assertEqual(len(engine.runtime.windows), previous_calls)
        evidence.onset = SimpleNamespace(is_onset=False)
        evidence.audio_hop_index += 1
        with patch.object(engine.evidence, "process", return_value=evidence):
            engine.process(np.full(HOP, .5, np.float32), start_sample=engine.last_end)
        np.testing.assert_array_equal(engine.runtime.windows[-1][0][-2 * HOP:-HOP], .25)
        self.assertEqual(engine.decoder.last_audio_onset, physical_attack_hop)
        self.assertEqual(engine.skipped_inferences, 1)

    def test_six_notes_limit_and_reattack_off_before_on(self):
        cfg = PolyphonicDecoderConfig(**META["decoder"])
        decoder = PolyphonicDecoder(cfg)
        frame = np.full(37, .99, np.float32)
        zeros = np.zeros(37, np.float32)
        initial = decoder.step(frame, frame, audio_hop_index=0, audio_onset=True)
        self.assertEqual(len(initial), 6)
        pitch = initial[0].pitch
        frame.fill(0)
        frame[pitch - 40] = .99
        decoder.step(frame, zeros, audio_hop_index=1, audio_onset=False)
        reattack = decoder.step(frame, frame, audio_hop_index=14, audio_onset=True)
        self.assertEqual([e.kind for e in reattack if e.pitch == pitch], ["note_off", "note_on"])
        # Persistent high pitch probability cannot retrigger without a new attack.
        later = decoder.step(frame, frame, audio_hop_index=40, audio_onset=False)
        self.assertFalse(any(e.kind == "note_on" for e in later))


class SchedulingTests(unittest.TestCase):
    def test_overload_is_bounded_and_preserves_capture_clock(self):
        capture = LatestAudioQueue(capacity=4, max_backlog=1)
        for n in range(20):
            capture.callback(np.full((HOP, 1), n, np.float32), HOP)
        block = capture.get()
        self.assertEqual(block.start_sample, 19 * HOP)
        self.assertTrue(block.discontinuity)
        self.assertEqual(capture.callback_drops, 16)
        self.assertEqual(capture.backlog_drops, 3)
        self.assertEqual(capture.next_sample, 20 * HOP)
        self.assertTrue(capture.queue.empty())

    def test_driver_errors_advance_clock_and_mark_discontinuity(self):
        capture = LatestAudioQueue()
        captured = capture.last_valid_callback_at
        capture.callback(np.zeros((HOP, 1)), HOP, status=True)
        self.assertEqual(capture.last_valid_callback_at, captured)
        capture.callback(np.zeros((123, 1)), 123)
        capture.callback(np.zeros((HOP, 1)), HOP)
        block = capture.get()
        self.assertEqual(block.start_sample, HOP + 123)
        self.assertTrue(block.discontinuity)
        self.assertEqual(capture.status_events, 1)
        self.assertEqual(capture.invalid_blocks, 1)

    def test_timing_storage_is_bounded_and_total_count_survives(self):
        stats = TimingStats(capacity=10)
        for i in range(100):
            stats.add(float(i))
        self.assertEqual(len(stats.values), 10)
        self.assertEqual(stats.report()["count"], 100)
        self.assertEqual(stats.report()["mean_ms"], 49.5)

    def test_partial_wav_eof_is_not_padded_into_future_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tail.wav"
            with wave.open(str(path), "wb") as out:
                out.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
                out.writeframes(np.zeros((2 * HOP + 7, 2), "<i2").tobytes())
            blocks = wav_blocks(path)
            info = next(blocks)
            self.assertEqual(info["sample_count"], 519)
            self.assertEqual(info["tail_samples_without_inference"], 7)
            self.assertEqual(len(list(blocks)), 2)

    def test_midi_cleanup_attempts_every_note_and_panic_after_driver_failure(self):
        sink = EventSink(output_format="none")
        import mido
        sink.mido = mido
        calls = []
        class Port:
            def send(self, message):
                calls.append(message)
                if message.type == "note_off" and message.note == 52:
                    raise RuntimeError("driver failure")
            def close(self):
                calls.append("closed")
        sink.port = Port()
        sink.sent_pitches = {52, 60}
        errors = sink.close()
        self.assertEqual(len(errors), 1)
        self.assertEqual(calls[-1], "closed")
        self.assertEqual([x.control for x in calls[:-1] if x.type == "control_change"], [64, 120, 123])
        self.assertEqual([x.note for x in calls[:-1] if x.type == "note_off"], [52, 60])

    def test_pinned_vendor_and_model_hashes(self):
        root = Path(__file__).resolve().parents[1]
        provenance = json.loads((root / "src/causal_note/live_reference/PROVENANCE.json").read_text())
        for row in provenance["files"]:
            self.assertEqual(hashlib.sha256((root / row["local_path"]).read_bytes()).hexdigest(), row["local_sha256"])

    def test_microphone_stall_and_runtime_failure_both_finalize_notes(self):
        for failed_model in (False, True):
            with self.subTest(failed_model=failed_model):
                engine = LiveEngine(FakeModel(), META, prime_silence=True)
                for i in range(3):
                    engine.process(np.full(HOP, .1, np.float32), start_sample=i * HOP)
                if failed_model:
                    def fail(*args):
                        raise RuntimeError("inference failure")
                    engine.runtime.infer = fail
                class Stream:
                    active = False
                    latency = .01
                    stopped = closed = False
                    def __init__(self, **kwargs):
                        self.callback = kwargs["callback"]
                    def start(self):
                        if failed_model:
                            for _ in range(2):
                                self.callback(np.full((HOP, 1), .1, np.float32), HOP)
                    def stop(self):
                        self.stopped = True
                    def close(self):
                        self.closed = True
                audio = SimpleNamespace(check_input_settings=lambda **kw: None,
                                        InputStream=Stream, query_devices=lambda *a: {"name": "fake"})
                args = SimpleNamespace(bundle=DEFAULT_BUNDLE, threads=2, gain=1.,
                    calibration_s=1., wav=None, format="none", midi_port=None, channel=1,
                    events=None, quiet=True, device=None, duration_s=None,
                    realtime=False, inject_stall_ms=0, report=None)
                # Start at the matching callback sample clock for this active session.
                capture = LatestAudioQueue()
                capture.next_sample = engine.last_end
                with patch.dict("sys.modules", {"sounddevice": audio}), \
                     patch("scripts.live_notes.LiveEngine.from_bundle", return_value=engine), \
                     patch("scripts.live_notes.LatestAudioQueue", return_value=capture):
                    report = run(args)
                self.assertEqual(report["stop_reason"], "error")
                self.assertIn("inference failure" if failed_model else "Flux microphone interrompu", report["error"])
                self.assertFalse(report["active_notes_at_end"])
                self.assertEqual(report["events"]["note_on"], report["events"]["note_off"])
                self.assertEqual(report["cleanup_errors"], [])


if __name__ == "__main__":
    unittest.main()
