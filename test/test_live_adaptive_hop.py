"""Safety tests for adaptive NOTE S613 hop (trained baseline hop=256)."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from causal_note.strict_live import (
    CausalPitchFeatures, CausalSelection, HOP, SR,
    Selection, StrictLiveStream,
)
from scripts.live_hop_tuning import (
    parse_hop_size, choose_cpu_hop, benchmark_hop, resolve_hop,
)


class Head:
    metadata = {"selection": {"onset": .6}}
    def predict(self, x):
        p = np.zeros((49, 3), np.float32)
        p[12, :2] = .99
        k = np.zeros(7, np.float32)
        k[1] = 1
        return p, k


class AdaptiveHopTests(unittest.TestCase):
    def test_manual_hops_and_validation(self):
        for hop in (128, 256, 512):
            self.assertEqual(parse_hop_size(str(hop)), hop)
        self.assertEqual(parse_hop_size("auto"), "auto")
        for wrong in ("64", "257", "1024", "bad", "0"):
            with self.assertRaises(ValueError):
                parse_hop_size(wrong)

    def test_time_normalized_history_and_bounded_memory(self):
        rng = np.random.default_rng(18)
        signal = rng.normal(0, .05, 140 * HOP).astype(np.float32)
        for hop, history, lags, recent in [
            (128, 64, (4, 8, 16, 32), 16),
            (256, 32, (2, 4, 8, 16), 8),
            (512, 16, (1, 2, 4, 8), 4),
        ]:
            f = CausalPitchFeatures(hop=hop)
            self.assertEqual(f.lags, lags)
            self.assertEqual(f.recent_frames, recent)
            for start in range(0, len(signal), 333):
                f.push(signal[start:start+333], start_sample=start)
            self.assertLessEqual(len(f.history), history)
            self.assertLessEqual(len(f.tail), 8192)
            self.assertEqual(f.received, len(signal))

    def test_first_time_confirmation_release_and_refractory_stay_physical(self):
        for hop, release, confirmation in [(128, 10, 2), (256, 5, 1), (512, 2, 1)]:
            c = CausalSelection(Selection(), hop=hop)
            self.assertEqual(c.release_steps, release)
            self.assertEqual(c.confirmation_steps, confirmation)
            p = np.zeros((49, 3), np.float32)
            p[12, :2] = .98
            k = np.eye(7, dtype=np.float32)[1]
            events = []
            for n in range(1, confirmation + 2):
                events.extend(c.step(p, k, n * hop, n * hop))
            note_on = [e for e in events if e["type"] == "note_on"]
            self.assertEqual(len(note_on), 1)
            self.assertEqual(note_on[0]["decision_sample"], confirmation * hop)

    def test_causal_future_mutation_for_each_hop(self):
        rng = np.random.default_rng(19)
        audio = rng.normal(0, .03, 64 * HOP).astype(np.float32)
        cutoff = 32 * HOP
        changed = audio.copy()
        changed[cutoff:] += rng.normal(0, 0.7, len(audio)-cutoff)
        for hop in (128, 256, 512):
            def predict(x):
                stream = StrictLiveStream(Head(), hop=hop)
                events = []
                for start in range(0, len(x), 733):
                    events.extend(stream.push(x[start:start+733], start_sample=start))
                events.extend(stream.finish())
                return [e for e in events if e["decision_sample"] <= cutoff]
            stable = predict(audio)
            perturbed = predict(changed)
            self.assertEqual(
                [(v["type"],v["pitch"],v["decision_sample"]) for v in stable],
                [(v["type"],v["pitch"],v["decision_sample"]) for v in perturbed])

    def test_auto_cpu_rule_lowest_healthy_hop_and_fallback(self):
        levels = {
            128: {"p95_ms": 1.2},
            256: {"p95_ms": 1.0},
            512: {"p95_ms": 1.0},
        }
        self.assertEqual(choose_cpu_hop(levels), 128)
        levels[128]["p95_ms"] = 3.0
        self.assertEqual(choose_cpu_hop(levels), 256)
        levels[256]["p95_ms"] = 5.0
        self.assertEqual(choose_cpu_hop(levels), 512)
        self.assertEqual(choose_cpu_hop({
            128: {"p95_ms": 100.},
            256: {"p95_ms": 100.},
            512: {"p95_ms": 100.},
        }), 512)

    def test_benchmark_works_without_real_sound_device(self):
        vals = benchmark_hop(Head(), frames=16)
        self.assertEqual(set(vals), {128, 256, 512})
        for hop in vals:
            self.assertGreaterEqual(vals[hop]["p95_ms"], 0)
        with patch("scripts.live_hop_tuning.benchmark_hop", return_value={
            128: {"p95_ms": 1.0}, 256: {"p95_ms": .8}, 512: {"p95_ms": .8},
        }):
            selected, info = resolve_hop(Head(), "auto")
        self.assertEqual(selected, 128)
        self.assertEqual(info["method"], "cpu_preflight")

    def test_reference_256_bitwise_same(self):
        rng = np.random.default_rng(614)
        audio = rng.normal(0, .02, HOP*10).astype(np.float32)
        base = StrictLiveStream(Head())
        stated = StrictLiveStream(Head(), hop=256)
        for start in range(0, len(audio), 256):
            x = audio[start:start+256]
            self.assertEqual(base.push(x, start_sample=start),
                             stated.push(x, start_sample=start))
        self.assertEqual(base.finish(), stated.finish())


if __name__ == "__main__":
    unittest.main()
