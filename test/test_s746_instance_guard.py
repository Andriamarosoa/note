"""Unit tests for S746 native event-preserving mux, no learned model required."""
import unittest
import numpy as np
from causal_note.s746_instance_guard import (
    InstanceGate, extra_pairs, onset_features, fit_event_labels, merge_events)


def evt(kind, pitch, sample, reason):
    return dict(type=kind, pitch=pitch, decision_sample=sample,
                available_sample=sample, time=sample / 44100.,
                reason=reason, velocity=96 if kind == "note_on" else 0)


class InstanceMuxTests(unittest.TestCase):
    def setUp(self):
        self.base = [evt("note_on", 40, 1024, "native"),
                     evt("note_off", 40, 4096, "s707_causal_hazard")]
        self.candidate = [(evt("note_on", 41, 2048, "s743_multifundamental"),
                           evt("note_off", 41, 3072, "s707_causal_hazard"))]
        self.k = np.zeros((8, 7), dtype=np.float32)
        self.k[:, 2:] = .2
        self.ids = [(1, 1)]

    def test_native_stream_exactly_retained_and_extra_added(self):
        merged, detail = merge_events(self.base, self.candidate, [0.99],
                                      self.ids, self.k, InstanceGate(.8, .2))
        self.assertEqual(detail["accepted"], 1)
        self.assertEqual([e for e in merged if e in self.base], self.base)
        self.assertEqual(sum(e["type"] == "note_on" for e in merged), 2)

    def test_no_posterior_or_low_risk_produces_exact_original(self):
        self.assertEqual(merge_events(self.base, self.candidate, [.5],
                         self.ids, self.k, InstanceGate(.8, .2))[0], self.base)
        self.assertEqual(merge_events(self.base, self.candidate, [.99],
                         self.ids, np.zeros_like(self.k),
                         InstanceGate(.8, .2))[0], self.base)

    def test_source_onset_preempts_extra_instead_of_losing_source(self):
        extra = [(evt("note_on", 40, 1024, "s743_multifundamental"),
                  evt("note_off", 40, 3000, "s707_causal_hazard"))]
        merged, diag = merge_events(self.base, extra, [.99],
                                    [(0, 0)], self.k, InstanceGate(.5, .1))
        self.assertEqual(diag["accepted"], 0)  # Source on wins same timestamp.
        self.assertEqual(merged, self.base)
        future = [evt("note_on", 40, 2048, "native"),
                  evt("note_off", 40, 4096, "s707_causal_hazard")]
        merged, diag = merge_events(future, extra, [.99],
                                    [(0, 0)], self.k, InstanceGate(.5, .1))
        self.assertEqual(diag["preempted"], 1)
        self.assertTrue(any(e["reason"] == "s746_source_priority" for e in merged))
        self.assertEqual([e for e in merged if e in future], future)

    def test_extra_pairs_excludes_native_ons(self):
        full = [self.base[0], self.candidate[0][0], self.candidate[0][1],
                self.base[1]]
        self.assertEqual(extra_pairs(full), self.candidate)

    def test_causal_features_ignore_all_future_offsets(self):
        ends = np.arange(1, 9) * 1024
        inputs = np.random.default_rng(1).random((8, 49, 5)).astype("float32")
        heads = np.ones((8, 49, 6), np.float32)
        f, indices = onset_features(self.candidate, ends, inputs, heads, self.k)
        modified = [(self.candidate[0][0],
                     evt("note_off", 41, 8192, "s707_causal_hazard"))]
        f2, ids2 = onset_features(modified, ends, inputs, heads, self.k)
        np.testing.assert_array_equal(f, f2)
        self.assertEqual(indices, ids2)

    def test_fit_label_requires_correct_complete_new_note(self):
        on, off = self.candidate[0]
        truth = [dict(pitch=41, time=on["time"], duration=(
                      off["time"] - on["time"]))]
        self.assertEqual(fit_event_labels(self.candidate, truth, set()).tolist(), [1])
        self.assertEqual(fit_event_labels(self.candidate, truth, {0}).tolist(), [0])


if __name__ == "__main__":
    unittest.main()
