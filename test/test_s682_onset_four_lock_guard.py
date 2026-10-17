"""S682 protects onset, Exact-K, S675 and full note jointly."""
import unittest
import numpy as np
from dataclasses import replace
from causal_note.strict_live import Selection,HOP,SR,CausalSelection
from scripts.s682_onset_four_lock_guard import (
    configurations,decoded_exact_k,safe_full,full_aggregate,aggregate
)


class FourLockTests(unittest.TestCase):
    def setUp(self):
        self.base=Selection(onset=.60,active_on=.50,
                            active_off=.25,release_frames=5)

    def test_every_variant_only_edits_onset_and_sustain_activation(self):
        options=configurations(self.base)
        self.assertIn("native",options)
        self.assertGreater(len(options),5)
        for key,val in options.items():
            self.assertEqual(val.max_notes,self.base.max_notes)
            self.assertEqual(val.count_confidence,self.base.count_confidence)
            self.assertEqual(val.active_off,self.base.active_off)
            self.assertEqual(val.release_frames,self.base.release_frames)
            self.assertEqual(val.refractory_frames,self.base.refractory_frames)
            self.assertEqual(val.onset_reset,self.base.onset_reset)

    def test_K_counts_include_removal_and_chord(self):
        events=[
            dict(type="note_on",pitch=40,decision_sample=HOP),
            dict(type="note_on",pitch=52,decision_sample=HOP),
            dict(type="note_off",pitch=40,decision_sample=4*HOP),
            dict(type="note_off",pitch=52,decision_sample=5*HOP),
        ]
        truth=[
            dict(time=HOP/SR,duration=3*HOP/SR,pitch=40),
            dict(time=HOP/SR,duration=4*HOP/SR,pitch=52),
        ]
        ends=np.arange(1,9)*HOP
        score=decoded_exact_k(events,truth,ends)
        self.assertEqual(score["frames"],8)
        self.assertGreaterEqual(score["correct"],7)

    def test_no_offset_regression_allowed_even_with_onset_gain(self):
        original=dict(correct=100,eligible=120,
                      conditional_offset_accuracy=100/120,
                      full_note_F1=.45,long_correct=30)
        good=dict(correct=101,eligible=121,
                  conditional_offset_accuracy=101/121,
                  full_note_F1=.451,long_correct=30)
        bad=dict(good,correct=102,
                 conditional_offset_accuracy=.81)
        self.assertTrue(safe_full(original,good))
        self.assertFalse(safe_full(original,bad))

    def test_baseline_event_selector_unchanged(self):
        config=configurations(self.base)["native"]
        native=CausalSelection(self.base)
        candidate=CausalSelection(config)
        p=np.zeros((49,3),np.float32);p[0,:2]=1.0
        k=np.zeros(7,np.float32);k[1]=1
        self.assertEqual(native.step(p,k,HOP,HOP),
                         candidate.step(p,k,HOP,HOP))


if __name__=="__main__":
    unittest.main()
