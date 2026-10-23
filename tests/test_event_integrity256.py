import unittest
from types import SimpleNamespace

import numpy as np

from onset256.event_confidence import EventConfidenceHead
from onset256.fn_recovery import FNRecoverySelector


def event(kind, sample=256):
    return dict(type=kind, pitch=52, decision_sample=sample)


class TestRecoveryEventIntegrity(unittest.TestCase):
    def select(self, selector, proposals, sample=256):
        return selector.select(end_sample=sample, main_raw=(), main_accepted=(),
                               proposals=proposals,
                               recovery_probability=np.ones(49, np.float32))

    def test_release_does_not_swallow_same_block_restrike(self):
        s = FNRecoverySelector(.5)
        picked = self.select(s, [event("note_off"), event("note_on")])
        self.assertEqual(picked, (event("note_on"),))
        self.assertEqual(s.last_added[12], 256)

    def test_release_does_not_start_a_future_inhibition(self):
        s = FNRecoverySelector(.5)
        self.assertEqual(self.select(s, [event("note_off")]), ())
        self.assertEqual(s.last_added[12], -10**12)
        picked = self.select(s, [event("note_on", 512)], 512)
        self.assertEqual(picked, (event("note_on", 512),))

    def test_duplicate_note_on_still_emitted_once(self):
        s = FNRecoverySelector(.5)
        self.assertEqual(len(self.select(s, [event("note_on"), event("note_on")])), 1)

    def test_candidate_archive_without_type_keeps_its_meaning(self):
        e = event("note_on")
        del e["type"]
        self.assertEqual(self.select(FNRecoverySelector(.5), [e]), (e,))

    def test_paired_audit_reproduces_the_original_bug(self):
        from scripts.onset_event_integrity256 import LegacyRecovery
        old = self.select(LegacyRecovery(.5), [event("note_off"), event("note_on")])
        self.assertEqual(old, (event("note_off"),))


class ConstantEstimator:
    n_features_in_ = 69

    def predict_proba(self, x):
        return np.tile([.2, .8], (len(x), 1))


class TestEventHeadCadence(unittest.TestCase):
    def test_no_confirmation_hop_and_no_backdating(self):
        h = EventConfidenceHead(ConstantEstimator(), cut=.7, selected_model="test")
        q = h.predict(np.zeros((49, 69), np.float32), sample=256)
        e = SimpleNamespace(pitch=52, sample=256)
        self.assertEqual(h.select((e,), q, sample=256), (e,))
        with self.assertRaises(ValueError):
            h.predict(np.zeros((49, 69), np.float32), sample=768)
        with self.assertRaises(ValueError):
            h.select((SimpleNamespace(pitch=52, sample=512),), q, sample=256)
