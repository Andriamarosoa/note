"""Code-audit regression tests for the immutable 69.258433% Onset PCM256.
These tests intentionally document a *known duplicate inference* in the HGB
wrapper rather than patching or promoting historical scientific models.
"""
import unittest
import numpy as np

from onset256.acoustics import HOP, WINDOWS
from onset256.live_event_stream import OnsetDecision
from onset256.learned_hgb_live import ArbitratedAcousticOnsetStream
from onset256.learned_hgb_arbiter import GatePolicy, LearnedHGBArbiter


class CountingModel:
    n_features_in_ = 195

    def __init__(self):
        self.calls = 0

    def predict_proba(self, x):
        self.calls += 1
        return np.column_stack((
            np.ones(len(x), dtype=np.float32),
            np.zeros(len(x), dtype=np.float32)
        ))


class ReferenceStub:
    def __init__(self):
        self.risk = type("Risk", (), {})()
        self.risk.acoustic_evidence = np.zeros((49, 195), np.float32)


class AcousticStub:
    """Model the already-verifiable first Onset predict_proba call."""
    def __init__(self):
        self.model = CountingModel()
        self.reference = ReferenceStub()
        self.last_reference = None
        self.received = 0

    def push(self, pcm, *, start_sample):
        assert start_sample == self.received
        assert np.asarray(pcm).shape == (256,)
        # AcousticOnsetStream.push already evaluates the trained direct head.
        self.model.predict_proba(self.reference.risk.acoustic_evidence)
        self.received += 256
        self.last_reference = OnsetDecision(self.received, ())
        return OnsetDecision(self.received, ())

    def finish(self):
        return ()


class OnsetHGBCodeAuditTests(unittest.TestCase):
    def test_double_predict_proba_on_each_hop_even_when_hgb_disabled(self):
        """CODE DEFECT: downstream HGB wrapper discards first acoustic score."""
        acoustic = AcousticStub()
        wrapper = ArbitratedAcousticOnsetStream(
            acoustic, None, GatePolicy())
        for i in range(4):
            out = wrapper.push(np.zeros(256, dtype=np.float32),
                               start_sample=i * 256)
            self.assertEqual(out.sample, (i+1)*256)
            self.assertEqual(len(out.intents), 0)
            # Exactly the observed current-code behavior, not an endorsement.
            self.assertEqual(acoustic.model.calls, (i+1)*2)
        self.assertEqual(acoustic.received, 4*256)

    def test_hgb_cannot_recover_onset_outside_reference_candidate_set(self):
        class AlwaysPositive:
            n_features_in_ = 207
            def predict_proba(self, x):
                return np.column_stack((
                    np.zeros(len(x), np.float32),
                    np.ones(len(x), np.float32)
                ))
        gate = LearnedHGBArbiter(AlwaysPositive(),
                                 GatePolicy(cut=.5, gap_hops=1))
        q = np.ones(49, np.float32)
        evidence = np.zeros((49, 195), np.float32)
        got, added = gate.step(q,evidence,(),(),end_sample=256)
        self.assertEqual(got, ())
        self.assertEqual(added, ())
        got, added = gate.step(q,evidence,(44,),(),end_sample=512)
        self.assertEqual(got,(44,))
        self.assertEqual(added,(44,))
        self.assertNotIn(40,got)

    def test_exact_decision_hop_but_long_causal_fft_history(self):
        self.assertEqual(HOP,256)
        self.assertEqual(WINDOWS,(2048,4096,8192))
        self.assertGreater(max(WINDOWS),HOP)

    def test_signed_retraining_drift_requires_distinct_reference(self):
        # Archive (69.206204%) != newly replayed control (69.258433%).
        archived=(7585,3068,3682)
        replay=(7607,3093,3660)
        self.assertNotEqual(archived,replay)
        def f1(x):
            tp,fp,fn=x
            return 2*tp/(2*tp+fp+fn)
        self.assertAlmostEqual(f1(archived),.6920620437956204)
        self.assertAlmostEqual(f1(replay),.6925843310420176)

    def test_arbitration_not_promoted_by_three_extra_false_positives(self):
        original=(7607,3093,3660)
        candidate=(7607,3096,3660)
        def f1(t):
            tp,fp,fn=t
            return 2*tp/(2*tp+fp+fn)
        self.assertLess(f1(candidate), f1(original))
        self.assertAlmostEqual(100*(f1(candidate)-f1(original)),
                               -.00945722800694826)


if __name__ == "__main__":
    unittest.main()
