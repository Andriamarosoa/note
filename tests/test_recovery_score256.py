import unittest
from types import SimpleNamespace
import numpy as np

from onset256.fn_recovery import FNRecoverySelector
from scripts.onset_recovery_hgb256 import eligibility, selected_mask, EventMatcher, extra_events
from scripts.onset_event_integrity256 import counts


class TestRecoveryScore(unittest.TestCase):
    def test_event_proxy_matches_live_threshold_and_refractory_boundaries(self):
        samples = np.asarray([256, 512, 4096, 4352, 4608, 8192, 8448])
        pitches = np.full(len(samples), 60)
        native = [(4608, 60)]
        q = np.asarray([.9, .95, .95, .95, .95, .95, .95], np.float32)
        proposals = dict(samples=samples, pitches=pitches,
                         eligible=eligibility(samples, pitches, native))
        mask = selected_mask(proposals, q, .9)
        self.assertFalse(mask[0])  # float32(.9) is below the exact float64 cut.
        arbiter = FNRecoverySelector(.9)
        actual = []
        for sample in range(256, 8449, 256):
            indices = np.flatnonzero(samples == sample)
            p = np.zeros(49, np.float32)
            events = []
            for i in indices:
                p[20] = q[i]
                events.append(dict(type="note_on", pitch=60, decision_sample=sample))
            main = [SimpleNamespace(pitch=60, sample=sample)] if sample == 4608 else []
            actual.extend(arbiter.select(end_sample=sample, main_raw=main,
                main_accepted=main, proposals=events, recovery_probability=p))
        self.assertEqual([int(s) for s in samples[mask]], [e['decision_sample'] for e in actual])

    def test_cached_matcher_preserves_one_to_one_assignment(self):
        row = dict(main=[dict(sample=256, pitch=60)], truth=[
            dict(time=.02, pitch=60), dict(time=.08, pitch=60), dict(time=.12, pitch=61)])
        p = dict(samples=np.asarray([512, 3072, 5120]), pitches=np.asarray([60,60,61]))
        scorer = EventMatcher(row, p)
        for i in range(8):
            mask = np.asarray([bool(i & (1 << k)) for k in range(3)])
            self.assertEqual(scorer.score(mask), counts(row['main']+extra_events(p,mask),row['truth']))
