"""Causal state regressions and event-level preservation, not logit checks."""
import unittest
import numpy as np

from causal_note.strict_live import CausalSelection, Selection
from causal_note.s702_event_selection import (
    candidate_track, render_candidates, onset_keys, paired_onsets,
)
from scripts.s613_train_strict_live import decode, event_score


class S702Tests(unittest.TestCase):
    def fixture(self, n=100):
        rng = np.random.default_rng(702)
        post = rng.random((n, 49, 3), dtype=np.float32)
        k = rng.random((n, 7), dtype=np.float32)
        k /= k.sum(1, keepdims=True)
        x = rng.random((n, 49, 39), dtype=np.float32)
        scores = rng.random((n, 49), dtype=np.float32)
        return np.arange(1, n + 1) * 256, post, k, x, scores

    def test_noop_exact_event_parity(self):
        ends, post, k, x, scores = self.fixture(220)
        for config in (Selection(), Selection(onset=.6, active_on=.5,
                       confirmation_frames=2, count_confidence=.15)):
            rows, _, _ = candidate_track(ends, post, k, post, x, scores, config)
            result = render_candidates(ends, post, rows, np.ones(len(rows)),
                                       config, int(ends[-1]))
            self.assertEqual(result, decode(ends, post, k, config, int(ends[-1])))

    def test_early_rescue_cannot_disarm_later_native(self):
        ends, post, k, x, scores = self.fixture(60)
        post[:] = 0
        post[:, :, 0] = .05
        post[:, 0, 1] = .8
        post[10:36, 0, 0] = .25
        post[35, 0, 0] = .8
        boosted = post.copy()
        boosted[10, 0, 0] = .625
        config = Selection(onset=.6, active_on=.5)
        base = decode(ends, post, k, config, int(ends[-1]))
        legacy = decode(ends, boosted, k, config, int(ends[-1]))
        self.assertNotIn((40, int(ends[35])), onset_keys(legacy))
        rows, _, _ = candidate_track(ends, post, k, boosted, x, scores, config)
        fixed = render_candidates(ends, post, rows, np.ones(len(rows)), config,
                                  int(ends[-1]), rescue_threshold=.5,
                                  protect_native=True)
        self.assertTrue(onset_keys(base).issubset(onset_keys(fixed)))
        self.assertIn((40, int(ends[10])), onset_keys(fixed))

    def test_rescues_cannot_take_native_polyphony_slots(self):
        ends, post, k, x, scores = self.fixture(60)
        post[:] = 0
        post[:, :7, 1] = .8
        boosted = post.copy()
        boosted[3, :6, 0] = .9
        post[30, 6, 0] = .9
        boosted[30, 6, 0] = .9
        config = Selection(onset=.6)
        rows, _, _ = candidate_track(ends, post, k, boosted, x, scores, config)
        result = render_candidates(ends, post, rows, np.ones(len(rows)), config,
                                   int(ends[-1]), rescue_threshold=.5,
                                   protect_native=True)
        self.assertIn((46, int(ends[30])), onset_keys(result))
        active = set()
        for event in result:
            if event['type'] == 'note_on':
                active.add(event['pitch'])
            else:
                active.discard(event['pitch'])
            self.assertLessEqual(len(active), 6)

    def test_prefix_features_and_outputs_are_causal(self):
        ends, post, k, x, scores = self.fixture(180)
        config = Selection(onset=.6)
        boosted = post.copy()
        boosted[:, :, 0] = np.maximum(boosted[:, :, 0], scores)
        a, ax, _ = candidate_track(ends, post, k, boosted, x, scores, config)
        cut = 97
        b, bx, _ = candidate_track(ends[:cut], post[:cut], k[:cut],
                                  boosted[:cut], x[:cut], scores[:cut], config)
        before = [i for i, row in enumerate(a) if row['frame'] < cut]
        self.assertEqual(b, [a[i] for i in before])
        np.testing.assert_array_equal(bx, ax[before])
        qa = np.clip(ax[:, 0], 0, 1)
        qb = qa[before]
        ea = render_candidates(ends, post, a, qa, config, int(ends[-1]),
                               native_threshold=.3, rescue_threshold=.4)
        eb = render_candidates(ends[:cut], post[:cut], b, qb, config,
                               int(ends[cut-1]), native_threshold=.3,
                               rescue_threshold=.4)
        keep = lambda ev: [e for e in ev if e['decision_sample'] <= ends[cut-1]
                           and e['reason'] != 'stream_end']
        self.assertEqual(keep(ea), keep(eb))

    def test_event_audit_exposes_loss_behind_net_gain(self):
        truth = [dict(pitch=40, time=t, duration=.1) for t in (.2, .5, .8)]
        make = lambda ts: [dict(type='note_on', pitch=40, time=t,
                                decision_sample=round(t*44100)) for t in ts]
        result = paired_onsets(make([.2]), make([.5,.8]), truth, event_score)
        self.assertEqual((result['recovered'], result['lost'], result['delta_tp']),
                         (2,1,1))

    def test_protected_native_survives_random_rescues(self):
        ends, post, k, x, scores = self.fixture(300)
        config = Selection(onset=.6)
        original = post.copy()
        boosted = post.copy()
        boosted[:, :, 0] = np.maximum(boosted[:, :, 0], scores)
        rows, _, base = candidate_track(ends, post, k, boosted, x, scores, config)
        output = render_candidates(ends, post, rows, np.ones(len(rows)), config,
                                   int(ends[-1]), rescue_threshold=.5,
                                   protect_native=True)
        self.assertTrue(onset_keys(base).issubset(onset_keys(output)))
        np.testing.assert_array_equal(post, original)


if __name__ == '__main__':
    unittest.main()
