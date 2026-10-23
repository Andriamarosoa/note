import unittest
import numpy as np

from onset256.learned_hgb_arbiter import (
    CausalHGBContext, DIM, GatePolicy, LearnedHGBArbiter,
    candidate_rows, replay_gate, replay_precomputed)


class ScoreModel:
    n_features_in_ = DIM
    def predict_proba(self, x):
        p = np.clip(x[:, 195], 0., 1.)
        return np.column_stack((1. - p, p))


class LearnedHGBArbiterTests(unittest.TestCase):
    def inputs(self, n=48):
        rng = np.random.default_rng(8721)
        q = rng.random((n, 49), dtype=np.float32)
        x = rng.random((n, 49, 195), dtype=np.float32)
        baseline = rng.random((n, 49)) < .025
        direct = rng.random((n, 49)) < .018
        return q, x, baseline, direct

    def test_cached_and_live_candidate_rows_identical(self):
        q, x, baseline, direct = self.inputs()
        cached, loc = candidate_rows(q, x, baseline, direct)
        observer = CausalHGBContext()
        rows, where = [], []
        for f in range(len(q)):
            h = tuple(np.flatnonzero(baseline[f]) + 40)
            d = tuple(np.flatnonzero(direct[f]) + 40)
            tmp = observer.step(q[f], x[f], h, d,
                                end_sample=(f+1)*256)
            rows.extend(tmp)
            where.extend((f, p) for p in h)
        np.testing.assert_array_equal(cached, np.asarray(rows, np.float32))
        self.assertEqual(loc, where)
        self.assertEqual(cached.shape[1], 207)

    def test_frozen_probabilities_equals_actual_live_decoder(self):
        q, x, baseline, direct = self.inputs(128)
        rows, _ = candidate_rows(q, x, baseline, direct)
        probabilities = ScoreModel().predict_proba(rows)[:, 1]
        for gap in (1, 7, 14):
            for cut in (.1, .5, .9):
                p = GatePolicy(cut=cut, gap_hops=gap)
                fast = replay_precomputed(direct, baseline, probabilities, p)
                live = replay_gate(q, x, baseline, direct, ScoreModel(), p)
                np.testing.assert_array_equal(fast, live)
                self.assertTrue(np.all(live[direct]))

    def test_each_decision_uses_no_future_and_no_added_hold(self):
        q, x, baseline, direct = self.inputs(32)
        cut = GatePolicy(cut=.5, gap_hops=7)
        a = replay_gate(q, x, baseline, direct, ScoreModel(), cut)
        q2, x2 = q.copy(), x.copy()
        q2[16:] = 0.
        x2[16:] = 0.
        b = replay_gate(q2, x2, baseline, direct, ScoreModel(), cut)
        np.testing.assert_array_equal(a[:16], b[:16])
        stream = LearnedHGBArbiter(ScoreModel(), cut)
        with self.assertRaises(ValueError):
            stream.step(q[0], x[0], [], [], end_sample=512)

    def test_old_acoustic_decision_is_not_vetoed(self):
        model = ScoreModel()
        stream = LearnedHGBArbiter(model, GatePolicy(cut=.99))
        q = np.zeros(49, np.float32)
        x = np.zeros((49, 195), np.float32)
        a, added = stream.step(q, x, [40, 41], [40, 52], end_sample=256)
        self.assertEqual(a, (40, 52))
        self.assertEqual(added, ())
        self.assertEqual(stream.context.end, 256)

    def test_new_hgb_proposal_without_future(self):
        model = ScoreModel()
        stream = LearnedHGBArbiter(model, GatePolicy(cut=.5, gap_hops=7))
        q = np.zeros(49, np.float32)
        q[1] = .99
        x = np.zeros((49, 195), np.float32)
        all_pitches, added = stream.step(q, x, [41], [], end_sample=256)
        self.assertEqual(all_pitches, (41,))
        self.assertEqual(added, (41,))
        # A repeated HGB attack is not emitted as a new event 1 hop later.
        all_pitches, added = stream.step(q, x, [41], [], end_sample=512)
        self.assertEqual(added, ())
        self.assertEqual(all_pitches, ())

    def test_non_finite_inputs_refused(self):
        q = np.zeros(49, np.float32)
        q[0] = np.nan
        with self.assertRaises(ValueError):
            CausalHGBContext().step(
                q, np.zeros((49, 195), np.float32),
                [40], [], end_sample=256)


if __name__ == "__main__":
    unittest.main()
