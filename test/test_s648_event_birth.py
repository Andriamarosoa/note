"""S648 original S614 and true causal note-birth guarantee."""
from __future__ import annotations
import unittest
import numpy as np

from causal_note.strict_live import HOP,Selection
from causal_note.s648_event_birth import (
    frame_features,causal_features,new_birth_targets,
    sampled_training_events,decode_birth,BirthPolicy,stream_predict_only
)
from scripts.s613_train_strict_live import decode


class SyntheticScorer:
    def predict_proba(self,x):
        # Only a pitch with a strong first harmonic should receive high q.
        q=np.where(x[:,0]>.6,.995,.01)
        return np.column_stack((1-q,q))


def fixture():
    t=30
    row=dict(ends=np.arange(1,t+1,dtype=np.int64)*HOP,
       x=np.zeros((t,49,126),np.float32),
       p=np.zeros((t,49,3),np.float32),
       k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(t,7)).copy(),
       truth=[dict(pitch=52,time=2*HOP/44100.,duration=.050)],
       selection=dict(onset=.6,active_on=.5,release_frames=3,refractory_frames=10),
       n=t*HOP)
    row["x"][2:7,12,0]=.97
    return row

class S648Tests(unittest.TestCase):
    def test_original_bytewise_unchanged_and_birth_can_emit(self):
        row=fixture()
        baseline=decode(row["ends"],row["p"],row["k"],
                        Selection(**row["selection"]),row["n"])
        a,info=decode_birth(row,SyntheticScorer(),BirthPolicy())
        self.assertEqual(a,baseline)
        self.assertEqual(info["emitted_candidates"],0)
        b,detail=decode_birth(row,SyntheticScorer(),BirthPolicy(.90,.10))
        self.assertEqual(sum(e["type"]=="note_on" for e in b),1)
        self.assertEqual(sum(e["type"]=="note_off" for e in b),1)
        self.assertEqual(detail["emitted_candidates"],1)
        self.assertTrue(all(e["decision_sample"]<=e["available_sample"] for e in b))

    def test_truth_labels_are_event_beginnings_not_occupancy(self):
        row=fixture()
        labels,starts=new_birth_targets(row)
        self.assertLessEqual(int(labels.sum()),2)
        self.assertEqual(len(starts),1)
        self.assertEqual(int(labels[:,12].sum()),2)
        feature=causal_features(row)
        x,y,counts=sampled_training_events(row,np.random.default_rng(648))
        self.assertEqual(x.shape[1],feature.shape[-1])
        self.assertGreater(int(y.sum()),0)

    def test_future_audio_cannot_change_feature_prefix(self):
        row=fixture()
        before=causal_features(row).copy()
        row["x"][19:]=42.
        row["p"][19:,:,0]=.99
        after=causal_features(row)
        np.testing.assert_array_equal(before[:19],after[:19])

    def test_per_frame_probability_equals_batched_head(self):
        row=fixture()
        f=causal_features(row)
        q=stream_predict_only(f[3],SyntheticScorer())
        self.assertEqual(q.shape,(49,))
        self.assertEqual(float(q[12]),.995)
        self.assertEqual(float(q[0]),.01)

    def test_bounded_selector_choice_and_no_restrike_burst(self):
        row=fixture()
        row["x"][2:10,:,0]=.99
        events,info=decode_birth(row,SyntheticScorer(),BirthPolicy(.90,.10))
        self.assertLessEqual(sum(e["type"]=="note_on" for e in events),2)
        self.assertTrue(info["emitted_candidates"]<=2)

if __name__=="__main__":
    unittest.main()
