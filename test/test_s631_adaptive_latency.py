"""S631 adaptive latency evidence: frame causality, variable hold, MIDI lifecycle."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import Selection,HOP
from causal_note.s631_adaptive_latency import (
    WAIT,EMIT,REJECT,CandidateTracker,DecisionPolicy,
    candidate_features,FEATURE_DIM,LearnedAdaptiveSelector)


class StubActions:
    classes_=np.array([0,1,2])
    def predict_proba(self,x):
        output=[]
        for v in x:
            elapsed=v[-15]  # could vary by architecture; classifier below
            score=float(v[65]) if len(v)>65 else 0.
            if score>.55:output.append([.03,.96,.01])
            elif score<.22:output.append([.05,.02,.93])
            else:output.append([.92,.05,.03])
        return np.asarray(output,np.float32)


class AdaptivePolicyTests(unittest.TestCase):
    def test_feature_dimensions_and_only_past(self):
        tracker=CandidateTracker(DecisionPolicy())
        p=np.zeros((49,3),np.float32)
        x=np.zeros((49,126),np.float32)
        k=np.eye(7,dtype=np.float32)[1]
        p[8,0]=.4;p[8,1]=.5
        event=tracker.step(p,x,k)
        self.assertEqual(len(event),1)
        self.assertEqual(event[0][2].shape,(FEATURE_DIM,))
        self.assertEqual(event[0][1],0)
        tracker.step(p,x,k)
        self.assertEqual(tracker.frame,1)
    def test_variable_episodic_age_and_bounded_memory(self):
        t=CandidateTracker(DecisionPolicy(max_frames=6))
        x=np.zeros((49,126),np.float32)
        k=np.eye(7,dtype=np.float32)[1]
        p=np.zeros((49,3),np.float32)
        p[7,0]=.5;p[7,1]=.5
        ages=[]
        for _ in range(10):
            rows=t.step(p,x,k)
            ages.extend([age for index,age,*_ in rows])
        self.assertEqual(ages,[0,1,2,3,4,5])
        self.assertEqual(len(t.pending),0)
    def test_future_mutation_cannot_change_prefix(self):
        rng=np.random.default_rng(631)
        p=rng.random((50,49,3)).astype(np.float32)*.7
        x=rng.random((50,49,126)).astype(np.float32)
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[2],(50,7))
        def play(p,x):
            tr=CandidateTracker()
            result=[]
            for i in range(len(p)):
                result.extend((i,idx,age,v.copy()) for idx,age,v,*_ in tr.step(p[i],x[i],k[i]))
            return [(i,idx,age,v) for i,idx,age,v in result if i<25]
        before=play(p,x)
        alteredp=p.copy();alteredx=x.copy()
        alteredp[25:]=0;alteredx[25:]*=5.
        after=play(alteredp,alteredx)
        self.assertEqual(len(before),len(after))
        for a,b in zip(before,after):
            self.assertEqual(a[:3],b[:3])
            np.testing.assert_array_equal(a[3],b[3])
    def test_invalid_policy(self):
        with self.assertRaises(ValueError):DecisionPolicy(max_frames=100)
        with self.assertRaises(ValueError):DecisionPolicy(emit_threshold=1.)
        with self.assertRaises(ValueError):CandidateTracker().step(
            np.zeros((48,3)),np.zeros((49,126)),np.ones(7))


if __name__=="__main__":
    unittest.main()
