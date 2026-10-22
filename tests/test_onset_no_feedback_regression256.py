"""Regression proving S69D error-head veto cannot create phantom NOTE-ONs.

Run #39741883968: CAL only saw 189 FP/45 TP vetoed in the original
proposals, but online zeroing Onset posterior produced 4675 FP (>4600).
The independent third head MUST keep the re-arm/active/refractory selector
state bit-for-bit identical to the frozen S613 main Onset trajectory.
"""
import unittest
import numpy as np
from onset256.acoustics import HOP
from onset256.runtime import Stream
from onset256.selector import CausalSelection
from onset256.heads import OnsetHeadSelector

class PersistentMain:
    metadata={"selection":{}}
    def __init__(self):self.calls=0
    def predict(self,feat):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0]=(.85,.9,0.)
        return p,np.ones(7,np.float32)/7

class Specialist:
    def __init__(self):self.calls=0
    def predict_proba(self,history):
        self.calls+=1
        if history.shape!=(49,39):raise AssertionError("39D trained256 drift")
        return np.tile([.4,.6],(49,1))

class SequenceRisk:
    cut=.5
    requires_selector_state=True
    trained_hop_samples=256
    def __init__(self,sequence):self.sequence=sequence;self.calls=0
    def predict_correct_probability(self,f,p,k,r,history,selector_state):
        assert selector_state.shape==(49,6)
        v=self.sequence[self.calls] if self.calls<len(self.sequence) else .9
        self.calls+=1
        q=np.ones(49,np.float32)*.9
        q[0]=v
        return q

def notes(decisions,axis):
    return [(e.pitch,e.sample) for d in decisions for e in getattr(d,axis)]

class NoFeedbackSelectorTests(unittest.TestCase):
    def test_original_onset_changes_state_when_risk_zeroed(self):
        original=CausalSelection()
        old_gated=CausalSelection()
        k=np.ones(7,np.float32)/7
        normal=np.zeros((49,3),np.float32)
        normal[0]=(.85,.9,0.)
        unchanged=[];zeroed=[]
        for i in range(12):
            end=(i+1)*256
            a=normal.copy()
            b=normal.copy()
            if i==1:b[0,0]=0.  # EXACT problematic S69D design
            unchanged.extend(x for x in original.step(a,k,end,end)
                             if x["type"]=="note_on")
            zeroed.extend(x for x in old_gated.step(b,k,end,end)
                          if x["type"]=="note_on")
        self.assertEqual([x["decision_sample"] for x in unchanged],[256])
        self.assertEqual([x["decision_sample"] for x in zeroed],[256,2816])
        self.assertFalse(original.armed[0])
        self.assertFalse(old_gated.armed[0])  # re-trigger consumed armed
        self.assertEqual(original.last_on[0],256)
        self.assertEqual(old_gated.last_on[0],2816)

    def test_new_output_only_contest_no_duplicate_phantom_attacks(self):
        risk=SequenceRisk([.9,.1]+[.9]*13)
        native=Stream(PersistentMain(),Specialist())
        checked=Stream(PersistentMain(),Specialist(),risk_model=risk)
        pcm=np.zeros(15*HOP,np.float32)
        split=8*HOP
        baseline=native.push(pcm[:split],start_sample=0)+native.push(pcm[split:],start_sample=split)
        gated=checked.push(pcm[:split],start_sample=0)+checked.push(pcm[split:],start_sample=split)
        self.assertEqual(notes(baseline,"intents"),notes(gated,"intents"))
        self.assertEqual(notes(gated,"intents"),[(40,256)])
        self.assertEqual(notes(gated,"predecision_intents"),[(40,256)])
        self.assertEqual(checked.onset_heads.error.calls,15)
        self.assertEqual(checked.calls,15)
        np.testing.assert_array_equal(checked.base.active,checked.risk_selector.active)
        np.testing.assert_array_equal(checked.base.armed,checked.risk_selector.armed)
        np.testing.assert_array_equal(checked.base.last_on,checked.risk_selector.last_on)
        self.assertEqual(checked.risk_selector.last_end,15*256)

    def test_contest_first_true_attack_but_no_late_birth(self):
        risk=SequenceRisk([.01]+[.99]*12)
        checked=Stream(PersistentMain(),Specialist(),risk_model=risk)
        decisions=tuple(d for i in range(13) for d in
                        checked.push(np.zeros(HOP,np.float32),start_sample=i*HOP))
        self.assertEqual(notes(decisions,"intents"),[(40,256)])
        self.assertEqual(notes(decisions,"predecision_intents"),[])
        self.assertEqual(len(decisions[0].head_doubts),1)
        self.assertTrue(decisions[0].head_doubts[0].contested)
        self.assertTrue(checked.base.active[0] and checked.risk_selector.active[0])
        self.assertEqual(checked.base.last_on[0],256)

    def test_only_current_nonlookahead_risk_and_no_other_pitch_birth(self):
        risk=SequenceRisk([.01,.99])
        s=Stream(PersistentMain(),Specialist(),risk_model=risk)
        self.assertEqual(s.push(np.zeros(HOP-1,np.float32),start_sample=0),())
        self.assertEqual(risk.calls,0)
        d1=s.push(np.zeros(1,np.float32),start_sample=HOP-1)
        self.assertEqual(d1[0].sample,256)
        self.assertEqual(d1[0].predecision_intents,())
        d2=s.push(np.zeros(HOP,np.float32),start_sample=HOP)
        self.assertEqual(d2[0].sample,512)
        self.assertEqual(d2[0].predecision_intents,())
        self.assertEqual(risk.calls,2)

    def test_selection_preserves_posteriors_and_returns_only_original(self):
        select=OnsetHeadSelector(.5)
        original=[dict(type="note_on",pitch=40,decision_sample=256),
                  dict(type="note_off",pitch=40,decision_sample=256),
                  dict(type="note_on",pitch=52,decision_sample=256)]
        c=np.ones(49,np.float32)
        c[0]=.1
        actual=select.challenge_emitted_onsets(original,c)
        self.assertEqual(actual,(original[2],))
        self.assertEqual(original[0]["type"],"note_on")
        self.assertEqual(len(original),3)
        with self.assertRaises(ValueError):
            select.challenge_emitted_onsets(original,np.full(49,np.nan,np.float32))

if __name__=="__main__":unittest.main()
