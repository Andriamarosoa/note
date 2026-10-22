"""Frozen-native256 falsifiable FN audit, prior to any FN rescue head training."""
import unittest
import numpy as np
from onset256.acoustics import HOP,SR
from scripts.audit_onset_fn_candidates256 import (
    FNExplorer,categorize,PROPOSAL_CONFIGS,
    HYPOTHESIS_LOW_ONSET_FRACTION)

class DummyNative:
    metadata={"selection":{}}
    def __init__(self,on=.15,sustain=.25):
        self.calls=0;self.on=on;self.sustain=sustain
    def predict(self,features):
        self.calls+=1
        assert features.shape==(49,126)
        p=np.zeros((49,3),np.float32)
        p[0]=[self.on,self.sustain,0.]
        return p,np.ones(7,np.float32)/7

class DummyRescue:
    def __init__(self):self.calls=0
    def predict_proba(self,context):
        self.calls+=1
        assert context.shape==(49,39)
        return np.tile(np.array([.4,.6],np.float32),(49,1))

class GenuinePCM256FNAuditTests(unittest.TestCase):
    def test_audit_is_noninvasive_and_candidates_are_causal(self):
        native=DummyNative()
        s701=DummyRescue()
        observed=FNExplorer(native,s701)
        decisions=observed.push(np.zeros(3*HOP,np.float32),start_sample=0)
        self.assertEqual([d.sample for d in decisions],[256,512,768])
        self.assertEqual(native.calls,3)
        self.assertEqual(s701.calls,3)
        self.assertEqual(len([e for d in decisions for e in d.intents]),0)
        self.assertEqual(len(observed.proposed["moderate"]),1)
        self.assertEqual(len(observed.proposed["sensitive"]),1)
        self.assertEqual(len(observed.proposed["cautious"]),0)
        self.assertEqual(observed.proposed["moderate"][0]["decision_sample"],HOP)
        self.assertEqual(observed.proposal_evidence["moderate"][0].shape,(69,))
        self.assertEqual(len(observed.frame_onset),3)
        self.assertEqual(observed.calls,3)
        original=observed.frame_onset[0][0]
        self.assertAlmostEqual(float(original),.15,places=6)
        self.assertEqual(observed.base.last_on[0],-10**12)
        observed.close_proposals()
    def test_false_negative_matches_causal_onset_candidate_only_posthoc(self):
        explorer=FNExplorer(DummyNative(),DummyRescue())
        decisions=explorer.push(np.zeros(3*HOP,np.float32),start_sample=0)
        # Ground truth introduced for audit AFTER model has done 3 decisions.
        truth=[dict(pitch=40,time=HOP/SR,duration=.1)]
        score,types,rows=categorize(truth,decisions,explorer)
        self.assertEqual(score,dict(tp=0,fp=0,fn=1))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["cause_evidence_category"],
                         "main_onset_below_calibrated_threshold")
        self.assertEqual(types["below_main_onset_threshold"],1)
        self.assertTrue(types["threshold_model_limitation_supported"])
        self.assertTrue(rows[0]["candidate_oracle_recoverable"]["sensitive"])
        self.assertTrue(rows[0]["candidate_oracle_recoverable"]["moderate"])
        self.assertFalse(rows[0]["candidate_oracle_recoverable"]["cautious"])
    def test_no_audio_until_complete_sample_boundary(self):
        model=FNExplorer(DummyNative(),DummyRescue())
        self.assertEqual(model.push(np.zeros(HOP-1,np.float32),start_sample=0),())
        self.assertEqual(model.calls,0)
        out=model.push(np.zeros(1,np.float32),start_sample=HOP-1)
        self.assertEqual([e.sample for e in out],[HOP])
        self.assertEqual(model.calls,1)
        with self.assertRaises(ValueError):
            model.push(np.zeros(1,np.float32),start_sample=0)
    def test_main_reference_invariant_when_candidate_threshold_is_lower(self):
        p=FNExplorer(DummyNative(on=.9,sustain=.9),DummyRescue())
        results=p.push(np.zeros(2*HOP,np.float32),start_sample=0)
        self.assertEqual([(d.sample,e.pitch) for d in results for e in d.intents],
                         [(256,40)])
        self.assertEqual(len(p.proposed["sensitive"]),1)
        self.assertEqual(len(p.proposed["moderate"]),1)
        self.assertEqual(len(p.proposed["cautious"]),1)
        self.assertEqual(p.base.armed[0],False)
        self.assertFalse(p.proposals["sensitive"].armed[0])
    def test_signed_audit_hypothesis_is_pre_registered(self):
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
        self.assertEqual(HYPOTHESIS_LOW_ONSET_FRACTION,.5)
        self.assertEqual(tuple(PROPOSAL_CONFIGS),("sensitive","moderate","cautious"))
        self.assertGreater(PROPOSAL_CONFIGS["moderate"]["onset"],
                           PROPOSAL_CONFIGS["sensitive"]["onset"])
if __name__=="__main__":unittest.main()
