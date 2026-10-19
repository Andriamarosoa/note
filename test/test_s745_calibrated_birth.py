"""S745 FIT-only prior correction, K-tail and acoustic risk safety tests."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s745_calibrated_birth_router import (
    fit_prior_correction,corrected_positive_probability,
    CalibratedBirthPolicy,decode_s745)
from scripts.s745_calibrated_fit_cal_loops import CHOICES


class CalibratedBirthTests(unittest.TestCase):
    def setUp(self):
        self.n=4
        self.ends=np.arange(1,self.n+1)*4*HOP
        self.p=np.zeros((self.n,49,3),np.float32)
        self.k=np.zeros((self.n,7),np.float32)
        self.k[:,0]=.45;self.k[:,1]=.4;self.k[:,2]=.15
        self.heads=np.repeat(self.p[:,:,0,None],6,axis=-1)
        self.ctx=np.zeros((self.n,49,16),np.float32)
        self.p_birth=np.zeros((self.n,49),np.float32)
        self.policy=GuidedPolicy(count_confidence=1.1,guide_confidence=.7)

    def test_fit_corrected_odds_use_only_training_counts(self):
        fact,stats=fit_prior_correction(10,1000,10,100,4.)
        self.assertGreater(fact,0)
        self.assertLess(fact,1)
        self.assertAlmostEqual(stats["natural_prior"],.01)
        self.assertEqual(stats["truth_source"],"FIT80 ONLY")
        p=corrected_positive_probability(
            np.array([0,.2,.5,.8,1],np.float32),fact)
        self.assertAlmostEqual(float(p[0]),0)
        self.assertAlmostEqual(float(p[-1]),1)
        self.assertTrue(np.all(np.diff(p)>=0))
        self.assertLess(float(p[2]),.5)
        with self.assertRaises(ValueError):
            fit_prior_correction(2,100,50,40,4.)

    def test_model_risk_alone_does_not_create_note(self):
        self.p_birth[0,0]=.99
        ev,d=decode_s745(
            self.ends,self.p,self.k,self.heads,self.ctx,
            self.p_birth,self.policy,
            CalibratedBirthPolicy(min_probability=.01,k_tail=.2),
            int(self.ends[-1]))
        self.assertEqual(d["added_births"],0)
        self.assertFalse(any(e["type"]=="note_on" for e in ev))

    def test_positive_model_and_physical_fundamental_and_K_joint(self):
        self.p[0,0,0]=.14
        self.p[0,0,1]=.60
        self.ctx[0,0,4]=.20
        self.ctx[0,0,7]=.50
        self.heads[0,0,0]=.14
        self.heads[0,0,3]=.55
        self.p_birth[0,0]=.12
        ex=CalibratedBirthPolicy(
            min_probability=.05,k_tail=.40,
            min_acoustic=.13,min_sustain=.35,
            min_rise=.015,mask_delta=.04)
        events,diag=decode_s745(
            self.ends,self.p,self.k,self.heads,self.ctx,
            self.p_birth,self.policy,ex,int(self.ends[-1]))
        ons=[e for e in events if e["type"]=="note_on"]
        self.assertEqual(len(ons),1)
        self.assertEqual(ons[0]["pitch"],40)
        self.assertEqual(ons[0]["available_sample"],self.ends[0])
        self.assertEqual(diag["added_births"],1)
        self.assertTrue(all(int(k)%2==1 for k in map(int,diag["masks"])))

    def test_future_audio_cannot_change_prior_note_birth(self):
        self.p[0,0,0]=.14
        self.p[0,0,1]=.60
        self.heads[0,0,3]=.55
        self.p_birth[0,0]=.12
        first,_=decode_s745(
            self.ends,self.p,self.k,self.heads,self.ctx,
            self.p_birth,self.policy,CalibratedBirthPolicy(
                min_probability=.02,k_tail=.3),int(self.ends[-1]))
        changed=self.p_birth.copy()
        changed[-1,:]=.9
        second,_=decode_s745(
            self.ends,self.p,self.k,self.heads,self.ctx,
            changed,self.policy,CalibratedBirthPolicy(
                min_probability=.02,k_tail=.3),int(self.ends[-1]))
        front=lambda events:[(e["type"],e["pitch"],e["decision_sample"])
            for e in events if e["decision_sample"]<self.ends[-1]]
        self.assertEqual(front(first),front(second))

    def test_twelve_prespecified_real_causal_trials(self):
        self.assertEqual(len(CHOICES),12)
        self.assertTrue(all(0<v.min_probability<1 for _,v in CHOICES.values()))


if __name__=="__main__":
    unittest.main()
