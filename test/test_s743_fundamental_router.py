"""S743 causal multi-fundamental candidate, 32-mask contract and MIDI tests."""
import unittest
from types import SimpleNamespace
import numpy as np
from causal_note.strict_live import HOP, SR
from causal_note.s708_cardinality_selection import GuidedPolicy, decode_guided
from causal_note.s743_fundamental_router import (
    candidate_heads,MultiHeadBirthPolicy,
    select_fragment_subset,decode_s743,MASKS)


class S743RealArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.t=4
        self.p=np.zeros((self.t,49,3),np.float32)
        self.native=self.p.copy()
        self.context=np.zeros((self.t,49,16),np.float32)
        self.raw=np.zeros((self.t,49,288),np.float32)
        self.scene=[SimpleNamespace(
            novelty_unshared=np.zeros(49,np.float32),
            supports=np.zeros(49,np.int32)) for _ in range(self.t)]
        self.ends=np.arange(1,self.t+1)*4*HOP
        self.k=np.zeros((self.t,7),np.float32)
        self.k[:,0]=.45;self.k[:,1]=.40;self.k[:,2]=.15
        self.policy=GuidedPolicy(count_confidence=1.1,
                                  guide_confidence=.7)

    def test_all_six_fundamental_heads_frozen_h0_and_causal_future(self):
        self.p[0,0,0]=.13;self.p[:,0,1]=.65
        self.context[0,0,4]=.3
        self.context[0,0,5]=.01
        self.context[0,0,8]=.1
        self.scene[0].supports[0]=3
        self.scene[0].novelty_unshared[0]=.25
        self.raw[0,0,192:216]=np.arange(24)
        head=candidate_heads(
            self.p,self.native,self.context,self.raw,self.scene)
        self.assertEqual(head.shape,(self.t,49,6))
        np.testing.assert_array_equal(head[:,:,0],self.p[:,:,0])
        self.assertGreater(head[0,0,2],head[0,0,0])
        self.assertGreater(head[0,0,3],head[0,0,0])
        self.assertEqual(len(MASKS),32)
        other=self.p.copy()
        other[-1]=.9
        current=candidate_heads(
            other,self.native,self.context,self.raw,self.scene)
        np.testing.assert_array_equal(head[:-1],current[:-1])

    def test_dynamically_chosen_mask_one_per_fragment_not_per_song(self):
        p=self.p[0]
        ctx=self.context[0]
        head=np.zeros((49,6),np.float32)
        head[:,0]=.1
        head[:,1:]=.1
        pol=MultiHeadBirthPolicy(mask_delta=.05)
        self.assertEqual(select_fragment_subset(head,p,ctx,pol),1)
        head[2,2]=.45
        self.assertEqual(select_fragment_subset(head,p,ctx,pol),5)
        head[9,5]=.55
        self.assertEqual(select_fragment_subset(head,p,ctx,pol),37)
        self.assertIn(select_fragment_subset(
            head,p,ctx,MultiHeadBirthPolicy(
                mask_strategy="conservative",mask_delta=.05)),MASKS)

    def test_native_k_alone_cannot_create_extra_midi_note(self):
        offers=candidate_heads(self.p,self.native,self.context,
                               self.raw,self.scene)
        ev,di=decode_s743(self.ends,self.p,self.k,offers,self.context,
              self.policy,MultiHeadBirthPolicy(k_tail=.2),
              int(self.ends[-1]))
        self.assertEqual(di["added_births"],0)
        self.assertFalse(any(x["type"]=="note_on" for x in ev))

    def test_unexplained_independent_pitch_can_be_offered_no_backdating(self):
        # H3 residual evidence boosts weak E2 onset with Sustain support.
        self.p[0,0,0]=.12
        self.p[0,0,1]=.70
        self.p[1:,0,0]=.09
        self.p[1:,0,1]=.70
        self.context[0,0,4]=.3
        self.context[0,0,8]=.12
        self.context[0,0,5]=.01
        self.scene[0].supports[0]=3
        self.scene[0].novelty_unshared[0]=.4
        offers=candidate_heads(self.p,self.native,self.context,
                               self.raw,self.scene)
        p=MultiHeadBirthPolicy(k_tail=.50,min_onset=.16,
                      min_sustain=.30,mask_strategy="adaptive",
                      mask_delta=.04,min_rise=.02)
        events,diag=decode_s743(
            self.ends,self.p,self.k,offers,self.context,
            self.policy,p,int(self.ends[-1]))
        self.assertTrue(diag["added_births"]>=1)
        self.assertEqual(events[0]["pitch"],40)
        self.assertEqual(events[0]["available_sample"],self.ends[0])
        self.assertTrue(all(x["decision_sample"]==x["available_sample"]
                            for x in events))
        alternate=self.p.copy()
        alternate[-1,1,0]=.8
        altered_offers=candidate_heads(alternate,self.native,self.context,
                                        self.raw,self.scene)
        trial,_=decode_s743(
            self.ends,alternate,self.k,altered_offers,self.context,
            self.policy,p,int(self.ends[-1]))
        self.assertEqual([(e["type"],e["pitch"],e["decision_sample"])
                for e in events if e["decision_sample"]<self.ends[-1]],
                [(e["type"],e["pitch"],e["decision_sample"])
                for e in trial if e["decision_sample"]<self.ends[-1]])

    def test_reference_protected_on_malformed_input(self):
        offers=candidate_heads(self.p,self.native,self.context,
                               self.raw,self.scene)
        with self.assertRaises(ValueError):
            decode_s743(self.ends,self.p,self.k,offers[:2],
                        self.context,self.policy,
                        MultiHeadBirthPolicy(),int(self.ends[-1]))
        with self.assertRaises(ValueError):
            MultiHeadBirthPolicy(mask_strategy="future").validate()


if __name__=="__main__":
    unittest.main()
