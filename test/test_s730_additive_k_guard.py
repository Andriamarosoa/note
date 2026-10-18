"""S730 must never change an original correct S708 event."""
import unittest
import numpy as np
from causal_note.s730_additive_k_guard import additive_rescue


def event(kind,p,t,reason="original"):
    return dict(type=kind,pitch=p,decision_sample=t,
                available_sample=t,time=t/44100.,velocity=96 if kind=="note_on" else 0,
                reason=reason)


class S730AdditiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(256,12801,256)
        self.k=np.zeros((len(self.ends),7),np.float32)
        self.k[:,1]=.95

    def test_no_proposals_returns_original_exactly(self):
        base=[event("note_on",40,1024),event("note_off",40,8192)]
        out,stats=additive_rescue(base,[],self.ends,self.k)
        self.assertEqual(out,base)
        self.assertEqual(stats["added"],0)

    def test_predicted_two_independent_sources_can_add_true_E3_without_removing_E2(self):
        base=[event("note_on",40,1024),event("note_off",40,8192)]
        proposal=[event("note_on",52,4096),event("note_off",52,7168)]
        self.k[:,1]=0;self.k[:,2]=.93
        out,stats=additive_rescue(base,proposal,self.ends,self.k,confidence=.50)
        self.assertEqual(stats["added"],1)
        self.assertEqual([e for e in out if e["reason"]=="original"],base)
        self.assertEqual([x["pitch"] for x in out if x["type"]=="note_on"],[40,52])
        self.assertEqual(sum(x["type"]=="note_off" for x in out),2)

    def test_ExactK_one_rejects_extra_harmonic_impostor(self):
        base=[event("note_on",40,1024),event("note_off",40,8192)]
        proposed=[event("note_on",52,4096),event("note_off",52,7168)]
        out,state=additive_rescue(base,proposed,self.ends,self.k)
        self.assertEqual(out,base);self.assertEqual(state["added"],0)

    def test_original_onset_preempts_old_addon_without_dropping_reference(self):
        base=[event("note_on",40,1024),event("note_on",52,6144),
              event("note_off",52,7168),event("note_off",40,8192)]
        proposed=[event("note_on",52,4096),event("note_off",52,7680)]
        self.k[:,1]=0;self.k[:,2]=.95
        out,stats=additive_rescue(base,proposed,self.ends,self.k,confidence=.50)
        self.assertEqual(stats["added"],1)
        self.assertEqual([x for x in out if x["reason"]=="original"],base)
        self.assertEqual([x["type"] for x in out if x["pitch"]==52],
                         ["note_on","note_off","note_on","note_off"])
        self.assertTrue(any(x["reason"]=="s730_preempted_by_original_onset" for x in out))

    def test_future_K_cannot_justify_past_attack(self):
        base=[event("note_on",40,1024),event("note_off",40,8192)]
        cand=[event("note_on",52,2048),event("note_off",52,3072)]
        self.k[self.ends>=4096,1]=0
        self.k[self.ends>=4096,2]=.9
        out,state=additive_rescue(base,cand,self.ends,self.k)
        self.assertEqual(out,base)

    def test_invalid_abi_and_event_order_rejected(self):
        with self.assertRaisesRegex(ValueError,"7-way"):
            additive_rescue([],[],self.ends,np.zeros((5,7)))
        with self.assertRaisesRegex(ValueError,"unsorted"):
            additive_rescue([event("note_off",40,8192),event("note_on",40,1024)],
                            [],self.ends,self.k)


if __name__=="__main__":
    unittest.main()
