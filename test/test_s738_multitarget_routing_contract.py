"""S738 target isolation, causal evidence and strict six-score contracts."""
import unittest
import numpy as np

from causal_note.s738_multitarget_routing_contract import (
    METRICS,StageState,acoustic_other_target_heads,apply_one_target,
    require_sequential_training,ensure_valid_adapters)
from causal_note.s736_fragment_mask import MASKS

class ReuseExactKSelectionContract(unittest.TestCase):
    def setUp(self):
        n=4
        self.post=np.full((n,49,3),.2,np.float32)
        self.native=np.full_like(self.post,.3)
        self.ctx=np.zeros((n,49,16),np.float32)
        self.k=np.tile(np.array([.05,.65,.30,0,0,0,0],np.float32),(n,1))
        self.eligible=np.ones((n,49),bool)

    def test_sustain_uses_all_32_and_preserves_onset_release(self):
        heads=acoustic_other_target_heads(
            "sustain",self.post,self.native,self.ctx,self.k)
        self.assertEqual(heads.shape,(4,49,6))
        masks=np.asarray([1,3,31,63],np.uint8)
        updated=apply_one_target(self.post,"sustain",heads,self.eligible,masks)
        self.assertTrue(np.array_equal(updated[:,:,0],self.post[:,:,0]))
        self.assertTrue(np.array_equal(updated[:,:,2],self.post[:,:,2]))
        self.assertTrue(np.array_equal(updated[0],self.post[0]))
        self.assertGreater(float(updated[1,0,1]),float(self.post[1,0,1]))

    def test_release_not_alias_of_official_s707_offset(self):
        heads=acoustic_other_target_heads(
            "release",self.post,self.native,self.ctx,self.k)
        revised=apply_one_target(self.post,"release",heads,self.eligible,
                                 np.asarray([63,1,3,5],np.uint8))
        self.assertTrue(np.array_equal(revised[:,:,:2],self.post[:,:,:2]))
        with self.assertRaises(ValueError):
            acoustic_other_target_heads(
                "offset",self.post,self.native,self.ctx,self.k)
        with self.assertRaises(NotImplementedError):
            ensure_valid_adapters("offset")
        with self.assertRaises(NotImplementedError):
            ensure_valid_adapters("pitch_identity")

    def test_no_future_fragment_dependency(self):
        head1=acoustic_other_target_heads(
            "release",self.post,self.native,self.ctx,self.k)
        changed=self.post.copy();changed[-1]=.9
        head2=acoustic_other_target_heads(
            "release",changed,self.native,self.ctx,self.k)
        self.assertTrue(np.array_equal(head1[:-1],head2[:-1]))

    def test_sequential_targets_only(self):
        self.assertTrue(require_sequential_training("onset",set()))
        with self.assertRaises(RuntimeError):
            require_sequential_training("release",{"onset"})
        self.assertTrue(require_sequential_training(
            "release",{"onset","sustain"}))

    def test_six_scores_all_must_exceed_eighty_not_just_onset(self):
        good={k:.81 for k in METRICS}
        state=StageState("onset","a"*64,True,good,
                         {k:.75 for k in METRICS},True,True)
        self.assertTrue(state.target_achieved())
        regressed=dict(good, offset_accuracy=.72)
        self.assertFalse(StageState("onset","a"*64,True,regressed,
                     {k:.75 for k in METRICS},True,True).target_achieved())
        misses=dict(good, offset_accuracy=.79)
        self.assertFalse(StageState("onset","a"*64,True,misses,
                     {k:.75 for k in METRICS},True,True).target_achieved())
        self.assertFalse(StageState("onset","a"*64,True,good,
                     {k:.75 for k in METRICS},False,True).eligible())

if __name__=="__main__":
    unittest.main()
