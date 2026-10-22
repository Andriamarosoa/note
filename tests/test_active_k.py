"""S828 private Active-K only; no other characteristic modules."""
import unittest
import numpy as np
from active_exact_k.score import active_score,original_count,targets_at_ends,totals
from active_exact_k.readout import proposed,MODES,SHIFTS
from active_exact_k.acoustics import HOP
class PrivateActiveK(unittest.TestCase):
    def test_gt_active_only(self):
        p=np.array([5,2,1,0],np.uint8)
        y=np.array([0,2,2,0],np.uint8)
        self.assertEqual(active_score(p,y),dict(correct=1,frames=2,undercount=1,overcount=0))
    def test_causal256_and_only_own_axis(self):
        self.assertEqual(HOP,256)
        import active_exact_k.stream as x
        self.assertFalse(any(hasattr(x,k) for k in ("OnsetDecoder","FullNoteDecoder","ExactKDecoder","OffsetDecoder","FramePitchDecoder")))
    def test_decoder_rejects_legacy_1024_grid(self):
        from active_exact_k.stream import PrivateS708GuidedActiveK
        with self.assertRaisesRegex(ValueError, "256-sample"):
            PrivateS708GuidedActiveK(object(),decision_hop=1024)
        decoder=PrivateS708GuidedActiveK(object(),decision_hop=256)
        self.assertEqual(decoder.decision_hop,256)
        with self.assertRaisesRegex(ValueError, "256-sample"):
            decoder.step(np.zeros((49,3),np.float32),np.ones(7,np.float32)/7,1024)
    def test_private_proposals(self):
        probs=np.full((6,7),.01,np.float32);probs[:,2]=.94
        old=np.ones(6,np.uint8)
        self.assertTrue(np.all(proposed(probs,old,"learned_private",0)==2))
        self.assertTrue(np.all(proposed(probs,old,"s708_private",0)==1))
if __name__=="__main__":unittest.main()
