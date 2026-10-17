"""S677 predictable lead-target labels, no physical event backdating."""
import unittest
import numpy as np
from causal_note.strict_live import HOP
from scripts.s677_batch_aware_hazard import (
    lead_target,TRAIN_OFFSET_LEAD_HOPS,choose_offset
)
class BatchAwareLabelTests(unittest.TestCase):
    def test_training_target_only_two_hops_early(self):
        self.assertEqual(TRAIN_OFFSET_LEAD_HOPS,2)
        positions=np.arange(8,17)
        np.testing.assert_array_equal(lead_target(positions,12),
                                      (positions>=10).astype(np.uint8))
    def test_prediction_only_after_native_input(self):
        ends=np.arange(1,100)*HOP
        data=dict(ends=ends,notes=[dict(start=3,pitch=40,
                                         native_end=90,next_start=97)])
        frames=np.arange(10,90)
        p=np.ones(len(frames),np.float32)
        for batch in (1,4):
            v=choose_offset(data,0,frames,p,.6,"hazard",batch_hops=batch,persistence=1)
            self.assertGreaterEqual(v,11*HOP/44100)
if __name__=="__main__":unittest.main()
