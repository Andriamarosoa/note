"""S694 frozen-K gate and native MIDI event noninterference proof."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s694_extra_note_quality_K2 import (
    accepted_by_quality_and_K,accepted_merge,POLICIES
)
def e(kind,pitch,frame):
    return dict(type=kind,pitch=pitch,decision_sample=frame*HOP,time=frame*HOP/SR)
class S694Tests(unittest.TestCase):
    def item(self):
        native=[e("note_on",40,5),e("note_off",40,30)]
        extra=[(e("note_on",52,10),e("note_off",52,25))]
        k=np.zeros((50,7),np.float32);k[:,1]=1.
        return dict(ends=np.arange(1,51)*HOP,k=k,
                    proposals=extra,original=native)
    def test_refuse_extra_when_native_count_says_K1(self):
        item=self.item()
        verdict=accepted_by_quality_and_K(item,np.asarray([.99]),
            "gate_0.55_k2_0.10")
        self.assertFalse(verdict.any())
        merged=accepted_merge(item["original"],item["proposals"],
                              verdict.astype(np.float32),.5)
        self.assertEqual(merged,item["original"])
    def test_use_predicted_K2_not_labels(self):
        item=self.item()
        item["k"][:,1]=.1;item["k"][:,2]=.9
        verdict=accepted_by_quality_and_K(item,np.asarray([.99]),
            "gate_0.65_k2_0.40")
        self.assertTrue(verdict[0])
        events=accepted_merge(item["original"],item["proposals"],
                              verdict.astype(np.float32),.5)
        self.assertEqual(sum(e["type"]=="note_on" for e in events),2)
        self.assertEqual([e for e in events if e["pitch"]==40],item["original"])
    def test_higher_K_rejects_weak_predicted_quality(self):
        item=self.item()
        item["k"][:,1]=.1;item["k"][:,2]=.9
        verdict=accepted_by_quality_and_K(item,np.asarray([.54]),
            "gate_0.55_k2_0.10")
        self.assertFalse(verdict.any())
    def test_no_future_decisions(self):
        item=self.item()
        item["k"][:,1]=0;item["k"][:,2]=1
        before=accepted_by_quality_and_K(item,np.asarray([.65]),
             "gate_0.55_k2_0.25")
        item["k"][20:]=0
        after=accepted_by_quality_and_K(item,np.asarray([.65]),
             "gate_0.55_k2_0.25")
        np.testing.assert_array_equal(before,after)
    def test_policies_frozen(self):
        self.assertIn("gate_0.72_k2_0.40",POLICIES)
if __name__=="__main__":unittest.main()
