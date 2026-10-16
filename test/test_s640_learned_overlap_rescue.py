"""S640 leakage, feature ABI and causal selection tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP,Selection,CausalSelection,load_heads
from scripts.live_strict_notes import DEFAULT_MODEL
from scripts.s640_learned_overlap_rescue import (
    FEATURE_NAMES,feature_vector,get_candidates,relay,iter_acoustic,
    FROZEN_GATES,PRIMARY_GATE
)


class LearnedOverlapSelectorTests(unittest.TestCase):
    def test_expected_causal_feature_width_and_identity(self):
        x=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        b=np.zeros_like(p)
        k=np.zeros(7,np.float32);k[2]=1
        b[12,:2]=[.91,.8]
        a=get_candidates(x,p,k,b,True,.6)
        self.assertEqual(list(a),[12])
        self.assertEqual(a[12].shape,(len(FEATURE_NAMES),))
        self.assertFalse(get_candidates(x,p,k,b,False,.6))
        self.assertEqual(FROZEN_GATES,(.35,.50,.70,.85))
        self.assertEqual(PRIMARY_GATE,.70)

    def test_strict_gating_does_not_mutate_original_and_preserves_K(self):
        n=np.zeros((49,3),np.float32)
        b=np.zeros_like(n);b[12,:2]=[.9,.85]
        state=CausalSelection(Selection(onset=.6))
        rejected,num=relay(n,b,state,{12:.65},.7)
        self.assertEqual(num,0)
        np.testing.assert_array_equal(rejected,n)
        accepted,num=relay(n,b,state,{12:.75},.7)
        self.assertEqual(num,1)
        self.assertAlmostEqual(float(accepted[12,0]),.6,places=6)
        np.testing.assert_array_equal(n,np.zeros_like(n))
        state.active[12]=True
        no_double,num=relay(n,b,state,{12:.99},.7)
        self.assertEqual(num,0)
        state.active[12]=False
        state.armed[12]=False
        no_arm,num=relay(n,b,state,{12:.99},.7)
        self.assertEqual(num,0)

    def test_physical_source_future_cannot_change_past_features(self):
        model=load_heads(DEFAULT_MODEL)
        t=np.arange(75*HOP)/SR
        source=np.asarray(.18*np.sin(2*np.pi*98*t),np.float32)
        change=source.copy()
        change[70*HOP:]=0
        rows=list(iter_acoustic(source,model,batch_frames=24))
        changed=list(iter_acoustic(change,model,batch_frames=24))
        self.assertEqual(len(rows),75)
        self.assertEqual(len(changed),75)
        for idx in (0,16,63,69):
            self.assertEqual(rows[idx][0],changed[idx][0])
            for j in (1,2,3,4):
                np.testing.assert_array_equal(rows[idx][j],changed[idx][j])
        self.assertFalse(rows[62][5])
        self.assertTrue(rows[63][5])


if __name__=="__main__":
    unittest.main()
