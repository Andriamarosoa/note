"""S735 static acoustic-mask contract. No GuitarSet benchmark inferred."""
import unittest
import numpy as np
from scripts.s735_audit_per_candidate_combinations import (
    BASE,DYNAMIC,NAMES,COMBOS,masks,per_row_combinations,assemble_modified)

class ComboAuditTests(unittest.TestCase):
    def test_32_unique_H0_fallback_subsets(self):
        m=masks()
        self.assertEqual(m.shape,(32,6))
        self.assertTrue(np.all(m[:,0]==1))
        self.assertTrue(np.all(m[0,1:]==0))
        self.assertTrue(np.all(m[-1,:]==1))
        self.assertEqual(len({tuple(r) for r in m}),32)
        self.assertEqual(len(NAMES),33)

    def test_each_row_selects_different_mask(self):
        n=3
        p=np.array([[.10,.88,.05,.05,.05,.05],
                    [.10,.05,.89,.05,.05,.05],
                    [.10,.05,.05,.05,.05,.91]],np.float32)
        w=np.full((n,6),.05,np.float32)
        w[np.arange(n),[1,2,5]]=.75
        w/=w.sum(1,keepdims=True)
        pooled,potential,winner=per_row_combinations(p,w)
        self.assertEqual(pooled.shape,(3,32))
        self.assertTrue(np.allclose(pooled[:,0],p[:,0]))
        self.assertEqual(tuple(potential.shape),(3,32))
        self.assertEqual(len(winner),3)
        self.assertGreater(len(set(winner.tolist())),1)

    def test_midi_boost_never_changes_K_sustain_release(self):
        post=np.full((3,49,3),.01,np.float32)
        post[:,:,1]=.65
        post[:,:,2]=.1
        frame=np.array([0,1,2])
        pitch=np.array([0,12,24])
        heads=np.zeros((3,6),np.float32)
        heads[:,0]=.1
        pooled=np.full((3,32),.65,np.float32)
        potential=np.full((3,32),.8,np.float32)
        ctx=np.zeros((3,49,16),np.float32)
        ctx[:,:,1]=.7
        ctx[:,:,2]=.1
        ctx[:,:,4]=.15
        changed,stats=assemble_modified(post,frame,pitch,heads,
                       pooled,potential,np.array([1,2,4]),ctx,DYNAMIC)
        self.assertEqual(stats["boosted"],3)
        self.assertEqual(stats["unique_masks"],3)
        self.assertTrue(np.array_equal(post[:,:,1:],changed[:,:,1:]))
        self.assertTrue(np.all(changed[frame,pitch,0]>=.78))

if __name__=="__main__":unittest.main()
