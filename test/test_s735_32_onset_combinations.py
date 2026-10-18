"""S735 synthetic ABI tests; these do not claim GuitarSet accuracy."""
import unittest
import numpy as np
from scripts.s735_audit_32_onset_combinations import (
    ALL,mask_posterior,dynamic_posterior)

class CombinationContract(unittest.TestCase):
    def setUp(self):
        self.base=np.zeros((3,49,3),np.float32)
        self.base[:,:,0]=.1
        self.base[:,:,1]=.9
        self.base[:,:,2]=.3
        self.heads=np.zeros((3,49,6),np.float32)
        self.heads[:,:,0]=.1
        self.heads[0,0,1]=.83
        self.heads[1,1,3]=.88
        self.heads[2,2,5]=.92
        self.eligible=np.zeros((3,49),bool)
        self.eligible[0,0]=self.eligible[1,1]=self.eligible[2,2]=True

    def test_all_32_combinations_exact_once(self):
        self.assertEqual(len(ALL),32)
        self.assertEqual(len(set(ALL)),32)
        self.assertTrue(all(x & 1 for x in ALL))
        self.assertEqual(ALL[0],1)
        self.assertEqual(ALL[-1],63)

    def test_fixed_subset_changes_only_target_and_available_head(self):
        subset=mask_posterior(self.base,self.heads,self.eligible,1|(1<<1))
        self.assertEqual(float(subset[0,0,0]),float(self.heads[0,0,1]))
        self.assertTrue(np.array_equal(subset[:,:,1:],self.base[:,:,1:]))
        self.assertEqual(float(subset[1,1,0]),float(self.base[1,1,0]))
        self.assertEqual(float(subset[2,2,0]),float(self.base[2,2,0]))

    def test_individual_pitch_may_use_individual_subset(self):
        chosen=np.ones((3,49),np.uint8)
        chosen[0,0]|=1<<1
        chosen[1,1]|=1<<3
        chosen[2,2]|=1<<5
        result=dynamic_posterior(self.base,self.heads,self.eligible,chosen)
        for t,p,h in ((0,0,1),(1,1,3),(2,2,5)):
            self.assertEqual(float(result[t,p,0]),float(self.heads[t,p,h]))
        self.assertTrue(np.array_equal(result[:,:,1:],self.base[:,:,1:]))
        untouched=~self.eligible
        self.assertTrue(np.array_equal(result[:,:,0][untouched],self.base[:,:,0][untouched]))

if __name__=="__main__":
    unittest.main()
