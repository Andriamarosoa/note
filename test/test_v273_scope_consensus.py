"""Fixed decoding contract tests for S56 on synthetic decisions."""
import unittest
import numpy as np
from scripts.audit_v273_scope_consensus import joint_consensus


class TestS56Consensus(unittest.TestCase):
    def setUp(self):
        self.b=np.array([0,1,3,5])
        self.s54=np.array([2,0,4,1])
        self.p=np.zeros((4,36),int)
        self.p[0,:34]=1
        self.p[0,34:]=0
        self.p[1,:30]=0
        self.p[1,30:]=1
        self.p[2,:]=2
        self.p[3,:]=5

    def test_target_consensus_and_excluded_baseline(self):
        out=joint_consensus(self.p,self.b,self.s54,32,False)
        np.testing.assert_array_equal(out,[1,1,3,5])

    def test_specialist_k2_only_when_consensus_keeps(self):
        out=joint_consensus(self.p,self.b,self.s54,36,True)
        np.testing.assert_array_equal(out,[2,1,3,5])

    def test_no_change_without_consensus(self):
        out=joint_consensus(self.p,self.b,self.s54,36,False)
        np.testing.assert_array_equal(out,self.b)

    def test_invalid_threshold_refused(self):
        with self.assertRaises(Exception):
            joint_consensus(self.p,self.b,self.s54,20,False)


if __name__=='__main__':
    unittest.main()
