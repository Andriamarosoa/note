import unittest
import numpy as np
from causal_note.strict_live import SR
from scripts.s776_k5k6_count_rootcauses import truth_count,confusion,measurements

class S776KCountTests(unittest.TestCase):
    def test_truth_counts_and_cap_k6(self):
        clock=np.array([256,512,768],np.int64)
        truth=[dict(time=0.,duration=512/SR,pitch=40+i) for i in range(7)]
        self.assertEqual(truth_count(clock,truth).tolist(),[6,0,0])
    def test_k5_k6_underestimation_preserved(self):
        target=np.array([5,6,6,4,0],np.int8)
        predicted=np.array([4,4,6,5,0],np.int8)
        mat=confusion(target,predicted)
        self.assertEqual(int(mat[5,4]),1)
        self.assertEqual(int(mat[6,4]),1)
        self.assertEqual(int(mat[6,6]),1)
        v=measurements(mat,None,target)
        self.assertEqual(v["per_k"][6]["n"],2)
        self.assertEqual(v["per_k"][6]["correct"],1)
        self.assertEqual(v["per_k"][6]["under"],1)
    def test_invalid_k_fails(self):
        with self.assertRaises(ValueError):
            confusion([7],[6])
if __name__=="__main__":
    unittest.main()
