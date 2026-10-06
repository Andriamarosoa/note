import unittest
import numpy as np
from scripts.audit_v273_symmetric_h2_reliability_gate import feats,account

class SymmetricH2GateTests(unittest.TestCase):
    def test_features(self):
        steps=np.arange(-6,7)
        p=np.zeros((13,7),float); p[:,2]=.6; p[:,3]=.3; p[:,0]=.1
        pr=p.argmax(1)
        x,n=feats(steps,p,pr)
        self.assertEqual(x.shape,(15,))
        self.assertGreater(n["mean_margin"],0)
    def test_account(self):
        rows=[{"true_k":2},{"true_k":3}]
        q=account(rows,[True,False])
        self.assertEqual(q["net"],1)

if __name__=="__main__": unittest.main()
