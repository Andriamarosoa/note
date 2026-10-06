import unittest
import numpy as np
from scripts.summarize_v273_symmetric_pitch_tta import decision, metrics

class SymmetricPitchTTATests(unittest.TestCase):
    def row(self,true_k=2,p2=.7,p3=.2):
        steps=np.arange(-6,7)
        p=np.zeros((13,7),float); p[:,2]=p2; p[:,3]=p3; p[:,0]=1-p2-p3
        return {"steps":steps,"p":p,"pred":p.argmax(1),"true_k":true_k}
    def test_mean_margin(self):
        self.assertTrue(decision(self.row(),"mean_margin",3))
    def test_lcb_margin(self):
        self.assertTrue(decision(self.row(),"lcb_margin",6))
    def test_accounting(self):
        q=metrics([self.row(2),self.row(3)],"mean_margin",2)
        self.assertEqual(q["corrections"],1); self.assertEqual(q["regressions"],1)
        self.assertEqual(q["net"],0)

if __name__=="__main__": unittest.main()
