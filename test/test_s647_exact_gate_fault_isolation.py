"""S647 exact high-q gate decomposition and no lookahead."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,Selection
from scripts.s647_exact_gate_fault_isolation import diagnose


class IsolationTests(unittest.TestCase):
    def test_separate_physically_incompatible_and_active_events(self):
        n=20
        p=np.zeros((n,49,3),np.float32)
        p[:,5,0]=.01;p[:,5,1]=.9
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
        row=dict(ends=np.arange(1,n+1)*HOP,
                 p=p,k=k,truth=[dict(pitch=45,time=HOP/44100.,duration=.07)])
        q=np.zeros((n,49),np.float32);q[:3,5]=.999
        a=diagnose(row,q,Selection())
        self.assertGreater(a["rising_q"],0)
        self.assertEqual(a["rising_q_onset_above_002"],0)
        self.assertEqual(a["rising_q_inactive_and_both"],0)
        self.assertEqual(a["rising_q_active"],0)
        self.assertGreater(a["rising_q_inactive_low_onset"],0)
    def test_future_mutation_preserves_prefix(self):
        n=20
        p=np.zeros((n,49,3),np.float32)
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7)).copy()
        row=dict(ends=np.arange(1,n+1)*HOP,
                 p=p,k=k,truth=[])
        q=np.zeros((n,49),np.float32);q[:4,8]=.999
        a=diagnose(row,q,Selection())
        q[11:]=.999
        b=diagnose(row,q,Selection())
        self.assertGreaterEqual(b["rising_q"],a["rising_q"])
        self.assertEqual(a["rising_q_near_true_attack"],0)
if __name__=="__main__":unittest.main()
