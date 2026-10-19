"""S781 forensic contracts; no CAL tuning or score promotion."""
import unittest
import numpy as np
from scripts.s781_k_tail_composition_audit import tail_histogram,by_composition

class S781ForensicTests(unittest.TestCase):
    def fixture(self):
        return [
            dict(composition="BN2",texture="comp",
                 gt=np.array([5,6,6,1],np.int8),baseline=np.array([4,4,5,1],np.int8),
                 trial=np.array([5,4,6,1],np.int8)),
            dict(composition="BN2",texture="solo",
                 gt=np.array([5,6,2],np.int8),baseline=np.array([4,3,2],np.int8),
                 trial=np.array([4,6,2],np.int8)),
            dict(composition="Jazz2",texture="comp",
                 gt=np.array([5,6,0],np.int8),baseline=np.array([3,3,0],np.int8),
                 trial=np.array([5,5,0],np.int8)),
            dict(composition="Jazz2",texture="solo",
                 gt=np.array([5,6,1],np.int8),baseline=np.array([5,2,1],np.int8),
                 trial=np.array([5,6,1],np.int8))
        ]

    def test_histogram_exact_k6_tail(self):
        rows=self.fixture()
        hist=tail_histogram(rows,"trial")
        self.assertEqual(sum(hist["6"]),5)
        self.assertEqual(hist["6"][6],3)
        self.assertEqual(hist["6"][4],1)
        self.assertEqual(hist["6"][5],1)
        self.assertEqual(sum(hist["5"]),4)

    def test_composition_and_texture_are_separate(self):
        d=by_composition(self.fixture(),"trial")
        self.assertEqual(set(d),{"BN2","Jazz2"})
        self.assertEqual(d["BN2"]["overall"]["per_k"][6]["n"],3)
        self.assertEqual(d["BN2"]["texture"]["comp"]["metrics"]["per_k"][6]["n"],2)
        self.assertEqual(d["BN2"]["texture"]["solo"]["metrics"]["per_k"][6]["n"],1)
        self.assertEqual(d["Jazz2"]["tail_histogram"]["6"][6],1)

    def test_no_shadow_prediction_mutation(self):
        rows=self.fixture()
        snapshot=[(r["gt"].copy(),r["trial"].copy()) for r in rows]
        by_composition(rows,"trial")
        for r,(gt,pred) in zip(rows,snapshot):
            np.testing.assert_array_equal(r["gt"],gt)
            np.testing.assert_array_equal(r["trial"],pred)

if __name__=="__main__":
    unittest.main()
