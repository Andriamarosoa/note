"""S741 FIT K-only neural selection: independent trainable K0-K6 contract."""
import unittest
import numpy as np
from scripts.s741_train_exactk_fragments import tensors,train_one


class KTrainingUnitTests(unittest.TestCase):
    def test_k0_to_k6_legal_and_fit_loop(self):
        n=5
        k=np.tile(np.array([.1,.6,.3,0,0,0,0],np.float32),(n,1))
        item=dict(features=np.zeros((n,37),np.float32),
                  descriptors=np.zeros((n,32,12),np.float32),
                  k=k,target=np.array([0,1,0,2,0],np.int64),
                  risk=np.zeros((n,32,2),np.float32))
        audit=np.zeros((32,7,3),np.float32)
        features,descriptors,stats,kk,legal,targets,risk=tensors(
            item,audit,np.zeros(37),np.ones(37))
        self.assertEqual(tuple(legal.shape),(n,32,7))
        self.assertTrue(bool(legal.all()))
        self.assertEqual(tuple(stats.shape),(n,32,7,3))
        rounds=[]
        model=train_one(741,[item,item],[audit,audit],
                        np.zeros(37),np.ones(37),
                        lambda mdl,seed,epoch,loss: rounds.append(
                            (seed,epoch,loss)))
        self.assertEqual(len(rounds),3)
        self.assertTrue(all(np.isfinite(row[2]) for row in rounds))
        self.assertEqual(rounds[-1][1],6)


if __name__=="__main__":
    unittest.main()
