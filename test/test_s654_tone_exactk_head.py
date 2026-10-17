"""S654 Tone K multi-head regression and exact original posterior contract tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR
from scripts.s654_tone_exatk_head import (
    physical_rows,metrics,models,MODEL_KINDS,project,train_cv,TARGET_K
)

class ToneCountTests(unittest.TestCase):
    def test_annotated_target_K_and_causal_posterior_features(self):
        ends=np.arange(1,200)*256
        a=np.zeros((199,7),np.float32)
        b=np.zeros_like(a);c=np.zeros_like(a)
        a[:,1]=1.;b[:,2]=1.;c[:,3]=1.
        truths=[
          dict(time=.30,pitch=40,duration=.30),
          dict(time=.31,pitch=52,duration=.30),
          dict(time=.32,pitch=59,duration=.30),
          dict(time=.56,pitch=64,duration=.20)]
        rows=physical_rows(ends,a,b,c,truths)
        self.assertEqual(rows["features"].shape,(4,54))
        self.assertEqual(len(rows["truth"]),4)
        self.assertEqual(list(rows["truth"])[:3],[3,3,3])
        self.assertEqual(int(rows["original_pred"][0]),1)
        self.assertEqual(project(rows["features"],"linear_simple").shape,(4,21))
        self.assertEqual(project(rows["features"],"linear").shape,(4,54))

    def test_no_fake_90_accuracy_from_unbalanced_k(self):
        y=np.array([1,1,2,3,4,5,6])
        p=np.array([1,1,1,1,1,1,1])
        d=metrics(y,p)
        self.assertEqual(d["n"],7)
        self.assertEqual(d["correct"],2)
        self.assertAlmostEqual(d["accuracy"],2/7)
        self.assertEqual(d["per_true_K"]["6"]["correct"],0)

    def test_training_only_groups_leave_compositions_out(self):
        rng=np.random.default_rng(654)
        rows=[]
        for comp in range(9):
            n=130
            y=np.resize(np.array([1,2,3,4,5,6],np.int8),n)
            x=rng.normal(size=(n,54)).astype(np.float32)
            x[:,0:7]=.02
            for i,k in enumerate(y):x[i,int(k)]=.95
            rows.append(dict(composition=f"C{comp}",features=x,
                             truth=y,original_pred=np.ones(n,np.int8)))
        fit,choice,winner,cv=train_cv(rows)
        self.assertIn(winner,MODEL_KINDS)
        self.assertIn(choice,(*MODEL_KINDS,"S614_original_K"))
        self.assertEqual(len(cv["train_models"]),len(MODEL_KINDS))
        self.assertEqual(cv["train_original"]["n"],9*130)
        self.assertEqual(len(cv["train_models"][winner]["folds"]),5)

if __name__=="__main__":
    unittest.main()
