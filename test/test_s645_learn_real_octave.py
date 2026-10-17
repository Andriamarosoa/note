"""S645 real-only octave discriminator group-CV and feature ABI tests."""
import unittest
import numpy as np

from scripts.s645_learn_real_octave import (
    small_indices,project,training_rows,oof_select_model,evaluate_source_set,
)


class NaturalOctaveHeadTests(unittest.TestCase):
    def test_compact_indices_from_only_present_spectral_features(self):
        x=np.arange(173,dtype=np.float32)[None,:]
        indices=small_indices()
        self.assertEqual(len(indices),len(set(indices)))
        self.assertTrue(all(0<=i<173 for i in indices))
        self.assertEqual(project(x,"logistic_harmonic_subset").shape,
                         (1,len(indices)))
        self.assertEqual(project(x,"logistic_173D").shape,(1,173))

    def test_train_only_group_cv_refuses_shared_compositions(self):
        rng=np.random.default_rng(645)
        records=[]
        for j in range(9):
            x=rng.normal(size=(30,173)).astype(np.float32)
            y=np.zeros(30,np.int8)
            y[:5]=1
            x[:,7]+=y*2
            records.append(dict(
                name=f"TRAIN_comp{j}",composition=f"comp{j}",
                rows=[dict(target=int(y[i]),feature=x[i])
                      for i in range(30)]))
        x,y,g=training_rows(records)
        self.assertEqual((len(x),int(y.sum())),(270,45))
        method,choices,folds=oof_select_model(x,y,g)
        self.assertIn(method,choices)
        self.assertEqual(len(folds),5)
        for f in folds:
            self.assertFalse(set(f["train_compositions"])&
                             set(f["held_compositions"]))
        for opt in choices.values():
            self.assertIn(opt["threshold"],
                          (.08,.12,.16,.20,.25,.30,.35,.40,.45,.50,
                           .55,.60,.65,.70,.75,.80,.85,.90,.95))

    def test_validation_has_no_annotation_as_classifier_input(self):
        class AlwaysNo:
            def predict_proba(self,x):
                return np.stack((np.ones(len(x)),np.zeros(len(x))),axis=-1)
        row1=dict(low_pitch=40,upper_pitch=52,true_chord_K=1,
                  at_s=1.,target=0,feature=np.zeros(173,np.float32))
        row2=dict(low_pitch=49,upper_pitch=61,true_chord_K=3,
                  at_s=2.,target=1,feature=np.ones(173,np.float32))
        records=[dict(name="GuitarSetDEV",composition="held",
                      rows=[row1,row2])]
        m=evaluate_source_set(records,AlwaysNo(),"logistic_173D",.5)
        self.assertEqual(m["overall"]["FN"],1)
        self.assertEqual(m["overall"]["TN"],1)
        self.assertEqual(m["overall"]["FP"],0)
        self.assertEqual(m["by_K"]["onset_K3plus"]["positives"],1)

if __name__=="__main__":
    unittest.main()
