"""Two 69D native256 isolated training resampling ablations, issue #144."""
import unittest
from collections import Counter
import numpy as np
from onset256.acoustics import HOP,SR
from scripts.train_onset_fp_track_sampling256 import (
    MODES,samples_train_only,fit_neural,REAL_FROZEN_69D,
    REFERENCE_FOUR)
from scripts.replay_onset_fp_track_sampling256 import (
    FROZEN_HISTORICAL_69_F1,FROZEN_HISTORICAL_69_FP)

class TrackSamplingControls(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(144)
        self.rows=[]
        for track in range(48):
            n=150 if track==0 else 90
            for j in range(n):
                x=rng.standard_normal(74).astype(np.float32)
                y=int(j%3!=0)
                self.rows.append(dict(X=x,y=y,track=f"train_{track:02d}",
                                      comp=("BN1","Funk2","Jazz1")[track%3]))
        self.train_indices=list(range(len(self.rows)))
        for i in range(11):
            self.rows.append(dict(
                X=np.full(74,900.+i,np.float32),
                y=1,track="outer_00",comp="Rock3"))
    def test_shared_frozen_historical_reference_and_256(self):
        self.assertEqual(HOP,256)
        self.assertEqual(SR,44100)
        self.assertEqual(REAL_FROZEN_69D,dict(tp=8127,fp=4519,fn=3140))
        self.assertEqual(REFERENCE_FOUR,dict(tp=8160,fp=4605,fn=3107))
        self.assertEqual(FROZEN_HISTORICAL_69_FP,4519)
        self.assertAlmostEqual(FROZEN_HISTORICAL_69_F1,.6797139631162966)
        self.assertEqual(MODES,("track_uniform","track_class"))
    def test_track_uniform_is_deterministic_and_never_samples_outer(self):
        i,stats=samples_train_only(self.rows,self.train_indices,"track_uniform",144)
        j,_=samples_train_only(self.rows,self.train_indices,"track_uniform",144)
        np.testing.assert_array_equal(i,j)
        self.assertEqual(len(i),len(self.train_indices))
        self.assertEqual(stats["original_unique_tracks"],48)
        self.assertTrue(stats["unique_examples_reused"]<len(i))
        self.assertTrue(all(self.rows[k]["comp"]!="Rock3" for k in i))
        counts=Counter(self.rows[k]["track"] for k in i)
        self.assertTrue(all(counts[f"train_{k:02d}"]>30 for k in range(48)))
    def test_track_class_sample_has_even_true_false_but_preserves_train_size(self):
        sel,stat=samples_train_only(self.rows,self.train_indices,"track_class",144)
        labs=Counter(self.rows[i]["y"] for i in sel)
        self.assertEqual(sum(labs.values()),len(self.train_indices))
        self.assertLessEqual(abs(labs[0]-labs[1]),1)
        self.assertEqual(stat["mode"],"track_class")
        self.assertTrue(all(i<len(self.train_indices) for i in sel))
    def test_fits_actual_mlp69_weights_not_boolean_mask(self):
        rng=np.random.default_rng(143)
        X=[]
        for k in range(240):
            vec=rng.standard_normal(74).astype(np.float32)
            X.append(dict(X=vec,y=int(vec[0]+.5*vec[4]>0),
                          track=f"train_{k%48:02d}",comp="BN1"))
        model=fit_neural(X,list(range(240)),144)
        self.assertEqual(model.n_features_in_,69)
        self.assertEqual(model.predict_proba(
            np.zeros((49,69),np.float32)).shape,(49,2))
    def test_reject_incompatible_unregistered_sampling(self):
        with self.assertRaises(ValueError):
            samples_train_only(self.rows,self.train_indices,"adaptive_veto",144)
if __name__=="__main__":unittest.main()
