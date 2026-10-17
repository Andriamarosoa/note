"""S648 harmonic/temporal novelty and consistent S641 retrain tests."""
import unittest
from collections import deque
import numpy as np

from scripts.s648_harmonic_temporal_residual import (
    WIDTH,TEMPORAL_COLUMNS,PATTERNS,novelty_features,historical,
    matrices,build_model,select_on_train
)
from scripts.s640_learned_overlap_rescue import FEATURE_NAMES
from scripts.s641_fundamental_consistency import ADDITIONAL_NAMES


class HarmonicResidualTests(unittest.TestCase):
    def test_shapes_finite_and_no_future(self):
        a=np.ones((49,126),np.float32)
        b=np.zeros_like(a)
        native=np.zeros((49,3),np.float32)
        n=novelty_features(a,b,b,native,3)
        self.assertEqual(len(n),len(TEMPORAL_COLUMNS))
        self.assertEqual(len(n),WIDTH)
        self.assertTrue(np.isfinite(n).all())
        with self.assertRaises(ValueError):
            novelty_features(a,b,np.zeros((49,124)),native,3)
        with self.assertRaises(ValueError):
            novelty_features(a,b,b,native,37)
        fs=matrices(np.zeros(len(FEATURE_NAMES),np.float32),
                    np.zeros(len(ADDITIONAL_NAMES),np.float32),n)
        self.assertEqual(set(fs),set(PATTERNS))
        self.assertEqual(fs["control41"].shape,(len(FEATURE_NAMES),))
        self.assertEqual(fs["competitive65"].shape,
                         (len(FEATURE_NAMES)+len(ADDITIONAL_NAMES),))
        self.assertEqual(fs["temporal41"].shape,(len(FEATURE_NAMES)+WIDTH,))
        self.assertEqual(fs["temporal65"].shape,
                         (len(FEATURE_NAMES)+len(ADDITIONAL_NAMES)+WIDTH,))

    def test_native_independent_fundamental_attack_not_octave_alias(self):
        cur=np.zeros((49,126),np.float32)
        early=cur.copy()
        mid=cur.copy()
        pitch=4
        cur[pitch,[0,8,16]]=[3,3,3]
        cur[pitch+12,[0,8,16]]=[.5,.5,.5]
        native=np.zeros((49,3),np.float32)
        features=novelty_features(cur,mid,early,native,pitch)
        # 6 sources x 2 + 6 contrasts per window = 18 positions.
        for j in range(3):
            self.assertAlmostEqual(float(features[j*18]),3.,places=5)
            self.assertAlmostEqual(float(features[j*18+12]),2.5,places=5)
        # Previous-frame history cannot mutate once generated.
        history=deque(maxlen=16)
        history.append(mid.copy())
        old4,old12=historical(history,cur)
        expected=novelty_features(cur,old4,old12,native,pitch)
        history.append(np.full_like(cur,77.))
        np.testing.assert_array_equal(
            expected,novelty_features(cur,old4,old12,native,pitch))

    def test_train_only_composition_folds_and_abstention(self):
        rng=np.random.default_rng(648)
        records=[]
        prototype=matrices(np.zeros(len(FEATURE_NAMES),np.float32),
                           np.zeros(len(ADDITIONAL_NAMES),np.float32),
                           np.zeros(WIDTH,np.float32))
        for j in range(9):
            n=110
            y=np.zeros(n,np.int8)
            y[:8]=1
            mats={}
            for name,v in prototype.items():
                m=rng.normal(size=(n,len(v))).astype(np.float32)
                m[:8,0]+=2
                mats[name]=m
            records.append(dict(composition=f"comp{j}",y=y,matrices=mats))
        fits,diagnostics,winner=select_on_train(records)
        self.assertIn(winner,PATTERNS)
        for name in PATTERNS:
            self.assertIn(diagnostics[name]["selected"]["threshold"],
                          (.70,.80,.85,.90,.92,.95,.97,.99,1.01))
            self.assertEqual(diagnostics[name]["TRAIN_candidates"],990)
            self.assertEqual(diagnostics[name]["TRAIN_positives"],72)
            self.assertTrue(hasattr(fits[name]["model"],"predict_proba"))


if __name__=="__main__":
    unittest.main()
