"""S641 tests: inverse octave labels are not evidence; native f0 check is causal."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP,load_heads
from causal_note.three_view_pitch import remap_pitch_probabilities
from scripts.live_strict_notes import DEFAULT_MODEL
from scripts.s641_fundamental_consistency import (
    original_fundamental_evidence,extended_features,
    FEATURE_NAMES,ADDITIONAL_NAMES,ALL_NAMES,
    train_only_group_threshold,iter_acoustic
)


class FundamentalConsistencyTests(unittest.TestCase):
    def test_pitch_up_and_inverse_remap_is_identity_only_on_valid_bins(self):
        p=np.zeros((49,3),np.float32)
        p[12,0]=.92 # pitched +12
        remapped=remap_pitch_probabilities(p,12)
        self.assertEqual(float(remapped[0,0]),float(p[12,0]))
        self.assertEqual(float(remapped[12,0]),0.)
        self.assertEqual(float(remapped[0,1]),0.)
        # Mapping alone cannot confirm physical presence of any note.

    def test_known_independent_fundamental_vs_upper_harmonic(self):
        x=np.zeros((49,126),np.float32)
        i=8
        for j in range(3):
            x[i,8*j]=3.2
            x[i,8*j+1]=1.1
            x[i,8*j+3]=.6
            x[i+12,8*j]=1.3
            x[i-1,8*j]=.4
            x[i+1,8*j]=.5
            x[i,57+j]=.8
            x[i+12,57+j]=.2
            x[i,24+8*j]=.8
            x[i+12,24+8*j]=.3
        v=original_fundamental_evidence(x,i)
        self.assertEqual(v.shape,(24,))
        self.assertEqual(len(ALL_NAMES),len(FEATURE_NAMES)+24)
        np.testing.assert_allclose(v[0::8],[2.1]*3,rtol=1e-5)
        np.testing.assert_allclose(v[2::8],[2.7]*3,rtol=1e-5)
        np.testing.assert_allclose(v[6::8],[.6]*3,rtol=1e-5)
        p=np.zeros((49,3),np.float32)
        k=np.zeros(7,np.float32);k[2]=1
        e=extended_features(x,p,p,k,i)
        self.assertEqual(e.shape,(len(ALL_NAMES),))

    def test_physical_fundamental_support_resists_missing_f0(self):
        x=np.zeros((49,126),np.float32)
        for j in range(3):
            x[2,8*j]=.2
            x[2,8*j+1]=2.3
            x[2+12,8*j]=2.3
            x[2-1,8*j]=1.
            x[2+1,8*j]=1.
        evidence=original_fundamental_evidence(x,2)
        self.assertTrue((evidence[0::8]<0).all())
        self.assertTrue((evidence[2::8]<0).all())
        self.assertTrue((evidence[3::8]<0).all())

    def test_train_only_composition_group_cv(self):
        rng=np.random.default_rng(641)
        x=rng.normal(size=(90,5)).astype(np.float32)
        groups=np.repeat(np.array([f"G{i}" for i in range(9)],dtype=object),10)
        y=np.zeros(90,np.int8)
        y[::6]=1
        cutoff,meta=train_only_group_threshold(x,y,groups)
        self.assertIn(cutoff,(*(.70,.80,.85,.90,.92,.95,.97,.99),1.01))
        self.assertEqual(len(meta["folds"]),5)
        for fold in meta["folds"]:
            self.assertFalse(set(fold["training_compositions"])&
                             set(fold["validation_compositions"]))

    def test_last_future_perturbation_does_not_modify_prior_native_features(self):
        heads=load_heads(DEFAULT_MODEL)
        t=np.arange(77*HOP)/SR
        pcm=np.asarray(.16*np.sin(2*np.pi*164.8*t),np.float32)
        second=pcm.copy();second[71*HOP:]*=.01
        a=list(iter_acoustic(pcm,heads,batch_frames=19))
        b=list(iter_acoustic(second,heads,batch_frames=23))
        self.assertEqual(len(a),77)
        for i in (0,30,63,70):
            self.assertEqual(a[i][0],b[i][0])
            evidence_a=original_fundamental_evidence(a[i][1],5)
            evidence_b=original_fundamental_evidence(b[i][1],5)
            np.testing.assert_allclose(evidence_a,evidence_b,atol=1e-7,rtol=1e-6)


if __name__=="__main__":
    unittest.main()
