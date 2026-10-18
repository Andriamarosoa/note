"""S723 32D feature ABI: causal-only, low-edge mask, old model exact parity."""
import unittest
import numpy as np
import torch

from causal_note.s723_polyphonic_novelty import (
    BASE,EXTRA,TOTAL,causal_novelty_track,augmented_track,init_from_s708)
from causal_note.s705_temporal_heads import TemporalHeads


class S723FeaturesTest(unittest.TestCase):
    def test_dimension_and_zero_future_dependency(self):
        rng=np.random.default_rng(723)
        data=rng.normal(size=(19,49,288)).astype(np.float32)
        f=causal_novelty_track(data)
        self.assertEqual(f.shape,(19,49,32))
        self.assertEqual(augmented_track(data).shape,(19,49,TOTAL))
        after=data.copy()
        after[12:]*=100
        np.testing.assert_array_equal(causal_novelty_track(after)[:12],f[:12])
        np.testing.assert_array_equal(f[0,:,:24],np.zeros((49,24),np.float32))

    def test_pitch_local_transient_not_global_flux_broadcast(self):
        x=np.zeros((3,49,288),np.float32)
        x[1,12,192]=2.
        f=causal_novelty_track(x)
        self.assertEqual(float(f[1,12,0]),2.)
        self.assertEqual(float(f[1,13,0]),0.)
        self.assertEqual(float(f[2,12,0]),0.)
        self.assertEqual(float(f[1,12,28]),2./3.)
        self.assertEqual(float(f[1,12,31]),0.)

    def test_missing_lower_pitch_not_clamped_to_E2_false_comparison(self):
        x=np.zeros((1,49,288),np.float32)
        x[0,:,141]=np.arange(1,50)/50.
        f=causal_novelty_track(x)
        # MIDI40 E2 and all pitches <52 have no octave below our ABI.
        np.testing.assert_allclose(f[0,:12,24],x[0,:12,141])
        np.testing.assert_allclose(f[0,:19,25],x[0,:19,141])
        np.testing.assert_array_equal(f[0,:12,26],0.)
        np.testing.assert_array_equal(f[0,:19,27],0.)
        self.assertAlmostEqual(float(f[0,12,24]),float(x[0,12,141]-x[0,0,141]),places=7)
        self.assertAlmostEqual(float(f[0,19,25]),float(x[0,19,141]-x[0,0,141]),places=7)
        self.assertTrue(np.all(f[0,12:,26]==1))
        self.assertTrue(np.all(f[0,19:,27]==1))

    def test_initial_new320d_model_matches_original288d_all_three_heads(self):
        rng=np.random.default_rng(723)
        old=TemporalHeads(np.zeros(BASE,np.float32),np.ones(BASE,np.float32))
        new=init_from_s708(old,np.zeros(EXTRA,np.float32),np.ones(EXTRA,np.float32))
        x=np.asarray(rng.uniform(-1,1,(1,12,49,BASE)),np.float32)
        v=torch.ones((1,12))
        x_new=np.concatenate((x,rng.normal(size=(1,12,49,EXTRA)).astype(np.float32)),axis=-1)
        old.eval();new.eval()
        with torch.inference_mode():
            p0,k0=old(torch.from_numpy(x),v)
            p1,k1=new(torch.from_numpy(x_new),v)
        torch.testing.assert_close(p0,p1,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(k0,k1,atol=1e-5,rtol=1e-5)
        self.assertEqual(new.encode[0].weight.shape[1],320)
        self.assertTrue(torch.all(new.encode[0].weight[:,288:]==0))

    def test_invalid_audio_or_feature_contract_fails(self):
        with self.assertRaisesRegex(ValueError,"288"):
            causal_novelty_track(np.ones((1,49,320),np.float32))
        x=np.zeros((2,49,288),np.float32);x[1,22,99]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            causal_novelty_track(x)


if __name__=="__main__":
    unittest.main()
