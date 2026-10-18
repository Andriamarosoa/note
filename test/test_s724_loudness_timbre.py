"""S724 auditory hypothesis regression fixtures, not claims of real-mic accuracy."""
import unittest
import numpy as np
import torch
from causal_note.s724_loudness_timbre import harmonic_timbre_features,augmented_track
from causal_note.s723_polyphonic_novelty import init_from_s708
from causal_note.s705_temporal_heads import TemporalHeads


class S724LoudnessTimbreTest(unittest.TestCase):
    def test_abi_causality_and_per_pitch_audio_only(self):
        rng=np.random.default_rng(724)
        x=rng.uniform(0,3,(17,49,288)).astype(np.float32)
        a=harmonic_timbre_features(x)
        self.assertEqual(a.shape,(17,49,32))
        self.assertEqual(augmented_track(x).shape,(17,49,320))
        x[12:]*=5.
        np.testing.assert_array_equal(a[:12],harmonic_timbre_features(x)[:12])
        np.testing.assert_array_equal(a[0,:,:8],np.zeros((49,8),np.float32))
        self.assertTrue(np.all(a[:,:12,30]==0.))
        self.assertTrue(np.all(a[:,:19,31]==0.))

    def test_distinguishes_new_E3_harmonics_from_louder_E2_only(self):
        base=np.zeros((3,49,288),np.float32)
        # E2 parent (MIDI40 index 0) has eight partials at three FFT scales.
        base[:,0,192:216]=4.
        louder=base.copy()
        louder[2,0,192:216]+=1.
        # In alternative recording a newly plucked E3 adds 2nd,4th,6th,
        # 8th partials of E2. These overlap physically at integer multiples.
        newE3=base.copy()
        parent=newE3[2,0,192:216].reshape(3,8)
        parent[:,[1,3,5,7]]+=1.
        features_louder=harmonic_timbre_features(louder)
        features_new=harmonic_timbre_features(newE3)
        e3=52-40
        # 24..26 are 3 parent octave even/odd flux differences.
        np.testing.assert_allclose(features_louder[2,e3,24:27],0.,atol=1e-6)
        np.testing.assert_allclose(features_new[2,e3,24:27],1.,atol=1e-6)
        self.assertTrue(np.all(features_new[2,e3,30:]==[1,0]))
        # If no new E3 was played the cue stays absent at time1.
        np.testing.assert_allclose(features_new[1,e3,24:27],0.,atol=1e-6)
        # Global gain changes do not force an E3 onset from this cue.
        np.testing.assert_allclose(features_louder[2,e3,8:24],0.,atol=1e-6)

    def test_timbre_shape_invariant_to_uniform_log_amplitude_shift(self):
        x=np.zeros((4,49,288),np.float32)
        rng=np.random.default_rng(724)
        spectrum=rng.uniform(3,8,(4,49,24)).astype(np.float32)
        x[:,:,192:216]=spectrum
        shifted=x.copy()
        shifted[:,:,192:216]+=2.5
        a=harmonic_timbre_features(x)
        b=harmonic_timbre_features(shifted)
        # Signed shape, normalized timbre, parent-interval contrast remain equal.
        np.testing.assert_allclose(a[:,:,8:30],b[:,:,8:30],atol=3e-6,rtol=2e-6)
        # This is an approximation in real PCM: log1p floor & changed pluck
        # technique violate perfectly uniform log-domain spectral changes.

    def test_E2_and_19_semitone_missing_parent_mask_not_fake_E2_parent(self):
        x=np.zeros((2,49,288),np.float32)
        x[1,0,192:216]=np.arange(24,dtype=np.float32)
        f=harmonic_timbre_features(x)
        self.assertTrue(np.all(f[:, :12,24:27]==0))
        self.assertTrue(np.all(f[:, :19,27:30]==0))
        self.assertTrue(np.all(f[:, 12:,30]==1))
        self.assertTrue(np.all(f[:, 19:,31]==1))

    def test_original_s708_is_equivalent_at_new_320D_initialization(self):
        old=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32))
        trial=init_from_s708(old,np.zeros(32,np.float32),np.ones(32,np.float32))
        old.eval();trial.eval()
        rng=np.random.default_rng(724)
        base=rng.normal(size=(1,9,49,288)).astype(np.float32)
        extras=rng.normal(size=(1,9,49,32)).astype(np.float32)
        with torch.inference_mode():
            a,k=old(torch.from_numpy(base))
            b,j=trial(torch.from_numpy(np.concatenate((base,extras),axis=-1)))
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(k,j,rtol=1e-5,atol=1e-5)

    def test_Rejects_corrupt_inputs(self):
        with self.assertRaisesRegex(ValueError,"288"):
            harmonic_timbre_features(np.zeros((2,49,320),np.float32))
        z=np.zeros((2,49,288),np.float32)
        z[0,1,196]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            harmonic_timbre_features(z)


if __name__=="__main__":
    unittest.main()
