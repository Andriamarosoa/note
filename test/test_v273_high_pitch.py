import unittest
import numpy as np

from scripts.v273_high_pitch_map import FIR, SAMPLE_RATE, octave_up, time_frequency_map
from scripts.v273_residual_map import time_frequency_map as previous_map
from scripts.v273_high_pitch_registers import assigned_pitches,masks
from scripts.v273_high_pitch_native import treatment_maps


class HighPitchFeatureTests(unittest.TestCase):
    def test_octave_up_doubles_tone_frequency(self):
        n=32768
        x=np.sin(2*np.pi*220.5*np.arange(n)/SAMPLE_RATE)
        y=octave_up(x)
        spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y))))
        f=np.fft.rfftfreq(len(y),1/SAMPLE_RATE)[spectrum.argmax()]
        self.assertLess(abs(f-441.),SAMPLE_RATE/len(y))
        self.assertEqual(len(y),n//2)

    def test_aliasing_is_suppressed(self):
        n=16384
        x=np.sin(2*np.pi*15000*np.arange(n)/SAMPLE_RATE)
        y=octave_up(x)[128:-128]
        self.assertLess(np.sqrt(np.mean(y*y)),1e-3)

    def test_filter_matches_direct_linear_convolution(self):
        x=np.random.default_rng(3).normal(size=(2,256))
        expected=np.array([np.convolve(row,FIR,mode='same')[1::2] for row in x])
        np.testing.assert_allclose(octave_up(x),expected,atol=1e-12)

    def test_normal_channels_match_previous_control(self):
        x=np.random.default_rng(4).normal(0,.02,20000).astype(np.float32)
        old=previous_map(x,np.zeros_like(x),7000)
        new=time_frequency_map(x,7000)
        np.testing.assert_array_equal(new[:,:,[0,2]],old[:,:,[0,2]])
        control=time_frequency_map(x,7000,include_pitch=False)
        np.testing.assert_array_equal(control,new[:,:,[0,0,2,2]])

    def test_future_and_extra_past_cannot_change_features(self):
        origin=5000
        x=np.random.default_rng(5).normal(0,.01,18000).astype(np.float32)
        expected=time_frequency_map(x,origin)
        changed=x.copy()
        changed[:origin-3100]=999
        changed[origin+2788:]=np.nan
        np.testing.assert_array_equal(expected,time_frequency_map(changed,origin))
        np.testing.assert_array_equal(expected,time_frequency_map(x[:origin+2788],origin))

    def test_edge_padding_and_silence_are_finite(self):
        x=np.zeros(512,np.float32)
        np.testing.assert_array_equal(time_frequency_map(x,0),np.zeros((31,64,4),np.float32))
        self.assertTrue(np.isfinite(time_frequency_map(np.ones(512,np.float32),0)).all())

    def test_invalid_audio_is_rejected(self):
        with self.assertRaises(ValueError): octave_up(np.ones(7))
        with self.assertRaises(ValueError): octave_up(np.array([1,np.nan]))
        with self.assertRaises(ValueError): time_frequency_map(np.full(6000,np.nan),0)

    def test_assignment_preserves_original_tie_rule_and_unassigned_events(self):
        groups=[np.array([1000]),np.array([1100])]
        p,k,missing=assigned_pitches(groups,[(1050,40.1),(1100,72.1),(5000,60.)])
        np.testing.assert_array_equal(k,[1,1])
        self.assertEqual(p[0,0],40)
        self.assertEqual(p[1,0],72)
        self.assertEqual(missing,1)

    def test_registers_and_k3_are_not_confused_with_silence(self):
        p=np.array([[40,60,72,-1,-1,-1],[48,-1,-1,-1,-1,-1],[-1]*6])
        m=masks(np.array([3,1,0]),p)
        np.testing.assert_array_equal(m['k3_bass'],[True,False,False])
        np.testing.assert_array_equal(m['any_high'],[True,False,False])
        np.testing.assert_array_equal(m['only_mid'],[False,True,False])
        self.assertFalse(m['only_bass'][2])

    def test_control_cannot_receive_compressed_channels(self):
        x=np.random.default_rng(18).uniform(size=(2,31,64,4)).astype(np.float32)
        control=treatment_maps(x,'observed_only')
        x[:,:,:,[1,3]]=50
        np.testing.assert_array_equal(control,treatment_maps(x,'observed_only'))
        np.testing.assert_array_equal(x,treatment_maps(x,'with_high_pitch'))
        with self.assertRaises(RuntimeError):treatment_maps(x,'with_residual')


if __name__=='__main__': unittest.main()
