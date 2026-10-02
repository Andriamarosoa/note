import unittest
import numpy as np

from scripts.v273_high_pitch_map import (
    SAMPLE_RATE, COMPRESSED_SAMPLE_RATE, compress_context, time_frequency_map
)
from scripts.v273_residual_map import time_frequency_map as previous_map
from scripts.v273_high_pitch_registers import assigned_pitches, masks
from scripts.v273_high_pitch_native import treatment_maps, rescue_maps


class HighPitchFeatureTests(unittest.TestCase):
    def test_pairwise_compression_matches_average_pooling(self):
        x=np.arange(16,dtype=np.float64).reshape(2,8)
        expected=.5*(x[...,0::2]+x[...,1::2])
        np.testing.assert_array_equal(compress_context(x),expected)

    def test_compression_preserves_physical_tone_frequency(self):
        n=32768
        f=220.5
        x=np.sin(2*np.pi*f*np.arange(n)/SAMPLE_RATE)
        y=compress_context(x)
        spectrum=np.abs(np.fft.rfft(y*np.hanning(len(y))))
        measured=np.fft.rfftfreq(len(y),1/COMPRESSED_SAMPLE_RATE)[spectrum.argmax()]
        self.assertLess(abs(measured-f),COMPRESSED_SAMPLE_RATE/len(y))

    def test_normal_channels_match_previous_control(self):
        x=np.random.default_rng(4).normal(0,.02,24000).astype(np.float32)
        old=previous_map(x,np.zeros_like(x),7000)
        new=time_frequency_map(x,7000)
        np.testing.assert_array_equal(new[:,:,[0,2]],old[:,:,[0,2]])
        control=time_frequency_map(x,7000,include_pitch=False)
        np.testing.assert_array_equal(control,new[:,:,[0,0,2,2]])

    def test_future_and_extra_past_cannot_change_features(self):
        origin=7000
        x=np.random.default_rng(5).normal(0,.01,22000).astype(np.float32)
        expected=time_frequency_map(x,origin)
        changed=x.copy()
        changed[:origin-5148]=999
        changed[origin+2788:]=np.nan
        np.testing.assert_array_equal(expected,time_frequency_map(changed,origin))
        np.testing.assert_array_equal(expected,time_frequency_map(x[:origin+2788],origin))

    def test_rescue_reacts_to_extra_past_while_normal_does_not(self):
        origin=7000
        x=np.zeros(22000,np.float32)
        base=time_frequency_map(x,origin)
        changed=x.copy()
        changed[origin-5000:origin-3200]=.5
        altered=time_frequency_map(changed,origin)
        np.testing.assert_array_equal(base[:,:,[0,2]],altered[:,:,[0,2]])
        self.assertGreater(float(np.max(np.abs(base[:,:,[1,3]]-altered[:,:,[1,3]]))),0.)

    def test_edge_padding_and_silence_are_finite(self):
        x=np.zeros(512,np.float32)
        np.testing.assert_array_equal(time_frequency_map(x,0),np.zeros((31,64,4),np.float32))
        self.assertTrue(np.isfinite(time_frequency_map(np.ones(512,np.float32),0)).all())

    def test_invalid_audio_is_rejected(self):
        with self.assertRaises(ValueError): compress_context(np.ones(7))
        with self.assertRaises(ValueError): compress_context(np.array([1,np.nan]))
        with self.assertRaises(ValueError): time_frequency_map(np.full(8000,np.nan),0)

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

    def test_main_path_cannot_receive_rescue_channels(self):
        x=np.random.default_rng(18).uniform(size=(2,31,64,4)).astype(np.float32)
        main=treatment_maps(x,'with_high_pitch')
        expected=x[:,:,:, [0,0,2,2]]
        np.testing.assert_array_equal(main,expected)
        x[:,:,:, [1,3]]=50
        np.testing.assert_array_equal(main,treatment_maps(x,'with_high_pitch'))

    def test_rescue_path_receives_only_compressed_context(self):
        x=np.random.default_rng(19).uniform(size=(2,31,64,4)).astype(np.float32)
        np.testing.assert_array_equal(rescue_maps(x),x[:,:,:, [1,3]])


if __name__=='__main__':
    unittest.main()
