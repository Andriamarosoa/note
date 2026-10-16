"""S616 physical-time normalization with frozen S613 256-bitwise contract.

No training/test labels are required: this is an acoustic-feature ABI test,
not a claim that 128/512 yield higher NOTE accuracy.
"""
import unittest
import numpy as np

from causal_note.strict_live import CausalPitchFeatures, feature_track, HOP, SR


class TimeAlignedFeatureTests(unittest.TestCase):
    def test_256_is_bitwise_same_as_unmodified_abi_short_rms(self):
        rng=np.random.default_rng(6516)
        audio=rng.normal(0,.03,5*8192).astype(np.float32)
        ends,a=feature_track(audio,block_size=256)
        ends2,b=feature_track(audio,block_size=701,hop=HOP)
        np.testing.assert_array_equal(ends,ends2)
        np.testing.assert_array_equal(a,b)
        idx=int(ends[-1])
        expected=np.log1p(100*np.sqrt(np.mean(audio[idx-256:idx]**2)))
        self.assertAlmostEqual(float(a[-1,0,-3]),float(expected),places=5)

    def test_short_rms_tracks_selected_physical_hop_window(self):
        rng=np.random.default_rng(1616)
        x=rng.normal(0,.08,12288).astype(np.float32)
        for hop in (128,256,512):
            ends, features=feature_track(x,block_size=2048,hop=hop)
            for k in (len(ends)//3, len(ends)-1):
                sample=int(ends[k])
                expected=np.log1p(
                    100*np.sqrt(np.mean(x[sample-hop:sample]**2)))
                self.assertAlmostEqual(
                    float(features[k,0,-3]),float(expected),places=5)

    def test_does_not_look_past_new_hop_frame(self):
        rng=np.random.default_rng(1808)
        a=rng.normal(0,.01,16384).astype(np.float32)
        b=a.copy()
        b[8192:]=rng.normal(0,.9,len(b)-8192)
        for hop in (128,256,512):
            ends1,f1=feature_track(a,hop=hop)
            ends2,f2=feature_track(b,hop=hop)
            before=ends1<=8192
            np.testing.assert_array_equal(ends1[before],ends2[before])
            np.testing.assert_array_equal(f1[before],f2[before])

    def test_normalized_flux_finite_for_all_hops(self):
        audio=np.random.default_rng(8192).normal(
            0,.04,16384).astype(np.float32)
        for hop in (128,256,512):
            _,frames=feature_track(audio,hop=hop)
            self.assertTrue(np.isfinite(frames).all())
            self.assertTrue((frames[:,:,-6:-3]>=0).all())


if __name__=="__main__":
    unittest.main()
