import unittest
import numpy as np
from scripts.s813_prepare_native256 import extract_track, targets_at_ends


class Native256Test(unittest.TestCase):
    def test_received_pcm_partition_and_future_suffix(self):
        rng = np.random.default_rng(813)
        t = np.arange(8192, dtype=np.float32)/44100
        pcm = (.13*np.sin(2*np.pi*82.4069*t)+.08*np.sin(2*np.pi*164.8138*t)).astype(np.float32)
        ends, original = extract_track(pcm)
        for block in (64,333,4096):
            other_ends, other = extract_track(pcm, block_size=block)
            np.testing.assert_array_equal(ends, other_ends)
            np.testing.assert_array_equal(original, other)
        altered = pcm.copy()
        altered[4096:] = rng.normal(0,.2,len(pcm)-4096)
        _, future = extract_track(altered)
        np.testing.assert_array_equal(original[ends<=4096], future[ends<=4096])
        self.assertFalse(np.array_equal(original[ends>4096], future[ends>4096]))
        np.testing.assert_array_equal(ends, np.arange(256,8193,256))
        self.assertEqual(original.shape, (32,49,222))

    def test_annotation_half_open_intervals_and_unison_count(self):
        ends = np.array([256,512,768,1024])
        notes = [dict(pitch=40,time=256/44100,duration=512/44100),
                 dict(pitch=40,time=512/44100,duration=256/44100)]
        y,k = targets_at_ends(notes,ends)
        np.testing.assert_array_equal(y[:,0],[1,1,0,0])
        np.testing.assert_array_equal(k,[1,2,0,0])
        self.assertFalse(y[:,1:].any())


if __name__ == '__main__':
    unittest.main()
