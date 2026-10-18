import unittest
import numpy as np
from causal_note.strict_live import SR,HOP
from causal_note.s706_sharp_spectra import sharp_track


class SharpTests(unittest.TestCase):
    def test_audio_prefix_is_invariant(self):
        audio=np.random.default_rng(706).normal(size=30*4*HOP).astype(np.float32)*.03
        ends=np.arange(1,31)*4*HOP
        full=sharp_track(audio,ends)
        truncated=sharp_track(audio[:ends[12]],ends[:13])
        np.testing.assert_array_equal(full[:13],truncated)
        changed=audio.copy();changed[ends[12]:]=4
        np.testing.assert_array_equal(full[:13],sharp_track(changed,ends)[:13])

    def test_adjacent_bass_notes_do_not_share_maximum(self):
        n=8192;hz=440*2**((40-69)/12)
        pcm=np.sin(2*np.pi*hz*np.arange(n)/SR).astype(np.float32)
        sharp=sharp_track(pcm,np.array([n]))[0]
        # Last resolution, first harmonic: its own frequency beats F2.
        self.assertGreater(sharp[0,16],sharp[1,16]+.3)
        self.assertEqual(sharp.shape,(49,96))

    def test_future_audio_request_rejected(self):
        with self.assertRaises(ValueError):
            sharp_track(np.zeros(1000),np.array([1024]))


if __name__=='__main__':
    unittest.main()
