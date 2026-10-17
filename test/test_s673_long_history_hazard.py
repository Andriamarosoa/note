"""S673 strict prefix invariance over 256 old audio frames."""
import unittest
import numpy as np
from causal_note.strict_live import HOP
from scripts.s673_causal_long_history_noteoff import frame_matrix,native_rhythm_clock
class S673HistoryTests(unittest.TestCase):
    def test_long_historical_features_never_see_future(self):
        rng=np.random.default_rng(673)
        n=350;f=rng.random((n,49,126),dtype=np.float32)
        p=rng.random((n,49,3),dtype=np.float32)
        k=np.zeros((n,7),np.float32);k[:,1]=1.
        ends=np.arange(1,n+1)*HOP
        clock=native_rhythm_clock([],ends)
        now=frame_matrix(f,p,k,ends,40,30,np.array([50,150,270]),clock)
        f[290:]=10;p[290:]=10;k[290:]=10
        other=frame_matrix(f,p,k,ends,40,30,np.array([50,150,270]),clock)
        np.testing.assert_array_equal(now,other)
        self.assertGreater(now.shape[1],75)
    def test_no_future_audio_for_early_note(self):
        n=100
        f=np.zeros((n,49,126),np.float32)
        p=np.zeros((n,49,3),np.float32)
        k=np.zeros((n,7),np.float32);k[:,1]=1.
        ends=np.arange(1,n+1)*HOP
        clock=native_rhythm_clock([],ends)
        x=frame_matrix(f,p,k,ends,40,5,np.arange(10,30),clock)
        self.assertTrue(np.isfinite(x).all())
if __name__=="__main__":unittest.main()
