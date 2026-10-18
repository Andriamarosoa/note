"""Ensure S729 equal-budget control blocks added phase features *before* FIT."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from scripts.s729_phase_source_exactk_training import PairedWindows


class S729PairedFitTests(unittest.TestCase):
    def test_phase_off_and_phase_on_use_identical_source_fit_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"source.joblib"
            n=9
            x=np.zeros((n,49,320),np.float16)
            x[:,:,288:]=2.
            yy=np.zeros((n,49,3),np.uint8)
            kk=np.zeros(n,np.int64)
            joblib.dump(dict(x=x,y=yy,k=kk,name="mock_FIT"),p)
            mean=np.ones(32,np.float32)*.25
            none=PairedWindows([p],mean,enable_memory=False)
            phase=PairedWindows([p],mean,enable_memory=True)
            off_y=none[0];on_y=phase[0]
            a,ya,ka,va,ma=off_y
            b,yb,kb,vb,mb=on_y
            valid=va.bool()
            np.testing.assert_array_equal(a[valid,:,288:].numpy(),np.full((n,49,32),.25,np.float32))
            np.testing.assert_array_equal(b[valid,:,288:].numpy(),np.full((n,49,32),2.,np.float32))
            np.testing.assert_array_equal(ya.numpy(),yb.numpy())
            np.testing.assert_array_equal(ka.numpy(),kb.numpy())
            np.testing.assert_array_equal(va.numpy(),vb.numpy())
            np.testing.assert_array_equal(ma.numpy(),mb.numpy())

if __name__=="__main__":
    unittest.main()
