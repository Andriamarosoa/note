"""S729 phase evidence is true causal and specific to independent octave reset."""
import unittest
import numpy as np
import torch
from causal_note.s729_phase_overlap_evidence import (
    CausalPhaseSourceEvidence,phase_track,augmented_track,disabled_phase_track)
from causal_note.s723_polyphonic_novelty import init_from_s708
from causal_note.s705_temporal_heads import TemporalHeads


class S729PhaseTests(unittest.TestCase):
    def _fixture(self):
        x=np.zeros((6,49,288),np.float32)
        x[:,0,192:216]=4.
        x[1:,0,240:264]=1.  # phase-stable E2 partials
        x[:5,0,127]=.98     # E2 owner detected in prior native sustain
        return x

    def test_new_phase_reset_only_on_even_parent_harmonics_suggests_E3(self):
        original=self._fixture()
        global_change=original.copy()
        global_change[4,0,240:264]=-1. # all E2 partials reset
        new=original.copy()
        parts=new[4,0,240:264].reshape(3,8)
        parts[:,[1,3,5,7]]=-1. # only even E2 harmonic phases reset
        a=phase_track(global_change)
        b=phase_track(new)
        e3=12
        self.assertTrue(np.all(a[4,e3,24:27]==0.))
        self.assertTrue(np.all(b[4,e3,24:27]>.1))
        self.assertEqual(float(b[0,e3,24]),0.)
        self.assertGreater(float(b[4,e3,30]),0.)

    def test_no_owner_no_octave_creation(self):
        x=self._fixture();x[:,:,127]=0.
        x[4,0,240:264].reshape(3,8)[:,[1,3,5,7]]=-1.
        p=phase_track(x)
        self.assertTrue(np.all(p[:,:,24:30]==0.))
        self.assertTrue(np.all(p[:,:,30:]==0.))

    def test_all_at_once_equals_incremental_and_past_unchanged_by_future(self):
        r=np.random.default_rng(729)
        x=r.uniform(0,3,(16,49,288)).astype(np.float32)
        x[:,:,240:264]=r.uniform(-1,1,(16,49,24))
        x[:,:,127]=r.uniform(0,1,(16,49))
        a=phase_track(x)
        m=CausalPhaseSourceEvidence()
        for i in range(len(x)):
            np.testing.assert_array_equal(a[i],m.step(x[i]))
        alt=x.copy();alt[10:]*=2.
        np.testing.assert_array_equal(a[:10],phase_track(alt)[:10])
        self.assertEqual(augmented_track(x).shape,(16,49,320))

    def test_phase_confidence_is_low_when_amplitude_zero(self):
        x=np.zeros((2,49,288),np.float32)
        x[1,:,240:264]=-1.
        np.testing.assert_array_equal(phase_track(x),np.zeros((2,49,32),np.float32))

    def test_original_S708_exact_k_and_pitch_head_epoch0_equal_320D(self):
        old=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32))
        new=init_from_s708(old,np.zeros(32,np.float32),np.ones(32,np.float32))
        r=np.random.default_rng(729)
        x=r.normal(size=(1,10,49,288)).astype(np.float32)
        w=r.normal(size=(1,10,49,32)).astype(np.float32)
        old.eval();new.eval()
        with torch.inference_mode():
            p,k=old(torch.from_numpy(x))
            q,j=new(torch.from_numpy(np.concatenate((x,w),axis=-1)))
        torch.testing.assert_close(p,q,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(k,j,atol=1e-5,rtol=1e-5)
        self.assertEqual(j.shape,(1,10,7))

    def test_disabled_phase_copies_baseline_and_preserves_mean(self):
        x=np.zeros((4,49,288),np.float32)
        mean=np.arange(32,dtype=np.float32)
        y=disabled_phase_track(x,mean)
        np.testing.assert_array_equal(y[:,:,:288],x)
        np.testing.assert_array_equal(y[2,24,288:],mean)
        with self.assertRaisesRegex(ValueError,"288"):
            phase_track(np.zeros((1,49,320),np.float32))


if __name__=="__main__":
    unittest.main()
