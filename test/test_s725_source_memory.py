"""S725 source-memory fixtures: E2 sustain/new E3 vs louder E2, streaming, gates."""
import unittest
import numpy as np
import torch
from causal_note.s725_source_memory import (
    BASE,EXTRA,TOTAL,CausalSourceMemory,source_memory_track,
    augmented_track,disabled_memory_track)
from causal_note.s723_polyphonic_novelty import init_from_s708
from causal_note.s705_temporal_heads import TemporalHeads


class S725MemoryTests(unittest.TestCase):
    def _base(self):
        x=np.zeros((6,49,288),np.float32)
        # E2 spectral template and native Sustain for 2 earlier decisions.
        x[:,0,192:216]=4.
        x[:4,0,127]=.95
        return x

    def test_E3_new_attack_has_octave_owned_innovation_when_E2_continues(self):
        original=self._base()
        gain=original.copy()
        gain[4,0,192:216]+=1.
        new=original.copy()
        new[4,0,192:216].reshape(3,8)[:,[1,3,5,7]]+=1.
        fa=source_memory_track(gain)
        fb=source_memory_track(new)
        e3=12
        # Columns 24:27 track unexpected *even*-parent partial movement.
        np.testing.assert_allclose(fa[4,e3,24:27],0.,atol=1e-6)
        self.assertTrue(np.all(fb[4,e3,24:27]>.1))
        self.assertTrue(np.all(fb[4,e3,30]>.0))
        # A new E3 is *hypothesized*, not automatically created.
        self.assertEqual(float(fb[3,e3,24]),0.)

    def test_unobserved_parent_does_not_create_false_source_ownership(self):
        x=np.zeros((4,49,288),np.float32)
        x[:,0,192:216]=3.
        x[3,0,192:216].reshape(3,8)[:,[1,3,5,7]]+=3.
        f=source_memory_track(x)
        self.assertTrue(np.all(f[:,:,30:]==0.))
        self.assertTrue(np.all(f[:,:,24:30]==0.))

    def test_real_stream_state_equal_all_at_once_and_future_perturbation(self):
        rng=np.random.default_rng(725)
        x=rng.uniform(0,5,(31,49,288)).astype(np.float32)
        x[:,:,127]=rng.uniform(0,1,(31,49))
        f=source_memory_track(x)
        stream=CausalSourceMemory()
        for t in range(len(x)):
            np.testing.assert_array_equal(f[t],stream.step(x[t]))
        xx=x.copy();xx[18:]*=7.
        np.testing.assert_array_equal(f[:18],source_memory_track(xx)[:18])
        self.assertEqual(augmented_track(x).shape,(31,49,TOTAL))
        np.testing.assert_array_equal(augmented_track(x)[:,:,:BASE],x)

    def test_memory_free_control_is_exactly_mean_constant(self):
        rng=np.random.default_rng(725)
        x=rng.normal(size=(7,49,BASE)).astype(np.float32)
        mean=rng.normal(size=EXTRA).astype(np.float32)
        a=disabled_memory_track(x,mean)
        self.assertEqual(a.shape,(7,49,TOTAL))
        np.testing.assert_array_equal(a[:,:,:BASE],x)
        np.testing.assert_array_equal(a[0,5,BASE:],mean)
        np.testing.assert_array_equal(a[-1,48,BASE:],mean)
        self.assertFalse(np.any(np.diff(a[:,:,BASE:],axis=0)))

    def test_old_S708_logits_identical_before_fit_and_count_K_preserved(self):
        rng=np.random.default_rng(725)
        old=TemporalHeads(np.zeros(BASE,np.float32),np.ones(BASE,np.float32))
        trial=init_from_s708(old,np.zeros(EXTRA,np.float32),np.ones(EXTRA,np.float32))
        old.eval();trial.eval()
        x=rng.normal(size=(1,13,49,BASE)).astype(np.float32)
        extra=rng.normal(size=(1,13,49,EXTRA)).astype(np.float32)
        with torch.inference_mode():
            a,ak=old(torch.from_numpy(x))
            b,bk=trial(torch.from_numpy(np.concatenate((x,extra),axis=-1)))
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(ak,bk,atol=1e-5,rtol=1e-5)
        self.assertEqual(bk.shape,(1,13,7))

    def test_corrupt_or_noncausal_input_rejected(self):
        with self.assertRaisesRegex(ValueError,"288"):
            source_memory_track(np.zeros((4,49,320),np.float32))
        x=np.zeros((4,49,BASE),np.float32);x[1,0,193]=np.nan
        with self.assertRaisesRegex(ValueError,"finite"):
            source_memory_track(x)


if __name__=="__main__":
    unittest.main()
