"""S827 proof: 256 real S708 signal path isolated from five other axes."""
import copy
import unittest

import numpy as np

from causal_note.strict_live import SR,HOP,feature_track
from causal_note.s704_joint_heads import acoustic_features
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s708_cardinality_selection import GuidedPolicy,GuidedSelector
from causal_note.s705_temporal_heads import TemporalHeads
from causal_note.s812_historical_axis_h0 import (
    SixHistoricalH0,checked_h1_view,AXES,signature
)
from causal_note.s827_s708_activek_only import (
    PrivateS708GuidedActiveK,S827S708ActiveKOnly
)


class StubNative:
    def predict(self,f):
        a=np.asarray(f,np.float32)
        p=np.zeros((49,3),np.float32)
        p[:,0]=np.clip(a[:,0]/10.,0,1)
        p[:,1]=.55
        p[:,2]=.1
        k=np.zeros(7,np.float32)
        k[2]=1
        return p,k


def wave(hops=12):
    t=np.arange(hops*256,dtype=np.float64)/SR
    return (.06*np.sin(2*np.pi*82.4*t)+
            .1*np.sin(2*np.pi*164.8*t)).astype(np.float32)


def policy():
    return GuidedPolicy(onset=.65,sustain_on=.45,sustain_off=.30,
                        release=.80,release_steps=1,count_confidence=.99,
                        blend=.75,guide_confidence=.70,
                        weak_onset=.2,weak_sustain=.65,rise=.04,trim=True)


def model():
    import torch
    torch.manual_seed(827)
    m=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32))
    m.eval()
    torch.set_num_threads(1)
    return m


def tuples(decisions):
    return [(d.decision_sample,d.count,d.model_count,
             d.model_confidence) for d in decisions]


class TestS827OnlyActiveK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nn=model()

    def make(self,capture=False):
        return S827S708ActiveKOnly(StubNative(),self.nn,policy(),
                                 capture_features=capture)

    def test_original_guided_decoder_exact_state_parity_at_1024(self):
        a=GuidedSelector(policy())
        b=PrivateS708GuidedActiveK(policy(),decision_hop=1024)
        rng=np.random.default_rng(827)
        for i in range(1,61):
            p=rng.uniform(0,1,(49,3)).astype(np.float32)
            k=np.zeros(7,np.float32);k[int(rng.integers(0,7))]=1.
            a.step(p,k,i*1024)
            got=b.step(p,k,i*1024)
            self.assertEqual(got.count,int(a.active.sum()))
            np.testing.assert_array_equal(a.active,b.active)
            np.testing.assert_array_equal(a.armed,b.armed)
            np.testing.assert_array_equal(a.low,b.low)
            np.testing.assert_array_equal(a.last_on,b.last_on)
            self.assertEqual(a.lower_count_steps,b.lower_count_steps)
            self.assertEqual(a.last_end,b.last_end)
        self.assertEqual(b.step_calls,60)

    def test_original_288d_feature_abi_every_256(self):
        audio=wave(10)
        ends,feat=feature_track(audio,block_size=256,hop=HOP)
        native=StubNative()
        post=np.stack([native.predict(f)[0] for f in feat])
        k=np.stack([native.predict(f)[1] for f in feat])
        ref=np.concatenate((acoustic_features(feat,post,k),
                            sharp_track(audio,ends)),-1)
        self.assertEqual(ref.shape,(10,49,288))
        private=self.make(capture=True)
        for off in range(0,len(audio),256):
            self.assertEqual(len(private.push(audio[off:off+256],start_sample=off)),1)
        np.testing.assert_allclose(np.stack(private.feature_debug),
                                   ref,rtol=3e-6,atol=3e-6)
        self.assertEqual(private.model_calls,10)
        self.assertEqual(private.diagnostics()["decision_hop_samples"],256)

    def test_callback_partition_invariance_256_and_no_future(self):
        audio=wave(16)
        results=[]
        for block in (128,256,365,777,2048,4096):
            stream=self.make()
            rows=[];sample=0
            while sample<len(audio):
                section=audio[sample:sample+block]
                rows.extend(stream.push(section,start_sample=sample))
                sample+=len(section)
            self.assertEqual(len(rows),16)
            self.assertEqual(stream.temporal.steps,16)
            self.assertEqual(stream.decoder.step_calls,16)
            self.assertEqual(tuple(x.decision_sample for x in rows),
                             tuple(range(256,4097,256)))
            results.append(tuples(rows))
        for result in results[1:]:
            self.assertEqual(result,results[0])
        modified=audio.copy();modified[8*256:]*=-1
        a,b=self.make(),self.make()
        x=[];y=[]
        for off in range(0,len(audio),256):
            x.extend(a.push(audio[off:off+256],start_sample=off))
            y.extend(b.push(modified[off:off+256],start_sample=off))
        self.assertEqual(tuples(x[:8]),tuples(y[:8]))

    def test_only_activek_overlay_all_other_axes_and_source_midi_bit_exact(self):
        midi=[dict(type="note_on",pitch=40,decision_sample=256,
            available_sample=256,time=256/SR,reason="source"),
            dict(type="note_off",pitch=40,decision_sample=1024,
            available_sample=1024,time=1024/SR,reason="source")]
        reference=signature(midi)
        bank=SixHistoricalH0()
        for t in range(256,1537,256):
            h0=bank.step(t,[x for x in midi if x["available_sample"]==t])
            own=PrivateS708GuidedActiveK(policy(),decision_hop=256)
            # H1 shadow overlay changes ONE axis result, never source.
            replacement=checked_h1_view(h0,("S827_private_count",5),"active_exact_k")
            for axis in AXES:
                if axis!="active_exact_k":
                    self.assertEqual(replacement[axis],h0[axis])
            self.assertEqual(signature(midi),reference)
        for axis in AXES:
            self.assertEqual(signature(bank.lanes[axis].events),reference)
        self.assertTrue(all(self.make().diagnostics()[k] is False for k in
           ("public_midi_emission","other_five_axes_mutated","source_model_retrained_to_256")))

    def test_reject_nonfinite_discontinuity_and_non256_decoding(self):
        a=self.make()
        with self.assertRaisesRegex(ValueError,"discontinuous"):
            a.push(wave(1),start_sample=7)
        with self.assertRaisesRegex(ValueError,"finite"):
            a.push(np.asarray([np.nan],np.float32),start_sample=0)
        with self.assertRaisesRegex(ValueError,"non-contiguous"):
            PrivateS708GuidedActiveK(policy()).step(
                np.zeros((49,3),np.float32),np.eye(7)[0],1024)


if __name__=="__main__":
    unittest.main()
