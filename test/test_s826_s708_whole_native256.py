"""S826 real S708 architecture at exact native 256; synthetic safety checks."""
import unittest
from unittest.mock import patch
import numpy as np
from causal_note.strict_live import SR,HOP,feature_track
from causal_note.s704_joint_heads import acoustic_features,JointPolicy,JointSelector
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s708_cardinality_selection import GuidedPolicy,GuidedSelector
from causal_note.s712_s708_live import S708StrictLiveStream
from causal_note.s826_s708_whole_native256 import S826S708Native256Stream

class StubHeads:
    def predict(self,x):
        a=np.asarray(x,np.float32)
        p=np.zeros((49,3),np.float32);p[:,1]=.5;p[:,2]=.1
        p[:,0]=np.clip(a[:,0]/10.,0,1)
        k=np.zeros(7,np.float32);k[0]=1
        return p,k

class StubTemporal:
    def eval(self):return self
    def __call__(self,x):
        import torch
        t=x.shape[1]
        p=torch.full((1,t,49,3),-5.,dtype=torch.float32)
        k=torch.full((1,t,7),-8.,dtype=torch.float32)
        k[:,:,0]=12.
        p[:,:,:,1]=5.
        return p,k

POLICY=GuidedPolicy(onset=.75,sustain_on=.5,sustain_off=.3,
  release=.85,release_steps=1,count_confidence=1.1,blend=.7,
  guide_confidence=.5,weak_onset=.15,weak_sustain=.55,rise=.04,trim=False)

def wave(hops=12):
    n=hops*HOP
    t=np.arange(n,dtype=np.float32)/SR
    return (.08*np.sin(2*np.pi*110*t)+.05*np.sin(2*np.pi*164.81*t)).astype(np.float32)

class S826OriginalS708Strict256Tests(unittest.TestCase):
    def make(self):
        return S826S708Native256Stream(StubHeads(),StubTemporal(),POLICY,capture_features=True)

    def test_same_288_channel_full_s708_features_each256(self):
        pcm=wave(10)
        ends,native=feature_track(pcm,block_size=256,hop=HOP)
        stub=StubHeads()
        pred=[stub.predict(f) for f in native]
        post=np.asarray([v[0] for v in pred])
        count=np.asarray([v[1] for v in pred])
        expected=np.concatenate((
            acoustic_features(native,post,count),
            sharp_track(pcm,ends)),axis=-1)
        assert expected.shape==(10,49,288)
        engine=self.make()
        observed=[]
        def capture(x):
            observed.append(x.copy())
            return np.zeros((49,3),np.float32),np.eye(7,dtype=np.float32)[0]
        engine.temporal.step=capture
        for offset in range(0,len(pcm),256):
            engine.push(pcm[offset:offset+HOP],start_sample=offset)
        self.assertEqual(engine.model_calls,10)
        self.assertEqual(engine.s826_decision_ends,list(range(256,256*11,256)))
        np.testing.assert_array_equal(np.asarray(engine.s826_source_288d),
                                      np.asarray(observed))
        for i,(got,want) in enumerate(zip(observed,expected)):
            np.testing.assert_allclose(got,want,rtol=3e-6,atol=3e-6,
                err_msg=f"original S708 288D frontend mismatch native256 frame {i}")
        self.assertEqual(engine.diagnostics()["decision_hop"],256)

    def test_256_head_calls_and_transport_invariance(self):
        pcm=wave(12)
        runs=[]
        for size in (128,256,381,1024,3072):
            engine=self.make();all_events=[];start=0
            while start<len(pcm):
                chunk=pcm[start:start+size]
                all_events.extend(engine.push(chunk,start_sample=start))
                start+=len(chunk)
            all_events+=engine.finish()
            self.assertEqual(engine.model_calls,12)
            self.assertEqual(len(engine.s826_decision_ends),12)
            self.assertEqual(engine.diagnostics()["original_frozen_checkpoint_training_hop"],1024)
            self.assertEqual(engine.diagnostics()["added_lookahead_samples"],0)
            self.assertEqual(engine.diagnostics()["added_hold_samples"],0)
            for e in all_events:
                self.assertGreaterEqual(e["available_sample"],e["decision_sample"])
                self.assertEqual(e["decision_sample"]%256,0)
            runs.append([(e["type"],e["pitch"],e["decision_sample"]) for e in all_events])
        self.assertTrue(all(x==runs[0] for x in runs))

    def test_legacy_1024_architecture_unchanged(self):
        pcm=wave(12)
        old=S708StrictLiveStream(StubHeads(),StubTemporal(),POLICY)
        events=[]
        for offset in range(0,len(pcm),HOP):
            events.extend(old.push(pcm[offset:offset+HOP],start_sample=offset))
        events+=old.finish()
        self.assertEqual(old.model_calls,3)
        self.assertEqual(old.diagnostics()["decision_hop"],1024)
        dec=GuidedSelector(POLICY)
        for step in range(1,4):
            x=np.zeros((49,3));k=np.zeros(7);k[0]=1
            dec.step(x,k,step*1024)
        self.assertEqual(dec.last_end,3072)
        with self.assertRaisesRegex(ValueError,"Non-contiguous"):
            GuidedSelector(POLICY,decision_hop_samples=256).step(
                np.zeros((49,3)),np.eye(7)[0],1024)

    def test_no_future_audio_or_s708_midi_sidecar(self):
        signal=wave(8)
        modified=signal.copy();modified[4*256:]*=-1.
        prefix=[]
        for w in (signal,modified):
            stream=self.make()
            for offset in range(0,len(w),HOP):
                stream.push(w[offset:offset+HOP],start_sample=offset)
            prefix.append(np.asarray(stream.s826_source_288d[:4]))
        np.testing.assert_array_equal(prefix[0],prefix[1])
        with self.assertRaisesRegex(ValueError,"discontinuous"):
            self.make().push(signal[:HOP],start_sample=HOP)

if __name__=="__main__":
    unittest.main()
