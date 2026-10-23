"""S836: exact S706 sharp feature parity, causal firewall and cost instrumentation."""
from __future__ import annotations
import unittest
import numpy as np

from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s836_incremental_sharp256 import IncrementalSharp256
from causal_note.s834_native256_decoders import Fragment,AXES
from causal_note.s834_s809_six_axis_pcm256 import SixIndependentAxes


def synth(hops=128):
    n=np.arange(256*hops, dtype=np.float64)
    time=n/44100
    wave=(.23*np.sin(2*np.pi*82.4069*time)
          +.07*np.sin(2*np.pi*164.8138*time)
          +.05*np.sin(2*np.pi*246.9417*time))
    wave[:2048]=0
    wave[16*256:17*256]*=1.7
    wave[43*256:55*256]*=.12
    wave+=np.random.default_rng(836).normal(0,.0005, wave.size)
    return wave.astype(np.float32)


def compare_all(audio):
    engine=IncrementalSharp256()
    raw=[];cached=[]
    for end in range(256,len(audio)+1,256):
        tail=audio[max(0,end-8448):end]
        ref_ends=np.array([end] if end==256 else [end-256,end],np.int64)
        r=sharp_track(tail,ref_ends if end<=8448 else
                      np.array([len(tail)] if end==256 else [len(tail)-256,len(tail)],np.int64))[-1]
        c=engine.step(tail,end=end)
        np.testing.assert_array_equal(r,c,
            err_msg="S836 changed protected 96D sharp acoustics at sample %d" % end)
        raw.append(r);cached.append(c)
    return raw,cached


class S836IncrementalSharpTests(unittest.TestCase):
    def test_every_256_frame_is_bit_exact_including_8192_rollover(self):
        src=synth(96)
        raw,cached=compare_all(src)
        self.assertEqual(len(raw),96)
        self.assertEqual(len(cached),96)

    def test_prefix_causality_and_future_mutation(self):
        original=synth(72)
        mutant=original.copy()
        mutant[45*256:]=np.random.default_rng(22).normal(0,.04,len(mutant)-45*256)
        raw,_=compare_all(original)
        changed,_=compare_all(mutant)
        for a,b in zip(raw[:45],changed[:45]):
            np.testing.assert_array_equal(a,b)

    def test_missing_audio_or_wrong_clock_is_rejected_without_state_change(self):
        a=synth(2)
        obj=IncrementalSharp256()
        with self.assertRaises(ValueError):
            obj.step(a[:255],end=256)
        with self.assertRaises(ValueError):
            obj.step(a[:256],end=512)
        self.assertEqual(obj.calls,0)
        obj.step(a[:256],end=256)
        with self.assertRaises(ValueError):
            obj.step(np.full(512,np.nan,np.float32),end=512)
        self.assertEqual(obj.calls,1)
        obj.step(a,end=512)
        self.assertEqual(obj.calls,2)

    def test_six_decoders_see_only_bit_identical_snapshots(self):
        audio=synth(20)
        old=SixIndependentAxes()
        new=SixIndependentAxes()
        cache=IncrementalSharp256()
        for end in range(256,len(audio)+1,256):
            tail=audio[max(0,end-8448):end]
            n=len(tail)
            reference=sharp_track(tail,[n] if end==256 else [n-256,n])[-1]
            optimized=cache.step(tail,end=end)
            np.testing.assert_array_equal(reference,optimized)
            # Identical acoustic evidence enters private independent axes;
            # the reference synthetic six-head decision must stay unchanged.
            post=np.zeros((49,3),np.float32)
            post[:,0]=np.clip(reference[:,0]/50,0,1)
            post[:,1]=np.clip(reference[:,24]/50,0,1)
            post[:,2]=np.clip(reference[:,48]/50,0,1)
            pnew=np.zeros((49,3),np.float32)
            pnew[:,0]=np.clip(optimized[:,0]/50,0,1)
            pnew[:,1]=np.clip(optimized[:,24]/50,0,1)
            pnew[:,2]=np.clip(optimized[:,48]/50,0,1)
            k=np.array([1,0,0,0,0,0,0],np.float32)
            dec0,sel0=old.step(Fragment(end,post,k))
            dec1,sel1=new.step(Fragment(end,pnew,k))
            self.assertEqual(dec0,dec1)
            self.assertEqual(sel0,sel1)
            self.assertEqual(set(dec0),set(AXES))


    def test_real_s834_step_invokes_temporal_net_in_both_engines(self):
        from collections import deque
        from types import SimpleNamespace
        from causal_note.s834_s809_native256_audio import S834S809Native256
        class DummyNative:
            def predict(self, features):
                p=np.zeros((49,3),np.float32)
                k=np.zeros(7,np.float32); k[0]=1.
                return p,k
        class DummyTemporal:
            def __init__(self):self.calls=0
            def step(self, x):
                assert x.shape==(49,288)
                self.calls+=1
                p=np.zeros((49,3),np.float32)
                k=np.zeros(7,np.float32);k[0]=1.
                return p,k
        engines=[]
        for mode in ("reference","incremental"):
            obj=object.__new__(S834S809Native256)
            obj.native_heads=DummyNative()
            obj.native_feat=deque(maxlen=65)
            obj.native_post=deque(maxlen=65)
            obj.native_k=deque(maxlen=65)
            obj.temporal=DummyTemporal()
            obj.decoder=SimpleNamespace(policy=SimpleNamespace(blend=.55))
            obj.audio=np.zeros(256,np.float32)
            obj.six=SixIndependentAxes()
            obj.sharp_engine=mode
            obj.sharp_cache=IncrementalSharp256() if mode=="incremental" else None
            obj.model_calls=0
            obj.model_call_ms=deque(maxlen=10)
            obj.last_decision=0
            obj.source_k=[]
            obj.source_post=[]
            obj.decisions={}
            obj.selection={}
            end,decisions,selections,snapshot=obj._step(256,np.zeros((49,126),np.float32))
            self.assertEqual(end,256)
            self.assertEqual(obj.temporal.calls,1)
            self.assertEqual(obj.model_calls,1)
            self.assertEqual(set(decisions),set(AXES))
            self.assertEqual(obj.s836_last_input_288d.shape,(49,288))
            engines.append((decisions,selections,obj.s836_last_input_288d.copy()))
        self.assertEqual(engines[0][0],engines[1][0])
        self.assertEqual(engines[0][1],engines[1][1])
        np.testing.assert_array_equal(engines[0][2],engines[1][2])


if __name__=="__main__":
    unittest.main()
