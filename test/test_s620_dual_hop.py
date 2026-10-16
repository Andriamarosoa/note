"""S620 invariants: past-only dual-hops, full reference parity and MIDI leases.

Synthetic tests are NOT evidence of real-guitar F1 or independent exact-K.
"""
from __future__ import annotations

import unittest
import numpy as np

from causal_note.dual_hop import (
    DualHopConfig, DualHopLiveStream, DualSourceEventMerger,
    fast_candidate_mask, novelty_gated_fast_heads,
)
from causal_note.strict_live import HOP, StrictLiveStream
from scripts.s620_dualhop_eval import replay


class Head:
    metadata={"selection":{"onset":.6,"active_on":.5}}
    def predict(self, f):
        p=np.zeros((49,3),np.float32)
        p[12,0]=.8
        p[12,1]=.8
        k=np.zeros((7,),np.float32);k[1]=1
        return p,k


def event_key(e):
    return e["type"],e["pitch"],e["decision_sample"],e["reason"]


class S620Tests(unittest.TestCase):
    def test_model_reference_events_unchanged_in_off_and_shadow(self):
        audio=np.random.default_rng(620).normal(0,.03,15130).astype(np.float32)
        for block in (128,256,333,2048,8192):
            orig=StrictLiveStream(Head())
            off=DualHopLiveStream(Head(),config=DualHopConfig(mode="off"))
            shadow=DualHopLiveStream(Head(),config=DualHopConfig(mode="shadow"))
            outputs=[[],[],[]]
            for i in range(0,len(audio),block):
                b=audio[i:i+block]
                outputs[0].extend(orig.push(b,start_sample=i))
                outputs[1].extend(off.push(b,start_sample=i))
                outputs[2].extend(shadow.push(b,start_sample=i))
            for a,stream in zip(outputs,(orig,off,shadow)):
                a.extend(stream.finish())
            reference=[event_key(e) for e in outputs[0]]
            self.assertEqual(reference,[event_key(e) for e in outputs[1]])
            self.assertEqual(reference,[event_key(e) for e in outputs[2]])
            self.assertTrue(all(e["available_sample"]>=e["decision_sample"]
                                for e in outputs[2]))
            self.assertLessEqual(len(shadow.fast_front.tail),8192)

    def test_merger_single_audible_note_even_when_both_sources_own_it(self):
        m=DualSourceEventMerger()
        def e(typ,t):
            return dict(type=typ,pitch=52,decision_sample=t,
                        available_sample=t,time=t/44100,
                        reason="test",velocity=96 if typ=="note_on" else 0)
        self.assertEqual(len(m.step("fast",e("note_on",128))),1)
        self.assertEqual(m.step("reference",e("note_on",256)),[])
        self.assertEqual(m.step("fast",e("note_off",640)),[])
        self.assertEqual(len(m.step("reference",e("note_off",768))),1)
        self.assertFalse(m.current[52-40])
        self.assertEqual(m.suppressed_duplicates,1)

    def test_guard_never_leaks_future_audio_into_previous_decisions(self):
        rng=np.random.default_rng(2020)
        base=rng.normal(0,.05,48*HOP).astype(np.float32)
        mutated=base.copy()
        mutated[24*HOP:]=rng.normal(0,.9,len(base)-24*HOP)
        for size in (128,256,512,2048):
            outputs=[]
            for track in (base,mutated):
                s=DualHopLiveStream(Head(),config=DualHopConfig(mode="guard"))
                e=[]
                for i in range(0,len(track),size):
                    e.extend(s.push(track[i:i+size],start_sample=i))
                e.extend(s.finish())
                outputs.append([event_key(z) for z in e
                                if z["decision_sample"]<=24*HOP])
            self.assertEqual(*outputs)

    def test_fast_pitch_novelty_filters_false_siblings(self):
        x=np.zeros((49,126),np.float32)
        p=np.zeros((49,3),np.float32)
        p[12,:2]=[.54,.8]
        p[20,:2]=[.62,.8]
        x[12,60]=.8
        x[20,60]=.02
        cfg=DualHopConfig(mode="guard",fast_onset=.5,novelty_ratio=.5,
                          novelty_min=.04)
        filt,mask=novelty_gated_fast_heads(x,p,cfg)
        self.assertTrue(mask[12])
        self.assertFalse(mask[20])
        self.assertEqual(float(filt[20,0]),0.)
        self.assertEqual(float(filt[12,0]),float(p[12,0]))

    def test_eval_replay_matches_transport_for_fake_heads(self):
        # Build EXACT same two frontends once, and compare reference/fast
        # selector replay against transport mode using identical tensors.
        from causal_note.strict_live import CausalPitchFeatures, SR
        rng=np.random.default_rng(615)
        audio=rng.normal(0,.07,8192).astype(np.float32)
        f128=CausalPitchFeatures(hop=128)
        f256=CausalPitchFeatures(hop=256)
        e128=[];x128=[];e256=[];x256=[]
        for i in range(0,len(audio),256):
            for t,x in f128.push(audio[i:i+256],start_sample=i):
                e128.append(t);x128.append(x)
            for t,x in f256.push(audio[i:i+256],start_sample=i):
                e256.append(t);x256.append(x)
        x128=np.stack(x128);x256=np.stack(x256)
        model=Head()
        p128=[];k128=[];p256=[];k256=[]
        for x in x128:
            p,k=model.predict(x);p128.append(p);k128.append(k)
        for x in x256:
            p,k=model.predict(x);p256.append(p);k256.append(k)
        dev=dict(end128=np.array(e128),end256=np.array(e256),
                 fast=np.stack(p128),kfast=np.stack(k128),
                 slow=np.stack(p256),kslow=np.stack(k256),frames=len(audio),
                 novelty={
                     "lag":np.maximum(x128[:,:,60:63],0).max(2),
                     "recent":np.maximum(x128[:,:,114:117],0).max(2)})
        for mode in ("off","shadow","guard"):
            cfg=DualHopConfig(mode=mode)
            stream=DualHopLiveStream(model,config=cfg)
            produced=[]
            for i in range(0,len(audio),128):
                produced+=stream.push(audio[i:i+128],start_sample=i)
            produced+=stream.finish()
            expected,_=replay(dev,model,cfg)
            self.assertEqual([event_key(e) for e in expected],
                             [event_key(e) for e in produced])


if __name__=="__main__":
    unittest.main()
