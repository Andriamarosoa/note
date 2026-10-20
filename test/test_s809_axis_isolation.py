"""S809: attacks on each of SIX independent heads, selectors and decoders.

The official S708 metric is NOT the isolated axis's SHADOW metric. This test
proves component-state non-interference; it cannot claim musical F1 gains.
"""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.s763_independent_decoders import (
    AXES,Fragment,DECISION_SAMPLES,AxisDecision,ExactKDecoder,
    OnsetDecoder,OffsetDecoder,ActiveKDecoder,FramePitchDecoder,FullNoteDecoder
)
from causal_note.s809_isolated_axis_pipeline import (
    HEAD_NAMES,SELECTOR_NAMES,AxisHeadBank,AxisSelector,AxisPipeline,
    SixIndependentAxes,default_pipelines,SCHEMA
)

def demo():
    result=[]
    for j in range(1,9):
        p=np.zeros((49,3),np.float32)
        p[0,1]=.75 if j<=3 else .06
        p[12,1]=.85 if j<=5 else .03
        p[0,0]=.98 if j in (1,4,7) else .18
        p[12,0]=.97 if j in (1,3,6) else .09
        p[0,2]=.98 if j in (4,5) else .05
        p[12,2]=.91 if j>=6 else .03
        k=np.zeros(7,np.float32);k[min(6,j%4)]=1.
        result.append(Fragment(j*DECISION_SAMPLES,p,k))
    return result

class MaliciousHead(AxisHeadBank):
    def propose(self,fragment):
        # Try to poison caller input and simultaneously return divergent data.
        fragment.post.setflags(write=True)
        fragment.post[:]=1.
        a,b=super().propose(fragment)
        return a,b

class ForcedSelector(AxisSelector):
    def __init__(self,axis):
        super().__init__(axis,mode="causal_contrast",threshold=0.)

class BadClockDecoder:
    def __init__(self,axis):self.axis=axis
    def step(self,fragment):
        return AxisDecision(self.axis,fragment.sample+DECISION_SAMPLES)

class ConstantKDecoder(ExactKDecoder):
    def step(self,f):
        return AxisDecision("exact_k",f.sample,count=0)

class IndependentAxesTests(unittest.TestCase):
    def compare(self,axis,mutated,require_difference=False):
        a=SixIndependentAxes()
        b=SixIndependentAxes(mutated)
        diffs=0
        for frame in demo():
            left,meta=a.step(frame)
            right,other=b.step(frame)
            self.assertEqual(set(left),set(AXES))
            for keep in AXES:
                if keep==axis:
                    diffs+=int(left[keep]!=right[keep])
                else:
                    self.assertEqual(left[keep],right[keep],
                       "changing "+axis+" unexpectedly mutated "+keep)
                    self.assertEqual(meta[keep],other[keep],
                       "changing "+axis+" modified foreign selector "+keep)
        if require_difference:self.assertGreater(diffs,0)
    def test_distinct_head_selector_decoder_per_axis(self):
        b=SixIndependentAxes()
        self.assertEqual(set(b.pipelines),set(AXES))
        self.assertEqual(len({id(p.head) for p in b.pipelines.values()}),6)
        self.assertEqual(len({id(p.selector) for p in b.pipelines.values()}),6)
        self.assertEqual(len({id(p.decoder) for p in b.pipelines.values()}),6)
        self.assertEqual(len(set(SELECTOR_NAMES.values())),6)
        self.assertEqual(len({x for y in HEAD_NAMES.values() for x in y}),12)
    def test_each_of_six_independent_selectors_can_change_in_isolation(self):
        for axis in AXES:
            pipes=default_pipelines()
            pipes[axis].selector=ForcedSelector(axis)
            self.compare(axis,pipes)
    def test_each_of_six_heads_can_be_mutated_without_foreign_damage(self):
        for axis in AXES:
            pipes=default_pipelines()
            pipes[axis].head=MaliciousHead(axis)
            self.compare(axis,pipes)
    def test_independent_decoder_can_change_one_axis_only(self):
        pipes=default_pipelines()
        pipes["exact_k"].decoder=ConstantKDecoder()
        self.compare("exact_k",pipes,require_difference=True)
        replacements={
            "active_exact_k":ActiveKDecoder(on=.99,off=.98),
            "onset":OnsetDecoder(threshold=1.,sustain=1.),
            "offset":OffsetDecoder(release=1.,low_sustain=0.),
            "frame_pitch":FramePitchDecoder(sustain=1.),
            "full_note":FullNoteDecoder(attack=1.,sustain=1.)
        }
        for axis,decoder in replacements.items():
            pipes=default_pipelines()
            pipes[axis].decoder=decoder
            self.compare(axis,pipes)
    def test_execution_order_invariance(self):
        a=SixIndependentAxes(order=AXES)
        b=SixIndependentAxes(order=tuple(reversed(AXES)))
        for f in demo():
            self.assertEqual(a.step(f),b.step(f))
    def test_future_change_cannot_rewrite_prefix(self):
        seq=demo();other=demo()
        post=np.ones((49,3),np.float32)
        k=np.zeros(7,np.float32);k[6]=1.
        other[-1]=Fragment(other[-1].sample,post,k)
        a=SixIndependentAxes()
        b=SixIndependentAxes()
        for f,g in zip(seq[:-1],other[:-1]):
            self.assertEqual(a.step(f),b.step(g))
    def test_illegal_shared_component_rejected(self):
        for component in ("head","selector","decoder"):
            p=default_pipelines()
            setattr(p["onset"],component,getattr(p["offset"],component))
            with self.assertRaises(ValueError):
                SixIndependentAxes(p)
    def test_foreign_head_selector_crosswire_rejected(self):
        with self.assertRaises(ValueError):
            AxisPipeline("onset",head=AxisHeadBank("offset"))
        with self.assertRaises(ValueError):
            AxisPipeline("onset",selector=AxisSelector("offset"))
    def test_invalid_foreign_or_future_event_rejected(self):
        p=default_pipelines()
        p["onset"].decoder=BadClockDecoder("onset")
        with self.assertRaises(AssertionError):
            SixIndependentAxes(p).step(demo()[0])
    def test_official_source_events_preserved_bitwise(self):
        original=[{"type":"note_on","pitch":40,"available_sample":1024,
          "decision_sample":1024,"time":1024/44100,"reason":"S708"},
          {"type":"note_off","pitch":40,"available_sample":2048,
          "decision_sample":2048,"time":2048/44100,"reason":"S707"}]
        copy=SixIndependentAxes.signed_midi_identity(original)
        self.assertEqual(original,copy)
        self.assertIsNot(original[0],copy[0])
        copy[0]["pitch"]=53
        self.assertEqual(original[0]["pitch"],40)

if __name__=="__main__":
    unittest.main()
