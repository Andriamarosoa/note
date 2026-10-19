"""S766 causal timer safety: source priority, immutable NoteOn, future invariance."""
import unittest
import numpy as np

from causal_note.strict_live import SR
from scripts.s766_causal_extra_note_timer_cal32 import (
    BLOCK,CAP_BLOCKS,GRID,RISKS,timer_limited_pairs,emit)
from scripts.s765_fit_independent_arbiter import DIM


def event(kind,pitch,block,reason):
    sample=block*BLOCK
    return dict(type=kind,pitch=pitch,available_sample=sample,
                decision_sample=sample,time=sample/SR,
                velocity=96 if kind=="note_on" else 0,reason=reason)


class AlwaysOne:
    classes_=np.array([0,1])
    def predict_proba(self,X):
        return np.stack([np.zeros(len(X)),np.ones(len(X))],axis=-1)


class TimerIsolationTests(unittest.TestCase):
    def example(self):
        source=[
            event("note_on",40,1,"s708_original_on_40"),
            event("note_off",40,2,"s707_original_off_40"),
            event("note_on",52,10,"s708_original_on_52"),
            event("note_off",52,13,"s707_original_off_52"),
        ]
        candidate=[(
            event("note_on",52,3,"s765_proposed_birth"),
            event("note_off",52,25,"s765_future_causal_release"),
        )]
        X=np.zeros((1,DIM),np.float32)
        k=np.zeros((40,7),np.float32);k[:,1]=1.
        indices=[(2,12)]
        heads=dict(onset=AlwaysOne(),complete=AlwaysOne())
        return source,candidate,X,indices,k,heads

    def test_sixteen_cap_duration_is_not_accepted(self):
        a,b,c,d,e,f=self.example()
        with self.assertRaises(ValueError):
            timer_limited_pairs(b,99)
        self.assertEqual(len(GRID),12)
        self.assertEqual(len(RISKS),3)
        self.assertEqual(len(CAP_BLOCKS),4)

    def test_short_timer_closes_extra_before_signed_future_onset(self):
        src,pairs,X,indices,k,heads=self.example()
        keep=[e.copy() for e in src]
        output,stats=emit(src,pairs,X,indices,k,heads,risk=.5,cap_blocks=4)
        original=[x for x in output if x["reason"].startswith(
            ("s708_original_","s707_original_"))]
        self.assertEqual(original,src)
        self.assertEqual(src,keep)
        self.assertEqual(stats["accepted"],1)
        self.assertEqual(stats["timer_capped"],1)
        self.assertEqual(stats["preempted"],0)
        timed=[x for x in output if x["reason"]=="s766_causal_timer_release"]
        self.assertEqual(len(timed),1)
        self.assertEqual(timed[0]["available_sample"],7*BLOCK)

    def test_long_duration_preempted_by_prioritized_signed_source(self):
        src,pairs,X,indices,k,heads=self.example()
        output,stats=emit(src,pairs,X,indices,k,heads,risk=.5,cap_blocks=32)
        self.assertEqual(stats["preempted"],1)
        self.assertEqual(stats["accepted"],1)
        self.assertEqual(stats["timer_capped"],0)
        originals=[e for e in output if e["reason"].startswith(
            ("s708_original_","s707_original_"))]
        self.assertEqual(originals,src)

    def test_note_on_never_depends_on_candidate_future_off(self):
        src,pairs,X,indices,k,heads=self.example()
        alt=[(pairs[0][0].copy(),event(
            "note_off",52,39,"different_future_noteoff"))]
        for cap in CAP_BLOCKS:
            a,_=timer_limited_pairs(pairs,cap)
            b,_=timer_limited_pairs(alt,cap)
            self.assertEqual(a[0][0],b[0][0])
            self.assertEqual(a[0][0]["available_sample"],3*BLOCK)

    def test_early_causal_release_precedes_timer(self):
        src,pairs,X,indices,k,heads=self.example()
        original_end=event("note_off",52,5,"s765_causal_release")
        result,stats=timer_limited_pairs([(pairs[0][0],original_end)],4)
        self.assertEqual(stats["timer_capped"],0)
        self.assertEqual(result[0][1],original_end)

    def test_invalid_candidate_end_rejected(self):
        src,pairs,X,indices,k,heads=self.example()
        invalid=event("note_off",52,3,"invalid_non_positive")
        with self.assertRaises(AssertionError):
            timer_limited_pairs([(pairs[0][0],invalid)],4)

if __name__=="__main__":
    unittest.main()
