"""S623 two-cadence evidence must never cause a second MIDI owner.

All synthetic tests protect causal computation but do not validate musical
F1 or true polyphonic Exact-K on independent songs.
"""
import unittest
import numpy as np

from causal_note.strict_live import Selection
from scripts.s623_fused_evidence import (
    EvidenceConfig, select_fast_evidence, run_one,
)


class FusedEvidenceTests(unittest.TestCase):
    def demo(self):
        e128=np.array([128,256,384,512],np.int64)
        e256=np.array([256,512],np.int64)
        p128=np.zeros((4,49,3),np.float32)
        p256=np.zeros((2,49,3),np.float32)
        k=np.zeros((2,7),np.float32);k[:,1]=1.
        novelty=np.zeros((4,49),np.float32)
        p128[0,12,:2]=[.8,.8]
        p128[1,12,:2]=[.9,.8]
        novelty[0,12]=1.
        novelty[1,12]=1.
        p256[0,12,:2]=[.48,.82]
        return dict(e128=e128,e256=e256,p128=p128,p256=p256,
                    k256=k,nov128=novelty,
                    nov256=novelty[[1,3]],frames=512)

    def test_fast_evidence_needs_same_pitch_attack_and_novelty(self):
        rec=self.demo()
        cfg=EvidenceConfig(mode="both",slow_weak_onset=.4,
                           fast_min_onset=.7,fast_novelty_ratio=.7)
        mask=select_fast_evidence(rec["p128"][:2],rec["nov128"][:2],cfg)
        self.assertEqual(np.flatnonzero(mask).tolist(),[12])
        rec["nov128"][1,12]=0
        other=select_fast_evidence(rec["p128"][:2],rec["nov128"][:2],cfg)
        self.assertFalse(other.any())

    def test_fast_context_boosts_single_weak_reference_note(self):
        rec=self.demo()
        selection=Selection(onset=.6,active_on=.5)
        original,info=run_one(rec,selection=selection)
        self.assertEqual([e for e in original if e["type"]=="note_on"],[])
        cfg=EvidenceConfig(mode="both",slow_weak_onset=.4,
                           fast_min_onset=.7,fast_novelty_ratio=.5)
        combined,meta=run_one(rec,cfg,selection=selection)
        on=[e for e in combined if e["type"]=="note_on"]
        self.assertEqual(len(on),1)
        self.assertEqual(on[0]["pitch"],52)
        self.assertEqual(on[0]["decision_sample"],256)
        self.assertEqual(on[0]["available_sample"],256)
        self.assertEqual(meta["promoted_pitch_frames"],1)
        self.assertEqual(len([e for e in combined if e["type"]=="note_off"]),1)

    def test_original_strong_attack_is_unchanged(self):
        rec=self.demo()
        rec["p256"][0,12,0]=.8
        cfg=EvidenceConfig(mode="either",slow_weak_onset=.4,
                           fast_min_onset=.4,fast_novelty_ratio=.5)
        original,_=run_one(rec,selection=Selection(onset=.6,active_on=.5))
        dual,meta=run_one(rec,cfg,selection=Selection(onset=.6,active_on=.5))
        self.assertEqual(original,dual)
        self.assertEqual(meta["promoted_pitch_frames"],0)

    def test_no_earlier_event_affected_by_mutating_future_frame(self):
        rec=self.demo()
        cfg=EvidenceConfig(mode="either",slow_weak_onset=.4,
                           fast_min_onset=.5,fast_novelty_ratio=.5)
        events,_=run_one(rec,cfg,selection=Selection(onset=.6,active_on=.5))
        rec["p128"][2:]=.99
        rec["nov128"][2:]=99
        rec["p256"][1]=.99
        changed,_=run_one(rec,cfg,selection=Selection(onset=.6,active_on=.5))
        t=256
        self.assertEqual(
            [(e["type"],e["pitch"],e["decision_sample"]) for e in events
             if e["decision_sample"]<=t],
            [(e["type"],e["pitch"],e["decision_sample"]) for e in changed
             if e["decision_sample"]<=t])


if __name__=="__main__":
    unittest.main()
