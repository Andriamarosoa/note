"""S618 FN-first criteria and causal threshold controlled-replay regression tests."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import Selection, HOP
from scripts.s618_missed_attack_sweep import (
    score_events, causes_of_misses, CONFIGS,
)
from scripts.s613_train_strict_live import event_score


class MissedAttackAuditTests(unittest.TestCase):
    def test_only_groundtruth_audit_classifies_head_miss(self):
        ends=np.array([256,512,768],dtype=np.int64)
        p=np.zeros((3,49,3),dtype=np.float32)
        p[:,0,0]=[.15,.45,.59]
        p[:,0,1]=.81
        truth=[dict(pitch=40,time=512/44100,duration=.6)]
        got=causes_of_misses(ends,p,truth,{0})
        self.assertEqual(got["causes"],{"onset_head_below_threshold":1})
        p[1,0,0]=.79
        p[1,0,1]=.15
        p[2,0,1]=.78
        got=causes_of_misses(ends,p,truth,{0})
        self.assertEqual(got["causes"],{"onset_sustain_not_simultaneously_ready":1})

    def test_reducing_onset_rescues_real_synthetic_event_without_future(self):
        ends=np.arange(1,9,dtype=np.int64)*HOP
        p=np.zeros((len(ends),49,3),dtype=np.float32)
        p[2,0,:2]=[.50,.85]
        k=np.zeros((len(ends),7),dtype=np.float32);k[:,1]=1.
        source=Selection(onset=.6,active_on=.5)
        baseline=score_events(ends,p,k,source,ends[-1])
        rescued=score_events(
            ends,p,k,Selection(onset=.45,active_on=.5),ends[-1])
        truth=[dict(pitch=40,time=3*HOP/44100,duration=.2)]
        baseline_score,_=event_score(baseline,truth)
        saved_score,_=event_score(rescued,truth)
        self.assertEqual(baseline_score["FN"],1)
        self.assertEqual(saved_score["TP"],1)
        event=next(e for e in rescued if e["type"]=="note_on")
        self.assertEqual(event["decision_sample"],3*HOP)
        self.assertEqual(event["available_sample"],3*HOP)

    def test_no_silent_change_to_release_or_refractory(self):
        original=Selection(onset=.6,active_on=.5)
        self.assertEqual(CONFIGS[0],("reference",.6,.5))
        self.assertTrue(all(c[1]<=original.onset and c[2]<=original.active_on for c in CONFIGS))
        self.assertEqual(len({c[0] for c in CONFIGS}),len(CONFIGS))


if __name__=="__main__":
    unittest.main()
