"""S792 unit tests: prove causal MIDI event emission and six-score gate."""
import unittest
import numpy as np
from scripts.s792_framepitch_midi256_replay import (
    replay_delayed_noteoffs,good_candidate,HOLD_BLOCKS,SR)
from scripts.s704_joint_metrics import SCORES

def event(kind,pitch,time,reason="s707_causal_hazard"):
    return dict(type=kind,pitch=pitch,available_sample=time,
        decision_sample=time,time=time/SR,velocity=64 if kind=="note_on" else 0,
        reason=reason)

class S792RealMidiTests(unittest.TestCase):
    def case(self):
        ends=np.arange(1,8)*1024
        events=[event("note_on",40,1024),event("note_off",40,2048),
            event("note_on",41,4096),event("note_off",41,5120)]
        mask=np.zeros((len(ends),49),bool)
        mask[:3,0]=True
        return events,ends,mask,7168

    def test_zero_hold_is_exact_original_midi(self):
        events,ends,mask,limit=self.case()
        got,stats=replay_delayed_noteoffs(events,ends,mask,0,limit)
        self.assertEqual(got,events)
        self.assertEqual(sum(stats.values()),0)

    def test_hold_preserves_noteons_and_emits_later_noteoff(self):
        events,ends,mask,limit=self.case()
        got,stats=replay_delayed_noteoffs(events,ends,mask,4,limit)
        self.assertEqual([(x["type"],x["pitch"]) for x in got],
                         [(x["type"],x["pitch"]) for x in events])
        first_off=[e for e in got if e["pitch"]==40 and e["type"]=="note_off"][0]
        self.assertEqual(first_off["available_sample"],4096)
        self.assertEqual(first_off["reason"],"expert_off")
        self.assertEqual(stats["held"],1)
        self.assertEqual(stats["extended"],1)
        self.assertTrue(all(e["available_sample"]>=e["decision_sample"] for e in got))

    def test_future_mask_changes_cannot_change_emissions_before_that_future(self):
        ev,ends,mask,limit=self.case()
        a,stats=replay_delayed_noteoffs(ev,ends,mask,4,limit)
        altered=mask.copy()
        altered[4:,:]=~altered[4:,:]
        b,_=replay_delayed_noteoffs(ev,ends,altered,4,limit)
        self.assertEqual([e for e in a if e["available_sample"]<ends[4]],
                         [e for e in b if e["available_sample"]<ends[4]])

    def test_restrike_flushes_previous_note_before_new_source_on(self):
        ends=np.arange(1,8)*1024
        ev=[event("note_on",40,1024),event("note_off",40,2048),
            event("note_on",40,3072),event("note_off",40,4096)]
        mask=np.ones((len(ends),49),bool)
        out,stat=replay_delayed_noteoffs(ev,ends,mask,4,7168)
        self.assertEqual([x["type"] for x in out],
                         ["note_on","note_off","note_on","note_off"])
        self.assertEqual(out[1]["available_sample"],3072)
        self.assertEqual(stat["retrigger"],1)

    def test_source_six_voice_limit_stays_six(self):
        ends=np.arange(1,8)*1024
        ev=[event("note_on",p,1024) for p in range(40,46)]
        ev+=[event("note_off",40,2048),event("note_on",46,3072)]
        ev+=[event("note_off",p,5120) for p in range(41,47)]
        mask=np.zeros((len(ends),49),bool)
        mask[:,0]=True
        out,stats=replay_delayed_noteoffs(ev,ends,mask,4,7168)
        self.assertGreater(stats["capacity"],0)
        self.assertEqual(sum(e["type"]=="note_on" for e in out),7)
        self.assertEqual(sum(e["type"]=="note_off" for e in out),7)

    def test_no_wrong_clock_or_oracle_input(self):
        ev,ends,mask,limit=self.case()
        with self.assertRaises(ValueError):
            replay_delayed_noteoffs(ev,ends,mask[:3],2,limit)
        with self.assertRaises(ValueError):
            replay_delayed_noteoffs(ev,ends,mask,999,limit)

    def test_six_metric_guard_blocks_even_one_regression(self):
        base={k:.75 for k in SCORES}
        candidate=dict(base,frame_pitch_f1=.76)
        bybase={"A":base,"B":base}
        bytrial={"A":candidate,"B":candidate}
        self.assertTrue(good_candidate(base,candidate,bybase,bytrial))
        candidate=dict(candidate,onset_f1=.74999)
        self.assertFalse(good_candidate(base,candidate,bybase,bytrial))

if __name__=="__main__":
    unittest.main()
