"""S752 falsification suite for NOTE's six workstreams (A through F).

Only source-only synthetic WAV/MIDI and post-hoc audit are used.
No altered S708 predictions, no opened HELD/TEST and no LIVE promotion.
"""
from __future__ import annotations
import unittest

import numpy as np

from causal_note.strict_live import HOP,SR
from causal_note.s706_sharp_spectra import sharp_track
from causal_note.s708_cardinality_selection import GuidedSelector, GuidedPolicy
from causal_note.s712_s708_live import S708StrictLiveStream
from scripts.s704_joint_metrics import metric
from scripts.s748_six_axis_failure_audit import axis_audit,paired_audit


def midi(kind,pitch,when):
    return dict(type=kind,pitch=int(pitch),available_sample=int(when),
        decision_sample=int(when),time=float(when/SR),
        reason="synthetic_attack" if kind=="note_on" else "synthetic_release",
        velocity=96 if kind=="note_on" else 0)


class SixRoleIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,101,dtype=np.int64)*HOP
        self.on=int(self.ends[8])
        self.off=int(self.ends[70])

    def events(self,pitches):
        notes=[midi("note_on",p,self.on) for p in pitches]
        offs=[midi("note_off",p,self.off) for p in pitches]
        return notes+offs

    def truth(self,pitches):
        return [dict(pitch=p,time=self.on/SR,
                     duration=(self.off-self.on)/SR) for p in pitches]

    def test_agent_b_octave_above_false_alarm_is_not_missed(self):
        # S748 only counted a WRONG LOWER octave. Wrong higher octave
        # on E2 is just as important for a physical harmonic confound.
        a=axis_audit(self.events([52]),self.truth([40]),self.ends)
        self.assertGreater(a["harmonic_false_alarms_near_miss"]["12"],0)
        self.assertGreater(
            a["harmonic_false_alarms_by_direction"]["12"]["false_above"],0)
        self.assertEqual(
            a["harmonic_false_alarms_by_direction"]["12"]["false_below"],0)

    def test_agent_b_octave_below_false_alarm_still_detected(self):
        a=axis_audit(self.events([40]),self.truth([52]),self.ends)
        self.assertGreater(a["harmonic_false_alarms_near_miss"]["12"],0)
        self.assertGreater(
            a["harmonic_false_alarms_by_direction"]["12"]["false_below"],0)

    def test_agent_d_true_k7_matches_clipped_signed_exact_k(self):
        # Ground truth has seven simultaneous labels, but signed S704
        # clips both actual/predicted counts to K6 in Exact-K.
        events=self.events([40,41,42,43,44,45])
        truth=self.truth([40,41,42,43,44,45,46])
        a=axis_audit(events,truth,self.ends)
        m=metric(events,truth,self.ends)
        self.assertGreater(a["k_gt6_frames"],0)
        self.assertEqual(a["per_true_k"]["6"]["under"],0)
        self.assertEqual(a["per_true_k"]["6"]["exact"],
                         a["per_true_k"]["6"]["true_frames"])
        for k,v in a["source_metrics"].items():
            self.assertAlmostEqual(v,m[k],places=12)

    def test_agent_d_orphan_noteoff_fails_closed(self):
        with self.assertRaises(ValueError):
            axis_audit([midi("note_off",40,self.off)],[],self.ends)

    def test_agent_c_exact_k_cannot_hallucinate_pitch_without_onset(self):
        selector=GuidedSelector(GuidedPolicy())
        p=np.zeros((49,3),np.float32)
        k=np.zeros(7,np.float32)
        k[6]=1.
        events=selector.step(p,k,4*HOP)
        self.assertFalse([e for e in events if e["type"]=="note_on"])

    def test_agent_a_pcm_future_mutation_does_not_change_past_dsp(self):
        n=12000
        t=np.arange(n,dtype=np.float64)/SR
        audio=(.1*np.sin(2*np.pi*82.407*t)).astype(np.float32)
        changed=audio.copy()
        changed[5000:]=np.clip(changed[5000:]+.3,-1.,1.)
        ends=np.array([1024,2048,3072,4096],np.int64)
        left=sharp_track(audio,ends)
        right=sharp_track(changed,ends)
        np.testing.assert_array_equal(left,right)

    def test_agent_a_s708_live_discontinuity_rejected_before_inference(self):
        live=S708StrictLiveStream.__new__(S708StrictLiveStream)
        live.closed=False
        live.received=4*HOP
        with self.assertRaises(ValueError):
            live.push(np.zeros(HOP,np.float32),start_sample=4*HOP+1)

    def test_agent_f_false_onset_must_fail_nonregression(self):
        original=self.events([40])
        truth=self.truth([40])
        bad=self.events([40,52])
        compared=paired_audit(original,bad,truth,self.ends)
        self.assertFalse(compared["strict_six_nonregression"])
        self.assertLess(compared["delta"]["onset_f1"],0.)

if __name__=="__main__":
    unittest.main()
