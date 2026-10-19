"""S753 causal head-vs-emitted MIDI K error partition.

Synthetic examples are not experimental CAL scores. Each test proves
one independent diagnostic invariant with exact known outcomes.
"""
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR
from scripts.s753_khead_decoder_anatomy import (
    decision_anatomy,aggregate_decision_audit)

def ev(kind,pitch,end):
    return dict(type=kind,pitch=pitch,available_sample=int(end),
        decision_sample=int(end),time=float(end/SR),
        reason="synthetic",velocity=96 if kind=="note_on" else 0)

class KHeadDecoderTests(unittest.TestCase):
    def setUp(self):
        self.ends=np.arange(1,13,dtype=np.int64)*4*HOP
        self.on=self.ends[1];self.off=self.ends[9]
        self.truth=[dict(pitch=p,time=self.on/SR,
            duration=(self.off-self.on)/SR) for p in (40,52)]
        self.single=[ev("note_on",40,self.on),ev("note_off",40,self.off)]
        self.k=np.zeros((len(self.ends),7),np.float32)
        self.k[:,2]=1.0

    def test_enough_head_but_under_emitted(self):
        r=decision_anatomy(self.ends,self.k,self.single,self.truth)
        p=r["decisions_by_true_k"]["2"]
        self.assertEqual(p["frames"],8)
        self.assertEqual(p["model_k_exact"],8)
        self.assertEqual(p["emitted_k_under"],8)
        self.assertEqual(p["head_enough_but_midi_under"],8)
        self.assertEqual(p["both_under"],0)
        report=aggregate_decision_audit([r,r])
        self.assertEqual(report["per_true_k"]["2"]["head_enough_but_midi_under"],16)
        self.assertEqual(report["per_true_k"]["2"]["model_k_exact_rate"],1)

    def test_head_under_and_decoder_under_partition(self):
        self.k[:]=0;self.k[:,1]=1.
        row=decision_anatomy(self.ends,self.k,self.single,self.truth)[
            "decisions_by_true_k"]["2"]
        self.assertEqual(row["emitted_k_under"],row["both_under"])
        self.assertEqual(row["head_enough_but_midi_under"],0)
        self.assertEqual(row["model_k_under"],8)

    def test_head_under_when_decoder_correct_is_counted(self):
        self.k[:]=0;self.k[:,1]=1.
        double=[ev("note_on",40,self.on),ev("note_on",52,self.on),
                ev("note_off",40,self.off),ev("note_off",52,self.off)]
        row=decision_anatomy(self.ends,self.k,double,self.truth)[
            "decisions_by_true_k"]["2"]
        self.assertEqual(row["emitted_k_exact"],8)
        self.assertEqual(row["head_under_but_midi_exact"],8)

    def test_truth_k7_clip_matches_signed_K6_protocol(self):
        self.truth=[dict(pitch=p,time=self.on/SR,
            duration=(self.off-self.on)/SR) for p in (40,41,42,43,44,45,46)]
        notes=[ev("note_on",p,self.on) for p in range(40,46)]
        offs=[ev("note_off",p,self.off) for p in range(40,46)]
        self.k[:]=0;self.k[:,6]=1
        res=decision_anatomy(self.ends,self.k,notes+offs,self.truth)
        self.assertEqual(res["true_gt6_decisions"],8)
        self.assertEqual(res["decisions_by_true_k"]["6"]["emitted_k_exact"],8)

    def test_k_distribution_is_probability_not_unscaled_logits(self):
        self.k[:,2]=3.
        with self.assertRaises(ValueError):
            decision_anatomy(self.ends,self.k,self.single,self.truth)

    def test_future_events_do_not_change_prefix(self):
        left=decision_anatomy(self.ends,self.k,self.single,self.truth)
        later=self.single+[ev("note_on",64,int(self.ends[-1])+1024),
                            ev("note_off",64,int(self.ends[-1])+2048)]
        right=decision_anatomy(self.ends,self.k,later,self.truth)
        self.assertEqual(left,right)

    def test_missing_noteoff_rejected(self):
        with self.assertRaises(AssertionError):
            decision_anatomy(self.ends,self.k,
                [ev("note_off",40,self.on)],self.truth)

if __name__=="__main__":
    unittest.main()
