"""S794 semantic tests of real MIDI256 Active Exact-K root-cause taxonomy."""
import unittest
import numpy as np

from scripts.s794_activek_failure_taxonomy import (
    diagnostic,prior_audio_lookups,summarize,REGISTERS
)
from causal_note.strict_live import SR

def event(kind,pitch,t):
    return dict(type=kind,pitch=pitch,decision_sample=t,
        available_sample=t,time=t/SR,reason="test",velocity=0)

class S794Tests(unittest.TestCase):
    def fixture(self):
        ends=np.array([256,512,768,1024,1280,1536,1792,2048,
                       2304,2560,2816,3072],np.int64)
        decisions=np.array([1024,2048,3072],np.int64)
        events=[
            event("note_on",40,1024),
            event("note_off",40,2048)]
        truth=[dict(pitch=40,time=1024/SR,duration=(2304-1024)/SR),
               dict(pitch=43,time=1280/SR,duration=(2304-1280)/SR)]
        post=np.zeros((3,49,3),np.float32)
        post[:,3,0]=.55
        post[:,3,1]=.60
        post[:,0,0]=.82
        post[:,0,1]=.75
        item=dict(ends=ends,decision_ends=decisions,samples=3072)
        return dict(item=item,post=post,events=events,truth=truth)

    def test_audio_use_never_advances_to_future_1024(self):
        ends=np.array([256,512,1024,1280,1536,2048],np.int64)
        d=np.array([1024,2048],np.int64)
        at,good=prior_audio_lookups(ends,d)
        np.testing.assert_array_equal(at,[-1,-1,0,0,0,1])
        np.testing.assert_array_equal(good,[False,False,True,True,True,True])

    def test_active_k_is_based_on_emitted_midi_not_k_argmax(self):
        row=diagnostic(self.fixture())
        self.assertEqual(row["original_counts"]["frames"],12)
        self.assertGreater(row["under_active_frames"],0)
        self.assertEqual(row["source_event_identity"]["native_events_checked"],2)
        self.assertEqual(row["future_audio_frames_used"],0)
        self.assertEqual(row["additional_scheduling_samples"],0)

    def test_missing_voice_and_low_register_cues(self):
        r=diagnostic(self.fixture())
        self.assertGreater(r["missed_pitch_frames"],0)
        self.assertGreater(r["expert_opportunity"]["bass_fundamentals"]
                ["missed_pitch_frames_with_cue"],0)
        self.assertGreater(r["expert_opportunity"]["poly_new_voice"]
                ["undercount_frames_with_missing_pitch_cue"],0)

    def test_per_k_conservation_and_summation(self):
        r=diagnostic(self.fixture())
        total=summarize([r])
        self.assertEqual(total["active_frames"],
            total["under_active_frames"]+total["over_active_frames"]+
            total["active_correct"])
        self.assertAlmostEqual(total["active_exact_k"],
                               r["scores"]["active_exact_k"])
        self.assertEqual(total["by_true_k"]["0"]["frames"],12-total["active_frames"])
        self.assertEqual(sum(v["frames"] for v in total["by_true_k"].values()),12)

    def test_future_acoustic_mutation_cannot_affect_earlier_frames(self):
        a=self.fixture()
        b=self.fixture()
        b["post"][1:,:,:]=.99
        da=diagnostic(a)
        db=diagnostic(b)
        self.assertEqual(da["scores"],db["scores"])
        self.assertEqual(da["original_counts"],db["original_counts"])
        # Current/future cue evidence can differ; MIDI source metrics cannot.
        self.assertEqual(da["future_audio_frames_used"],0)
        self.assertEqual(db["future_audio_frames_used"],0)

    def test_registers_partition_all_49_pitches(self):
        self.assertEqual(sum(hi-lo for lo,hi in REGISTERS.values()),49)
        self.assertEqual(len(REGISTERS),3)

if __name__=="__main__":
    unittest.main()
