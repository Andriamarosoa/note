"""S640 verify mechanism labels, deterministic causal windows, no oracle leakage."""
from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import HOP,SR,Selection,StrictLiveStream,load_heads
from scripts.s616_live_dev_ablation import MODEL
from scripts.s640_frozen_common_misses_acoustic_audit import (
    preliminary_miss_classification,reproduce_native_s614,S614_SHA)


def frame(at,onset=.1,sustain=.1,co=False,armed=True,
          refractory=True,gated=False,ready=False):
    o=np.zeros(49,np.float32);s=np.zeros(49,np.float32)
    p=np.zeros(49,np.float32)
    o[12]=onset;s[12]=sustain;p[12]=.25
    flags=np.zeros(49,bool);flags[12]=co
    eligible=np.zeros(49,bool);eligible[12]=ready
    blocked=np.zeros(49,bool);blocked[12]=gated
    armed_arr=np.ones(49,bool);armed_arr[12]=armed
    refractory_arr=np.ones(49,bool);refractory_arr[12]=refractory
    return dict(end=round(at*SR),onset=o,sustain=s,co_positive=flags,
        ready=eligible,gated=blocked,armed=armed_arr,
        refractory_ok=refractory_arr,fundamental=p)


class S640DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.note=dict(pitch=52,start_s=.1)
        self.cfg=Selection()

    def reason(self,rows,outputs=()):
        return preliminary_miss_classification(self.note,rows,outputs,self.cfg)

    def test_inadequate_onset_and_sustain(self):
        result=self.reason([frame(.10,.21,.8),frame(.13,.55,.8)])
        self.assertEqual(result["category"],"no_sufficient_onset_after_true_onset")
        result=self.reason([frame(.12,.9,.1)])
        self.assertEqual(result["category"],"onset_present_sustain_not_simultaneous")

    def test_time_misalignment_is_separate(self):
        signal=[frame(.12,.1,.1)]
        self.assertEqual(self.reason(signal,[(.080,52)])["category"],
                         "premature_same_pitch_output")
        self.assertEqual(self.reason(signal,[(.211,52)])["category"],
                         "delayed_same_pitch_output")

    def test_observable_selector_gate_is_not_hypothetical_inference(self):
        signal=[frame(.12,.9,.9,co=True,armed=False)]
        self.assertEqual(self.reason(signal)["category"],"candidate_disarmed")
        signal=[frame(.12,.9,.9,co=True,refractory=False)]
        self.assertEqual(self.reason(signal)["category"],"refractory_gate")
        signal=[frame(.12,.9,.9,co=True,gated=True,ready=True)]
        self.assertEqual(self.reason(signal)["category"],"polyphony_capacity_gate")
        signal=[frame(.12,.9,.9,co=True,ready=True)]
        self.assertEqual(self.reason(signal)["category"],"ready_but_not_matched")

    def test_audio_before_true_onset_cannot_pretend_as_causal_recovery(self):
        signal=[frame(.075,.9,.9,co=True,ready=True),
                frame(.120,.10,.10)]
        result=self.reason(signal)
        self.assertEqual(result["category"],"no_sufficient_onset_after_true_onset")
        self.assertGreater(result["highest_onset_pre"],.8)

    def test_original_checkpoint_events_are_identical(self):
        heads=load_heads(MODEL,expected_sha256=S614_SHA)
        audio=np.zeros(HOP*24,np.int16)
        frames,ours=reproduce_native_s614(audio,heads)
        original=StrictLiveStream(heads)
        native=[]
        for start in range(0,len(audio),HOP):
            native.extend(original.push(audio[start:start+HOP].astype(np.float32)/32768.,
                                        start_sample=start))
        native.extend(original.finish())
        self.assertEqual(ours,native)
        self.assertEqual(len(frames),24)


if __name__=="__main__":unittest.main()
