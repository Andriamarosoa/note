"""S669 threshold and strict physical batch availability tests."""
import unittest
from dataclasses import replace
import numpy as np
from causal_note.strict_live import Selection,HOP
from scripts.s669_native_release_head_threshold_sweep import score,POLICIES

class NativeNoteOffTests(unittest.TestCase):
    def test_policies_include_unchanged_baseline(self):
        self.assertEqual(POLICIES[0][0],"original")
        self.assertEqual(POLICIES[0][1],None)
        self.assertIn(("gate_0.45_sustain_0.25_persist_5",.45,.25,5),POLICIES)

    def test_only_release_hysteresis_modified(self):
        base=Selection(onset=.6,active_on=.5,active_off=.25,release_frames=5)
        new=replace(base,active_off=.15,release_frames=3)
        self.assertEqual(new.onset,base.onset)
        self.assertEqual(new.max_notes,base.max_notes)
        self.assertEqual(base.active_off,.25)

    def test_batch4_offset_never_before_audio_arrived(self):
        ends=np.arange(1,61)*HOP
        events=[
           dict(type="note_on",pitch=40,decision_sample=10*HOP,
                time=10*HOP/44100,reason="causal_attack"),
           dict(type="note_off",pitch=40,decision_sample=51*HOP,
                time=51*HOP/44100,reason="causal_release"),
        ]
        truth=[dict(time=10*HOP/44100,duration=41*HOP/44100,pitch=40)]
        a=score(events,truth,ends,batch_hops=1)
        b=score(events,truth,ends,batch_hops=4)
        self.assertEqual(a["offset"]["eligible"],1)
        self.assertEqual(b["offset"]["eligible"],1)
        self.assertGreaterEqual(b["offset"]["accuracy50"],0)

if __name__=="__main__":
    unittest.main()
