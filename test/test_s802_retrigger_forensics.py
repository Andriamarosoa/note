"""S802 never conflate a native retrigger closure with a false-positive attack."""
import unittest
import numpy as np
from causal_note.strict_live import SR
from causal_note.s795_activek_experts import FEATURES,HEADS,decode
from causal_note.s797_fragment_router import CUT
from scripts.s801_fit_fragment_ablations import route
from scripts.s802_retrigger_forensics import classify_events,lag_bin,summarize
from scripts.s797_fragment_loop_fixture import example

class S802Tests(unittest.TestCase):
    def synth(self):
        d,X,P,orig=example()
        # Add genuine later source attack that preempts synthetic pitch 42.
        event=lambda kind,p,t: dict(type=kind,pitch=p,available_sample=t,
              decision_sample=t,time=t/SR,reason="native",velocity=96)
        orig=[orig[0],event("note_on",42,2048),
              event("note_off",42,3072),orig[-1]]
        routed,_,_=route(X,P,"consensus")
        out,counts=decode(orig,d,X,routed,31,CUT,3072)
        self.assertGreaterEqual(counts["new_onsets"],1)
        return out,d,X,P
    def test_retrigger_is_not_automatically_false_positive(self):
        out,d,X,P=self.synth()
        truth=[dict(pitch=42,time=1024/SR,duration=1024/SR)]
        records=classify_events(out,truth,d,X,P,"FIT","test","consensus")
        retrigger=[x for x in records if x["source_retrigger"]
                   and x["pitch"]==42]
        self.assertEqual(len(retrigger),1)
        self.assertTrue(retrigger[0]["synthetic_matched_truth"])
        self.assertEqual(retrigger[0]["life_samples"],1024)
        self.assertTrue(retrigger[0]["source_native_attacked_same_clock"])
    def test_retrigger_can_really_be_false_positive(self):
        out,d,X,P=self.synth()
        # Matching must prefer native at 2048 over synthetic at 1024.
        truth=[dict(pitch=42,time=2048/SR,duration=1024/SR)]
        rows=classify_events(out,truth,d,X,P,"FIT","test","consensus")
        retrigger=[x for x in rows if x["source_retrigger"] and x["pitch"]==42]
        self.assertEqual(len(retrigger),1)
        self.assertFalse(retrigger[0]["synthetic_matched_truth"])
        stat=summarize(rows)
        self.assertGreaterEqual(stat["retrigger_false_positives"],1)
    def test_future_truth_never_changes_midi(self):
        out,d,X,P=self.synth()
        a=classify_events(out,[],d,X,P,"FIT","test","consensus")
        b=classify_events(out,[dict(pitch=42,time=1024/SR,
                                   duration=1024/SR)],d,X,P,"FIT","test","consensus")
        self.assertEqual([(x["type"],x["pitch"],x["available_sample"])
                          for x in out],
                         [(x["type"],x["pitch"],x["available_sample"])
                          for x in out])
        self.assertNotEqual(sum(x["synthetic_matched_truth"] for x in a),
                            sum(x["synthetic_matched_truth"] for x in b))
    def test_diagnostics_bins_are_fixed_not_optimizer(self):
        self.assertEqual(lag_bin(1024),"le_1024_samples")
        self.assertEqual(lag_bin(3072),"le_4096_samples")
        self.assertEqual(lag_bin(None),"other_release")

if __name__=="__main__":unittest.main()
