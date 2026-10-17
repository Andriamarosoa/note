"""S658 tests independent protocol, overlap stats and promotion veto."""
import unittest
from scripts.s658_natural_octave_tone_gate import (
    POLICIES,aggregate,choose_by_TRAIN,CALIB_GAIN_MIN,HELD_GAIN_MIN,
)

class NaturalGuitarSetToneGateTests(unittest.TestCase):
    @staticmethod
    def fake(comp,original=(7,3),other=(8,2)):
        stats={}
        for name in POLICIES:
            tp,fp=original if name=="S614_original" else other
            stats[name]=dict(score=dict(TP=tp,FP=fp,FN=10-tp,
                                        latency_ms=[]),
                             overlap_TP=tp,K2_TP=tp,K3plus_TP=tp,
                             E2_TP=0,E3_TP=tp,
                             new_onsets_vs_S614=max(0,tp-original[0]),
                             lost_onsets_vs_S614=max(0,original[0]-tp))
        return dict(composition=comp,stats=stats)

    def test_stats_for_all_original_and_conditional_selections(self):
        rows=[self.fake("A"),self.fake("B")]
        out=aggregate(rows)
        self.assertEqual(set(out),set(POLICIES))
        self.assertEqual(out["S614_original"]["score"]["TP"],14)
        self.assertEqual(out["gate_last4_050"]["score"]["TP"],16)
        self.assertEqual(out["gate_last4_050"]["score"]["FP"],4)

    def test_selection_must_confirm_on_distinct_held_compositions(self):
        calib=[self.fake(x) for x in "ABCDE"]
        held=[self.fake(x,other=(5,6)) for x in "FGHI"]
        selected,details=choose_by_TRAIN(calib,held)
        self.assertEqual(selected,"S614_original")
        self.assertEqual(len(details["held_compositions"]),4)
        self.assertEqual(details["held_success_compositions"],0)

    def test_reference_original_is_never_removed(self):
        self.assertEqual(POLICIES[0],"S614_original")
        self.assertEqual(len(POLICIES),6)
        self.assertGreater(CALIB_GAIN_MIN,0)
        self.assertGreater(HELD_GAIN_MIN,0)

if __name__=="__main__":
    unittest.main()
