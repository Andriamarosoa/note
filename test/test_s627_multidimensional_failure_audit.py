"""Checks S627 truth-vs-pred labels, original feature ABI and matched control bins."""
import unittest
import numpy as np

from scripts.s627_multidimensional_failure_audit import (
    feature_view,build_context,interval_relation,stratified_rates,
    matched_controls,posthoc_head_evidence,selector_trace,
    frame_metrics,time_bin,
)


def note(p,t,d):
    return dict(pitch=p,time=t,duration=d)


class S627Tests(unittest.TestCase):
    def test_real_schema_not_s619_guess_for_2hop_temporal_lag(self):
        x=np.zeros(126,np.float32)
        x[0]=.2;x[1]=.7;x[8]=.5;x[16]=.9
        x[54:57]=[.3,.9,.1]  # actual 2-hop original score-delta
        x[60:63]=[.1,.2,.8]  # actual 4-hop delta, NOT 2
        x[114:117]=[.4,.5,.6]
        x[120:123]=[.3,.1,.2]
        x[123]=.7
        z=feature_view(x)
        self.assertAlmostEqual(z["lag_2hop_score_delta"],.9,places=5)
        self.assertAlmostEqual(z["lag_4hop_score_delta"],.8,places=5)
        self.assertAlmostEqual(z["recent_positive_novelty"],.6,places=5)
        self.assertAlmostEqual(z["spectral_flux_mean"],.2,places=5)
        self.assertAlmostEqual(z["short_rms_log"],.7,places=5)

    def test_near_onset_and_active_pitch_counts_are_independent(self):
        notes=[note(40,.10,.6),note(52,.12,.45),note(53,.23,.30)]
        earlier=build_context(notes,1)
        later=build_context(notes,2)
        self.assertEqual(earlier["onset_K_raw"],2)
        self.assertEqual(earlier["active_K_raw"],2)
        self.assertEqual(later["onset_K_raw"],1)
        self.assertEqual(later["active_K_raw"],3)
        self.assertEqual(earlier["coactive_interval"],"harmonic_interval_12_19_24")
        self.assertEqual(later["coactive_interval"],"close_1_2")

    def test_harmonic_interval_is_only_an_interval_label(self):
        self.assertEqual(interval_relation(52,[40]),"harmonic_interval_12_19_24")
        self.assertEqual(interval_relation(53,[52,41]),"close_1_2")
        self.assertEqual(interval_relation(40,[]),"none")

    def test_truth_label_no_collision_with_false_positives(self):
        rows=[
          dict(source="A",corpus="GuitarSet",composition="song1",
               pitch=40,active_K="2",matched=False,p_onset=.1,
               p_sustain=.1,p_release=.1,h1=.1,h2_to_h1=.3,
               h234_to_h1=.3,lag_2hop_score_delta=.1,
               lag_4hop_score_delta=.1,recent_positive_novelty=.1,
               short_rms_log=.1,K_confidence=.3),
          dict(source="A",corpus="GuitarSet",composition="song1",
               pitch=40,active_K="2",matched=True,p_onset=.8,
               p_sustain=.8,p_release=.2,h1=.8,h2_to_h1=.7,
               h234_to_h1=.8,lag_2hop_score_delta=.8,
               lag_4hop_score_delta=.8,recent_positive_novelty=.8,
               short_rms_log=.8,K_confidence=.8),
        ]
        for r in rows:r["duration_s"]=.25
        controls=matched_controls(rows)
        self.assertEqual(controls["paired_samples"],1)
        self.assertEqual(controls["match_level"]["same_source_pitch_K"],1)
        self.assertAlmostEqual(controls["metrics"]["p_onset"]["median_FN_minus_TP"],-.7)

    def test_denominators_and_low_source_support_are_reported(self):
        rows=[
          dict(corpus="GuitarSet",composition="one",matched=True),
          dict(corpus="GuitarSet",composition="one",matched=False),
          dict(corpus="URMP",composition="two",matched=False),
        ]
        out=stratified_rates(rows,["corpus"],min_cases=4)
        g=next(x for x in out["corpus"] if x["condition"]=="GuitarSet")
        self.assertEqual(g["total"],2)
        self.assertEqual(g["missed"],1)
        self.assertTrue(g["limited_evidence"])

    def test_finite_causal_selection_state_before_current_frame(self):
        from causal_note.strict_live import Selection,HOP
        ends=np.array([HOP,2*HOP],np.int64)
        pp=np.zeros((2,49,3),np.float32)
        pp[0,0,:2]=[.9,.9]
        kk=np.zeros((2,7),np.float32);kk[:,1]=1
        events,state=selector_trace(ends,pp,kk,Selection(),2*HOP)
        self.assertFalse(state["active"][0,0])
        self.assertTrue(state["active"][1,0])
        self.assertTrue(any(e["type"]=="note_on" and e["pitch"]==40 for e in events))


if __name__=="__main__":
    unittest.main()
