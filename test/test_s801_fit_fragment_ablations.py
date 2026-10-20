"""S801 deterministic, prefix-causal masks + complete MIDI256 fixture checks."""
import unittest
import numpy as np
from scripts.s797_fragment_loop_fixture import example
from scripts.s801_fit_fragment_ablations import POLICIES,route,safe_all_six
from causal_note.s795_activek_experts import HEADS,decode
from causal_note.s797_fragment_router import CUT

class S801Ablations(unittest.TestCase):
    def test_all_policies_preserve_native_midi_zero_hold(self):
        d,X,P,original=example()
        for policy in POLICIES:
            with self.subTest(policy=policy):
                p,m,meta=route(X,P,policy)
                out,c=decode(original,d,X,p,0 if policy=="identity" else 31,
                             0. if policy=="identity" else CUT,3072)
                native=[(e["type"],e["pitch"],e["available_sample"])
                        for e in out if "s793_source_id" in e]
                expected=[(e["type"],e["pitch"],e["available_sample"])
                          for e in original]
                self.assertEqual(native,expected)
                self.assertEqual(c["max_added_native_delay_samples"],0)
                self.assertEqual(c["extra_hold_samples"],0)
                self.assertEqual(c["new_onsets"],c["new_offsets"])
                self.assertEqual(m.shape,(3,49))
                self.assertEqual(set(p),set(HEADS))
    def test_real_five_modes_and_guard_no_future_dependent(self):
        d,X,P,_=example()
        for policy in POLICIES:
            pp,mask,_=route(X,P,policy)
            altered=X.copy();altered[2:]=0.7
            alternate={h:v.copy() for h,v in P.items()}
            for h in HEADS:alternate[h][2:]=.77
            qq,mask2,_=route(altered,alternate,policy)
            self.assertTrue(np.array_equal(mask[:2],mask2[:2]))
            for h in HEADS:self.assertTrue(np.array_equal(pp[h][:2],qq[h][:2]))
    def test_no_regression_selection_rejects_any_axis(self):
        names=("onset_f1","exact_k","active_exact_k",
               "offset_accuracy","full_note_f1","frame_pitch_f1")
        b={k:.55 for k in names}
        self.assertTrue(safe_all_six(b,dict(b)))
        for k in names:
            candidate=dict(b);candidate[k]=.549
            self.assertFalse(safe_all_six(b,candidate))
    def test_protect_harmonic_parent_and_birth_offset(self):
        d,X,P,orig=example()
        routed,_,_=route(X,P,"harmonic_parent_guard")
        self.assertEqual(float(routed["harmonic"][0,52-40]),0.)
        gated,_,_=route(X,P,"offset_birth_guard")
        self.assertEqual(float(gated["general"][0,42-40]),.98)
    def test_undeclared_policy_rejected(self):
        d,X,P,_=example()
        with self.assertRaises(ValueError):route(X,P,"optimised_on_CAL")
if __name__=="__main__":unittest.main()
