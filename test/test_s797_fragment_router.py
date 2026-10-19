"""S797 independent checks of per-fragment masks and source timing."""
import unittest
import numpy as np
from causal_note.s795_activek_experts import HEADS
from causal_note.s797_fragment_router import MODES,route_probabilities
from scripts.s797_fragment_loop_fixture import example,trial
from causal_note.s793_zero_added_latency import SOURCE_ID

class S797FragmentTests(unittest.TestCase):
    def test_five_ablation_loops_and_spurious_harmonic_guard(self):
        expected={"identity":set(),"consensus":{42,55,60,52},
                  "harmonic_veto":{42,55,60},
                  "poly_gap":{42,55,60},"adaptive":{42,55,60}}
        for mode in MODES:
            with self.subTest(mode=mode):
                attacks,mask,meta,events,changes,original=trial(mode)
                self.assertEqual(attacks,expected[mode])
                self.assertEqual(changes["new_onsets"],len(attacks))
                self.assertEqual(changes["new_offsets"],len(attacks))
                native=[e for e in events if e.get(SOURCE_ID) is not None]
                self.assertEqual([(e["type"],e["pitch"],e["available_sample"])
                    for e in native],[(e["type"],e["pitch"],e["available_sample"])
                    for e in original])
                self.assertEqual(changes["max_added_native_delay_samples"],0)
                self.assertEqual(meta["extra_lookahead_samples"],0)
                self.assertEqual(meta["extra_hold_samples"],0)

    def test_adaptive_uses_distinct_masks_at_same_frame(self):
        attacks,mask,meta,*_=trial("adaptive")
        self.assertEqual(attacks,{42,55,60})
        self.assertNotEqual(int(mask[0,42-40]),int(mask[0,55-40]))
        self.assertNotEqual(int(mask[0,55-40]),int(mask[0,60-40]))
        self.assertEqual(int(mask[0,52-40]),0)

    def test_future_audio_cannot_change_previous_fragment(self):
        _,X,P,_=example()
        a,m,_=route_probabilities(X,P,"adaptive")
        X2=X.copy();X2[1:]=.77
        P2={k:v.copy() for k,v in P.items()}
        for p in P2.values():p[1:]=.94
        b,n,_=route_probabilities(X2,P2,"adaptive")
        np.testing.assert_array_equal(m[:1],n[:1])
        for name in HEADS:np.testing.assert_array_equal(a[name][:1],b[name][:1])

    def test_invalid_and_nonnumeric_inputs_fail_closed(self):
        _,X,P,_=example()
        with self.assertRaises(ValueError):route_probabilities(X,P,"oracle")
        with self.assertRaises(ValueError):route_probabilities(X,P,cut=.85)
        with self.assertRaises(ValueError):route_probabilities(X,dict(P,general=np.full((3,49),2.)))
        with self.assertRaises(ValueError):route_probabilities(X[:,:,:-1],P)
        with self.assertRaises(ValueError):route_probabilities(X,{k:v for k,v in P.items() if k!="bass"})
        XX=X.copy();XX[0,0,0]=np.nan
        with self.assertRaises(ValueError):route_probabilities(XX,P)

if __name__=="__main__":unittest.main()
