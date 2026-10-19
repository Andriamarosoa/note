"""S789 contracts: gaps, F1 reconstruction, six roles, sampling stability."""
import unittest
from scripts.s789_deep_data_microscope import (
    f1,ratio,wilson,min_correct_to_exceed_80,min_recovered_f1,
    track_texture,ROLES,aggregate,bootstrap_track_offset)

class S789MicroscopeTests(unittest.TestCase):
    def test_target_is_strictly_over_80(self):
        self.assertEqual(min_correct_to_exceed_80(2241,2898),78)
        self.assertEqual(min_correct_to_exceed_80(78366,126304),22678)
        self.assertEqual(min_correct_to_exceed_80(56413,100243),23782)
        self.assertGreater(min_recovered_f1(2913,849,1170),0)

    def test_f1_and_missed_note_recovery(self):
        self.assertAlmostEqual(f1(2913,849,1170),.7426386233269598)
        recovered=min_recovered_f1(2913,849,1170)
        self.assertGreater(f1(2913+recovered,849,1170-recovered),.8)
        self.assertLessEqual(f1(2913+recovered-1,849,1170-recovered+1),.8)

    def test_six_independent_roles_mandatory(self):
        self.assertEqual(set(ROLES),
            {"data","acoustics","architecture","training","metrics","live"})
        self.assertEqual(track_texture("04_BN2-119-G_comp"),"comp")
        self.assertEqual(track_texture("04_BN2-119-G_solo"),"solo")
        with self.assertRaises(AssertionError):
            track_texture("unknown")

    def test_wilson_and_bootstrap_track(self):
        left,right=wilson(70,100)
        self.assertLess(left,.7)
        self.assertGreater(right,.7)
        records=[]
        for i in range(5):
            m=dict(tp=5,fp=1,fn=1,onset_f1=f1(5,1,1),
                 offset_correct=9+i,offset_eligible=15,
                 offset_accuracy=(9+i)/15,
                 frame_pitch_tp=50,frame_pitch_fp=5,frame_pitch_fn=5,
                 frame_pitch_f1=f1(50,5,5),
                 k_correct=7,frames=10,exact_k=.7,
                 active_k_correct=5,active_frames=10,
                 active_exact_k=.5,full_note_f1=.3)
            records.append(dict(name=f"{i}_BN2_comp",metrics={"candidate":m},
                         composition="BN2"))
        result=aggregate(records)
        self.assertEqual(result["n_tracks"],5)
        ci=bootstrap_track_offset(records,789)
        self.assertLessEqual(ci[0],result["offset_accuracy"])
        self.assertGreaterEqual(ci[1],result["offset_accuracy"])

if __name__=="__main__":
    unittest.main()
