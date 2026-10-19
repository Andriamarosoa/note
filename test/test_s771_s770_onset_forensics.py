import unittest
from scripts.s771_s770_onset_forensics import tp_needed,fp_allowed,audit

class TestS771(unittest.TestCase):
    def test_strict_80_integer_bounds(self):
        self.assertEqual(tp_needed(791,4083),3250)
        self.assertEqual(fp_allowed(2887,4083),247)
        self.assertEqual(791-fp_allowed(2887,4083),544)

    def test_no_cal_metric_is_promotion(self):
        # Synthetic miniature model report with 65 registered combinations.
        grid=[]
        for q in (0.,.2,.3,.4,.5):
            grid.append((q,None,0.))
            for b in (.6,.7,.8,.9):
                for rise in (0.,.025,.1):
                    grid.append((q,b,rise))
        self.assertEqual(len(grid),65)
        sc=lambda tp,fp: dict(TP=tp,FP=fp,FN=4083-tp,
            f1=2*tp/(tp+fp+4083),precision=tp/(tp+fp),recall=tp/4083)
        rows={str(c):sc(2913,849) for c in grid}
        base={'onset_f1':rows[str(grid[0])]['f1'],'exact_k':.6204,
              'active_exact_k':.5627,'offset_accuracy':.7732,
              'full_note_f1':.5713,'frame_pitch_f1':.7646,
              'tp':2913,'fp':849,'fn':1170,'frames':126304,'active_frames':100243}
        x=dict(schema="S770_GROUPED_JOINT_ONSET_SHADOW",
            fit_recordings_done=80,cal_recordings_done=32,promoted=False,
            source_midi_immutable=True,ONSET_only_not_MIDI=True,
            TEST_opened=False,HELD_opened=False,
            GRID=[str(k) for k in grid],
            FIT=dict(onset_oof=rows,folds={str(i):{} for i in range(5)},
                     selected_frozen=str(grid[0])),
            CAL=dict(onset_shadow_grid=rows,selected_frozen=str(grid[0]),
                     native_onset=sc(2913,849)),
            source={k:base[k] for k in ("onset_f1","exact_k",
              "active_exact_k","offset_accuracy","full_note_f1",
              "frame_pitch_f1")})
        y=audit(x,{"frozen_candidate":{"CAL":base}})
        self.assertEqual(y['number_of_combinations'],65)
        self.assertTrue(y['signed_source_parity_valid'])
        self.assertFalse(y['promoted'])
        self.assertEqual(y['true_onsets'],4083)

if __name__=="__main__":
    unittest.main()
