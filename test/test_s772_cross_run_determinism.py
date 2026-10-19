import unittest
from scripts.s772_cross_run_determinism import compare

class TestS772(unittest.TestCase):
    def test_identical_self_compare_remains_identical(self):
        d=dict(TP=9,FP=2,FN=1,f1=.85,precision=.8,recall=.9)
        row=dict(schema="S770_GROUPED_JOINT_ONSET_SHADOW",
                 GRID=[str(i) for i in range(65)],
                 fit_recordings_done=80,cal_recordings_done=32,
                 promoted=False,selected="S708_frozen",
                 FIT=dict(folds={str(i):None for i in range(5)},
                          counts=dict(train=100),
                          selected_frozen="0",
                          onset_oof={str(i):d.copy() for i in range(65)}),
                 CAL=dict(frozen_score=d.copy(),
                          onset_shadow_grid={str(i):d.copy() for i in range(65)}),
                 source={k:.8 for k in ("onset_f1","exact_k","active_exact_k",
                         "offset_accuracy","full_note_f1","frame_pitch_f1")},
                 source_reference_parity=dict(valid=True,delta={}))
        x=compare(row,row)
        self.assertEqual(x["cal_scores_changed"],0)
        self.assertEqual(x["fit_scores_changed"],0)
        self.assertTrue(x["frozen_config_identical"])
        self.assertFalse(x["promoted"])

    def test_one_candidate_diff_is_preserved_not_hidden(self):
        import copy
        d=dict(TP=9,FP=2,FN=1,f1=.85,precision=.8,recall=.9)
        row=dict(schema="S770_GROUPED_JOINT_ONSET_SHADOW",
                 GRID=[str(i) for i in range(65)],fit_recordings_done=80,
                 cal_recordings_done=32,promoted=False,selected="S708_frozen",
                 FIT=dict(folds={str(i):None for i in range(5)},
                          counts={},selected_frozen="0",
                          onset_oof={str(i):d.copy() for i in range(65)}),
                 CAL=dict(frozen_score=d.copy(),
                          onset_shadow_grid={str(i):d.copy() for i in range(65)}),
                 source={k:.8 for k in ("onset_f1","exact_k","active_exact_k",
                         "offset_accuracy","full_note_f1","frame_pitch_f1")},
                 source_reference_parity=dict(valid=True,delta={}))
        changed=copy.deepcopy(row)
        changed["CAL"]["onset_shadow_grid"]["1"]["FP"]=3
        x=compare(row,changed)
        self.assertEqual(x["cal_scores_changed"],1)
        self.assertEqual(x["exact_changed_cal"]["1"]["FP"],1)
        self.assertFalse(x["promoted"])

if __name__=="__main__":
    unittest.main()
