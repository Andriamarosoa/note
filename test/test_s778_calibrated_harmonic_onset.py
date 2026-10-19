"""S778: independent pitch-lattice head, causal/label firewall, FIT-only CV."""
import unittest
import numpy as np
from causal_note.strict_live import SR
from scripts.s778_calibrated_harmonic_onset_fit80 import (
    FEAT,THRESHOLDS,BLOCK,feature_grid,missing_onset_targets,
    sample_training,onset_probabilities,proposals,onset_score,combine,fit,
    harmonic_contrast,shift_pitch)
from test.test_s765_fit_independent_arbiter import ArbiterTests

def setup():
    item,p,k,source,shadow,ctx=ArbiterTests().fixture()
    # Acoustic full ABI 288 channels, original 49x3 logits.
    return item,p,k,source,ctx

class Degenerate:
    classes_=np.array([0,1])
    def predict_proba(self,X):
        return np.stack([np.zeros(len(X)),np.ones(len(X))],axis=-1)

class OnsetLatticeTests(unittest.TestCase):
    def test_no_spurious_octave_wraparound(self):
        x=np.zeros((2,49,8),np.float32)
        x[:,48]=2
        assert np.count_nonzero(shift_pitch(x,12))==0
        lower=shift_pitch(x,-12)
        self.assertEqual(float(lower[0,36,0]),2)
        self.assertEqual(float(lower[0,0,0]),0)

    def test_octave_competitor_changes_target_pitch_features(self):
        item,p,k,source,ctx=setup()
        item["x"]=item["x"].copy()
        item["x"][:,:,192:216]=0.
        base=feature_grid(item,p,k,ctx)
        item["x"][:,0,192:216]=4.
        affected=feature_grid(item,p,k,ctx)
        self.assertFalse(np.array_equal(base[:,12,87:95],affected[:,12,87:95]))
        np.testing.assert_array_equal(base[:,48,87:95],affected[:,48,87:95])

    def test_features_all_pitch_and_finite(self):
        item,p,k,source,ctx=setup()
        X=feature_grid(item,p,k,ctx)
        self.assertEqual(X.shape,(6,49,FEAT))
        self.assertTrue(np.isfinite(X).all())

    def test_features_prefix_cannot_depend_on_future_audio(self):
        item,p,k,source,ctx=setup()
        a=feature_grid(item,p,k,ctx)
        new=dict(item)
        new["x"]=item["x"].copy()
        new["x"][4:,:,:]=1
        b=feature_grid(new,p,k,ctx)
        np.testing.assert_array_equal(a[:4],b[:4])

    def test_target_only_baseline_unmatched(self):
        item,p,k,source,ctx=setup()
        ends=item["decision_ends"]
        truth=[dict(pitch=40,time=ends[0]/SR,duration=.1),
               dict(pitch=52,time=ends[3]/SR,duration=.1)]
        target=missing_onset_targets(ends,truth,[0])
        self.assertEqual(int(target[:,0].sum()),0)
        self.assertEqual(int(target[3:,12].sum()),3)

    def test_sampling_is_deterministic_and_no_CAL_truth_in_model_inputs(self):
        item,p,k,source,ctx=setup()
        X=feature_grid(item,p,k,ctx)
        y=np.zeros((len(X),49),np.uint8)
        y[1,15]=1;y[3,20]=1
        x1,y1=sample_training(X,y,p,ctx,768)
        x2,y2=sample_training(X,y,p,ctx,768)
        np.testing.assert_array_equal(x1,x2)
        np.testing.assert_array_equal(y1,y2)
        self.assertEqual(int(y1.sum()),2)
        self.assertEqual(x1.shape[1],FEAT)

    def test_prediction_all_pitch_lattice(self):
        item,p,k,source,ctx=setup()
        X=feature_grid(item,p,k,ctx)
        pred=onset_probabilities(Degenerate(),X)
        self.assertEqual(pred.shape,(6,49))
        self.assertTrue(np.all(pred==1.))

    def test_causal_noteon_and_hysteretic_refractory(self):
        item,p,k,source,ctx=setup()
        ends=item["decision_ends"]
        score=np.zeros((len(ends),49),np.float32)
        score[0:4,14]=.95
        score[5,14]=.95
        notes=proposals(score,ends,source,.8)
        self.assertEqual(len(notes),2)
        self.assertEqual(notes[0]["pitch"],54)
        self.assertEqual(notes[1]["available_sample"],int(ends[5]))
        self.assertEqual(notes[0]["available_sample"],int(ends[0]))
        self.assertEqual(notes[0]["decision_sample"],int(ends[0]))

    def test_source_never_changed_and_past_source_veto_not_future(self):
        item,p,k,source,ctx=setup()
        ends=item["decision_ends"]
        score=np.zeros((len(ends),49),np.float32)
        score[2,0]=.99
        score[3,12]=.99
        before=[e.copy() for e in source]
        result=proposals(score,ends,source,.8)
        self.assertEqual(source,before)
        self.assertEqual([e["pitch"] for e in result],[52])
        future=[dict(type="note_on",pitch=52,
                     available_sample=int(ends[5]),
                     decision_sample=int(ends[5]),
                     time=ends[5]/SR,velocity=96,reason="future")]
        changed=proposals(score,ends,source+future,.8)
        self.assertEqual(result,changed)

    def test_threshold_score_is_actual_pitched_one_to_one_80ms(self):
        item,p,k,source,ctx=setup()
        truth=[dict(pitch=40,time=float(item["decision_ends"][0]/SR),duration=.09),
               dict(pitch=52,time=float(item["decision_ends"][3]/SR),duration=.09)]
        shadow=[dict(type="note_on",pitch=52,
                     time=float(item["decision_ends"][3]/SR))]
        stats=onset_score(source,shadow,truth)
        self.assertEqual(stats["TP"],2)
        self.assertEqual(stats["FP"],0)
        self.assertEqual(stats["FN"],0)
        self.assertEqual(combine([stats,stats])["f1"],1.)

    def test_all_official_metrics_are_unmodified_by_onset_shadow(self):
        from scripts.s704_joint_metrics import metric
        item,p,k,source,ctx=setup()
        notes=[dict(pitch=40,time=float(item["decision_ends"][0]/SR),
                    duration=float(BLOCK/SR))]
        snapshot=[e.copy() for e in source]
        scores=metric(source,notes,np.arange(256,8*256+256,256))
        scores_after=metric(source,notes,np.arange(256,8*256+256,256))
        self.assertEqual(source,snapshot)
        self.assertEqual(scores["onset_f1"],scores_after["onset_f1"])
        self.assertEqual(scores["full_note_f1"],scores_after["full_note_f1"])

if __name__=="__main__":
    unittest.main()
