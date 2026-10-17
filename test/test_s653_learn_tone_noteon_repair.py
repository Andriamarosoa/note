"""S653 Tone head training loop regression, causality and original-K tests."""
import unittest
import numpy as np

from causal_note.strict_live import Selection,CausalSelection,HOP
from scripts.s653_learn_tone_noteon_repair import (
    FEATURE_VERSION,mask_candidates,feature_for_pitch,propose,
    training_candidates,models,project,fit_training,MODEL_NAMES,
)

class ToneNoteOnRepairTests(unittest.TestCase):
    def setUp(self):
        self.cfg=Selection()
        self.a=np.zeros((49,3),np.float32)
        self.b=np.zeros_like(self.a)
        self.c=np.zeros_like(self.a)
        self.k=np.zeros(7,np.float32);self.k[1]=1.

    def test_filtered_views_propose_only_missing_pitch_candidates(self):
        i=0
        self.a[i,:2]=(.15,.2)
        self.b[i,:2]=(.8,.82)
        ids=mask_candidates(self.a,self.b,self.c,self.cfg)
        self.assertEqual(list(ids),[i])
        self.a[i,0]=.88
        self.assertEqual(list(mask_candidates(self.a,self.b,self.c,self.cfg)),[])
        self.a[i,0]=.15
        self.b[i,:2]=(.3,.1)
        self.assertEqual(list(mask_candidates(self.a,self.b,self.c,self.cfg)),[])

    def test_Tone_candidate_features_have_no_annotation_or_future(self):
        self.a[0,0]=.2;self.b[0,:2]=(.72,.82)
        f=feature_for_pitch(self.a,self.b,self.c,self.k,self.k,self.k,
                            self.a.copy(),self.b.copy(),self.c.copy(),0)
        self.assertEqual(f.shape,(49,))
        self.assertTrue(np.isfinite(f).all())
        self.assertEqual(project(f[None,:],"K_only").shape[0],1)
        self.assertEqual(set(models()),set(MODEL_NAMES))

    def test_original_posterior_and_original_ExactK_unchanged(self):
        self.a[0,:2]=(.20,.30);self.b[0,:2]=(.87,.75)
        saved=self.a.copy()
        state=CausalSelection(self.cfg)
        ids=mask_candidates(self.a,self.b,self.c,self.cfg)
        enhanced,accepted=propose(self.a,self.b,self.c,self.cfg,ids,[True],state)
        self.assertEqual(accepted,1)
        np.testing.assert_array_equal(self.a,saved)
        self.assertGreaterEqual(float(enhanced[0,0]),self.cfg.onset)
        self.assertEqual(float(enhanced[2,0]),float(self.a[2,0]))
        events=state.step(enhanced,self.k,HOP,HOP)
        self.assertEqual([e["pitch"] for e in events if e["type"]=="note_on"],[40])

    def test_group_CV_training_choices_only_train_labels(self):
        rng=np.random.default_rng(653)
        records=[]
        for comp in range(9):
            x=rng.standard_normal((70,49)).astype(np.float32)
            y=np.zeros(70,np.int8);y[:7]=1
            x[:7,2]+=3.
            records.append(dict(x=x,y=y,composition=f"COMP{comp}"))
        fitted,cv,winner,selected,counts=fit_training(records)
        self.assertEqual(counts["n"],630)
        self.assertEqual(counts["positive"],63)
        self.assertIn(winner,MODEL_NAMES)
        self.assertIn(selected,(*MODEL_NAMES,"S614_original"))
        self.assertEqual(set(fitted),set(MODEL_NAMES))
        for policy in cv.values():
            self.assertIn(policy["best"]["threshold"],
                          (.50,.60,.70,.75,.80,.85,.90,.93,.96,.98,1.01))

if __name__=="__main__":
    unittest.main()
