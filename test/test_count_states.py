import unittest
import numpy as np
from causal_note.count_states import (SilencePolicy, acoustic_evidence, target_state,
    encode_states, decode_classes, decode_probabilities, state_metrics, UNLABELLED)
from scripts.v273_silence_state_model import require_training_population


class CountStateTests(unittest.TestCase):
    def setUp(self):
        self.policy=SilencePolicy(-60,-50)
        self.audio=np.zeros(16000,np.float32)
        self.origin=6000

    def evidence(self):
        return acoustic_evidence(self.audio,self.origin,self.policy)

    def test_silence_is_separate_from_continuing_and_foreign_notes(self):
        evidence=self.evidence()
        self.assertEqual(target_state(0,evidence,0),(-1,True))
        self.assertEqual(target_state(0,evidence,1),(0,True))
        self.assertEqual(target_state(0,evidence,3),(0,True))
        self.assertEqual(target_state(1,evidence,0),(1,True))

    def test_audible_unannotated_sound_is_zero_not_silence(self):
        self.audio.fill(.02)
        self.assertEqual(target_state(0,self.evidence(),0),(0,True))

    def test_brief_peak_and_quiet_note_are_protected(self):
        self.audio[self.origin]=.004
        self.assertFalse(self.evidence()['quiet'])
        self.audio.fill(.0011)
        self.assertFalse(self.evidence()['quiet'])
        self.audio.fill(.00001)
        self.assertEqual(target_state(0,self.evidence(),1),(0,True))

    def test_padding_never_creates_silence_training_examples(self):
        evidence=acoustic_evidence(self.audio,0,self.policy)
        self.assertEqual(target_state(0,evidence,0),(UNLABELLED,False))
        self.assertEqual(target_state(0,evidence,1),(0,True))

    def test_original_support_cannot_read_future_or_extra_past(self):
        expected=self.evidence()
        self.audio[:self.origin-3100]=np.nan
        self.audio[self.origin+2788:]=np.nan
        self.assertEqual(expected,self.evidence())

    def test_boundary_sample_inside_support_is_observed(self):
        self.audio[self.origin+2787]=.1
        self.assertFalse(self.evidence()['quiet'])
        self.audio.fill(0)
        self.audio[self.origin-3100]=.1
        self.assertFalse(self.evidence()['quiet'])

    def test_class_codec_including_unsigned_input(self):
        states=np.arange(-1,7)
        np.testing.assert_array_equal(decode_classes(encode_states(states)),states)
        np.testing.assert_array_equal(decode_classes(np.array([0,1,7],np.uint8)),[-1,0,6])
        np.testing.assert_array_equal(decode_probabilities(np.eye(8)),states)
        for bad in ([-2],[7],[1.5],[True]):
            with self.assertRaises(ValueError):encode_states(bad)
        with self.assertRaises(ValueError):decode_probabilities(np.ones((2,7))/7)

    def test_metrics_do_not_treat_minus_one_as_a_negative_count(self):
        result=state_metrics(np.array([-1,0,2,3]),np.array([0,-1,2,-1]))
        self.assertEqual(result['state_exact'],.25)
        self.assertEqual(result['onset_count_exact'],.75)
        self.assertEqual(result['polyphonic']['exact'],.5)
        self.assertEqual(result['undercount'],1)
        self.assertEqual(result['overcount'],0)
        self.assertEqual(result['sounding_zero_as_silence'],1)
        self.assertEqual(result['silence_as_sounding_zero'],1)
        self.assertEqual(np.asarray(result['confusion_true_by_predicted']).shape,(8,8))

    def test_no_fake_training_without_a_silence_population(self):
        states=np.array([0,1,0,2]);mask=np.ones(4,bool)
        with self.assertRaisesRegex(ValueError,'silence'):
            require_training_population(states,mask,[0,1],[2,3])
        states=np.array([-1,0,-1,0,UNLABELLED]);mask=np.array([1,1,1,1,0],bool)
        fit,val=require_training_population(states,mask,[0,1,4],[2,3])
        np.testing.assert_array_equal(fit,[0,1]);np.testing.assert_array_equal(val,[2,3])


if __name__=='__main__':unittest.main()
