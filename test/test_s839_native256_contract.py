"""Regression guards against relabeling S708 1024-trained weights as PCM256."""
import copy
import unittest

import numpy as np

from causal_note.s839_native256_contract import (
    CONTRACT, HOP, Native256Violation, Exact256Stream,
    provenance_from_observed_training, verify_native256,
)


class StrictNativePCM256Tests(unittest.TestCase):
    def fresh(self):
        return provenance_from_observed_training(
            observed_train_shapes=[(8,1,256),(2,1,256)],
            actual_step_samples=256)

    def test_new_training_data_and_inference_both_256(self):
        p=self.fresh()
        self.assertEqual(p["train_input_samples"],256)
        self.assertEqual(p["train_hop_samples"],256)
        self.assertEqual(p["inference_input_samples"],256)
        self.assertEqual(p["inference_hop_samples"],256)
        verify_native256(p,live_input_samples=256,model_tensor_samples=256)

    def test_old_s708_1024_training_is_blocked_even_if_replayed_at_256(self):
        p=self.fresh()
        p["train_hop_samples"]=1024
        with self.assertRaises(Native256Violation):
            verify_native256(p,live_input_samples=256)
        p=self.fresh()
        p["pretraining_source_hop_samples"]=1024
        with self.assertRaises(Native256Violation):
            verify_native256(p)
        with self.assertRaises(Native256Violation):
            provenance_from_observed_training(
                observed_train_shapes=[(16,1,256)],
                actual_step_samples=256,
                teacher_training_hop_samples=1024)

    def test_fake_metadata_or_large_windows_rejected(self):
        p=self.fresh()
        for key,value in (
            ("largest_direct_pcm_window_samples",1024),
            ("train_input_samples",1024),
            ("inference_hop_samples",128),
            ("lookahead_samples",1),
            ("added_wait_samples",256),
            ("uses_frozen_1024_teacher",True),
            ("trained_from_random_initialization",False),
            ("target_axis","exact_k"),
        ):
            bad=copy.deepcopy(p);bad[key]=value
            with self.subTest(key=key),self.assertRaises(Native256Violation):
                verify_native256(bad)
        bad=self.fresh()
        del bad["train_input_samples"]
        with self.assertRaises(Native256Violation):
            verify_native256(bad)

    def test_training_observations_reject_larger_or_tiny_blocks(self):
        for n in (128,512,1024):
            with self.subTest(n=n),self.assertRaises(Native256Violation):
                provenance_from_observed_training(
                    observed_train_shapes=[(8,1,n)],
                    actual_step_samples=256)
        with self.assertRaises(Native256Violation):
            provenance_from_observed_training(
                observed_train_shapes=[(8,1,256)],
                actual_step_samples=1024)
        with self.assertRaises(Native256Violation):
            provenance_from_observed_training(
                observed_train_shapes=[],
                actual_step_samples=256)
        with self.assertRaises(Native256Violation):
            provenance_from_observed_training(
                observed_train_shapes=[(8,1,256)],
                actual_step_samples=256,pretrained=True)

    def test_stream_requires_one_consecutive_pcm256_and_no_wait(self):
        n=[]
        def predictor(block):
            n.append(block.copy())
            return {"onset":float(block.max())}
        stream=Exact256Stream(predictor,self.fresh())
        result=stream.step(np.zeros(256,np.float32),start_sample=0)
        self.assertEqual(result[0],256)
        self.assertEqual(len(n),1)
        with self.assertRaises(Native256Violation):
            stream.step(np.ones(1024,np.float32),start_sample=256)
        with self.assertRaises(Native256Violation):
            stream.step(np.ones(256,np.float32),start_sample=512)
        self.assertEqual(len(n),1)
        self.assertEqual(stream.step(np.ones(256,np.float32),
                                    start_sample=256)[0],512)
        self.assertEqual(len(n),2)


if __name__=="__main__":
    unittest.main()
