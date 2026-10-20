"""S810 integrity tests: six fitted-head objects cannot share weights or outputs.

These tests do not claim that S810 scores are final S704 musical metrics.
"""
import unittest
import numpy as np
from causal_note.s763_independent_decoders import AXES
from causal_note.s810_trained_six_axis_shadow import (
    TrainedAxis, self_feature_dimension, count_features, FEATURE_COLUMNS,
)

class FakeModel:
    def __init__(self, n,labels):
        self.n=n
        self.classes_=np.asarray(labels)
    def predict(self,x):
        assert x.shape[1]==self.n
        return np.full(len(x),self.classes_[-1],dtype=np.uint8)
    def predict_proba(self,x):
        assert x.shape[1]==self.n
        return np.tile(np.array([[.1,.9]]),(len(x),1))

class SixIndependentFittedTests(unittest.TestCase):
    def test_every_axis_has_its_own_dimension_and_selector(self):
        self.assertEqual(len(AXES),6)
        self.assertEqual(self_feature_dimension("exact_k"),61)
        self.assertEqual(self_feature_dimension("active_exact_k"),61)
        for axis in AXES:
            expected=61 if axis in ("exact_k","active_exact_k") else len(FEATURE_COLUMNS[axis])
            self.assertEqual(self_feature_dimension(axis),expected)

    def test_private_fitted_predictions_independent(self):
        axes={}
        for axis in AXES:
            n=self_feature_dimension(axis)
            labels=[0,1,2,3] if axis=="full_note" else [0,1]
            axes[axis]=TrainedAxis(axis,FakeModel(n,labels),axis)
        x=np.zeros((4,49,23),np.float32)
        before={axis:head.decide(x.copy()) for axis,head in axes.items()}
        self.assertEqual(len({v.selector for v in before.values()}),6)
        axes["active_exact_k"].model=FakeModel(61,[0,4])
        after={axis:head.decide(x.copy()) for axis,head in axes.items()}
        for axis in AXES:
            if axis!="active_exact_k":
                np.testing.assert_array_equal(before[axis].values,after[axis].values)
            self.assertFalse(after[axis].values.flags.writeable)
        self.assertFalse(np.array_equal(before["active_exact_k"].values,
                                        after["active_exact_k"].values))

    def test_future_frame_cannot_rewrite_prediction_prefix(self):
        head=TrainedAxis("frame_pitch",FakeModel(20,[0,1]),"sha")
        x=np.zeros((4,49,23),np.float32)
        changed=x.copy();changed[3:]=1.
        a=head.decide(x).values
        b=head.decide(changed).values
        np.testing.assert_array_equal(a[:3],b[:3])

    def test_bad_acoustic_abi_rejected(self):
        with self.assertRaises(ValueError):
            count_features(np.zeros((4,40,23)))
        with self.assertRaises(ValueError):
            count_features(np.zeros((4,49,15)))

if __name__=="__main__":unittest.main()
