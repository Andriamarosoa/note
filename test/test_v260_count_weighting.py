import importlib.util
import unittest

import numpy as np

from scripts import train_v250_count_only as baseline
from scripts import train_v260_count_weighting as v


class WeightingTests(unittest.TestCase):
    def test_weighted_control_matches_v25(self):
        labels = np.array([0]*100 + [1]*20 + [2,3,4,5,6])
        np.testing.assert_array_equal(v.arm_weights('weighted',labels), baseline.class_weights(labels))

    def test_uniform_including_absent_classes(self):
        for labels in (np.array([0]*100+[1]), np.arange(7)):
            np.testing.assert_array_equal(v.arm_weights('uniform',labels),np.ones(7))

    def test_validation_uses_fit_table(self):
        fit = np.array([0]*100+[1,2,3,4,5,6])
        validation = np.array([6]*100+[0])
        table = v.arm_weights('weighted',fit)
        np.testing.assert_array_equal(table[validation],baseline.class_weights(fit)[validation])
        self.assertFalse(np.allclose(table,v.arm_weights('weighted',validation)))

    def test_unknown_arm_rejected(self):
        with self.assertRaises(ValueError):v.arm_weights('typo',np.arange(7))
        with self.assertRaises(ValueError):v.build_model('typo',v.SEED)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow runs in CI preflight')
class GraphTests(unittest.TestCase):
    def test_identical_initial_models_and_weighted_loss(self):
        import tensorflow as tf
        snapshots=[]
        labels=np.array([0,1,3,6],dtype=np.int32)
        for arm in v.ARMS:
            model=v.build_model(arm,v.SEED)
            snapshots.append(model.get_weights())
            rng=np.random.default_rng(260)
            inputs={t.name.split(':')[0]:rng.normal(size=(4,)+tuple(int(d) for d in t.shape[1:])).astype(np.float32) for t in model.inputs}
            inputs['candidate_mask'][:]=0
            inputs['candidate_mask'][:,:8]=1
            weight=v.arm_weights(arm,np.array([0]*100+[1,3,6]))[labels]
            probability=model(inputs,training=False).numpy()
            expected=float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(labels,probability).numpy()*weight))
            actual=float(model.test_on_batch(inputs,labels,sample_weight=weight))
            self.assertAlmostEqual(actual,expected,places=5)
            self.assertTrue(np.isfinite(model.train_on_batch(inputs,labels,sample_weight=weight)))
        self.assertEqual(len(snapshots[0]),len(snapshots[1]))
        for a,b in zip(*snapshots):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
