import importlib.util
import unittest

import numpy as np

from scripts import train_v250_count_only as v


class CountProtocolTests(unittest.TestCase):
    def test_weights_are_normalized_on_fit_labels(self):
        fit=np.array([0,0,0,1,2,6])
        table=v.class_weights(fit)
        self.assertAlmostEqual(float(table[fit].mean()),1.,places=6)
        self.assertTrue(np.isfinite(table).all())
        self.assertGreater(table[6],table[0])

    def test_weights_do_not_use_validation_distribution(self):
        fit=np.array([0,0,1,2])
        before=v.class_weights(fit)
        validation=np.array([6]*1000)
        _=before[validation]
        np.testing.assert_array_equal(before,v.class_weights(fit))

    def context(self):
        return {'k':np.zeros(6),'outer_idx':np.array([4,5]),'meta_fit_idx':np.array([0,1]),
                'meta_val_idx':np.array([2,3]),'final_fit_idx':np.array([0,1,2,3])}

    def test_disjoint_partitions(self):
        self.assertEqual(len(v.validate_partitions(self.context())),4)

    def test_outer_leakage_is_rejected(self):
        ctx=self.context();ctx['final_fit_idx']=np.array([0,1,2,4])
        with self.assertRaisesRegex(RuntimeError,'leakage'):v.validate_partitions(ctx)

    def test_missing_fit_row_is_rejected(self):
        ctx=self.context();ctx['meta_fit_idx']=np.array([0])
        with self.assertRaisesRegex(RuntimeError,'cover'):v.validate_partitions(ctx)

    def test_cardinality_confusion_and_false_births(self):
        r=v.cardinality(np.array([0,0,2,3]),np.array([0,1,1,3]))
        self.assertEqual(r['k0_false_birth_rows'],1)
        self.assertEqual(r['poly_exact'],.5)
        self.assertEqual((r['under'],r['over']),(1,1))
        self.assertEqual(np.sum(r['confusion_true_by_predicted']),4)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow graph tests run in CI preflight')
class CountGraphTests(unittest.TestCase):
    def test_extreme_ordinal_distributions(self):
        import tensorflow as tf
        logits=np.full((7,6),-80.,dtype=np.float32)
        for k in range(7):logits[k,:k]=80.
        p=v.ordinal_probabilities(tf,tf.constant(logits)).numpy()
        np.testing.assert_allclose(p,np.eye(7),atol=1e-6)
        np.testing.assert_allclose(p.sum(1),1,atol=1e-6)

    def test_paired_trunks_and_finite_training(self):
        import tensorflow as tf
        snapshots=[]
        for arm in v.ARMS:
            model=v.build_model(arm,v.SEED)
            names={x.name for x in model.layers}
            self.assertNotIn('candidate_subset',names)
            self.assertNotIn('event_set',names)
            snapshots.append({layer.name:layer.get_weights() for layer in model.layers
                              if layer.weights and layer.name not in ('cardinality','v250_ordinal_logits')})
            rng=np.random.default_rng(250)
            inputs={t.name.split(':')[0]:rng.normal(size=(4,)+tuple(int(d) for d in t.shape[1:])).astype(np.float32) for t in model.inputs}
            inputs['candidate_mask'][:]=0;inputs['candidate_mask'][:,:8]=1
            labels=np.array([0,1,3,6],dtype=np.int32)
            before=model(inputs,training=False).numpy()
            np.testing.assert_allclose(before.sum(1),1,atol=1e-5)
            with tf.GradientTape() as tape:
                out=model(inputs,training=False)
                loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,out))
            grads=tape.gradient(loss,model.trainable_weights)
            self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in grads))
            self.assertGreater(float(tf.linalg.global_norm(grads)),0.)
            value=model.train_on_batch(inputs,labels)
            self.assertTrue(np.isfinite(value).all())
            self.assertEqual(model.output_shape,(None,7))
        self.assertEqual(set(snapshots[0]),set(snapshots[1]))
        for name in snapshots[0]:
            for a,b in zip(snapshots[0][name],snapshots[1][name]):np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
