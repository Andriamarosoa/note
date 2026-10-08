import unittest
import numpy as np
import tensorflow as tf

from scripts.learn_v273_catalogue import CatalogueCritic, train_catalogue
from scripts.learn_v273_group_consensus import train_consensus, consensus_loss
from scripts.prepare_v273_learned_corrector import fit_corrector, infer_corrector, project_actions
from scripts.v273_catalogue_contract import joint_groups
from test import test_v273_group127_neural as group_tests


class CatalogueNeuralTests(unittest.TestCase):
    def test_control_reproduces_actual_consensus_training(self):
        x,y,b = group_tests.GroupNeuralTests().fixture()
        old = train_consensus(x,y,b,x,'pooled_ce',epochs=2)
        new = train_catalogue(x,y,b,x,epochs=2)
        self.assertEqual(new[-1].count_params(),12994)
        for k in old[2]: np.testing.assert_array_equal(old[2][k],new[2][k])
        np.testing.assert_array_equal(old[0],new[0])

    def test_extended_critic_gradients_and_full_training(self):
        x,y,b = group_tests.GroupNeuralTests().fixture()
        votes = np.concatenate([x['member_votes'],np.roll(x['member_votes'][:,:1],1,axis=2)],1)
        props,desc,_ = joint_groups(votes,b)
        rng = np.random.default_rng(13)
        x = dict(x,member_votes=votes,proposal=props,
                 group_features=np.concatenate([desc,rng.random((len(b),255,9),dtype=np.float32)],2))
        model = CatalogueCritic()
        with tf.GradientTape() as tape:
            out = model(x); loss,*_ = consensus_loss(out,y,b,props,'pooled_ce')
        for g in tape.gradient(loss,model.trainable_variables):
            self.assertIsNotNone(g); self.assertTrue(np.isfinite(g.numpy()).all())
        self.assertEqual(model.count_params(),13538)
        pred,mask,out,history,_ = train_catalogue(x,y,b,x,epochs=2)
        self.assertTrue(np.isfinite(out['expected_gain']).all())
        np.testing.assert_allclose(out['class_probability'].sum(1)+out['other_probability'],1,atol=1e-6)
        for i in range(len(b)):
            for k in np.unique(props[i]):
                g = out['expected_gain'][i,props[i]==k]
                np.testing.assert_array_equal(g,np.repeat(g[0],len(g)))

    def test_real_corrector_training_and_label_free_inference(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(49,43)); y = np.arange(49)%7; b = 2+np.arange(49)%3
        model,scaler,history = fit_corrector(raw[:35],y[:35],b[:35],epochs=2)
        self.assertEqual(model.count_params(),5575)
        p = infer_corrector(model,scaler,raw[35:],b[35:])
        self.assertEqual(p.shape,(14,7)); self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(project_actions(p,b[35:]).sum(1),1,atol=1e-6)
        self.assertEqual([h['epoch'] for h in history],[1,2])


if __name__ == '__main__': unittest.main()
