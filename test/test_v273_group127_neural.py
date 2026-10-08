import unittest
import numpy as np
import tensorflow as tf

from scripts.v273_group127_contract import joint_groups, outcome_labels
from scripts.learn_v273_group127 import JointGroupCritic, event_loss, train_groups


class GroupNeuralTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(41);n=21;b=np.tile([2,3,4],7)
        votes=rng.dirichlet(np.ones(5),size=(n,7)).astype(np.float32)
        votes[:,0]=np.eye(5)[b-2]
        votes[:,1]=np.eye(5)[np.full(n,3)]
        proposals,features,_=joint_groups(votes,b)
        features=np.concatenate([features,rng.uniform(size=(n,127,9)).astype(np.float32)],2)
        inputs=dict(group_features=features,context=rng.normal(size=(n,43)).astype(np.float32),
            baseline=np.eye(7,dtype=np.float32)[b],member_votes=votes,proposal=proposals)
        return inputs,np.tile(np.arange(7),3).astype(np.int32),b

    def test_normalized_joint_outcomes_with_one_baseline_risk(self):
        x,y,b=self.fixture();out=JointGroupCritic()(x)
        probability=out['outcome_probability'].numpy();changing=x['proposal']!=b[:,None]
        np.testing.assert_allclose(probability.sum(2),1,atol=1e-6)
        self.assertTrue((probability >= -1e-7).all())
        for i in range(len(b)):
            np.testing.assert_array_equal(probability[i,changing[i],1],np.repeat(out['baseline_correct'].numpy()[i],changing[i].sum()))
        np.testing.assert_array_equal(probability[~changing],np.tile([0.,0.,1.],((~changing).sum(),1)))
        np.testing.assert_array_equal(out['expected_gain'].numpy(),probability[:,:,0]-probability[:,:,1])

    def test_loss_is_joint_likelihood_averaged_by_event(self):
        x,y,b=self.fixture();out=JointGroupCritic()(x)
        loss,_,_=event_loss(out,y,b,x['proposal'])
        labels=outcome_labels(x['proposal'],y,b)
        probability=out['outcome_probability'].numpy()
        observed=np.take_along_axis(probability,labels[:,:,None],axis=2).squeeze(2)
        change=x['proposal']!=b[:,None]
        self.assertTrue(change.any(1).all())
        expected=np.mean(np.sum(-np.log(observed)*change,axis=1)/change.sum(1))
        self.assertAlmostEqual(float(loss),float(expected),places=5)

    def test_neural_gradients_include_group_interactions_and_audits(self):
        x,y,b=self.fixture();x={k:tf.convert_to_tensor(v) for k,v in x.items()}
        model=JointGroupCritic()
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x['group_features'])
            out=model(x,training=True);loss,_,_=event_loss(out,y,b,x['proposal'])
        for grad in tape.gradient(loss,model.trainable_variables):
            self.assertIsNotNone(grad);self.assertTrue(np.isfinite(grad.numpy()).all())
        feature_grad=tape.gradient(loss,x['group_features']).numpy()
        self.assertGreater(float(np.abs(feature_grad[...,:55]).sum()),0)
        self.assertGreater(float(np.abs(feature_grad[...,55:]).sum()),0)

    def test_actual_short_training_and_decision(self):
        x,y,b=self.fixture()
        pred,mask,out,history,model=train_groups(x,y,b,x,epochs=2)
        self.assertEqual(pred.shape,(len(y),));self.assertTrue(np.isin(pred,[2,3,4,5,6]).all())
        self.assertTrue(np.isfinite(out['outcome_probability']).all())
        self.assertEqual([h['epoch'] for h in history],[1,2]);self.assertGreater(model.count_params(),0)
        self.assertTrue(((mask>=0)&(mask<=127)).all())


if __name__=='__main__':unittest.main()
