import unittest
import numpy as np
import tensorflow as tf

from scripts.learn_v273_group127 import JointGroupCritic
from scripts.learn_v273_group_consensus import GroupConsensus, consensus_loss, consensus_outputs, train_consensus, MODES
from scripts.v273_group127_contract import joint_groups, choose_groups
from test import test_v273_group127_neural as original


class ConsensusTests(unittest.TestCase):
    def fixture(self):
        return original.GroupNeuralTests().fixture()

    def test_equal_verdicts_have_identical_utilities_and_parameter_count(self):
        x,y,b=self.fixture();old=JointGroupCritic();old(x)
        for mode in MODES:
            model=GroupConsensus(mode);o=model(x)
            self.assertEqual(model.count_params(),old.count_params())
            self.assertEqual(model.count_params(),12994)
            p=o['outcome_probability'].numpy();g=o['expected_gain'].numpy()
            np.testing.assert_allclose(p.sum(2),1,atol=1e-6)
            for i in range(len(b)):
                for k in np.unique(x['proposal'][i]):
                    values=g[i,x['proposal'][i]==k]
                    np.testing.assert_array_equal(values,np.repeat(values[0],len(values)))
            if mode=='pooled_ce':
                np.testing.assert_allclose(o['class_probability'].numpy().sum(1)+o['other_probability'].numpy(),1,atol=1e-6)

    def test_permuting_group_order_preserves_class_probabilities(self):
        x,y,b=self.fixture();permutation=np.random.default_rng(17).permutation(127)
        shuffled={k:v[:,permutation] if k in ('group_features','proposal') else v for k,v in x.items()}
        for mode in MODES:
            model=GroupConsensus(mode);a=model(x);c=model(shuffled)
            np.testing.assert_allclose(a['class_probability'],c['class_probability'],atol=1e-6)

    def test_other_prevents_forced_probability_one_for_only_wrong_alternative(self):
        proposals=np.full((1,127),4,np.int32);b=np.array([2]);logits=tf.zeros((1,127))
        out=consensus_outputs(logits,proposals,tf.zeros(1),b,'pooled_ce')
        self.assertAlmostEqual(float(out['conditional_correct'][0,0]),.5)
        self.assertAlmostEqual(float(out['other_probability'][0]),.25)
        loss,_,_=consensus_loss(out,np.array([3]),b,proposals,'pooled_ce')
        self.assertAlmostEqual(float(loss),-np.log(.25),places=6)
        none=np.full((1,127),2,np.int32)
        empty=consensus_outputs(logits,none,tf.zeros(1),b,'pooled_ce')
        np.testing.assert_array_equal(empty['expected_gain'],np.zeros((1,127)))
        self.assertTrue(np.isfinite(float(consensus_loss(empty,np.array([3]),b,none,'pooled_ce')[0])))

    def test_seven_way_synergy_remains_selectable_without_member_veto(self):
        v=np.zeros((1,7,5),np.float32);v[0,0,0]=1
        for j in range(1,7):v[0,j,1]=.4;v[0,j,2+(j-1)%3]=.6
        props,_,_=joint_groups(v,np.array([2]))
        self.assertFalse((props[:,[0,1,3,7,15,31,63]]==3).any())
        self.assertEqual(props[0,126],3)
        for mode in MODES:
            raw=tf.constant(np.where(props==3,5.,-5.).astype(np.float32))
            out=consensus_outputs(raw,props,tf.constant([-5.]),np.array([2]),mode)
            prediction,mask=choose_groups(out['expected_gain'].numpy(),props,np.array([2]))
            self.assertEqual(prediction[0],3);self.assertGreater(mask[0],0)
            self.assertGreater(float(out['expected_gain'][0,126]),0)

    def test_categorical_loss_is_event_likelihood_including_unavailable_truth(self):
        x,y,b=self.fixture();out=GroupConsensus('pooled_ce')(x)
        probability=out['class_probability'].numpy()[np.arange(len(y)),y]
        probability=np.where(out['available'].numpy()[np.arange(len(y)),y] | (y==b),probability,out['other_probability'].numpy())
        loss,_,_=consensus_loss(out,y,b,x['proposal'],'pooled_ce')
        self.assertAlmostEqual(float(loss),float(-np.log(probability).mean()),places=5)

    def test_connected_finite_gradients_and_real_short_training(self):
        x,y,b=self.fixture()
        for mode in MODES:
            tensors={k:tf.convert_to_tensor(v) for k,v in x.items()};model=GroupConsensus(mode)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(tensors['group_features']);out=model(tensors)
                loss,_,_=consensus_loss(out,y,b,tensors['proposal'],mode)
            for g in tape.gradient(loss,model.trainable_variables):
                self.assertIsNotNone(g);self.assertTrue(np.isfinite(g.numpy()).all())
            grad=tape.gradient(loss,tensors['group_features']).numpy()
            self.assertGreater(np.abs(grad[...,:55]).sum(),0);self.assertGreater(np.abs(grad[...,55:]).sum(),0)
            pred,mask,out,history,_=train_consensus(x,y,b,x,mode,epochs=2)
            self.assertEqual(len(pred),len(y));self.assertTrue(np.isfinite(out['expected_gain']).all())
            self.assertEqual([h['epoch'] for h in history],[1,2])


if __name__=='__main__':unittest.main()
