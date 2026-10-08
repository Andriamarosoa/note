import unittest
import numpy as np
import tensorflow as tf

from scripts.learn_v273_catalogue import CatalogueCritic,train_catalogue
from scripts.learn_v273_contextual_risk import ContextualRiskCritic
from scripts.v273_catalogue_contract import group_masks,direct_dim,joint_groups
from scripts.v273_contextual_risk_contract import audit_profiles,retention
from test import test_v273_group127_neural as group_tests


class ContextualRiskTests(unittest.TestCase):
    def test_risk_can_use_group_audit_association_lost_by_raw_mean(self):
        count=8;groups=255;offset=direct_dim(count)
        identity=group_masks(count)
        features=np.zeros((1,groups,offset+9),np.float32)
        features[0,:,:8]=identity
        features[0,:,offset+3]=identity[:,0]
        features[0,:,offset+4]=1-identity[:,0]
        x=dict(group_features=features,baseline=np.eye(7,dtype=np.float32)[[3]],
            context=np.zeros((1,43),np.float32),member_votes=np.full((1,8,5),.2,np.float32),
            proposal=np.full((1,groups),2,np.int32))
        # Reversing this odd alternating pattern would leave it unchanged.
        permuted=features.copy();permuted[...,offset:]=np.roll(features[...,offset:],1,axis=1)
        self.assertFalse(np.array_equal(features,permuted))
        np.testing.assert_array_equal(features[...,offset:].mean(1),permuted[...,offset:].mean(1))
        other=dict(x,group_features=permuted)
        old=CatalogueCritic();old(x)
        np.testing.assert_array_equal(old(x)['baseline_correct'],old(other)['baseline_correct'])
        new=ContextualRiskCritic();new(x)
        for variable in new.trainable_variables:variable.assign(tf.zeros_like(variable))
        w=new.group_hidden.kernel.numpy();w[0,0]=2.;w[offset+3,0]=2.;new.group_hidden.kernel.assign(w)
        w=new.group_interaction.kernel.numpy();w[0,0]=1.;new.group_interaction.kernel.assign(w)
        w=new.baseline_hidden.kernel.numpy();w[-32,0]=1.;new.baseline_hidden.kernel.assign(w)
        w=new.baseline_logit.kernel.numpy();w[0,0]=1.;new.baseline_logit.kernel.assign(w)
        self.assertGreater(float(tf.reduce_max(tf.abs(new(x)['baseline_correct']-new(other)['baseline_correct']))),1e-4)
        new.zero_group_risk_context=True
        np.testing.assert_array_equal(new(x)['baseline_correct'],new(other)['baseline_correct'])

    def test_training_simplex_and_fixed_weight_risk_intervention(self):
        x,y,b=group_tests.GroupNeuralTests().fixture()
        votes=np.concatenate([x['member_votes'],np.roll(x['member_votes'][:,:1],1,axis=2)],1)
        props,desc,_=joint_groups(votes,b)
        x=dict(x,member_votes=votes,proposal=props,
            group_features=np.concatenate([desc,np.random.default_rng(15).random((len(b),255,9),dtype=np.float32)],2))
        pred,mask,out,history,model=train_catalogue(x,y,b,x,epochs=2,model_factory=ContextualRiskCritic)
        self.assertEqual(model.count_params(),14562)
        np.testing.assert_allclose(out['class_probability'].sum(1)+out['other_probability'],1,atol=1e-6)
        original=model(x);weights=[w.numpy().copy() for w in model.weights]
        model.zero_group_risk_context=True;ablated=model(x)
        np.testing.assert_array_equal(original['pooled_class_logits'],ablated['pooled_class_logits'])
        for old,new in zip(weights,model.weights):np.testing.assert_array_equal(old,new)
        for i in range(len(b)):
            for k in np.unique(props[i]):
                gain=out['expected_gain'][i,props[i]==k]
                np.testing.assert_array_equal(gain,np.repeat(gain[0],len(gain)))

    def test_profile_uses_complete_groups_and_preserves_success_cost(self):
        props=np.array([[2,2],[2,2],[3,3]])
        audits=np.zeros((3,2,9));audits[:,:,7]=1
        audits[0,:,3:5]=[[.1,.5],[.6,.2]]
        audits[1,:,3:5]=[[.1,.5],[.2,.4]]
        p=np.array([2,2,3]);b=np.array([3,3,3]);y=np.array([2,3,3])
        profiles=audit_profiles(props,audits,p,b)
        self.assertEqual(profiles.tolist(),['mixed_by_complete_group','local_rates_favor_regression_for_all_groups','keep'])
        comparison=retention(y,b,p,b)
        self.assertEqual(comparison['corrections_lost'],1)
        self.assertEqual(comparison['regressions_avoided'],1)


if __name__=='__main__':unittest.main()
