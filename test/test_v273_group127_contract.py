import unittest
import numpy as np

from scripts.v273_group127_contract import (
    GROUP_MASKS, GROUP_SIZES, seven_candidates, joint_groups, outcome_labels,
    choose_groups, local_audits, assemble_group_outer,
)
from scripts.v273_selector_contract import ProducerPool
from test import test_v273_selector_contract as original_tests


class GroupContractTests(unittest.TestCase):
    def test_all_127_groups_and_optional_baseline(self):
        self.assertEqual(len(np.unique(GROUP_MASKS,axis=0)),127)
        np.testing.assert_array_equal(np.bincount(GROUP_SIZES.astype(int),minlength=8),[0,7,21,35,35,21,7,1])
        self.assertEqual(int(np.sum(GROUP_MASKS[:,0] == 0)),63)
        p=np.ones((2,6,5),np.float32)/5
        p[:,0]=np.eye(5)[[0,1]]
        v=seven_candidates(p)
        np.testing.assert_array_equal(v[:,6],p[:,1:].mean(1))
        pred,desc,mean=joint_groups(v,np.array([2,3]))
        self.assertEqual(desc.shape,(2,127,55))
        np.testing.assert_array_equal(pred[:,0],[2,3])
        np.testing.assert_allclose(mean.sum(2),1,atol=1e-6)

    def test_pair_can_succeed_when_both_members_fail(self):
        v=np.zeros((1,7,5),np.float32)
        v[0]=[.1,.4,.5,0,0]
        v[0,1]=[.1,.4,0,.5,0]
        pred,_,_=joint_groups(v,np.array([2]))
        np.testing.assert_array_equal(pred[0,[0,1,2]],[4,5,3])
        labels=outcome_labels(pred,np.array([3]),np.array([2]))
        np.testing.assert_array_equal(labels[0,[0,1,2]],[2,2,0])
        gain=np.full((1,127),-.2);gain[0,2]=.4
        chosen,mask=choose_groups(gain,pred,np.array([2]))
        np.testing.assert_array_equal(chosen,[3]);np.testing.assert_array_equal(mask,[3])
        singleton,_=choose_groups(gain,pred,np.array([2]),singletons=True)
        np.testing.assert_array_equal(singleton,[2])

    def test_seven_members_all_wrong_can_jointly_be_right(self):
        v=np.zeros((1,7,5),np.float32);v[0,0,0]=1
        for j in range(1,7):
            v[0,j,1]=.4;v[0,j,2+(j-1)%3]=.6
        pred,_,_=joint_groups(v,np.array([2]))
        singles=pred[:,[2**i-1 for i in range(7)]]
        self.assertFalse(np.any(singles == 3))
        self.assertEqual(int(pred[0,126]),3)

    def test_keep_neutral_and_ties_are_not_fake_corrections(self):
        proposal=np.array([[2,4,3]+[3]*124,[2,4,3]+[3]*124])
        labels=outcome_labels(proposal,np.array([3,0]),np.array([3,3]))
        np.testing.assert_array_equal(labels[0,:3],[1,1,2])
        np.testing.assert_array_equal(labels[1,:3],[2,2,2])
        pred,mask=choose_groups(np.zeros((2,127)),proposal,np.array([3,3]))
        np.testing.assert_array_equal(pred,[3,3]);np.testing.assert_array_equal(mask,[0,0])

    def test_local_audit_compares_joint_outcomes_and_rejects_self(self):
        raw=np.arange(6,dtype=float)[:,None]
        props=np.full((6,127),4);y=np.array([4,4,3,3,3,1]);b=np.full(6,3)
        qprops=np.full((1,127),4)
        a=local_audits(raw,np.array([[2.5]]),props,y,b,qprops,np.array([3]),np.arange(6),np.array([10]))
        np.testing.assert_allclose(a[0,0,:3],np.array([3,4,2])/9,atol=1e-7)
        np.testing.assert_allclose(a[0,0,3:6],(np.array([2,3,1])+12*np.array([3,4,2])/9)/18,atol=1e-7)
        with self.assertRaises(ValueError):
            local_audits(raw,np.array([[2.5]]),props,y,b,qprops,np.array([3]),np.arange(6),np.array([1]))

    def test_outer_and_receiver_labels_cannot_enter_their_inputs(self):
        x,y,b,f,ids,raw=original_tests.SelectorContractTests().fixture()
        train,held,tr,val,manifest=assemble_group_outer(ProducerPool(x,y,b,f,ids),raw,0)
        altered=y.copy();altered[f==0]=6-altered[f==0]
        ta,ha,_,_,_=assemble_group_outer(ProducerPool(x,altered,b,f,ids),raw,0)
        for name in train:
            np.testing.assert_array_equal(train[name],ta[name]);np.testing.assert_array_equal(held[name],ha[name])
        altered=y.copy();altered[f==1]=(altered[f==1]+1)%7
        ta,_,_,_,_=assemble_group_outer(ProducerPool(x,altered,b,f,ids),raw,0)
        for name in train:
            np.testing.assert_array_equal(train[name][f[tr]==1],ta[name][f[tr]==1])
        for record in manifest['inner_audits']:
            excluded={0,record['receiver_fold']}
            self.assertFalse(set(record['reference_folds'])&excluded)
            for producer in record['reference_expert_producers']:
                self.assertFalse(set(producer['train_folds'])&excluded)


if __name__=='__main__':unittest.main()
