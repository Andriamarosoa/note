import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import v273_group127_contract as original
from scripts.v273_catalogue_contract import group_masks, joint_groups, choose_groups, assemble_catalogue_outer
from scripts.prepare_v273_learned_corrector import project_actions, FrozenCorrector, CataloguePool
from scripts.prepare_v273_group127_producers import cache_key
from scripts.v273_selector_contract import ProducerPool, FOLDS, id_digest
from scripts.yourmt3_exactk_common import digest
from test import test_v273_selector_contract as selector_tests


def fixture_cache(y, b, f, ids, directory):
    """Synthetic producers depend on every fit label, to detect excluded leakage."""
    data = dict(truth=y, baseline=b, fold=f, eligible_global_index=ids)
    manifests = []
    for size in (1,2,3):
        for key in itertools.combinations(FOLDS, size):
            fit = np.isin(f, key)
            prior = (np.bincount(y[fit], minlength=7)+1).astype(np.float32)
            prior /= prior.sum()
            values = np.full((len(y),7), np.nan, np.float32); values[~fit] = prior
            data[cache_key(key)] = values
            manifests.append(dict(train_folds=list(key), train_global_ids=ids[fit].tolist(),
                                  train_id_sha256=id_digest(ids[fit])))
    directory.mkdir()
    np.savez_compressed(directory/'correctors.npz', **data)
    (directory/'manifest.json').write_text(json.dumps(dict(producers=manifests,
        cache_sha256=digest(directory/'correctors.npz'))))
    return FrozenCorrector(y,b,f,ids,directory)


class CatalogueContractTests(unittest.TestCase):
    def test_255_groups_keep_the_original_127_proposals(self):
        rng = np.random.default_rng(5)
        v = rng.dirichlet(np.ones(5), size=(13,8)).astype(np.float32)
        b = np.array([2,3,4]*4+[2]); masks = group_masks(8)
        self.assertEqual(len(masks), 255)
        np.testing.assert_array_equal(np.bincount(masks.sum(1).astype(int)), [0,8,28,56,70,56,28,8,1])
        props, desc, mean = joint_groups(v,b)
        old = original.joint_groups(v[:,:7],b)
        np.testing.assert_array_equal(props[:,:127],old[0])
        np.testing.assert_array_equal(mean[:,:127],old[2])
        self.assertEqual(desc.shape,(13,255,61))
        for a,c in zip(joint_groups(v[:,:7],b),old):
            np.testing.assert_array_equal(a,c)

    def test_eight_way_synergy_has_no_individual_veto(self):
        v = np.zeros((1,8,5),np.float32); v[0,0,0] = 1
        for j in range(1,8):
            v[0,j,1] = .4; v[0,j,2+(j-1)%3] = .6
        props,_,_ = joint_groups(v,np.array([2]))
        self.assertFalse((props[:,[2**i-1 for i in range(8)]]==3).any())
        self.assertEqual(props[0,-1],3)
        gain = np.full((1,255),-.3); gain[0,-1] = .4
        pred,mask = choose_groups(gain,props,np.array([2]))
        np.testing.assert_array_equal((pred,mask),[[3],[255]])
        pred,_ = choose_groups(gain,props,np.array([2]),singletons=True)
        self.assertEqual(pred[0],2)

    def test_action_projection_uses_only_observables_and_preserves_mass(self):
        p = np.array([[.1,.2,.1,.2,.1,.1,.2]],np.float32)
        q = project_actions(p,np.array([3]))
        np.testing.assert_allclose(q,[[.1,.5,.1,.1,.2]],atol=1e-7)
        with self.assertRaises(ValueError): project_actions(p,np.array([0]))

    def test_seven_candidate_assembly_reproduces_original(self):
        x,y,b,f,ids,raw = selector_tests.SelectorContractTests().fixture()
        old = original.assemble_group_outer(ProducerPool(x,y,b,f,ids),raw,0)
        new = assemble_catalogue_outer(CataloguePool(ProducerPool(x,y,b,f,ids)),raw,0)
        for a,c in zip(old[:2],new[:2]):
            for name in a: np.testing.assert_array_equal(a[name],c[name])
        self.assertEqual(old[-1],new[-1])

    def test_corrector_cache_rejects_fit_and_forbidden_rows(self):
        x,y,b,f,ids,raw = selector_tests.SelectorContractTests().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            corrector = fixture_cache(y,b,f,ids,Path(tmp)/'cache')
            at = np.flatnonzero(f==1)
            self.assertTrue(np.isfinite(corrector.probabilities({2,4},at,{0,1})).all())
            with self.assertRaises(ValueError): corrector.probabilities({0,2},at,{0,1})
            with self.assertRaises(ValueError): corrector.probabilities({1},at,{0})

    def test_outer_and_receiver_labels_cannot_enter_extended_inputs(self):
        x,y,b,f,ids,raw = selector_tests.SelectorContractTests().fixture()
        with tempfile.TemporaryDirectory() as tmp:
            def assemble(truth, name):
                c = fixture_cache(truth,b,f,ids,Path(tmp)/name)
                pool = CataloguePool(ProducerPool(x,truth,b,f,ids),c)
                return assemble_catalogue_outer(pool,raw,0)
            train,held,tr,_,_ = assemble(y,'base')
            changed = y.copy(); changed[f==0] = 6-changed[f==0]
            ta,ha,*_ = assemble(changed,'outer')
            for name in train:
                np.testing.assert_array_equal(train[name],ta[name])
                np.testing.assert_array_equal(held[name],ha[name])
            changed = y.copy(); changed[f==1] = (changed[f==1]+1)%7
            ta,*_ = assemble(changed,'receiver')
            for name in train:
                np.testing.assert_array_equal(train[name][f[tr]==1],ta[name][f[tr]==1])


if __name__ == '__main__': unittest.main()
