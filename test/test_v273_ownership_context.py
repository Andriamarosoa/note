import unittest
import numpy as np

from scripts.v273_ownership_context import owners_at_samples, owned_sample_mask, required_proposal_watermark
from scripts.audit_v273_design_contract import counterexample
from scripts.train_v102_source_time_assignment import _nearest_cluster_for_event


class OwnershipContextTest(unittest.TestCase):
    def test_native_inputs_identical_but_target_changes(self):
        r = counterexample()
        self.assertTrue(r['native_inputs_identical'])
        self.assertEqual([c['target_k'] for c in r['configurations']], [0, 1])

    def test_radius_and_lower_group_tie(self):
        groups = [np.asarray([1000]), np.asarray([1200])]
        np.testing.assert_array_equal(owners_at_samples(groups, np.asarray([118,117,1100,2082,2083])),
                                      [0,-1,0,1,-1])

    def test_matches_existing_supervision_with_duplicates_and_unsorted_groups(self):
        rng = np.random.default_rng(29)
        groups = [rng.integers(1000,10000,25) for _ in range(7)]
        samples = np.concatenate(groups)
        ids = np.repeat(np.arange(7),25)
        order = np.argsort(samples,kind='stable'); samples=samples[order]; ids=ids[order]
        queries = rng.integers(0,11000,300)
        expected=[]
        for q in queries:
            nearest = _nearest_cluster_for_event(int(q),samples,ids)
            expected.append(-1 if nearest is None else nearest[1])
        np.testing.assert_array_equal(owners_at_samples(groups,queries),expected)

    def test_future_proposals_are_required_for_complete_contract(self):
        groups=[np.asarray([10000,11764]),np.asarray([13300])]
        self.assertEqual(required_proposal_watermark(groups[0]),13528)
        with self.assertRaisesRegex(ValueError,'incomplete proposal context'):
            owned_sample_mask(groups,0,np.asarray([12600]),complete_through=12788)
        np.testing.assert_array_equal(owned_sample_mask(groups,0,np.asarray([12600]),complete_through=13528),[False])

    def test_certified_prefix_agrees_with_full_geometry_for_all_eligible_samples(self):
        groups=[np.asarray([1000,2764]),np.asarray([3300,8000]),np.asarray([9000])]
        queries=np.arange(118,3647,dtype=np.int64)
        watermark=required_proposal_watermark(groups[0])
        np.testing.assert_array_equal(owned_sample_mask(groups,0,queries,complete_through=watermark),
                                      owners_at_samples(groups,queries)==0)

    def test_full_proposals_matter_even_if_one_is_not_retained(self):
        full=[np.asarray([1000,2600]),np.asarray([2800])]
        truncated=[np.asarray([1000]),np.asarray([2800])]
        q=np.asarray([2630])
        self.assertEqual(owners_at_samples(full,q)[0],0)
        self.assertEqual(owners_at_samples(truncated,q)[0],1)

    def test_invalid_geometry_is_rejected(self):
        for groups in ([],[np.asarray([1.5])],[np.asarray([-1])]):
            with self.assertRaises(ValueError): owners_at_samples(groups,np.asarray([2]))
        with self.assertRaises(ValueError): owners_at_samples([np.asarray([1])],np.asarray([2]),radius=-1)


if __name__ == '__main__':
    unittest.main()
