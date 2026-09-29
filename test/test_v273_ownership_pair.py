import unittest
import numpy as np
from scripts.v273_ownership_experiment import geometry_for_track, frame_fraction, lookup, sorted_proposals
from scripts.v273_ownership_context import owners_at_samples
from scripts.train_v86_state_transition_proposals import _candidate_groups, PEAK_MERGE_SAMPLES
from scripts.audit_v273_ownership_pair import comparison


class OwnershipPairTest(unittest.TestCase):
    def test_temporal_compression_still_separates_the_counterexample(self):
        a=geometry_for_track([np.asarray([10000,11764]),np.asarray([13300])])[0][0]
        b=geometry_for_track([np.asarray([10000,11764]),np.asarray([13500])])[0][0]
        np.testing.assert_array_equal(a[:,0],b[:,0])
        self.assertTrue(np.any(a[:,1]!=b[:,1]))

    def test_fast_geometry_matches_exact_rule_and_frame_support(self):
        groups=[np.asarray([1500,1800,2900]),np.asarray([3500,3900]),np.asarray([7000])]
        g,losses,_,_=geometry_for_track(groups)
        for i,group in enumerate(groups):
            q=group[0]-1308+np.arange(4096)
            local=(np.abs(q[:,None]-group).min(1)<=882)&(q>=0)
            own=(owners_at_samples(groups,q)==i)&(q>=0)
            np.testing.assert_array_equal(g[i,:,0],frame_fraction(local))
            np.testing.assert_array_equal(g[i,:,1],frame_fraction(own))
            self.assertEqual(losses[i],np.sum(local&~own))

    def test_duplicate_positions_and_group_ties(self):
        groups=[np.asarray([1000,1200]),np.asarray([1000,1400])]
        flat,ids=sorted_proposals(groups);q=np.asarray([1000,1300,2200,2300])
        np.testing.assert_array_equal(lookup(flat,ids,q)[0],owners_at_samples(groups,q))

    def test_no_neighbors_yields_identical_channels(self):
        g,l,_,_=geometry_for_track([np.asarray([0,100,200])])
        np.testing.assert_array_equal(g[:,:,0],g[:,:,1]);self.assertEqual(l[0],0)
        self.assertEqual(g[0,0,0],0.)  # negative audio padding is never owned

    def test_merge_prefix_is_final_after_four_samples_plus_peak_lookahead(self):
        rng=np.random.default_rng(911)
        for kind in ('random','rising_chain'):
            presence=rng.uniform(0,.8,512).astype(np.float32)
            if kind=='rising_chain':
                presence[:]=0;presence[1::3]=np.linspace(.2,.9,len(presence[1::3]))
            mult=np.tile([.8,.1,.1],(len(presence),1))
            whole=_candidate_groups(dict(presence=presence,multiplicity=mult),.1)
            for end in (20,80,160,300,400):
                n=end+PEAK_MERGE_SAMPLES+2
                prefix=_candidate_groups(dict(presence=presence[:n],multiplicity=mult[:n]),.1)
                self.assertEqual([x for x in prefix if x['sample']<=end],
                                 [x for x in whole if x['sample']<=end])

    def test_audit_exposes_surcount_correction_and_small_k_regression(self):
        before=dict(global_index=np.arange(4),member=np.repeat('01_BN1-147-Gb_comp.jams',4),
            k=np.asarray([1,2,3,0]),predicted=np.asarray([1,3,3,1]),
            lost_eligible_samples=np.asarray([0,12,0,9]))
        after={**before,'predicted':np.asarray([0,2,3,0])}
        result=comparison(before,after)
        self.assertEqual(result['poly']['delta'],1)
        self.assertEqual(result['by_k']['1']['delta'],-1)
        self.assertEqual(result['by_competition']['competition']['all']['corrected'],2)
        self.assertEqual(result['by_competition']['no_competition']['all']['regressed'],1)
        self.assertEqual(result['error_direction_by_k']['1']['under_after'],1)
        self.assertEqual(result['error_direction_by_k']['2']['over_before'],1)
        self.assertEqual(result['error_direction_by_k']['2']['over_after'],0)


if __name__=='__main__':unittest.main()
