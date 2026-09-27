from pathlib import Path
import unittest
import numpy as np
from scripts.v273_native_protocol import load_config
from scripts.v273_window_experiment import partitions, batch_inputs, epoch_order
from scripts.audit_v273_window_pair import effect, metrics


class WindowPairTests(unittest.TestCase):
    def test_composition_safe_partitions(self):
        cfg=load_config(Path('analysis/v273-native-paired-config.json'))
        members=np.repeat(sorted(cfg['member_folds']),2)
        p=partitions(members,cfg)
        self.assertEqual(len(p['outer']),100)
        self.assertEqual(len(p['final_fit']),380)
        for key,forbidden in (('inner_fit',{0,3}),('inner_val',{1,2,3,4}),('final_fit',{3})):
            self.assertFalse({cfg['member_folds'][m] for m in members[p[key]]} & forbidden)
        with self.assertRaisesRegex(RuntimeError,'inventory'):
            partitions(members[:-2],cfg)

    def test_batch_is_exact_prefix_with_identical_other_inputs(self):
        rng=np.random.default_rng(7)
        cache=dict(sequence=rng.normal(size=(6,48,133)).astype(np.float16),
                   stats=rng.normal(size=(6,8)).astype(np.float16),mask=np.ones((6,48),np.uint8),
                   spectral=rng.normal(size=(6,31,64,3)).astype(np.float16))
        ids=np.asarray([5,0,3])
        a,b=batch_inputs(cache,ids,23),batch_inputs(cache,ids,31)
        for key in ('candidate_set','candidate_mask','cluster_stats'):
            np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(a['spectral_map'],b['spectral_map'][:,:23])

    def test_shuffle_pairing_and_coverage(self):
        ids=np.asarray([0,4,6,9,11])
        np.testing.assert_array_equal(epoch_order(ids,23,1),epoch_order(ids,23,1))
        np.testing.assert_array_equal(np.sort(epoch_order(ids,23,1)),ids)
        self.assertFalse(np.array_equal(epoch_order(np.arange(40),23,1),epoch_order(np.arange(40),23,2)))

    def test_error_audit_does_not_count_wrong_to_wrong_as_fixed(self):
        k=np.asarray([2,3,4,1,0]);before=np.asarray([2,2,3,1,1]);after=np.asarray([3,3,5,1,0])
        r=effect(k,before,after,np.ones(5,bool))
        self.assertEqual((r['corrected'],r['degraded'],r['wrong_to_wrong'],r['net_correct']),(2,1,1,1))
        for prediction in (before,after):
            m=metrics(k,prediction)
            self.assertEqual(m['correct']+m['over']+m['under'],len(k))


if __name__=='__main__':
    unittest.main()
