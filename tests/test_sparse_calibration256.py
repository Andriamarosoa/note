import unittest
import numpy as np
from onset256.acoustic_attack import AttackPolicy
from onset256.acoustic_calibration import decode_cache,count_cache,truth_arrays
from onset256.sparse_calibration import score_policies
from scripts.onset_acoustic_direct256 import policies


class SparseCalibrationTests(unittest.TestCase):
    def test_every_original_policy_matches_dense_full_stream_counts(self):
        rng=np.random.default_rng(236)
        grid=list(policies())+[AttackPolicy(enabled=False)]
        q=rng.random((95,49),dtype=np.float32)
        q[::3]=np.float32(.9);q[1::7]=np.float32(.15)
        baseline=rng.random(q.shape)<.10
        truth=[dict(time=float(t),pitch=int(p)) for t,p in zip(
            rng.uniform(0,.6,200),rng.integers(40,89,200))]
        cache={'a':dict(baseline=baseline,truth=truth_arrays(truth))}
        results=score_policies({'a':q},cache,grid)
        for policy,result in zip(grid,results):
            expected=count_cache(decode_cache(q,baseline,policy),*cache['a']['truth'])
            self.assertEqual(tuple(result['score'][k] for k in ('tp','fp','fn')),expected)

    def test_low_reference_emission_does_not_rearm_on_same_frame(self):
        q=np.zeros((30,49),np.float32);q[2:,0]=.95
        base=np.zeros(q.shape,bool);base[1,0]=True
        policy=AttackPolicy(use_baseline=True,gap_hops=2)
        truth=[dict(time=.1,pitch=40)]
        cache={'a':dict(baseline=base,truth=truth_arrays(truth))}
        actual=score_policies({'a':q},cache,[policy])[0]['score']
        expected=count_cache(decode_cache(q,base,policy),*cache['a']['truth'])
        self.assertEqual(tuple(actual[k] for k in ('tp','fp','fn')),expected)


if __name__=='__main__':unittest.main()
