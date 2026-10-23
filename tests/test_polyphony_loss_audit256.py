import tempfile
from pathlib import Path
import unittest

import numpy as np

from onset256.acoustic_attack import AttackPolicy
from onset256.feature_cache import compress_features, load_features
from scripts.audit_onset_polyphony_losses256 import (
    decoded_trace, loss_mechanism, paired_matching, note_context)


class PolyphonyLossAuditTests(unittest.TestCase):
    def test_net_losses_do_not_hide_new_recoveries(self):
        truth = [dict(time=.1,pitch=40,duration=.1),dict(time=.3,pitch=44,duration=.1)]
        old = [(17*256,40)]
        new = [(52*256,44)]
        _,_,lost,gained,before,after = paired_matching(old,new,truth)
        self.assertEqual(lost,{0});self.assertEqual(gained,{1})
        self.assertEqual(after['fn']-before['fn'],0)

    def test_accompaniment_label_is_not_polyphony(self):
        truth = [dict(time=0.,pitch=40,duration=.2),
                 dict(time=.1,pitch=52,duration=.2),
                 dict(time=.5,pitch=55,duration=.1)]
        self.assertEqual(note_context(truth,1)['polyphony'],'K2')
        self.assertEqual(note_context(truth,2)['polyphony'],'K1')

    def test_trace_identifies_refractory_without_emitting_late(self):
        q = np.zeros((30,49),np.float32);q[0,0]=.9;q[2,0]=.9
        base=np.zeros(q.shape,bool);base[2,0]=True
        policy=AttackPolicy(gap_hops=14,use_baseline=False)
        trace=decoded_trace(q,base,policy)
        result=loss_mechanism(dict(time=3*256/44100.,pitch=40),
            (3*256,40),q,base,trace,policy)
        self.assertEqual(result['rejection_at_original_boundary'],'refractory')
        # The earlier accepted event is also in tolerance, so the matcher,
        # rather than a fabricated acoustic failure, is explicitly reported.
        self.assertEqual(result['decoder_mechanism'],'one_to_one_matching_competition')
        self.assertEqual(int(trace['output'].sum()),1)

    def test_probability_plateau_exposes_rearming(self):
        q=np.zeros((50,49),np.float32);q[:,0]=.9
        base=np.zeros(q.shape,bool);base[40,0]=True
        policy=AttackPolicy(use_baseline=False)
        trace=decoded_trace(q,base,policy)
        result=loss_mechanism(dict(time=41*256/44100.,pitch=40),
            (41*256,40),q,base,trace,policy)
        self.assertEqual(result['decoder_mechanism'],'score_crosses_cut_but_not_rearmed')

    def test_reference_veto_is_not_called_weak_fundamental(self):
        q=np.zeros((50,49),np.float32);q[30,0]=.1
        base=np.zeros(q.shape,bool);base[30,0]=True
        policy=AttackPolicy(baseline_cut=.15)
        trace=decoded_trace(q,base,policy)
        result=loss_mechanism(dict(time=31*256/44100.,pitch=40),
            (31*256,40),q,base,trace,policy)
        self.assertEqual(result['rejection_at_original_boundary'],'reference_veto_cut')
        self.assertFalse(result['acoustic_cause_proven'])

    def test_compressed_cache_preserves_float32_bits(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'features_003.npy'
            x=np.random.default_rng(32).normal(size=(10,49,195)).astype(np.float32)
            x[0,0,0]=-0.
            np.save(path,x);compress_features(path)
            restored=load_features(d,3)
            np.testing.assert_array_equal(x.view(np.uint32),restored.view(np.uint32))
            self.assertFalse(path.exists())


if __name__=='__main__':
    unittest.main()
