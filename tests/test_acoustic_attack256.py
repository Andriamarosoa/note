import unittest
import numpy as np
from onset256.acoustic_attack import AttackPolicy, DirectAttackSelector, acoustic_matrix
from onset256.acoustic_calibration import decode_cache, count_cache, truth_arrays


class AcousticAttackTests(unittest.TestCase):
    def test_fires_without_old_candidate_or_future_confirmation(self):
        s=DirectAttackSelector(AttackPolicy(cut=.5,gap_hops=7))
        q=np.zeros(49,np.float32);q[0]=.8
        self.assertEqual(s.step(q,[],end_sample=256),(40,))
        self.assertEqual(s.step(q,[],end_sample=512),())
        self.assertEqual(s.finish(),())

    def test_restrike_after_rearming_and_causal_duplicate_merge(self):
        s=DirectAttackSelector(AttackPolicy(cut=.5,gap_hops=2))
        q=np.zeros(49,np.float32);q[0]=.8
        self.assertEqual(s.step(q,[40],end_sample=256),(40,))
        self.assertEqual(s.step(np.zeros(49,np.float32),[40],end_sample=512),())
        self.assertEqual(s.step(q,[],end_sample=768),(40,))

    def test_disabled_control_preserves_reference_even_with_short_gaps(self):
        s=DirectAttackSelector(AttackPolicy(enabled=False))
        for frame in range(3):
            self.assertEqual(s.step(np.zeros(49,np.float32),[40,52],
                                   end_sample=(frame+1)*256),(40,52))

    def test_shape_and_continuity(self):
        s=DirectAttackSelector(AttackPolicy())
        with self.assertRaises(ValueError):s.step(np.zeros(49),[],end_sample=512)
        with self.assertRaises(ValueError):acoustic_matrix(np.zeros((49,125)),np.zeros((49,69)))
        self.assertEqual(acoustic_matrix(np.zeros((49,126)),np.zeros((49,69))).shape,(49,195))

    def test_cached_calibration_equals_live_even_at_float_boundaries(self):
        rng=np.random.default_rng(452)
        q=rng.random((80,49),dtype=np.float32)
        q[::3]=np.float32(.9)
        baseline=rng.random(q.shape)<.03
        for policy in (AttackPolicy(cut=.9),AttackPolicy(use_baseline=False,gap_hops=3),
                       AttackPolicy(baseline_cut=.15,reset_ratio=.25),
                       AttackPolicy(enabled=False)):
            live=DirectAttackSelector(policy);observed=np.zeros(q.shape,bool)
            for frame in range(len(q)):
                got=live.step(q[frame],np.flatnonzero(baseline[frame])+40,
                              end_sample=(frame+1)*256)
                for pitch in got:observed[frame,pitch-40]=True
            np.testing.assert_array_equal(observed,decode_cache(q,baseline,policy))

    def test_fast_matching_equals_hungarian_with_ambiguous_neighbors(self):
        from scripts.onset_event_integrity256 import counts
        from scripts.onset_acoustic_direct256 import mask_events
        rng=np.random.default_rng(196)
        for _ in range(70):
            mask=rng.random((120,49))<.012
            truth=[dict(time=float(t),pitch=int(p)) for t,p in zip(
                rng.uniform(0,.8,80),rng.integers(40,89,80))]
            got=count_cache(mask,*truth_arrays(truth))
            expected=counts(mask_events(mask),truth)
            self.assertEqual(got,tuple(expected[k] for k in ('tp','fp','fn')))

    def test_training_observer_equals_existing_preselection_evidence(self):
        from onset256.acoustic_attack import AcousticEvidenceStream
        from onset256.compiled_inference import accelerated
        from onset256.model import load
        from onset256.runtime import Stream
        from onset256.error_evidence69 import hard_matrix
        class Observer:
            cut=0.;requires_selector_state=True;calls=0
            def predict_correct_probability(self,f,p,k,q,h,s):
                self.x=acoustic_matrix(f,hard_matrix(f,p,k,q,h,s));self.calls+=1
                return np.ones(49,np.float32)
        native,s701=load('model/live/strict_note/NOTE_strict_live.npz',
                         'model/onset256_research_39915038301/s701.joblib')
        observed=Observer();original=Stream(native,accelerated(s701),risk_model=observed)
        new=AcousticEvidenceStream(native,s701)
        rng=np.random.default_rng(457)
        audio=rng.normal(0,.08,256*36).astype(np.float32)
        for start in range(0,len(audio),256):
            pcm=audio[start:start+256]
            original.push(pcm,start_sample=start)
            np.testing.assert_array_equal(new.push(pcm,start_sample=start),observed.x)
        with self.assertRaises(ValueError):new.push(audio[:512],start_sample=len(audio))


if __name__=='__main__':unittest.main()
