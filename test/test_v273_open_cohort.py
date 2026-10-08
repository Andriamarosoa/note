import unittest
import numpy as np
from scripts.extract_v273_open_cohort import ownership
from scripts.yourmt3_exactk_common import assign_onsets
from scripts.extract_v273_harmonic_trajectory import _frames,features_from_spectrogram
from scripts.train_v273_open_cohort import piece_names,numpy_forward


class OpenCohortTest(unittest.TestCase):
    def test_ownership_matches_full_native_assignment(self):
        rng=np.random.default_rng(27405)
        for _ in range(10):
            starts=np.cumsum(rng.integers(2000,5000,10));ends=starts+rng.integers(0,1765,10)
            queries=np.arange(starts[0]-1200,ends[-1]+1200,13)
            _,assigned=assign_onsets(queries,starts,ends)
            for i in range(len(starts)):
                np.testing.assert_array_equal(ownership(starts,ends,i,queries),assigned==i)

    def test_ownership_tie_belongs_to_earlier_group(self):
        starts=np.array([1000,3000]);ends=np.array([1400,3200]);q=np.array([2200])
        self.assertTrue(ownership(starts,ends,0,q)[0])
        self.assertFalse(ownership(starts,ends,1,q)[0])

    def test_legacy_features_preserved(self):
        x=np.random.default_rng(27405).normal(size=18000)
        spec,times=_frames(x,8000)
        old=features_from_spectrogram(spec,times)
        new,states=features_from_spectrogram(spec,times,return_states=True)
        self.assertEqual(old,new);self.assertEqual(states.shape,(42,49));self.assertEqual(len(new),58)
        np.testing.assert_allclose(np.median(states.sum(1)),1.)

    def test_piece_groups_all_players_and_versions(self):
        p=piece_names(['00_BN1-147-Gb_comp.jams','04_BN1-147-Gb_solo.jams','01_BN2-131-B_comp.jams'])
        self.assertEqual(p[0],p[1]);self.assertNotEqual(p[0],p[2])

    def test_independent_forward_matches_tensorflow(self):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(27405)
        x=np.random.default_rng(27405).normal(size=(31,8)).astype(np.float32)
        model=tf.keras.Sequential([tf.keras.layers.Input(shape=(8,)),
            tf.keras.layers.Dense(128,activation='gelu'),tf.keras.layers.Dropout(.1),
            tf.keras.layers.Dense(64,activation='gelu'),tf.keras.layers.Dropout(.1),tf.keras.layers.Dense(7)])
        prob=tf.nn.softmax(model(x,training=False)).numpy()
        replay=numpy_forward(x,model.get_weights())
        np.testing.assert_allclose(prob,replay,atol=1e-6)
        np.testing.assert_array_equal(prob.argmax(1),replay.argmax(1))


if __name__=='__main__':unittest.main()
