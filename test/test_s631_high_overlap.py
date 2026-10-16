"""S631 immutable training data overlay unit/regression checks (no downloads)."""
import unittest
import numpy as np

from causal_note.strict_live import SR, feature_track
from causal_note.s627_joint_heads import targets
from scripts.s631_high_overlap_augmentation import (
    choose_pairs, mix_pair, shifted_labels, candidates, CLIP_SECONDS,
)


def specs():
    first=dict(name="train_background",composition="train_A",corpus="GuitarSet",
               split="train",seconds=6.,
               notes=[dict(time=1.5,duration=.8,pitch=59,track="A")])
    second=dict(name="train_high",composition="train_B",corpus="URMP",
                split="train",seconds=6.,
                notes=[dict(time=2.,duration=.5,pitch=83,track="B")])
    return first,second


class HighOverlapTests(unittest.TestCase):
    def test_mixing_real_waveforms_and_true_high_attack_above_sustain(self):
        first,second=specs()
        pairs=choose_pairs([first,second],1,np.random.default_rng(631))
        pair=pairs[0]
        self.assertEqual(pair["high"]["pitch"],83)
        self.assertEqual(pair["interval"],24)
        self.assertNotEqual(pair["first"]["composition"],pair["second"]["composition"])
        a=np.full(int(6*SR),.10,np.float32)
        b=np.full(int(6*SR),.20,np.float32)
        mixed,notes=mix_pair(pair,a,b)
        self.assertEqual(mixed.dtype,np.float32)
        self.assertEqual(len(mixed),int(round(CLIP_SECONDS*SR)))
        self.assertAlmostEqual(float(mixed[SR]),.065+.20*pair["secondary_gain"],places=5)
        self.assertAlmostEqual(float(mixed[0]),0.)
        upper=next(n for n in notes if n["pitch"]==83)
        lower=next(n for n in notes if n["pitch"]==59)
        self.assertGreater(upper["time"],lower["time"])
        self.assertLess(upper["time"],lower["time"]+lower["duration"])
        self.assertEqual(upper["track"],"high_overlay")
        self.assertTrue(np.max(np.abs(mixed))<1.)

    def test_superposed_tonal_pcm_creates_distinct_causal_pitch_targets(self):
        first,second=specs()
        pair=choose_pairs([first,second],1,np.random.default_rng(631))[0]
        t=np.arange(int(6*SR))/SR
        a=(.13*np.sin(2*np.pi*246.94*t)).astype(np.float32)
        b=(.12*np.sin(2*np.pi*987.77*t)).astype(np.float32)
        mixed,notes=mix_pair(pair,a,b)
        ends,features=feature_track(mixed)
        self.assertEqual(features.shape[1:],(49,126))
        y,k=targets(ends,notes)
        attack=round((.6+pair["gap_s"])*SR)
        frame=int(np.searchsorted(ends,attack,side="left"))
        self.assertEqual(y[frame,83-40],1.)
        self.assertTrue((k>=1).any())
        self.assertEqual(y[frame,59-40],0.)  # old pitch sustains; no new attack

    def test_do_not_treat_sustain_as_new_high_note(self):
        old=[dict(time=-.4,duration=1.7,pitch=83)]
        shifted=shifted_labels(old,0,CLIP_SECONDS,"background")
        self.assertEqual(shifted,[])  # sound can exist; onset is before crop

    def test_cannot_build_train_mixes_from_missing_high_or_one_group(self):
        a,b=specs()
        b["composition"]=a["composition"];b["corpus"]=a["corpus"]
        with self.assertRaises(ValueError):
            choose_pairs([a,b],1,np.random.default_rng(1))
        b["composition"]="different";b["corpus"]="URMP"
        b["notes"][0]["pitch"]=65
        with self.assertRaises(ValueError):
            choose_pairs([a,b],1,np.random.default_rng(1))

    def test_pair_generation_determinism(self):
        a,b=specs()
        p=choose_pairs([a,b],6,np.random.default_rng(631))
        q=choose_pairs([a,b],6,np.random.default_rng(631))
        self.assertEqual([(v["interval"],v["gap_s"],v["secondary_gain"])
            for v in p],[(v["interval"],v["gap_s"],v["secondary_gain"]) for v in q])
        self.assertTrue(all(v["high"]["pitch"]>=76 for v in p))


if __name__=="__main__":
    unittest.main()
