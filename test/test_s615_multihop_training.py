"""S615 synthetic smoke: no held-out labels in train and 256 ABI unchanged.

Synthetic fixtures verify the code only, never estimate real musical F1.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from causal_note.strict_live import HOP, feature_track, NativeLiveHeads, StrictLiveStream
from scripts.s615_prepare_multihop import prepare_one
from scripts.s615_train_multihop import source_data, fit


class S615Tests(unittest.TestCase):
    def make_source(self, root, split, number, *, seconds=1.1):
        ident=f"fixture_{split}_{number}"
        t=np.arange(int(seconds*44100),dtype=np.float64)/44100
        base=82.41 * (2 ** (number/12))
        y=(.09*np.sin(2*np.pi*base*t)*(t>=.20)
           +.065*np.sin(2*np.pi*(base*2)*t)*(t>=.48))
        y=np.asarray(y,dtype=np.float32)
        spec=dict(id=ident,name=ident,split=split,corpus="GuitarSet",
                  composition=f"{split}_song{number}")
        cache=root/"cache"
        cache.mkdir(parents=True,exist_ok=True)
        np.save(cache/f"{ident}.audio.npy",y,allow_pickle=False)
        notes=[dict(pitch=40+number,time=.20,duration=.65),
               dict(pitch=52+number,time=.48,duration=.4)]
        proof=dict(source=spec,annotations_read=split!="test",
                   prepared_samples=len(y))
        if split!="test":
            proof["labels"]=dict(notes=notes,sha256=["fixture_annotation_"+ident])
        (cache/f"{ident}.json").write_text(json.dumps(proof))
        return spec

    def test_feature_track_256_matches_reference_bitwise(self):
        x=np.random.default_rng(255).normal(0,.03,4096).astype(np.float32)
        a,b=feature_track(x)
        a2,b2=feature_track(x,hop=256)
        np.testing.assert_array_equal(a,a2)
        np.testing.assert_array_equal(b,b2)
        self.assertEqual(b.shape[1:],(49,126))

    def test_source_cache_and_refusal_to_read_test_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=self.make_source(root,"train",0)
            test=self.make_source(root,"test",1)
            for hop in (128,256,512):
                receipt=prepare_one(root,source,hop)
                self.assertEqual(receipt["hop"],hop)
                self.assertTrue(prepare_one(root,source,hop)["reused"])
                ends, feats, y,k,notes,n=source_data(root,source,hop)
                self.assertEqual(feats.shape,(n//hop,49,126))
                self.assertEqual(ends[0],hop)
                self.assertGreater(y[...,0].sum(),0)
                self.assertGreater(int(k.max()),0)
            with self.assertRaises(ValueError):
                prepare_one(root,test,256)

    def test_fit_one_epoch_multiple_hops_and_shared_candidate(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch training dependency not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            sources=( [self.make_source(root,"train",i) for i in (0,2,4)] +
                      [self.make_source(root,"dev",i) for i in (1,3)] +
                      [self.make_source(root,"test",5)] )
            (root/"protocol.json").write_text(json.dumps(dict(
                schema="S613_native_causal_harmonic_heads_v1",
                sample_rate=44100,sources=sources)))
            for s in sources:
                if s["split"]!="test":
                    for h in (128,256,512):
                        prepare_one(root,s,h)
            for mode in ("hop128","hop256","hop512","shared"):
                dst=root/"experiments"/mode
                report=fit(root,dst,mode,count_epochs=1,pitch_epochs=1,
                           max_pitch_rows=1600)
                self.assertEqual(report["status"],"EXPERIMENTAL_DEV_ONLY_NOT_PROMOTED")
                head=NativeLiveHeads(dst/"NOTE_multihop_candidate.npz")
                self.assertEqual(head.metadata["trained_hops"],
                                 [128,256,512] if mode=="shared" else [int(mode[3:])])
                self.assertNotIn("test", [row["split"] for row in
                    json.loads((dst/"training_sources.json").read_text())])
                x=np.zeros((49,126),np.float32)
                p,k=head.predict(x)
                self.assertEqual(p.shape,(49,3))
                self.assertEqual(k.shape,(7,))
                h=head.metadata["trained_hops"][0]
                stream=StrictLiveStream(head,hop=h)
                self.assertEqual(stream.hop,h)
                self.assertEqual(stream.selector.config.onset,
                                 head.metadata["selection_by_hop"][str(h)]["onset"])


if __name__ == "__main__":
    unittest.main()
