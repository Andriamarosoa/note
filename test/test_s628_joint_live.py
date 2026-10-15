"""S628 neutral forward parity + mutation of future audio."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from causal_note.s627_joint_heads import SCHEMA,INPUT_DIM,LATENT,Fusion
from causal_note.s628_joint_live import S628JointLiveStream
from causal_note.strict_live import NativeLiveHeads,StrictLiveStream,Selection,SR,HOP


def fake_joint(path):
    # Tiny synthetic export; no trained weights, for deterministic CI only.
    arrays=dict(schema=np.asarray(SCHEMA),
                metadata=np.asarray(json.dumps(dict(architecture="joint_new_pitches_Kevent"))),
                mean=np.zeros(INPUT_DIM,np.float32),
                scale=np.ones(INPUT_DIM,np.float32),
                encoder_w0=np.zeros((INPUT_DIM,64),np.float32),
                encoder_b0=np.zeros(64,np.float32),
                encoder_w1=np.zeros((64,LATENT),np.float32),
                encoder_b1=np.zeros(LATENT,np.float32),
                k_w=np.zeros((2*LATENT,7),np.float32),
                k_b=np.zeros(7,np.float32),
                pitch_w=np.zeros((LATENT+7,1),np.float32),
                pitch_b=np.zeros(1,np.float32))
    np.savez_compressed(path,**arrays)


class StubHeads:
    metadata={"selection":dict(onset=.6,active_on=.5)}
    def predict(self,feat):
        assert feat.shape==(49,126)
        p=np.zeros((49,3),np.float32)
        p[10,0]=.7;p[10,1]=.8
        k=np.eye(7,dtype=np.float32)[1]
        return p,k


class StreamTests(unittest.TestCase):
    def test_passthrough_matches_original_S614_on_same_audio(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"synthetic.npz"
            fake_joint(path)
            o=StrictLiveStream(StubHeads())
            j=S628JointLiveStream(StubHeads(),path,None,Fusion())
            rng=np.random.default_rng(628)
            audio=rng.standard_normal(256*70).astype(np.float32)*.02
            expected=[];actual=[]
            for start in range(0,len(audio),256):
                expected.extend(o.push(audio[start:start+256],start_sample=start))
                actual.extend(j.push(audio[start:start+256],start_sample=start))
            expected.extend(o.finish());actual.extend(j.finish())
            self.assertEqual(expected,actual)
            self.assertEqual(j.finish(),[])
    def test_future_mutation_does_not_change_past_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"synthetic.npz"
            fake_joint(path)
            audio=np.random.default_rng(628).random(256*60).astype(np.float32)*.04
            altered=audio.copy();altered[256*30:]*=-1
            def play(a):
                j=S628JointLiveStream(StubHeads(),path,None,Fusion(.4,.2,.25))
                out=[]
                for start in range(0,len(a),256):
                    out.extend(j.push(a[start:start+256],start_sample=start))
                return [x for x in out if x["decision_sample"]<=256*30]
            self.assertEqual(play(audio),play(altered))


if __name__=="__main__":unittest.main()
