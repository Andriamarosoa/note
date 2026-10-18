"""S713 independent tests: real causal NoteOn/NoteOff emission and accelerated S708 CPU parity."""
import unittest
from collections import deque
import numpy as np
import torch

from causal_note.strict_live import HOP, SR
from causal_note.s708_cardinality_selection import GuidedPolicy
from causal_note.s705_temporal_heads import TemporalHeads, predict_track
from causal_note.s712_s708_live import S708StrictLiveStream, IncrementalS708TemporalHead, latest_s708_acoustic_features, DECISION
from causal_note.s704_joint_heads import acoustic_features
from test.test_s712_s708_live import FrozenStubHeads, NativeTemporalStub


class S713LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def script_stream(self):
        policy=GuidedPolicy(onset=.75, sustain_on=.5,sustain_off=.3,
                            release=.85,release_steps=1,count_confidence=1.1,
                            blend=1., guide_confidence=1.1, weak_onset=.15,
                            weak_sustain=.55,rise=.04,trim=False)
        stream=S708StrictLiveStream(FrozenStubHeads(),NativeTemporalStub(),policy)
        calls=[]
        def inject(features):
            calls.append(features)
            p=np.zeros((49,3),np.float32)
            p[:,2]=.1
            if len(calls)<=2:
                p[52-40]=[.99,.99,.01]
            else:
                p[52-40]=[0.,0.,.99]
            k=np.zeros(7,np.float32)
            k[1]=1.
            return p,k
        stream.temporal.step=inject
        return stream

    def test_noteon_noteoff_from_the_same_emitted_state_in_various_pcm_chunks(self):
        rng=np.random.default_rng(713)
        x=(rng.normal(0,.03,DECISION*5)).astype(np.float32)
        expected=None
        for size in (128,256,333,2048,DECISION*5):
            stream=self.script_stream()
            events=[]
            at=0
            while at<len(x):
                chunk=x[at:at+size]
                events.extend(stream.push(chunk,start_sample=at))
                at+=len(chunk)
            events.extend(stream.finish())
            # Real selector must emit and release the very same pitch.
            pairs=[(e["type"],e["pitch"],e["decision_sample"],e["reason"])
                   for e in events]
            self.assertEqual(pairs,[
                ("note_on",52,DECISION,"joint_causal_attack"),
                ("note_off",52,DECISION*3,"joint_causal_release")])
            for event in events:
                self.assertGreaterEqual(event["available_sample"],
                                        event["decision_sample"])
                self.assertAlmostEqual(event["time"],
                                       event["available_sample"]/SR,places=9)
            self.assertEqual(stream.model_calls,5)
            self.assertFalse(stream.selector.active.any())
            if expected is None:
                expected=pairs
            else:
                self.assertEqual(pairs,expected)

    def test_finish_releases_an_active_note_without_duplicating_closures(self):
        stream=self.script_stream()
        signal=np.zeros(DECISION,dtype=np.float32)
        events=stream.push(signal,start_sample=0)
        self.assertEqual([e["type"] for e in events],["note_on"])
        end=stream.finish(reason="shutdown")
        self.assertEqual(len(end),1)
        self.assertEqual(end[0]["type"],"note_off")
        self.assertEqual(end[0]["pitch"],52)
        self.assertEqual(end[0]["reason"],"shutdown")
        self.assertEqual(stream.finish(),[])

    def test_latest_only_acoustic_matches_frozen_full_65_history(self):
        rng=np.random.default_rng(713)
        f=deque(maxlen=65);p=deque(maxlen=65);k=deque(maxlen=65)
        for index in range(82):
            f.append(rng.random((49,126),dtype=np.float32))
            p.append(rng.random((49,3),dtype=np.float32))
            k.append(rng.random(7,dtype=np.float32))
            want=acoustic_features(np.stack(f),np.stack(p),np.stack(k))[-1]
            got=latest_s708_acoustic_features(f,p,k)
            np.testing.assert_allclose(got,want,atol=1e-6,rtol=1e-6)

    def test_fast_temporal_head_matches_offline_reference_framewise(self):
        torch.manual_seed(713)
        rng=np.random.default_rng(713)
        model=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32),width=12).eval()
        data=(rng.standard_normal((52,49,288))*.25).astype(np.float32)
        with torch.inference_mode():
            p_ref,k_ref=predict_track(model,data,chunk=17)
            fast=IncrementalS708TemporalHead(model)
            p_fast,k_fast=zip(*(fast.step(x) for x in data))
        np.testing.assert_allclose(np.stack(p_fast),p_ref,atol=3e-6,rtol=3e-6)
        np.testing.assert_allclose(np.stack(k_fast),k_ref,atol=3e-6,rtol=3e-6)
        self.assertEqual(fast.steps,len(data))

    def test_fast_temporal_head_is_strictly_prefix_causal(self):
        torch.manual_seed(714)
        rng=np.random.default_rng(714)
        model=TemporalHeads(np.zeros(288,np.float32),np.ones(288,np.float32),width=8).eval()
        data=(rng.standard_normal((40,49,288))*.25).astype(np.float32)
        alternate=data.copy();alternate[20:]*=-1
        left=IncrementalS708TemporalHead(model)
        right=IncrementalS708TemporalHead(model)
        for a,b in zip(data,alternate):
            pa,ka=left.step(a);pb,kb=right.step(b)
            if left.steps<=20:
                np.testing.assert_array_equal(pa,pb)
                np.testing.assert_array_equal(ka,kb)


if __name__=="__main__":
    unittest.main()
