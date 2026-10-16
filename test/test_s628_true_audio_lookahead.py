"""S628: actual physically arrived future audio, never invented 5ms signal."""
import unittest
import numpy as np

from causal_note.lookahead import (
    LookaheadConfig,LookaheadLiveStream,
    combine_reference_and_future,replay_lookahead,
)
from causal_note.strict_live import StrictLiveStream,HOP,SR,Selection


class TrivialFrozenHead:
    metadata={"selection":dict(onset=.60,active_on=.50)}
    def predict(self,features):
        f=np.asarray(features)
        # NOTE peaks depend on FUTURE pcm, not only a timer delay.
        h=f[:,0]
        p=np.zeros((49,3),np.float32)
        p[0,0]=.80 if h[0]>.012 else .30
        p[0,1]=.9
        k=np.zeros(7,np.float32);k[1]=1.
        return p,k


def key(event):
    return event["type"],event["pitch"],event["decision_sample"],event["available_sample"]


class S628Tests(unittest.TestCase):
    def test_5ms_converts_to_220_actual_pcm_samples(self):
        conf=LookaheadConfig(milliseconds=5)
        self.assertEqual(conf.samples,220)
        self.assertAlmostEqual(conf.samples*1000/SR,4.9886621,places=5)
        with self.assertRaises(ValueError):
            LookaheadConfig(milliseconds=-1)
        with self.assertRaises(ValueError):
            LookaheadConfig(milliseconds=100)

    def test_delay_does_not_change_the_original_note_ownership_or_K(self):
        rng=np.random.default_rng(628)
        audio=rng.normal(0,.02,12345).astype(np.float32)
        # The frozen trivial model is PCM-dependent; shifted phase can
        # change its features. This check uses a PCM-independent head.
        class Fixed:
            metadata={"selection":dict(onset=.60,active_on=.50)}
            def predict(self,x):
                p=np.zeros((49,3),np.float32);p[0,:2]=[.9,.9]
                k=np.zeros(7,np.float32);k[1]=1
                return p,k
        for block in (128,256,333,2048,8192):
            normal=StrictLiveStream(Fixed())
            reference=[]
            for start in range(0,len(audio),block):
                reference+=normal.push(audio[start:start+block],start_sample=start)
            reference+=normal.finish()
            look,stats=replay_lookahead(
                audio,Fixed(),LookaheadConfig(milliseconds=5,mode="delay"),
                block_size=block)
            base=[(e["type"],e["pitch"],e["decision_sample"]) for e in reference]
            shifted=[(e["type"],e["pitch"],e["decision_sample"]) for e in look]
            self.assertEqual(base,shifted)
            for e in look:
                self.assertGreaterEqual(e["available_sample"],e["decision_sample"])
                self.assertAlmostEqual(e["time"],e["available_sample"]/SR)
            self.assertEqual(stats["actual_lookahead_samples"],220)

    def test_future_proposal_is_never_emitted_before_future_pcm_arrives(self):
        x=np.zeros(2048,dtype=np.float32)
        x[HOP:HOP+220]=.8
        stream=LookaheadLiveStream(TrivialFrozenHead(),
                 config=LookaheadConfig(milliseconds=5,mode="max_onset"))
        out=[]
        for start in range(0,len(x),HOP):
            chunk=stream.push(x[start:start+HOP],start_sample=start)
            if start==0:
                self.assertFalse(chunk) # cannot see t+220 yet
            out+=chunk
        out+=stream.finish()
        for e in out:
            self.assertLessEqual(e["decision_sample"]+220,e["available_sample"]
                if e["reason"]!="stream_end" else e["available_sample"])
            self.assertAlmostEqual(e["time"],e["available_sample"]/SR)

    def test_no_future_perturbation_changes_an_earlier_output(self):
        x=np.random.default_rng(10).normal(0,.04,15*HOP).astype(np.float32)
        y=x.copy()
        cutoff=9*HOP
        y[cutoff:]=.9
        for mode in ("delay","average","confirm","max_onset","future"):
            streams=[]
            for a in (x,y):
                s=LookaheadLiveStream(TrivialFrozenHead(),
                       config=LookaheadConfig(milliseconds=5,mode=mode))
                ev=[]
                for start in range(0,len(a),128):
                    ev+=s.push(a[start:start+128],start_sample=start)
                ev+=s.finish()
                streams.append([key(e) for e in ev
                                if e["available_sample"]<=cutoff])
            self.assertEqual(streams[0],streams[1],mode)

    def test_modes_do_not_mutate_original_head_probabilities(self):
        p=np.full((49,3),.3,np.float32)
        ahead=np.full((49,3),.8,np.float32)
        p[:,-1]=.5
        for mode in ("delay","average","confirm","max_onset","future"):
            out=combine_reference_and_future(p,ahead,mode)
            np.testing.assert_array_equal(p[:,2],out[:,2])
            self.assertTrue(np.all((out>=0)&(out<=1)))
        np.testing.assert_array_equal(p,combine_reference_and_future(p,ahead,"delay"))

    def test_large_unbounded_microphone_callback_splitting(self):
        audio=np.random.default_rng(620).normal(0,.02,12403).astype(np.float32)
        events1,_=replay_lookahead(audio,TrivialFrozenHead(),
             LookaheadConfig(milliseconds=5,mode="average"),block_size=256)
        events2,_=replay_lookahead(audio,TrivialFrozenHead(),
             LookaheadConfig(milliseconds=5,mode="average"),block_size=8000)
        # Both use same actual PCM and no future audio. Capture callback
        # changes physical event time, not the nominal acoustic decision.
        a=[(e["type"],e["pitch"],e["decision_sample"]) for e in events1]
        b=[(e["type"],e["pitch"],e["decision_sample"]) for e in events2]
        self.assertEqual(a,b)


if __name__=="__main__":
    unittest.main()
