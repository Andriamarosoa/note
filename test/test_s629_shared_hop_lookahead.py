"""S629 shared single-pass bounded future NOTE evidence regression tests."""
import unittest
import numpy as np
from causal_note.strict_live import SR,HOP,StrictLiveStream
from causal_note.lookahead import (
    LookaheadConfig,SharedHopLookaheadLiveStream,LookaheadLiveStream,
)


class FixedFrozenHeads:
    metadata={"selection":dict(onset=.60,active_on=.50)}
    def __init__(self):
        self.calls=0
    def predict(self,f):
        self.calls+=1
        p=np.zeros((49,3),np.float32)
        p[0,:2]=[.9,.9]
        k=np.zeros(7,np.float32);k[1]=1.
        return p,k


class S629Tests(unittest.TestCase):
    def run_pcm(self,stream,pcm,block):
        ev=[]
        for i in range(0,len(pcm),block):
            ev+=stream.push(pcm[i:i+block],start_sample=i)
        ev+=stream.finish()
        return ev

    def test_exact_actual_quantization_reported(self):
        for requested,expected_samples in [(2.5,256),(5.,256),(10.,512),(15.,768)]:
            s=SharedHopLookaheadLiveStream(
                FixedFrozenHeads(),config=LookaheadConfig(
                    milliseconds=requested,mode="average"))
            self.assertEqual(s.actual_samples,expected_samples)
            self.assertGreaterEqual(s.actual_samples,s.requested_samples)
            self.assertLess(s.actual_samples-s.requested_samples,HOP)
            self.assertEqual(s.diagnostics()["engine"],"shared_single_inference")
        self.assertAlmostEqual(512*1000/SR,11.60997732,places=5)

    def test_only_one_frozen_predict_call_per_hop(self):
        x=np.zeros(12*HOP,np.float32)
        h=FixedFrozenHeads()
        s=SharedHopLookaheadLiveStream(
            h,config=LookaheadConfig(milliseconds=10,mode="average"))
        ev=self.run_pcm(s,x,HOP)
        self.assertEqual(h.calls,12)
        self.assertEqual(s.stats["native_reference_frames"],12)
        self.assertEqual(s.stats["lookahead_confirmed_frames"],10)
        self.assertEqual(s.stats["EOF_baseline_unconfirmed_frames"],2)
        self.assertEqual(len([x for x in ev if x["type"]=="note_on"]),1)

    def test_zero_future_original_decision_is_preserved_in_delay_control(self):
        audio=np.random.default_rng(3).normal(0,.04,12345).astype(np.float32)
        for block in (128,256,512,2048,8192):
            h0=FixedFrozenHeads()
            base=self.run_pcm(StrictLiveStream(h0),audio,block)
            h1=FixedFrozenHeads()
            s=SharedHopLookaheadLiveStream(
                h1,config=LookaheadConfig(milliseconds=10,mode="delay"))
            changed=self.run_pcm(s,audio,block)
            def key(ev):
                return [(e["type"],e["pitch"],e["decision_sample"]) for e in ev]
            self.assertEqual(key(base),key(changed))
            self.assertEqual(h0.calls,h1.calls)
            for event in changed:
                self.assertGreaterEqual(event["available_sample"],event["decision_sample"])
                self.assertAlmostEqual(event["time"],event["available_sample"]/SR)
                if (not event["reason"].startswith("lookahead_eof_unconfirmed_")
                        and event["reason"]!="stream_end"):
                    self.assertGreaterEqual(event["available_sample"]-event["decision_sample"],512)

    def test_future_pcm_changes_cannot_affect_any_emitted_event_before_capture(self):
        pcm=np.random.default_rng(9).normal(0,.05,30*HOP).astype(np.float32)
        mutated=pcm.copy()
        cutoff=18*HOP
        mutated[cutoff:]=.90
        for mode in ("delay","average","confirm","max_onset","future"):
            out=[]
            for data in (pcm,mutated):
                stream=SharedHopLookaheadLiveStream(
                    FixedFrozenHeads(),config=LookaheadConfig(
                        milliseconds=10,mode=mode))
                result=self.run_pcm(stream,data,256)
                out.append([(e["type"],e["pitch"],e["decision_sample"],
                             e["available_sample"]) for e in result
                            if e["available_sample"]<=cutoff])
            self.assertEqual(out[0],out[1])

    def test_block_size_only_changes_delivery_time_not_musical_decisions(self):
        pcm=np.random.default_rng(63).normal(0,.05,12000).astype(np.float32)
        for mode in ("average","max_onset","future"):
            def run(block):
                return self.run_pcm(SharedHopLookaheadLiveStream(
                    FixedFrozenHeads(),config=LookaheadConfig(
                    milliseconds=10,mode=mode)),pcm,block)
            first=run(256);second=run(8192)
            def nominal(ev):
                return [(e["type"],e["pitch"],e["decision_sample"]) for e in ev]
            self.assertEqual(nominal(first),nominal(second),mode)


if __name__=="__main__":
    unittest.main()
