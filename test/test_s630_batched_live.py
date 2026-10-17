"""S630 frozen MLP/ONNX batched NoteOn must preserve per-hop decisions."""
import unittest
import numpy as np

from causal_note.strict_live import StrictLiveStream,HOP,SR
from causal_note.batched_live import BatchedLiveStream
from causal_note.lookahead import LookaheadConfig


class DummyVectorizedHeads:
    metadata={"selection":dict(onset=.60,active_on=.50)}
    def __init__(self):
        self.calls=0
        self.frames_processed=0
    def predict(self,features):
        self.calls+=1
        f=np.asarray(features)
        batched=f.ndim==3
        if not batched:f=f[None]
        self.frames_processed+=len(f)
        p=np.zeros((len(f),49,3),np.float32)
        # Every sound triggers MIDI attack only based on existing past.
        # Dynamic spectrogram features support original or batched.
        for i,row in enumerate(f):
            p[i,0,:2]=[.90,.85] if row[0,0]>.1 else [.10,.1]
        k=np.zeros((len(f),7),np.float32);k[:,1]=1.
        return (p,k) if batched else (p[0],k[0])


def run_stream(model,pcm,block):
    events=[]
    for i in range(0,len(pcm),block):
        events+=model.push(pcm[i:i+block],start_sample=i)
    events+=model.finish()
    return events


def nominal(events):
    return [(e["type"],e["pitch"],e["decision_sample"]) for e in events]


class S630Tests(unittest.TestCase):
    def test_batched_no_future_is_exact_reference_for_nominal_notes(self):
        pcm=np.random.default_rng(630).normal(0,.06,15321).astype(np.float32)
        for block in (128,256,512,1024,4096,8192):
            original=run_stream(StrictLiveStream(DummyVectorizedHeads()),pcm,block)
            for batch in (2,4,8):
                heads=DummyVectorizedHeads()
                stream=BatchedLiveStream(heads,batch_hops=batch)
                trial=run_stream(stream,pcm,block)
                self.assertEqual(nominal(trial),nominal(original))
                self.assertEqual(heads.frames_processed,len(pcm)//HOP)
                self.assertLessEqual(heads.calls,(len(pcm)//HOP+batch-1)//batch)
                for e in trial:
                    self.assertGreaterEqual(e["available_sample"],e["decision_sample"])
                    self.assertAlmostEqual(e["time"],e["available_sample"]/SR)
                self.assertEqual(stream.stats["frames_selected"],len(pcm)//HOP)
                self.assertEqual(stream.stats["frames_inferred"],len(pcm)//HOP)

    def test_full_future_branch_one_inference_per_batch(self):
        pcm=np.random.default_rng(11).normal(0,.05,256*11).astype(np.float32)
        heads=DummyVectorizedHeads()
        s=BatchedLiveStream(heads,batch_hops=4,
                            lookahead=LookaheadConfig(milliseconds=10,
                                                      mode="max_onset"))
        notes=run_stream(s,pcm,256)
        self.assertEqual(heads.calls,3)
        self.assertEqual(s.stats["frames_inferred"],11)
        self.assertEqual(s.stats["frames_selected"],9)
        self.assertEqual(s.stats["EOF_unconfirmed_frames"],2)
        self.assertEqual(s.actual_lookahead_samples,512)
        self.assertEqual(s.diagnostics()["max_intentional_batch_wait_ms"],3*HOP*1000/SR)
        for e in notes:
            if e["reason"]!="stream_end" and not e["reason"].startswith("lookahead_eof_unconfirmed_"):
                self.assertGreaterEqual(e["available_sample"]-e["decision_sample"],512)

    def test_no_predicted_future_before_callback_received(self):
        pcm=np.random.default_rng(12).normal(0,.05,32*HOP).astype(np.float32)
        altered=pcm.copy()
        at=16*HOP
        altered[at:]=1
        for mode in (None,"delay","average","confirm","max_onset","future"):
            seen=[]
            for data in (pcm,altered):
                cfg=LookaheadConfig(milliseconds=10,mode=mode) if mode else None
                s=BatchedLiveStream(DummyVectorizedHeads(),batch_hops=4,lookahead=cfg)
                events=run_stream(s,data,256)
                seen.append([(e["type"],e["pitch"],e["decision_sample"],
                              e["available_sample"]) for e in events
                             if e["available_sample"]<=at])
            self.assertEqual(seen[0],seen[1])

    def test_frozen_full_s613_model_batched_matches_original_within_fp_tolerance(self):
        from causal_note.strict_live import load_heads
        from scripts.live_strict_notes import DEFAULT_MODEL
        # Test uses same original freeze, fixture PCM no truth required.
        model=load_heads(DEFAULT_MODEL)
        pcm=(.03*np.sin(2*np.pi*82.41*np.arange(4096)/SR)).astype(np.float32)
        source=run_stream(StrictLiveStream(model),pcm,256)
        batched=run_stream(BatchedLiveStream(model,batch_hops=4),pcm,256)
        self.assertEqual(nominal(source),nominal(batched))
        self.assertEqual(len(batched),len(source))

    def test_invalid_batch_rejected(self):
        for value in (0,-1,33,1.5,"4"):
            with self.assertRaises(ValueError):
                BatchedLiveStream(DummyVectorizedHeads(),batch_hops=value)


if __name__=="__main__":
    unittest.main()
