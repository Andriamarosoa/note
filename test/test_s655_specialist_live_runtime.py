"""S655: exact native S614 causal streaming/batched specialist parity."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,feature_track,load_heads
from causal_note.s654_pitch_risk_selector import decode_with_specialist
from causal_note.s655_specialist_live_runtime import S655CausalRiskStream
from scripts.s616_live_dev_ablation import MODEL
from scripts.s638_note_midi_same_guitarset import S614_SHA

class Birth:
    def predict_proba(self,x):
        q=np.where(x[:,0]>.10,.98,.01)
        return np.column_stack((1-q,q))

class Risk:
    def __init__(self,p):self.p=p
    def predict_proba(self,x):
        arr=np.full(len(x),self.p,dtype=float)
        return np.column_stack((1-arr,arr))

class LiveRiskTests(unittest.TestCase):
    def run_once(self,p):
        native=load_heads(MODEL,expected_sha256=S614_SHA)
        rng=np.random.default_rng(655)
        pcm=rng.normal(0,.045,HOP*40).astype(np.float32)
        ends,x=feature_track(pcm)
        proba,k=native.predict(x)
        row=dict(ends=ends,x=x,p=proba,k=k,n=len(pcm),
                 selection=dict(native.metadata["selection"]))
        batch,stats=decode_with_specialist(row,Birth(),Risk(p),.5)
        stream=S655CausalRiskStream(native,Birth(),Risk(p),risk_threshold=.5)
        output=[]
        for start in range(0,len(pcm),HOP):
            output.extend(stream.push(pcm[start:start+HOP],start_sample=start))
        output.extend(stream.finish())
        self.assertEqual(batch,output)
        self.assertEqual(stats["risk_rejected"],stream.candidate_vetoes)
        self.assertEqual(stats["risk_accepted_frames"],stream.risk_acceptances)
        self.assertEqual(stats["actual_new_notes"],stream.assisted_noteons)
        self.assertEqual(sum(e["type"]=="note_on" for e in output),
                         sum(e["type"]=="note_off" for e in output))
        self.assertTrue(all(e["decision_sample"]<=e["available_sample"]
                            for e in output))

    def test_risk_accept_stream_parity(self):self.run_once(.99)
    def test_risk_reject_stream_parity(self):self.run_once(.01)

    def test_future_audio_mutation_never_changes_output_prefix(self):
        native=load_heads(MODEL,expected_sha256=S614_SHA)
        rng=np.random.default_rng(655)
        pcm=rng.normal(0,.05,HOP*36).astype(np.float32)
        future=pcm.copy()
        future[20*HOP:]=rng.normal(0,.2,len(future)-20*HOP)
        def play(audio):
            stream=S655CausalRiskStream(native,Birth(),Risk(.99),.5)
            output=[]
            for start in range(0,len(audio),HOP):
                output.extend(stream.push(audio[start:start+HOP],start_sample=start))
            return [e for e in output if e["available_sample"]<=20*HOP]
        self.assertEqual(play(pcm),play(future))

if __name__=="__main__":unittest.main()
