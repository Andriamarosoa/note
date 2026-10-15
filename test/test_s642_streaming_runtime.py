"""Verify S642 0..3 hop streamer equals causal batch decisions."""
import unittest
import numpy as np
from causal_note.strict_live import HOP,feature_track
from causal_note.s642_dual_learned import S642Policy,evaluated_policy
from causal_note.s642_streaming_runtime import S642CausalLive,S642EventScheduler

class Heads:
    metadata={"selection":{"onset":.6,"active_on":.5,"release_frames":3,"refractory_frames":10}}
    def __init__(self):self.t=0
    def predict(self,x):
        p=np.zeros((49,3),np.float32)
        if self.t<10:p[12]=[.3,.78,0.]
        self.t+=1
        return p,np.eye(7,dtype=np.float32)[1]

class Attack:
    def predict_proba(self,x):return np.column_stack([np.zeros(len(x)),np.ones(len(x))])

class Delay:
    def __init__(self,n=2):self.n=n
    def predict(self,x):return np.full(len(x),self.n,dtype=int)

class S642LiveParityTests(unittest.TestCase):
    def test_true_4_policy_parity_and_no_backdating(self):
        audio=np.zeros(25*HOP,np.float32)
        ends,x=feature_track(audio)
        baseline=Heads()
        p=[];k=[]
        for frame in x:
            out,count=baseline.predict(frame);p.append(out);k.append(count)
        row=dict(ends=ends,x=x,p=np.stack(p),k=np.stack(k),
                 n=len(audio),selection=Heads.metadata["selection"])
        for a in (False,True):
            for d in (False,True):
                policy=S642Policy(use_candidate=a,use_delay=d)
                expected,_=evaluated_policy(row,Attack(),Delay(),policy)
                stream=S642CausalLive(Heads(),Attack(),Delay(),policy)
                emitted=[]
                for start in range(0,len(audio),HOP):
                    emitted.extend(stream.push(audio[start:start+HOP],start_sample=start))
                emitted.extend(stream.finish())
                self.assertEqual(expected,emitted)
                self.assertTrue(all(e["decision_sample"]<=e["available_sample"]
                                    for e in emitted))
    def test_future_mutation_no_effect_to_past(self):
        original=np.zeros(25*HOP,np.float32)
        altered=original.copy()
        altered[15*HOP:]=.04
        def play(audio):
            stream=S642CausalLive(Heads(),Attack(),Delay(3),
                                  S642Policy(use_candidate=True,use_delay=True))
            emitted=[]
            for start in range(0,len(audio),HOP):
                emitted.extend(stream.push(audio[start:start+HOP],start_sample=start))
            return [e for e in emitted if e["available_sample"]<=15*HOP]
        self.assertEqual(play(original),play(altered))
    def test_illegal_hop_prediction_raises(self):
        s=S642EventScheduler(Delay(5),S642Policy(use_delay=True))
        with self.assertRaises(ValueError):
            s.add([dict(type="note_on",pitch=52,decision_sample=HOP,
                        available_sample=HOP,time=HOP/44100.)],
                  np.zeros((49,72),np.float32))
if __name__=="__main__":unittest.main()
