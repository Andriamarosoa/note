"""S632: parity of batch risk evaluation and actual causal PCM streaming."""
from __future__ import annotations
import hashlib
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from causal_note.strict_live import HOP,Selection,StrictLiveStream
from causal_note.s632_streaming_runtime import StreamingCostSelector,S632Live
from causal_note.s632_risk_policy import (
    RiskPolicy,causal_episodes,predicted_costs,apply_policy)


class ConstantCost:
    def __init__(self,value):
        self.value=float(value)
    def predict(self,x):
        return np.full(len(x),self.value,dtype=np.float64)


class StubHeads:
    metadata={"selection":{}}
    def predict(self,features):
        p=np.zeros((49,3),np.float32)
        p[10,0]=.31;p[10,1]=.58
        k=np.eye(7,dtype=np.float32)[1]
        return p,k


class StreamingRiskTests(unittest.TestCase):
    def test_original_baseline_plus_causal_dynamic_parity(self):
        n=55
        ends=np.arange(1,n+1)*HOP
        p=np.zeros((n,49,3),np.float32)
        p[:25,10,0]=.3;p[:25,10,1]=.5
        p[8:16,16,0]=.8;p[8:16,16,1]=.9
        p[26:44,10,0]=.3;p[26:44,10,1]=.6
        k=np.broadcast_to(np.eye(7,dtype=np.float32)[1],(n,7))
        x=np.zeros((n,49,126),np.float32)
        row=dict(ends=ends,p=p,k=k,x=x,n=n*HOP)
        _,frames=causal_episodes(row)
        models=[ConstantCost(.12),ConstantCost(.95),ConstantCost(1.3)]
        fixed_costs=predicted_costs(models,frames)
        for policy in [RiskPolicy(max_emit_cost=.45),
                       RiskPolicy(mode="replace",max_emit_cost=.45),
                       RiskPolicy(extra_wait_cost=-.1,max_emit_cost=.45)]:
            expected,statistics=apply_policy(row,fixed_costs,Selection(),policy)
            live=StreamingCostSelector(models,Selection(),policy)
            actual=[]
            for end,a,b,c in zip(ends,p,k,x):
                actual.extend(live.step(a,b,c,int(end),int(end)))
            actual.extend(live.finish(n*HOP))
            self.assertEqual(expected,actual)
            self.assertEqual(len(statistics["delays_ms"]),len(live.latencies))

    def test_verified_weight_loading_and_original_passthrough(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[];hashes=[]
            for i,cost in enumerate((2.8,.01,1.)):
                path=Path(folder)/f"cost{i}.joblib"
                joblib.dump(ConstantCost(cost),path)
                paths.append(path)
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            reference=StrictLiveStream(StubHeads())
            candidate=S632Live(StubHeads(),paths,hashes,RiskPolicy())
            audio=np.random.default_rng(632).normal(0,.02,256*45).astype(np.float32)
            a=[];b=[]
            for start in range(0,len(audio),256):
                block=audio[start:start+256]
                a.extend(reference.push(block,start_sample=start))
                b.extend(candidate.push(block,start_sample=start))
            a.extend(reference.finish());b.extend(candidate.finish())
            self.assertEqual(a,b)
            self.assertEqual(candidate.finish(),[])
            with self.assertRaises(ValueError):
                S632Live(StubHeads(),paths,["0"*64]*3,RiskPolicy())

    def test_mutated_future_does_not_change_prior_emissions(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[];hashes=[]
            for i,cost in enumerate((.16,.9,1.)):
                path=Path(folder)/f"model{i}.joblib"
                joblib.dump(ConstantCost(cost),path)
                paths.append(path)
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            audio=np.random.default_rng(6320).normal(0,.05,256*60).astype(np.float32)
            after=audio.copy();after[256*30:]*=-1
            def run(signal):
                engine=S632Live(StubHeads(),paths,hashes,RiskPolicy())
                events=[]
                for start in range(0,len(signal),256):
                    events.extend(engine.push(signal[start:start+256],start_sample=start))
                return [e for e in events if e["decision_sample"]<=256*30]
            self.assertEqual(run(audio),run(after))


if __name__=="__main__":
    unittest.main()
