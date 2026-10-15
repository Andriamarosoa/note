from __future__ import annotations
import unittest
import numpy as np
from causal_note.strict_live import HOP,Selection,CausalSelection
from causal_note.s630_phase_bridge import CausalPhaseBridge,PhasePolicy


class PastOnsetPhaseTests(unittest.TestCase):
    def test_delayed_sustain_uses_past_not_future(self):
        config=Selection(onset=.6,active_on=.5)
        bridge=CausalPhaseBridge(config,PhasePolicy(5,.6,.5))
        orig=CausalSelection(config)
        k=np.eye(7)[1].astype(np.float32)
        p=np.zeros((49,3),np.float32)
        p[12,0]=.9;p[12,1]=.1
        self.assertEqual(bridge.step(p,k,HOP),[])
        self.assertEqual(orig.step(p,k,HOP,HOP),[])
        p[12,0]=.25;p[12,1]=.85
        got=bridge.step(p,k,HOP*2)
        self.assertEqual([(e["type"],e["pitch"]) for e in got],[("note_on",52)])
        self.assertEqual(orig.step(p,k,HOP*2,HOP*2),[])
    def test_past_only_mutation(self):
        rng=np.random.default_rng(630)
        p=rng.random((75,49,3)).astype(np.float32)
        k=np.broadcast_to(np.eye(7)[1],(75,7))
        def run(arr):
            gate=CausalPhaseBridge(Selection(),PhasePolicy(7,.75,.5))
            events=[]
            for i in range(len(arr)):
                events.extend(gate.step(arr[i],k[i],(i+1)*HOP))
            return [v for v in events if v["decision_sample"]<=40*HOP]
        altered=p.copy();altered[40:]*=.01
        self.assertEqual(run(p),run(altered))
    def test_memory_bounded(self):
        v=CausalPhaseBridge(Selection(),PhasePolicy(3,.7,.5))
        p=np.zeros((49,3),np.float32)
        k=np.eye(7)[1]
        for i in range(20):v.step(p,k,(i+1)*HOP)
        self.assertEqual(len(v.history),3)


if __name__=="__main__":unittest.main()
